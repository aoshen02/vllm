# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from vllm import _custom_ops as ops
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )


def _rename(arg: Any, old: str, new: str) -> Any:
    return arg.replace(old, new) if isinstance(arg, str) else arg


def quantize_on_load(
    layer: torch.nn.Module,
    name: str,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    *scale_names: str,
) -> None:
    """Make `layer.<name>`, created in its quantized checkpoint layout, load
    high-precision checkpoint tensors: each arriving tensor is quantized by
    `quantize` into `(values, *scales)`, loaded into `layer.<name>` and
    `layer.<scale_names>` by their own loaders, as a quantized checkpoint is."""
    load = getattr(getattr(layer, name), "weight_loader", default_weight_loader)
    layer.quantizes_on_load = True

    def loader(param, loaded_weight, *args, **kwargs):
        # A layerwise reload counts the elements each loader copies, into meta
        # parameters: count only the values, the scales' loaders count theirs.
        with _disable_current_modes():
            device = current_platform.device_type if param.is_meta else param.device
            values, *scales = quantize(loaded_weight.to(device, layer.params_dtype))
            for scale_name, scale in zip(scale_names, scales):
                scale_param = getattr(layer, scale_name)
                # the MoE loader picks how to load a scale by its parameter name
                getattr(scale_param, "weight_loader", default_weight_loader)(
                    scale_param,
                    scale,
                    *(_rename(a, name, scale_name) for a in args),
                    **{k: _rename(v, name, scale_name) for k, v in kwargs.items()},
                )
        return load(param, values, *args, **kwargs)

    getattr(layer, name).weight_loader = loader


def _quantize_nvfp4(weight: torch.Tensor, amax: torch.Tensor) -> tuple:
    """`weight` to NVFP4 in the ModelOpt layout, its global scale from `amax`."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        FLOAT4_E2M1_MAX,
    )

    amax = amax.to(weight.device, torch.float32).clamp_min(1e-8)
    global_scale = FLOAT4_E2M1_MAX * torch.finfo(torch.float8_e4m3fn).max / amax
    values, scales = ops.scaled_fp4_quant(
        weight.contiguous(), global_scale, is_sf_swizzled_layout=False
    )
    return values, scales, (1.0 / global_scale).reshape(1)


def _nvfp4_on_load(layer: torch.nn.Module) -> None:
    quantize_on_load(
        layer,
        "w2_weight",
        lambda w: _quantize_nvfp4(w, w.abs().amax()),
        "w2_weight_scale",
        "w2_weight_scale_2",
    )
    # w1 and w3 of an expert share one global scale: the first to arrive
    # waits for the other
    amax = None
    quantize_on_load(
        layer,
        "w13_weight",
        lambda w: _quantize_nvfp4(w, w.abs().amax() if amax is None else amax),
        "w13_weight_scale",
        "w13_weight_scale_2",
    )
    load = layer.w13_weight.weight_loader
    pending: dict[int, tuple[str, torch.Tensor]] = {}

    def w13_loader(param, loaded_weight, weight_name, shard_id, expert_id, *a, **k):
        nonlocal amax
        amax = None
        if layer.moe_config.is_act_and_mul and shard_id in ("w1", "w3"):
            if (other := pending.pop(expert_id, None)) is None:
                pending[expert_id] = (shard_id, loaded_weight.clone())
                return True
            amax = max(other[1].abs().amax(), loaded_weight.abs().amax())
            amax = amax.to(layer.params_dtype)
            load(param, other[1], weight_name, other[0], expert_id, *a, **k)
        return load(param, loaded_weight, weight_name, shard_id, expert_id, *a, **k)

    layer.w13_weight.weight_loader = w13_loader


def checkpoint_method(
    layer: torch.nn.Module, spec: "QuantSpec", default_activation: Any
) -> "QuantizeMethodBase | None":
    """The method of the pre-quantized checkpoint an online scheme stands for,
    loading high-precision weights by quantizing them as they arrive; None if
    none fits, and the online method quantizes them after loading."""
    from compressed_tensors.quantization import QuantizationArgs

    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
        CompressedTensorsW8A8Fp8MoEMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8 import (  # noqa: E501
        CompressedTensorsW8A8Mxfp8MoEMethod,
    )
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
        Fp8MoEMethod,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.model_executor.layers.quantization.utils import quant_utils as qu
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        mxfp4_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    if spec.activation != default_activation or isinstance(layer, ParallelLMHead):
        return None
    weight, block = spec.weight, Fp8Config(weight_block_size=[128, 128])
    quantize = {
        qu.kFp8StaticTensorSym: ops.scaled_fp8_quant,
        qu.kFp8Static128BlockSym: per_block_cast_to_fp8,
        qu.kFp8StaticChannelSym: partial(
            ops.scaled_fp8_quant, use_per_token_if_dynamic=True
        ),
        qu.kMxfp8Static: mxfp8_e4m3_quantize,
        qu.kMxfp4Static: mxfp4_quantize,
    }.get(weight)
    if isinstance(layer, RoutedExperts):
        moe, prefixes = layer.moe_config, ("w13_", "w2_")
        if weight == qu.kFp8StaticTensorSym:
            method = Fp8MoEMethod(Fp8Config(), layer)
        elif weight == qu.kFp8Static128BlockSym:
            method = Fp8MoEMethod(block, layer)
        elif weight == qu.kFp8StaticChannelSym:
            channel = QuantizationArgs(type="float", strategy="channel")
            token = QuantizationArgs(type="float", strategy="token", dynamic=True)
            method = CompressedTensorsW8A8Fp8MoEMethod(channel, token, moe)
        elif weight == qu.kMxfp8Static:
            method = CompressedTensorsW8A8Mxfp8MoEMethod(moe)
        elif weight == qu.kMxfp4Static:
            method = Mxfp4MoEMethod(moe, spec.activation)
        elif weight == qu.kNvfp4Static:
            config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
            config.moe_activation_override = spec.activation
            method = ModelOptNvFp4FusedMoE(config, moe)
        else:
            return None
    elif isinstance(layer, LinearBase):
        prefixes = ("",)
        if weight == qu.kFp8StaticTensorSym:
            method = Fp8LinearMethod(Fp8Config())
        elif weight == qu.kFp8Static128BlockSym:
            method = Fp8LinearMethod(block)
        elif weight == qu.kFp8StaticChannelSym:
            method = ModelOptLinearMethod(
                QuantSpec(weight=qu.kFp8StaticTokenSym, activation=spec.activation),
                CkptCtx(),
            )

            def quantize(w):  # its channel scales are 1-D
                values, scales = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
                return values, scales.flatten()

        elif weight == qu.kMxfp8Static:
            method = ModelOptLinearMethod(
                QuantSpec(weight=qu.kMxfp8Static, activation=qu.kMxfp8Dynamic),
                CkptCtx(),
            )
        else:
            return None
    else:
        return None
    scale = "weight_scale_inv" if weight == qu.kFp8Static128BlockSym else "weight_scale"
    create_weights = method.create_weights

    def create_weights_quantized_on_load(layer, *args, **kwargs):
        create_weights(layer, *args, **kwargs)
        if weight == qu.kNvfp4Static:
            return _nvfp4_on_load(layer)
        for p in prefixes:
            quantize_on_load(layer, f"{p}weight", quantize, f"{p}{scale}")

    method.create_weights = create_weights_quantized_on_load  # type: ignore[method-assign]
    return method
