# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    _quantize_moe_weight_to_nvfp4,
    _quantize_nvfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def _rename(arg: Any, old: str, new: str) -> Any:
    return arg.replace(old, new) if isinstance(arg, str) else arg


def _load_quantized(layer: torch.nn.Module, name: str, *scale_names: str):
    """Load `(values, *scales)` as a checkpoint of that layout is loaded."""
    load = getattr(getattr(layer, name), "weight_loader", default_weight_loader)
    layer.quantizes_on_load = True

    def load_quantized(param, quantized, *args, **kwargs):
        values, *scales = quantized
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

    return load_quantized


def quantize_on_load(
    layer: torch.nn.Module, name: str, quantize: Callable, *scale_names: str
) -> None:
    """Make quantized `layer.<name>` load high-precision tensors via `quantize`."""
    load = _load_quantized(layer, name, *scale_names)

    def loader(param, loaded_weight, *args, **kwargs):
        weight = loaded_weight.to(param.device, layer.params_dtype).contiguous()
        return load(param, quantize(weight), *args, **kwargs)

    getattr(layer, name).weight_loader = loader


def _quantize_fp8_per_channel(weight: torch.Tensor) -> tuple:
    scale = online_fp8._fp8_channel_scale(weight_amax(weight, dim=-1, keepdim=True))
    return online_fp8._fp8_quant_per_channel(weight, scale), scale


def _quantize_int8_per_channel(weight: torch.Tensor) -> tuple:
    vmax = torch.iinfo(torch.int8).max
    amax = weight_amax(weight, dim=-1, keepdim=True).float()
    scale = (amax / vmax).clamp_min(torch.finfo(torch.float32).tiny)
    values = weight.float().div(scale).round().clamp(-vmax, vmax).to(torch.int8)
    return values, scale


def _nvfp4_on_load(layer: torch.nn.Module, method: Any) -> None:
    """w1 and w3 of an expert share one global scale: the first waits."""
    pending: dict[int, tuple[str, torch.Tensor, torch.Event]] = {}

    def loader(load, param, loaded_weight, weight_name, shard_id, expert_id, **kw):
        if layer._map_global_expert_id_to_local_expert_id(expert_id) == -1:
            return False
        weight = loaded_weight.to(param.device, layer.params_dtype, copy=True)
        shards = [(shard_id, weight)]
        if shard_id in ("w1", "w3") and layer.moe_config.is_act_and_mul:
            first = pending.pop(expert_id, None)
            if first is None or first[0] == shard_id:
                copied = torch.Event()
                copied.record()
                pending[expert_id] = (shard_id, weight, copied)
                return True
            first_id, first_weight, copied = first
            # held from another load, which may have run on another stream
            copied.wait()
            first_weight.record_stream(torch.accelerator.current_stream())
            shards.insert(0, (first_id, first_weight))
        stacked = torch.cat([w for _, w in shards])[None]
        values, scales, scale_2 = _quantize_moe_weight_to_nvfp4(stacked)
        n = len(shards)
        for (shard, _), v, s in zip(shards, values[0].chunk(n), scales[0].chunk(n)):
            ret = load(param, (v, s, scale_2), weight_name, shard, expert_id, **kw)
        return ret

    for p in ("w13_", "w2_"):
        names = (f"{p}weight", f"{p}weight_scale", f"{p}weight_scale_2")
        load = _load_quantized(layer, *names)
        getattr(layer, names[0]).weight_loader = partial(loader, load)

    process = method.process_weights_after_loading

    def process_weights_after_loading(layer: torch.nn.Module) -> None:
        if pending:
            experts = sorted(pending)
            pending.clear()
            raise RuntimeError(
                f"{layer.layer_name}: NVFP4 experts {experts} got only one of "
                "w1/w3; both must be loaded in the same update"
            )
        process(layer)

    method.process_weights_after_loading = process_weights_after_loading


def _nvfp4_linear_on_load(layer: torch.nn.Module, method: Any) -> None:
    """The fused shards of a layer (q/k/v, gate/up) share one global scale:
    all but the last wait."""
    load = _load_quantized(layer, "weight", "weight_scale", "weight_scale_2")
    num_shards = len(layer.logical_widths)
    pending: dict[Any, tuple[torch.Tensor, tuple, dict]] = {}

    def loader(param, loaded_weight, *args, **kwargs):
        weight = loaded_weight.to(param.device, layer.params_dtype, copy=True)
        shard_id = args[0] if args else kwargs.get("loaded_shard_id")
        if shard_id is None or num_shards == 1:
            return load(
                param, _quantize_nvfp4(weight, weight_amax(weight)), *args, **kwargs
            )
        pending[shard_id] = (weight, args, kwargs)
        if len(pending) < num_shards:
            return None
        shards = list(pending.values())
        pending.clear()
        amax = torch.stack([weight_amax(w).float() for w, _, _ in shards]).max()
        for w, a, kw in shards:
            ret = load(param, _quantize_nvfp4(w, amax), *a, **kw)
        return ret

    layer.weight.weight_loader = loader
    process = method.process_weights_after_loading

    def process_weights_after_loading(layer: torch.nn.Module) -> None:
        if pending:
            shards = sorted(map(str, pending))
            pending.clear()
            raise RuntimeError(
                f"NVFP4 linear got only shards {shards} of a fused layer; all "
                "of them must be loaded in the same update"
            )
        process(layer)

    method.process_weights_after_loading = process_weights_after_loading


def checkpoint_method(layer: torch.nn.Module, spec: Any, default: Any) -> Any:
    """The pre-quantized checkpoint method that `spec` stands for, if any."""
    from compressed_tensors.quantization import QuantizationArgs

    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 as ct_fp8  # noqa: E501
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as ct_int8  # noqa: E501
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8 as ct_mxfp8  # noqa: E501
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import fp8, modelopt, mxfp4
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        CompressedTensorsW8A16Fp8,
    )
    from vllm.model_executor.layers.quantization.input_hadamard import (
        InputHadamardLinearMethod,
        block_hadamard_rotate,
        hadamard_matrix,
    )
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        select_online_mxfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.quark.quark import QuarkLinearMethod
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )
    from vllm.model_executor.layers.quantization.utils import mxfp4_utils, mxfp8_utils
    from vllm.model_executor.layers.quantization.utils import quant_utils as qu
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    is_moe = isinstance(layer, RoutedExperts)
    moe: Any = getattr(layer, "moe_config", None)
    method: Any  # the checkpoint method of the matching pre-quantized scheme
    quantize: Callable[..., Any]
    weight, ckpt, hadamard = spec.weight, modelopt.CkptCtx(), spec.hadamard_size
    if not (is_moe or isinstance(layer, LinearBase)):
        return None
    is_mxfp4 = weight == qu.kMxfp4Static
    mxfp4_linear = is_mxfp4 and not is_moe
    # PTPC's only override is `activation: null`, weight-only FP8
    fp8_weight_only = (
        weight == qu.kFp8StaticChannelSym and spec.activation is None and not is_moe
    )
    if hadamard is not None and not mxfp4_linear:
        return None
    if spec.activation != default and not (is_mxfp4 or fp8_weight_only):
        return None
    block = weight == qu.kFp8Static128BlockSym
    if weight in (qu.kFp8StaticTensorSym, qu.kFp8Static128BlockSym):
        cfg = fp8.Fp8Config(weight_block_size=[128, 128] if block else None)
        method = fp8.Fp8MoEMethod(cfg, layer) if is_moe else fp8.Fp8LinearMethod(cfg)
        quantize = per_block_cast_to_fp8 if block else ops.scaled_fp8_quant
    elif weight == qu.kFp8StaticChannelSym and is_moe:
        channel = QuantizationArgs(type="float", strategy="channel")
        token = QuantizationArgs(type="float", strategy="token", dynamic=True)
        method = ct_fp8.CompressedTensorsW8A8Fp8MoEMethod(channel, token, moe)
        quantize = _quantize_fp8_per_channel
    elif fp8_weight_only:
        channel = QuantizationArgs(type="float", strategy="channel")
        layer.scheme = CompressedTensorsW8A16Fp8(channel, is_static_input_scheme=False)
        method = CompressedTensorsLinearMethod(None)  # type: ignore[arg-type]
        quantize = _quantize_fp8_per_channel
    elif weight == qu.kFp8StaticChannelSym:
        token_spec = QuantSpec(weight=qu.kFp8StaticTokenSym, activation=spec.activation)
        method = modelopt.ModelOptLinearMethod(token_spec, ckpt)

        def quantize(w):  # its channel scales are 1-D
            values, scales = _quantize_fp8_per_channel(w)
            return values, scales.flatten()

    elif weight == qu.kInt8StaticChannelSym and is_moe:
        channel = QuantizationArgs(type="int", num_bits=8, strategy="channel")
        token = QuantizationArgs(type="int", strategy="token", dynamic=True)
        method = ct_int8.CompressedTensorsW8A8Int8MoEMethod(channel, token, moe)
        quantize = _quantize_int8_per_channel
    elif weight == qu.kMxfp8Static and is_moe:
        method = ct_mxfp8.CompressedTensorsW8A8Mxfp8MoEMethod(moe)
        quantize = mxfp8_utils.mxfp8_e4m3_quantize
    elif weight == qu.kMxfp8Static:
        mx_spec = QuantSpec(weight=qu.kMxfp8Static, activation=qu.kMxfp8Dynamic)
        method = modelopt.ModelOptLinearMethod(mx_spec, ckpt)
        quantize = mxfp8_utils.mxfp8_e4m3_quantize
    elif weight == qu.kMxfp4Static and is_moe:
        method = mxfp4.Mxfp4MoEMethod(moe, spec.activation)
        # the online backend choice, which honours the activation key
        method.mxfp4_backend, method.experts_cls = select_online_mxfp4_moe_backend(
            moe, spec.activation
        )
        quantize = mxfp4_utils.mxfp4_quantize
    elif mxfp4_linear:
        if spec.activation not in QuarkOCP_MX.supported_activation_quant_keys:
            return None
        layer.scheme = QuarkOCP_MX(weight, spec.activation)
        method = QuarkLinearMethod(None)  # type: ignore[arg-type]
        quantize = mxfp4_utils.mxfp4_quantize
        if hadamard is not None:

            def quantize(w):  # the input is rotated by the same matrix
                rotation = hadamard_matrix(hadamard, w.device)
                return mxfp4_utils.mxfp4_quantize(block_hadamard_rotate(w, rotation))

    elif weight == qu.kNvfp4Static and not is_moe:  # weight-only (W4A16)
        nvfp4_spec = QuantSpec(weight=weight, activation=None)
        method = modelopt.ModelOptLinearMethod(
            nvfp4_spec, modelopt.CkptCtx(group_size=16)
        )
    elif weight == qu.kNvfp4Static and is_moe:
        config = modelopt.ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        config.moe_activation_override = spec.activation
        method = modelopt.ModelOptNvFp4FusedMoE(config, moe)
    else:
        return None
    scale = "weight_scale_inv" if block else "weight_scale"
    create_weights = method.create_weights

    def create_weights_quantized_on_load(layer, *args, **kwargs):
        create_weights(layer, *args, **kwargs)
        if weight == qu.kNvfp4Static and is_moe:
            return _nvfp4_on_load(layer, method)
        if weight == qu.kNvfp4Static:
            return _nvfp4_linear_on_load(layer, method)
        for p in ("w13_", "w2_") if is_moe else ("",):
            quantize_on_load(layer, f"{p}weight", quantize, f"{p}{scale}")

    method.create_weights = create_weights_quantized_on_load  # type: ignore[method-assign]
    if hadamard is not None:
        return InputHadamardLinearMethod(
            method, hadamard, rotate_weight=False, checkpoint_rotation=False
        )
    return method
