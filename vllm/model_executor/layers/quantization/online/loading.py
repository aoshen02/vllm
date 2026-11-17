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
    # the loaders as created: requantization stages checkpoint tensors through
    # the parameters' own `weight_loader` attributes afterwards
    scale_loaders = [
        getattr(getattr(layer, s), "weight_loader", default_weight_loader)
        for s in scale_names
    ]
    layer.quantizes_on_load = True

    def load_quantized(param, quantized, *args, **kwargs):
        values, *scales = quantized
        for scale_name, scale_load, scale in zip(scale_names, scale_loaders, scales):
            scale_param = getattr(layer, scale_name)
            # the MoE loader picks how to load a scale by its parameter name
            scale_load(
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
        if not is_moe:
            # kernels read the partition sizes, which online methods set and
            # some layers (the MoE router gate) do not
            if not hasattr(layer, "input_size_per_partition"):
                layer.input_size_per_partition = kwargs.get(
                    "input_size_per_partition", args[0] if args else None
                )
            if not hasattr(layer, "output_size_per_partition"):
                sizes: list[int] = kwargs.get(
                    "output_partition_sizes", args[1] if len(args) > 1 else []
                )
                layer.output_size_per_partition = sum(sizes)
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


def _requant_roles(
    source: Any, source_scheme: Any
) -> tuple[dict[str, tuple[str, str]], Callable] | None:
    """`(roles, dequantize)` for a checkpoint method that can be requantized on
    load: `roles` maps each checkpoint tensor the dequantization reads to `(the
    high-precision parameter it becomes, its part)`, and `dequantize(parts,
    dtype)` rebuilds one unsharded high-precision tensor from the parts."""
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )
    from vllm.model_executor.layers.quantization.quark.quark import QuarkLinearMethod
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkNvfp4MoEMethod,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Static,
    )

    # class: (packed values, global scale); compressed-tensors stores the
    # global encode scale, the others its reciprocal
    nvfp4_moe: dict[type, tuple[str, str]] = {
        CompressedTensorsW4A4Nvfp4MoEMethod: ("weight_packed", "weight_global_scale"),
        modelopt.ModelOptNvFp4FusedMoE: ("weight", "weight_scale_2"),
        QuarkNvfp4MoEMethod: ("weight", "weight_scale_2"),
    }
    if type(source) in nvfp4_moe:
        packed, global_name = nvfp4_moe[type(source)]
        invert = isinstance(source, CompressedTensorsW4A4Nvfp4MoEMethod)
        group_size = getattr(source, "group_size", None) or (
            source.quant_config.group_size
        )
        roles = {
            f"{p}{name}": (f"{p}weight", part)
            for p in ("w13_", "w2_")
            for name, part in (
                (packed, "values"),
                ("weight_scale", "scale"),
                (global_name, "global"),
            )
        }

        def dequantize_nvfp4(parts: dict, dtype: torch.dtype) -> torch.Tensor:
            global_scale = parts["global"].float().reshape(1)
            if invert:
                global_scale = 1.0 / global_scale
            return dequantize_to_dtype(
                parts["values"][None],
                parts["scale"][None],
                global_scale,
                dtype,
                block_size=group_size,
                swizzle=False,
            )[0]

        return roles, dequantize_nvfp4

    if (
        isinstance(source, modelopt.ModelOptLinearMethod)
        and source.wkey.key is kMxfp8Static
    ) or (
        isinstance(source, QuarkLinearMethod)
        and getattr(source_scheme, "weight_quant_key", None) == kMxfp8Static
    ):
        roles = {"weight": ("weight", "values"), "weight_scale": ("weight", "scale")}

        def dequantize_mxfp8(parts: dict, dtype: torch.dtype) -> torch.Tensor:
            return dequant_mxfp8_to_bf16(
                parts["values"].contiguous(), parts["scale"].contiguous()
            ).to(dtype)

        return roles, dequantize_mxfp8
    return None


def _moe_args(args: tuple, kwargs: dict) -> tuple:
    """`(weight_name, shard_id, expert_id)` of a MoE weight loader call."""
    names = ("weight_name", "shard_id", "expert_id")
    return tuple(args[i] if i < len(args) else kwargs[n] for i, n in enumerate(names))


def requantize_on_load(source: Any, target: Any, source_scheme: Any = None) -> Any:
    """`target`, a method quantizing high-precision weights on load, made to
    load the checkpoint tensors of `source`, a pre-quantized checkpoint method,
    or None if `source` cannot be dequantized on load.

    The tensors one dequantization needs (an expert shard's packed values,
    block scales and global scale; a linear shard's values and scales) arrive
    in separate loads: all but the last wait, then they are dequantized and
    loaded as one high-precision tensor. Checkpoint tensors the target does not
    hold stay meta parameters, never written. `source_scheme` is the layer
    scheme `source` creates its weights with."""
    from vllm.model_executor.layers.quantization.online.requantization import (
        skip_checkpoint_backend_selection,
    )

    found = _requant_roles(source, source_scheme)
    if found is None:
        return None
    roles, dequantize = found
    num_parts = len({part for _, part in roles.values()})
    is_moe = hasattr(target, "moe")
    create_weights = target.create_weights
    pending: dict[tuple, dict[str, tuple]] = {}
    # checkpoint-only names the target holds as attributes (a None activation
    # scale): the checkpoint loads the parameter, post-load processing reads
    # the attribute
    shadowed: dict[str, Any] = {}

    def create_weights_requantized_on_load(layer, *args, **kwargs):
        before = dict(layer._parameters)
        target_scheme = getattr(layer, "scheme", None)
        if source_scheme is not None:
            layer.scheme = source_scheme
        # the checkpoint method names the tensors the checkpoint loads
        with torch.device("meta"), skip_checkpoint_backend_selection():
            source.create_weights(layer, *args, **kwargs)
        checkpoint = {
            name: param
            for name, param in layer._parameters.items()
            if param is not None and before.get(name) is not param
        }
        for name in checkpoint:
            delattr(layer, name)
        if source_scheme is not None:
            layer.scheme = target_scheme
        create_weights(layer, *args, **kwargs)
        high_precision = {
            name: getattr(layer, name).weight_loader for name, _ in roles.values()
        }

        def stage(name: str, part: str, param, loaded_weight, *args, **kwargs):
            if is_moe:
                args = _moe_args(args, kwargs)
                shard_id, expert_id = args[1:]
                if layer._map_global_expert_id_to_local_expert_id(expert_id) == -1:
                    return False
                key: tuple = (name, shard_id, expert_id)
                kwargs = {"return_success": True}
            else:
                key = (name, args[0] if args else kwargs.get("loaded_shard_id"))
            copied = torch.Event()
            tensor = loaded_weight.to(getattr(layer, name).device, copy=True)
            copied.record()
            got = pending.setdefault(key, {})
            got[part] = (tensor, copied, args, kwargs)
            if len(got) < num_parts:
                return True
            del pending[key]
            for _, event, _, _ in got.values():
                # held from another load, which may have run on another stream
                event.wait()
            weight = dequantize(
                {part: tensor for part, (tensor, _, _, _) in got.items()},
                layer.params_dtype,
            )
            _, _, args, kwargs = got["values"]
            if is_moe:  # the MoE loader tells weights from scales by name
                args = (name, *args[1:])
            return high_precision[name](getattr(layer, name), weight, *args, **kwargs)

        def drop(param, loaded_weight, *args, **kwargs):
            # e.g. calibrated activation scales: the target computes its own
            if is_moe:
                expert_id = _moe_args(args, kwargs)[2]
                return layer._map_global_expert_id_to_local_expert_id(expert_id) != -1
            return None

        for name, param in checkpoint.items():
            if name in roles:
                loader: Callable = partial(stage, *roles[name])
            elif name in layer._parameters:
                continue  # the target loads it as it is (e.g. a bias)
            else:
                loader = drop
            if name not in layer._parameters:
                if hasattr(layer, name):
                    shadowed[name] = getattr(layer, name)
                    delattr(layer, name)
                layer.register_parameter(name, param)
            getattr(layer, name).weight_loader = loader

    process = target.process_weights_after_loading

    def process_weights_after_loading(layer: torch.nn.Module) -> None:
        if pending:
            incomplete = sorted(map(str, pending))
            pending.clear()
            raise RuntimeError(
                f"requantization got only part of the checkpoint tensors of "
                f"{incomplete}; all of them must be loaded in the same update"
            )
        placeholders = {name: layer._parameters.pop(name) for name in shadowed}
        for name, value in shadowed.items():
            setattr(layer, name, value)
        try:
            process(layer)
        finally:
            for name, param in placeholders.items():
                if getattr(layer, name, None) is not None:
                    raise RuntimeError(
                        f"requantization: the target sets {name}, which the "
                        "checkpoint also loads"
                    )
                delattr(layer, name)
                layer.register_parameter(name, param)

    target.create_weights = create_weights_requantized_on_load
    target.process_weights_after_loading = process_weights_after_loading
    return target
