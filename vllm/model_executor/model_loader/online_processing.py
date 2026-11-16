# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-start loading of online-quantized layers.

An online quantization method creates a layer's full-precision weights on the
meta device and calls `initialize_online_processing`. Its weight loaders then
buffer the checkpoint shards; once every element of the layer has arrived, the
layer is materialized, the shards are loaded and the layer's quant method
processes (quantizes) it, so only one layer is held in full precision at a
time. `finalize_online_processing` processes the layers that never completed
(padding, unloaded weights).
"""

import inspect
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

__all__ = [
    "finalize_online_processing",
    "finish_online_layer",
    "initialize_online_processing",
    "is_online_processing",
    "stop_online_processing",
]

# Tensors never loaded by a weight loader, so they do not count toward a layer.
SKIP_LOAD_TENSORS: set[str] = {
    "_expert_map",
    "expert_mask",
    "expert_global_to_physical",
    "expert_physical_to_global",
    "expert_local_to_global",
    "e_score_correction_bias",
}

_LOADER = "online_process_loader"


@dataclass
class _OnlineLoad:
    device: torch.device
    load_numel: int = 0
    loaded: list[tuple[str, inspect.BoundArguments]] = field(default_factory=list)


def _tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: t
        for name, t in (*layer._parameters.items(), *layer._buffers.items())
        if t is not None
    }


def _layer_numel(layer: torch.nn.Module) -> int:
    """Checkpoint elements the layer loads, excluding declared padding."""
    return sum(
        getattr(t, "weight_loader_numel", t.numel())
        for name, t in _tensors(layer).items()
        if name not in SKIP_LOAD_TENSORS
    )


def _weight_loader(tensor: torch.Tensor) -> Callable:
    return getattr(tensor, "weight_loader", default_weight_loader)


def _original_loader(tensor: torch.Tensor) -> Callable:
    loader = _weight_loader(tensor)
    while loader.__name__ == _LOADER:
        loader = loader.__wrapped__  # type: ignore[attr-defined]
    return loader


class _CopyCounter(TorchDispatchMode):
    """Counts the elements a weight loader writes with `copy_`, whatever views
    (e.g. `narrow`) it takes first."""

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default:
            self.copied_numel += args[0].numel()
        return func(*args, **(kwargs or {}))


def is_online_processing(layer: torch.nn.Module) -> bool:
    return "_online_load" in layer.__dict__


def stop_online_processing(layer: torch.nn.Module) -> None:
    """Leave the layer out of online processing; its wrapped loaders ignore
    further loads."""
    layer.__dict__.pop("_online_load", None)


def initialize_online_processing(layer: torch.nn.Module) -> None:
    """Buffer the layer's loads and process it once all of them arrived."""
    layer._online_load = _OnlineLoad(device=torch.get_default_device())
    _wrap_loaders(layer)


def _wrap_loaders(layer: torch.nn.Module) -> None:
    for name, tensor in _tensors(layer).items():
        if name not in SKIP_LOAD_TENSORS and _weight_loader(tensor).__name__ != _LOADER:
            tensor.weight_loader = _make_loader(layer, name)


def _make_loader(layer: torch.nn.Module, param_name: str) -> Callable:
    original = _original_loader(getattr(layer, param_name))
    signature = inspect.signature(original)

    @wraps(original, assigned=("__doc__", "__annotations__"))
    def online_process_loader(*args, **kwargs):
        state = layer.__dict__.get("_online_load")
        if state is None:
            # processed already: some quant configs load a weight more than
            # once (CT_WNA16 loads `weight_shape` per qkv partition)
            return None
        # layers may register parameters late (e.g. `bias`)
        _wrap_loaders(layer)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        state.loaded.append((param_name, bound))
        with _CopyCounter() as counter:
            ret = original(*bound.args, **bound.kwargs)
        # a loader may copy into its parameter more than once (an in-place
        # post-load transform); count at most the parameter's size
        param = bound.arguments.get("param")
        numel = counter.copied_numel
        if isinstance(param, torch.Tensor):
            numel = min(numel, getattr(param, "weight_loader_numel", param.numel()))
        state.load_numel += numel
        if state.load_numel >= _layer_numel(layer):
            finish_online_layer(layer)
        return ret

    return online_process_loader


def finish_online_layer(
    layer: torch.nn.Module, fill: Callable[..., None] | None = None
) -> None:
    """Materialize the layer, load its buffered shards (or `fill(tensor,
    name=...)` each tensor instead) and run its quant method's processing."""
    state = layer.__dict__.pop("_online_load")
    with state.device:
        for name, t in _tensors(layer).items():
            if t.is_meta and name not in SKIP_LOAD_TENSORS:
                # zeroed: loaders only write checkpoint regions (MoE padding)
                new = torch.zeros_like(t, device=state.device)
                new.__class__ = t.__class__
                new.__dict__ = t.__dict__.copy()
                setattr(layer, name, new)
    for t in _tensors(layer).values():
        if hasattr(t, "weight_loader"):
            t.weight_loader = _original_loader(t)
    if fill is not None:
        for name, t in _tensors(layer).items():
            fill(t, name=name)
    else:
        for name, bound in state.loaded:
            param = getattr(layer, name)
            bound.arguments["param"] = param
            param.weight_loader(*bound.args, **bound.kwargs)

    layer.__dict__.pop("_already_called_process_weights_after_loading", None)
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
        from vllm.model_executor.layers.quantization.online.moe_base import (
            OnlineMoEMethodBase,
        )
        from vllm.model_executor.model_loader.utils import (
            online_quantization_timing_context,
        )

        online = isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
        with (
            online_quantization_timing_context(layer, quant_method)
            if online
            else nullcontext()
        ):
            quant_method.process_weights_after_loading(layer)
        if hasattr(layer, "update_param_tp_status"):
            layer.update_param_tp_status()


def finalize_online_processing(model: torch.nn.Module) -> None:
    """Process the layers whose loads never completed (padding, weights the
    checkpoint does not hold)."""
    for layer in model.modules():
        if is_online_processing(layer):
            finish_online_layer(layer)
