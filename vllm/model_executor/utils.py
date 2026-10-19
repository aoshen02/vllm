# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""

import copy
import types
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from vllm.utils.torch_utils import is_torch_equal_or_newer

_weights_pre_processed: ContextVar[bool] = ContextVar(
    "weights_pre_processed", default=False
)


@contextmanager
def weights_already_processed():
    """Mark that weights are already in post-processed (runtime) format, so
    ``process_weights_after_loading`` skips tensor transforms (used by the
    weight cache IPC loader).
    """
    token = _weights_pre_processed.set(True)
    try:
        yield
    finally:
        _weights_pre_processed.reset(token)


def is_weights_pre_processed() -> bool:
    return _weights_pre_processed.get()


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.

    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # NOTE(woosuk): During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        # TODO(woosuk): Remove this hack once we have a better solution.
        from vllm.platforms import current_platform

        if current_platform.use_sync_weight_loader() and key == "weight_loader":
            value = current_platform.make_synced_weight_loader(value)
        setattr(weight, key, value)


# Runtime state, not model state: bound or rebuilt by the model runner, or
# per-step scratch. Never registered, at any attribute depth.
RUNTIME_TENSOR_ATTRS: dict[str, str] = {
    "kv_cache": "KV cache / SSM state, bound by the model runner",
    "eplb_state": "EPLB maps and load counters, owned by EplbState",
    "_replayssm_ring_start": "per-step speculative SSM replay state",
    "_replayssm_prev_num_accepted": "per-step speculative SSM replay state",
    "_replayssm_prev_query_len": "per-step speculative SSM replay state",
    "deepstack_input_embeds": "per-step multimodal staging buffer",
    "per_layer_embeddings": "per-step staging buffer",
    "topk_indices_buffer": "per-step sparse attention scratch",
    "_topk_indices_buffer": "per-step sparse attention scratch",
    "candidate_block_buffer": "per-step sparse indexer scratch",
    "_workspace_buffer": "attention workspace",
    "_nvfp4_fp8_out": "attention output scratch",
    "index_group": "per-step sparse indexer scratch",
    "diffusion_states": "per-request diffusion decoding state",
    "aot_compiled_fn": "torch.compile artifacts (example inputs)",
    "_prefill_topk_ragged_cache": "per-step sparse prefill scratch",
    "specialized_decode": "Kimi-K3 decode executor the model runner builds",
    "_cu_2": "per-step TurboQuant cu_seqlens scratch",
    "_cu_2_q": "per-step TurboQuant cu_seqlens scratch",
    "_cu_2_k": "per-step TurboQuant cu_seqlens scratch",
}

_MODULE_STATE = {"_parameters", "_buffers", "_modules"}


def storage_views(tensor: torch.Tensor) -> set[tuple]:
    """Storage views behind ``tensor``, through traceable wrapper subclasses."""
    if is_traceable_wrapper_subclass(tensor):
        names, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
        return set().union(*(storage_views(getattr(tensor, n)) for n in names))
    return {
        (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
    }


def _walk_unregistered(model: torch.nn.Module, include_meta: bool) -> list[tuple]:
    """``(owning module, path, tensor, holder, key)`` per unregistered tensor;
    ``holder[key]`` refers to it, ``holder`` None if it cannot be rebound
    (tuple item)."""
    registered: set[tuple] = set()
    for t in (*model.parameters(), *model.buffers()):
        registered |= storage_views(t)
    modules = {id(m) for m in model.modules()}
    seen: set[int] = set()
    found: list[tuple] = []

    def visit(owner: torch.nn.Module, value: Any, path: str, holder, key) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.is_meta:
                if include_meta:
                    found.append((owner, path, value, holder, key))
            else:
                views = storage_views(value)
                if not views <= registered:
                    registered.update(views)
                    found.append((owner, path, value, holder, key))
        elif isinstance(value, torch.nn.Module):
            # A module outside model.modules(), e.g. kept in a dict.
            for prefix, module in value.named_modules(prefix=path):
                if id(module) not in modules:
                    for tensors in (module._parameters, module._buffers):
                        for name, t in list(tensors.items()):
                            visit(owner, t, f"{prefix}.{name}", tensors, name)
                    visit_attrs(owner, module, f"{prefix}.")
        elif isinstance(value, (tuple, list)):
            container = value if isinstance(value, list) else None
            for i, item in enumerate(value):
                visit(owner, item, f"{path}.{i}", container, i)
        elif isinstance(value, dict):
            for k, item in list(value.items()):
                visit(owner, item, f"{path}.{k}", value, k)
        elif not isinstance(value, (type, types.ModuleType)) and isinstance(
            getattr(value, "__dict__", None), dict
        ):
            # Configs reach other models' layers (drafter forward context);
            # process groups hold communication buffers, not model state.
            if str(getattr(type(value), "__module__", "")).startswith(
                ("vllm.config", "vllm.distributed", "transformers")
            ):
                return
            for name, item in list(vars(value).items()):
                if name not in RUNTIME_TENSOR_ATTRS:
                    visit(owner, item, f"{path}.{name}", vars(value), name)

    def visit_attrs(owner: torch.nn.Module, module: torch.nn.Module, prefix: str):
        for name, value in list(vars(module).items()):
            if not (
                name in _MODULE_STATE
                or name in RUNTIME_TENSOR_ATTRS
                or name.startswith("_forward_")
            ):
                visit(owner, value, f"{prefix}{name}", vars(module), name)

    for module in model.modules():
        visit_attrs(module, module, "")
    return found


def find_unregistered_tensors(
    model: torch.nn.Module,
) -> list[tuple[torch.nn.Module, str, torch.Tensor]]:
    """Tensors reachable from module attributes (plain attributes, quant methods,
    kernels, attention impls, dataclasses, containers) whose storage is no
    parameter or buffer, as ``(owning module, attribute path, tensor)``."""
    return [found[:3] for found in _walk_unregistered(model, include_meta=False)]


def register_model_tensors(model: torch.nn.Module) -> None:
    """Register each tensor ``find_unregistered_tensors`` finds as a
    non-persistent ``_held_<path>`` buffer of its module, aliasing it.

    Re-run after a reload: a rebuilt tensor moves into the storage registered
    under the same name before, so captured addresses stay valid.
    """
    previous: dict[tuple[int, str], torch.Tensor] = {}
    for module in model.modules():
        for name in [n for n in module._buffers if n.startswith("_held_")]:
            previous[(id(module), name)] = module._buffers.pop(name)
            module._non_persistent_buffers_set.discard(name)
    for module, path, tensor, holder, key in _walk_unregistered(model, True):
        base = "_held_" + "".join(c if c.isalnum() else "_" for c in path)
        name, n = base, 0
        while name in module._buffers:  # "a.b" and "a_b" map to one name
            n += 1
            name = f"{base}_{n}"
        old = previous.get((id(module), name))
        if tensor.is_meta:
            # Built on meta, imported from the IPC weight cache: use the import.
            if old is None or (old.shape, old.stride(), old.dtype) != (
                tensor.shape,
                tensor.stride(),
                tensor.dtype,
            ):
                continue
            if holder is not None:
                is_param = isinstance(tensor, torch.nn.Parameter)
                holder[key] = torch.nn.Parameter(old, False) if is_param else old
            tensor = old
        elif old is not None:
            reuse_storage(old, tensor)
        module.register_buffer(name, tensor, persistent=False)


def reuse_storage(old: torch.Tensor, new: torch.Tensor) -> bool:
    """Copy ``new`` into ``old`` and point ``new`` at that storage, so captured
    addresses stay valid; returns False, doing nothing, if the layouts differ."""
    if not (
        type(new) is type(old) is torch.Tensor
        and new.data_ptr() != old.data_ptr()
        and new.shape == old.shape
        and new.stride() == old.stride()
        and new.dtype == old.dtype
        and new.device == old.device
    ):
        return False
    with torch.inference_mode():
        old.copy_(new)
        new.set_(old)
    return True


def replace_parameter(
    layer: torch.nn.Module,
    param_name: str,
    new_tensor: torch.Tensor | None,
    prefer_copy: bool = False,
):
    """Replace a parameter of a layer while maintaining the ability to reload the
    weight. Called within implementations of the `process_weights_after_loading` method.

    Custom attributes set on ``new_tensor`` (e.g. kernel dispatch flags such as
    ``is_shuffled``) are carried over to the replacement parameter, except
    ``weight_loader``, which is always taken from the existing parameter.

    Attributes of the existing parameter are otherwise dropped when a new
    parameter is registered, but kept when ``prefer_copy`` reuses it in place.

    This function should not be called on weights which are tied/shared

    Args:
        layer: Layer containing parameter to replace
        param_name: Name of parameter to replace
        new_tensor: New data of the new parameter, or None to set the parameter
            to None
        prefer_copy: If True and the existing parameter is compatible with
            ``new_tensor`` (same shape, dtype, and device), copy ``new_tensor``
            into the existing parameter in place rather than re-registering
            a new parameter. This preserves the parameter's storage address
            (``data_ptr``), which is required for captured CUDA graphs to
            remain valid across weight updates (e.g. in RL training loops).

    """
    # should not be used on a tied/shared param

    # If new_tensor is None, set the parameter to None
    if new_tensor is None:
        setattr(layer, param_name, None)
        return

    old_param: torch.nn.Parameter | None = getattr(layer, param_name, None)

    new_tensor_attrs = dict(new_tensor.__dict__)

    # `weight_loader` is the only attribute not ported over from new_tensor,
    # old_param.weight_loader only is supported. `_weight_loader` is
    # `BasevLLMParameter`'s backing field for its `weight_loader` property, so
    # a loader riding along under that name has to be dropped as well.
    new_tensor_attrs.pop("weight_loader", None)
    new_tensor_attrs.pop("_weight_loader", None)

    if isinstance(new_tensor, torch.nn.Parameter):
        new_tensor = new_tensor.data

    if (
        prefer_copy
        and old_param is not None
        and old_param.shape == new_tensor.shape
        and old_param.dtype == new_tensor.dtype
        and old_param.device == new_tensor.device
    ):
        old_param.copy_(new_tensor)
        for attr_name, attr in new_tensor_attrs.items():
            setattr(old_param, attr_name, attr)
        return

    new_param = torch.nn.Parameter(new_tensor, requires_grad=False)

    for attr_name, attr in new_tensor_attrs.items():
        setattr(new_param, attr_name, attr)

    if old_param is not None and hasattr(old_param, "weight_loader"):
        weight_loader = old_param.weight_loader
        set_weight_attrs(new_param, {"weight_loader": weight_loader})

    setattr(layer, param_name, new_param)


def get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    parent_map = getattr(model, "packed_modules_mapping", None)
    parent_map = copy.deepcopy(parent_map) if parent_map is not None else {}

    # don't infer mapping if the model has defined it explicitly.
    if parent_map:
        return parent_map

    # We only check main components instead of whole model submodules
    for child in model.children():
        child_map = getattr(child, "packed_modules_mapping", None)
        child_map = copy.deepcopy(child_map) if child_map is not None else {}

        if any((k in parent_map and parent_map[k] != v) for k, v in child_map.items()):
            raise ValueError(
                f"Can't update {type(model).__name__}'s packed_modules_mapping "
                f"safely because of conflicts from {type(child).__name__}."
            )
        else:
            parent_map.update(child_map)
    return parent_map


def get_moe_expert_mapping(
    model: torch.nn.Module,
) -> list[tuple[str, str, int, str]]:
    """Get the expert mapping from a model.

    It will be retrieved from the first module that has a `get_expert_mapping` method.
    If the model manually implements `get_expert_mapping`, it will be used.
    Otherwise, it will use the first RoutedExperts layer."""
    for _, module in model.named_modules():
        get_mapping = getattr(module, "get_expert_mapping", None)
        if get_mapping is not None:
            return get_mapping()
    raise ValueError("No module in the model has a `get_expert_mapping` method.")


def maybe_disable_graph_partition(current_backend: str) -> dict[str, bool]:
    if current_backend == "inductor" and is_torch_equal_or_newer("2.9.0.dev"):
        return {"graph_partition": False}
    else:
        return {}
