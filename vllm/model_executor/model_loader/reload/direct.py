# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload.

Every update is a full sync of checkpoint-format tensors. Instead of moving
layers to meta and rebuilding them, the loader writes straight into the live
tensors and the cold-start post-load processing runs once at the end:

- cold start records each tensor's checkpoint layout (before post-load
  processing) and every tensor's address (after it);
- ``begin`` views tensors whose layout post-load processing changed back to
  their checkpoint layout, on the same storage;
- ``finish`` reruns ``process_weights_after_loading`` for every module, as
  cold start does, raises if an address moved, and warns about checkpoint
  names cold start loaded that this update did not deliver.

A reload is exactly as correct as a cold start: no module is skipped because
it "received nothing" (a loader may bypass the parameter's ``weight_loader``,
and post-processing may read other modules' tensors). The only exception is
tensors the trainer deliberately does not sync, named by the regex
``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX`` (e.g. Engram): they keep their kernel
layout, and modules holding only such tensors skip post-processing.

Selected with ``VLLM_RELOAD_MODE=direct``.
"""

import os
from collections.abc import Iterable, Iterator

import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
    "finish_direct_reload",
    "record_checkpoint_layout",
    "record_load_names",
    "record_runtime_addresses",
]


def direct_reload_enabled() -> bool:
    return os.environ.get("VLLM_RELOAD_MODE", "layerwise") == "direct"


def _excluded() -> "re.Pattern[str] | None":
    """Names the trainer deliberately does not sync (e.g. Engram)."""
    pattern = os.environ.get("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", "")
    return re.compile(pattern) if pattern else None


def _loadable_tensors(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters and persistent buffers, each object once."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        tensors = list(module._parameters.items()) + [
            (n, b)
            for n, b in module._buffers.items()
            if n not in module._non_persistent_buffers_set
        ]
        for name, tensor in tensors:
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield (f"{module_name}.{name}" if module_name else name), tensor


def _addresses(model: torch.nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, t in model.named_parameters(remove_duplicate=False):
        if t.device.type == "cuda" and t.numel():
            out[name] = t.data_ptr()
    for name, t in model.named_buffers(remove_duplicate=False):
        if t is not None and t.device.type == "cuda" and t.numel():
            out[name] = t.data_ptr()
    return out


def record_load_names(model: torch.nn.Module) -> None:
    """Record the names passed through the model's own ``load_weights`` (cold
    start: before loading; reloads: from ``begin`` to ``finish``)."""
    names: set[str] = set()
    load_weights = model.load_weights

    def recording(weights: Iterable[tuple[str, torch.Tensor]], *args, **kwargs):
        def named():
            for name, tensor in weights:
                names.add(name)
                yield name, tensor

        return load_weights(named(), *args, **kwargs)

    model.load_weights = recording
    model._direct_load_names = names


def _pop_load_names(model: torch.nn.Module) -> set[str]:
    model.__dict__.pop("load_weights", None)  # back to the class method
    return model.__dict__.pop("_direct_load_names", set())


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout. Runs once; reloads reuse the cold-start record."""
    if getattr(model, "_direct_layout_recorded", False):
        return
    model._direct_layout_recorded = True
    model._direct_cold_names = _pop_load_names(model)
    registered: list[tuple[torch.nn.Module, str, bool, torch.Tensor]] = []
    for module_name, module in model.named_modules():
        for local, t, is_param in _own_tensors(module):
            if t.is_meta or hasattr(t, "_ckpt_layout"):
                continue
            t._ckpt_layout = (tuple(t.shape), t.stride(), t.dtype, t.storage_offset())
            registered.append((module, local, is_param, t))
    model._direct_registered = registered


def record_runtime_addresses(model: torch.nn.Module) -> None:
    """Cold start, after post-load processing: remember every address, and
    keep the checkpoint tensors post-load processing unregistered (deleted, or
    replaced under the same name) so each reload can register them again."""
    if hasattr(model, "_direct_addresses"):
        return
    model._direct_addresses = _addresses(model)
    dropped = [
        entry
        for entry in model.__dict__.pop("_direct_registered", [])
        if _registered(entry[0], entry[1]) is not entry[3]
    ]
    model._direct_dropped = dropped
    if dropped:
        live = {t.untyped_storage().data_ptr() for _, t in _loadable_tensors(model)}
        held = sum(
            t.untyped_storage().nbytes()
            for *_, t in dropped
            if t.untyped_storage().data_ptr() not in live
        )
        logger.info(
            "direct reload: %d checkpoint tensors are unregistered by post-load "
            "processing (e.g. %s); kept for reloads, %.2f GiB not shared with "
            "the runtime tensors",
            len(dropped),
            [f"{type(m).__name__}.{n}" for m, n, *_ in dropped[:4]],
            held / 2**30,
        )


def _own_tensors(module: torch.nn.Module):
    for n, p in module._parameters.items():
        if p is not None:
            yield n, p, True
    for n, b in module._buffers.items():
        if b is not None and n not in module._non_persistent_buffers_set:
            yield n, b, False


def _registered(module: torch.nn.Module, name: str) -> torch.Tensor | None:
    t = module._parameters.get(name)
    return t if t is not None else module._buffers.get(name)


def _register(module, name: str, is_param: bool, t: torch.Tensor | None) -> None:
    module._parameters.pop(name, None)
    module._buffers.pop(name, None)
    module.__dict__.pop(name, None)
    if t is None:
        return
    if is_param:
        module.register_parameter(name, t)
    else:
        module.register_buffer(name, t)


def _unsupported(model: torch.nn.Module) -> str | None:
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, (OnlineLinearBase, OnlineMoEMethodBase)):
            return f"{name}: online quantization ({type(method).__name__})"
    return None


@torch.no_grad()
def begin_direct_reload(model: torch.nn.Module) -> None:
    """Make every tensor writable by its checkpoint weight loader."""
    if not getattr(model, "_direct_layout_recorded", False):
        raise RuntimeError("direct reload: no checkpoint layout was recorded")
    if (reason := _unsupported(model)) is not None:
        raise NotImplementedError(f"direct reload: {reason}")

    # checkpoint names post-load processing deleted or replaced: register the
    # checkpoint tensors again so the model's loader finds them; the rerun at
    # finish unregisters them as cold start did
    reregistered = []
    for module, name, is_param, t in getattr(model, "_direct_dropped", []):
        current = _registered(module, name)
        reregistered.append((module, name, name in module._parameters, current))
        _register(module, name, is_param, t)
    model._direct_reregistered = reregistered

    excluded = _excluded()
    restored = 0
    for name, t in _loadable_tensors(model):
        layout = getattr(t, "_ckpt_layout", None)
        if layout is None:
            continue  # created by post-load processing: derived, rewritten there
        if excluded is not None and excluded.search(name):
            continue  # not synced: stays in kernel layout
        shape, stride, dtype, offset = layout
        if (tuple(t.shape), t.stride(), t.dtype, t.storage_offset()) != layout:
            storage = t.untyped_storage()
            span = 1 + sum((s - 1) * st for s, st in zip(shape, stride)) if shape else 1
            need = (offset + span) * dtype.itemsize
            if need > storage.nbytes():
                raise RuntimeError(
                    f"direct reload: {name} storage has {storage.nbytes()} bytes, "
                    f"its checkpoint layout needs {need}"
                )
            view = torch.empty(0, dtype=dtype, device=t.device)
            view.set_(storage, offset, shape, stride)
            t.data = view  # no zeroing: the full sync rewrites the checkpoint region
            restored += 1

    record_load_names(model)
    logger.info("direct reload: %d tensors viewed back to checkpoint layout", restored)


@torch.no_grad()
def finish_direct_reload(model: torch.nn.Module, model_config) -> None:
    """Rerun cold-start post-load processing, then check names and addresses."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    torch.accelerator.synchronize()  # engines may write on other streams
    received = _pop_load_names(model)

    # A module the trainer deliberately does not sync (every checkpoint tensor
    # matches the exclude regex) keeps its kernel layout and skips
    # post-processing, which would otherwise convert converted bytes again.
    excluded = _excluded()
    skipped: list[torch.nn.Module] = []
    for module_name, module in model.named_modules():
        tensors = list(module._parameters.items()) + [
            (n, b)
            for n, b in module._buffers.items()
            if n not in module._non_persistent_buffers_set
        ]
        own = [
            (f"{module_name}.{n}" if module_name else n, t)
            for n, t in tensors
            if t is not None and hasattr(t, "_ckpt_layout")
        ]
        if not own or excluded is None:
            continue
        if not all(excluded.search(name) for name, _ in own):
            continue
        module._direct_skip_pwal = True
        skipped.append(module)
    # a skipped module keeps its runtime registration (no rerun unregisters)
    skipped_ids = {id(m) for m in skipped}
    for module, name, is_param, current in model.__dict__.pop(
        "_direct_reregistered", []
    ):
        if id(module) in skipped_ids:
            _register(module, name, is_param, current)
    # One-shot "already processed" guards (R7) would make the rerun below a
    # no-op and leave the freshly written checkpoint layout in place.
    cleared = 0
    for module in model.modules():
        if module not in skipped and module.__dict__.pop(
            "_already_called_process_weights_after_loading", None
        ):
            cleared += 1
    if cleared:
        logger.info("direct reload: cleared %d one-shot post-load guards", cleared)

    device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.accelerator.current_accelerator(),
    )
    pinned = _snapshot_side_state(model, skipped)
    try:
        process_weights_after_loading(model, model_config, device)
    finally:
        for module in skipped:
            module.__dict__.pop("_direct_skip_pwal", None)
    _pin_side_state(pinned)
    # hand the receive buffers and post-load temporaries back to the driver:
    # the next update's NCCL group allocates outside the caching allocator
    torch.accelerator.empty_cache()
    if skipped:
        logger.info(
            "direct reload: %d excluded modules kept their kernel layout", len(skipped)
        )

    missing = sorted(
        n
        for n in model._direct_cold_names - received
        if excluded is None or not excluded.search(n)
    )
    current = _addresses(model)
    moved = sorted(n for n, p in model._direct_addresses.items() if current.get(n) != p)
    model._direct_report = {"missing": missing, "moved": moved}
    if missing:
        logger.warning(
            "direct reload: %d checkpoint names not received, e.g. %s",
            len(missing),
            missing[:8],
        )
    # kernels and captured graphs would keep reading the old storage
    if moved and os.environ.get("VLLM_DIRECT_RELOAD_ALLOW_MOVED") != "1":
        raise RuntimeError(f"direct reload: {len(moved)} tensors moved: {moved[:8]}")


def _snapshot_side_state(model: torch.nn.Module, skipped: list[torch.nn.Module]):
    """Before the rerun: each processed module's side references and its
    registered tensors that are not loaded from the checkpoint (kernel
    constants, derived scales), which post-load processing may reallocate."""
    from .side_state import collect_side_tensors

    out = []
    for module in model.modules():
        if module in skipped or getattr(module, "quant_method", None) is None:
            continue
        startup = {
            n: t.data
            for n, t in list(module._parameters.items()) + list(module._buffers.items())
            if t is not None and not hasattr(t, "_ckpt_layout")
        }
        out.append((module, collect_side_tensors(module), startup))
    return out


def _pin_side_state(pinned) -> None:
    """After the rerun: copy reallocated tensors back into their startup
    storage and redirect side references to it, so captured CUDA graphs read
    the new values."""
    from .side_state import pin_side_tensors

    for module, before, startup in pinned:
        temps: dict[str, torch.Tensor] = {}
        kernels: dict[str, torch.Tensor] = {}
        for n, old in startup.items():
            cur = module._parameters.get(n)
            is_param = cur is not None
            if cur is None:
                cur = module._buffers.get(n)
            if (
                cur is None
                or cur.data_ptr() == old.data_ptr()
                or cur.shape != old.shape
                or cur.stride() != old.stride()
                or cur.dtype != old.dtype
                or cur.device != old.device
            ):
                continue
            temps[n] = cur
            old.copy_(cur)
            if is_param:
                cur.data = old
                kernels[n] = cur
            else:
                module._buffers[n] = old
                kernels[n] = old
        pin_side_tensors(module, before, temps, kernels, logger.debug)
