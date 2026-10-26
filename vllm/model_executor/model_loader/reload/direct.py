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
  cold start does, and raises if an address moved.

Whether the update delivered every weight is checked by the weight transfer
engine, once per update, with the cold-start completeness check.

A reload is exactly as correct as a cold start: no module is skipped because
it "received nothing" (a loader may bypass the parameter's ``weight_loader``,
and post-processing may read other modules' tensors). The only exception is
tensors the trainer deliberately does not sync, named by the regex
``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX`` (e.g. Engram). It has to cover whole
modules with their submodules, which a reload leaves untouched: no view-back,
no re-registration, no post-processing.

Selected with ``VLLM_RELOAD_MODE=direct``.
"""

import os
from collections.abc import Iterator, Set

import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import is_weights_pre_processed

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
    "finish_direct_reload",
    "record_checkpoint_layout",
    "record_runtime_addresses",
    "unsynced_weights_regex",
]


_rerunning = False


def rerunning_post_load() -> bool:
    """Whether a direct reload is rerunning post-load processing."""
    return _rerunning


def direct_reload_enabled() -> bool:
    return os.environ.get("VLLM_RELOAD_MODE", "layerwise") == "direct"


def unsynced_weights_regex() -> "re.Pattern[str] | None":
    """Names the trainer deliberately does not sync (e.g. Engram); weight
    updates do not have to deliver them."""
    pattern = os.environ.get("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", "")
    return re.compile(pattern) if pattern else None


def _loadable_tensors(
    model: torch.nn.Module, skipped: Set[int] = frozenset()
) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters and persistent buffers, each object once, outside the
    modules whose ids are in ``skipped``."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        if id(module) in skipped:
            continue
        for name, tensor, _ in _own_tensors(module):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield (f"{module_name}.{name}" if module_name else name), tensor


def _addresses(model: torch.nn.Module) -> dict[str, tuple]:
    """Where kernels and captured graphs read each tensor, and how."""
    out: dict[str, tuple] = {}
    tensors = list(model.named_parameters(remove_duplicate=False))
    tensors += model.named_buffers(remove_duplicate=False)
    for name, t in tensors:
        if t is not None and t.device.type == "cuda" and t.numel():
            out[name] = (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype)
    return out


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout. Runs once; reloads reuse the cold-start record."""
    if getattr(model, "_direct_layout_recorded", False):
        return
    model._direct_layout_recorded = True
    # the IPC weight cache loads runtime layout: there is no checkpoint layout
    model._direct_pre_processed = is_weights_pre_processed()
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
        logger.info(
            "direct reload: %d checkpoint tensors are unregistered by post-load "
            "processing (e.g. %s); kept for reloads",
            len(dropped),
            [f"{type(m).__name__}.{n}" for m, n, *_ in dropped[:4]],
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

    if model._direct_pre_processed:
        return "weights loaded pre-processed (IPC weight cache)"
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, (OnlineLinearBase, OnlineMoEMethodBase)):
            return f"{name}: online quantization ({type(method).__name__})"
    return None


def _unsynced_modules(model: torch.nn.Module) -> set[int]:
    """Ids of the modules a reload leaves alone: every checkpoint tensor in
    their subtree matches ``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX``.

    Every other module reruns post-processing on its checkpoint tensors in
    checkpoint layout, which an unsynced tensor (kept in kernel layout) is
    not. So a subtree holding both synced and unsynced tensors raises when
    its root owns an unsynced tensor (it would convert it again), or runs a
    quantization method's post-processing (at cold start that read the
    subtree in checkpoint layout).
    """
    excluded = unsynced_weights_regex()
    if excluded is None:
        return set()
    dropped: dict[int, list[str]] = {}
    for module, name, *_ in getattr(model, "_direct_dropped", []):
        dropped.setdefault(id(module), []).append(name)
    skipped: set[int] = set()
    mixed: list[str] = []

    def visit(prefix: str, module: torch.nn.Module) -> tuple[bool, bool]:
        own = [n for n, t, _ in _own_tensors(module) if hasattr(t, "_ckpt_layout")]
        own = [prefix + n for n in own + dropped.get(id(module), [])]
        unsynced = [n for n in own if excluded.search(n)]
        any_unsynced, any_synced = bool(unsynced), len(unsynced) < len(own)
        for name, child in module.named_children():
            u, s = visit(f"{prefix}{name}.", child)
            any_unsynced |= u
            any_synced |= s
        if not any_synced:
            if any_unsynced:
                skipped.add(id(module))
        elif any_unsynced and (
            unsynced
            or isinstance(getattr(module, "quant_method", None), QuantizeMethodBase)
        ):
            mixed.append(prefix.rstrip(".") or "<root>")
        return any_unsynced, any_synced

    visit("", model)
    if mixed:
        raise RuntimeError(
            "direct reload: VLLM_DIRECT_RELOAD_EXCLUDE_REGEX must cover all of "
            "the checkpoint tensors under these modules, or none: "
            f"{mixed[:8]}"
        )
    return skipped


@torch.no_grad()
def begin_direct_reload(model: torch.nn.Module) -> None:
    """Make every tensor writable by its checkpoint weight loader."""
    if not getattr(model, "_direct_layout_recorded", False):
        raise RuntimeError("direct reload: no checkpoint layout was recorded")
    if (reason := _unsupported(model)) is not None:
        raise NotImplementedError(f"direct reload: {reason}")

    # modules the trainer does not sync keep their runtime state untouched
    skipped = model._direct_skipped = _unsynced_modules(model)

    # checkpoint names post-load processing deleted or replaced: register the
    # checkpoint tensors again so the model's loader finds them; the rerun at
    # finish unregisters them as cold start did
    for module, name, is_param, t in getattr(model, "_direct_dropped", []):
        if id(module) not in skipped:
            _register(module, name, is_param, t)

    restored = 0
    for name, t in _loadable_tensors(model, skipped):
        layout = getattr(t, "_ckpt_layout", None)
        if layout is None:
            continue  # created by post-load processing: derived, rewritten there
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

    logger.info("direct reload: %d tensors viewed back to checkpoint layout", restored)


@torch.no_grad()
def finish_direct_reload(model: torch.nn.Module, model_config) -> None:
    """Rerun cold-start post-load processing, then check addresses."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    torch.accelerator.synchronize()  # engines may write on other streams

    # Unsynced modules keep their kernel layout and skip post-processing,
    # which would otherwise convert converted bytes again.
    skipped_ids = model.__dict__.pop("_direct_skipped")
    skipped = [m for m in model.modules() if id(m) in skipped_ids]
    for module in skipped:
        module._direct_skip_pwal = True
    # One-shot "already processed" guards (R7) would make the rerun below a
    # no-op and leave the freshly written checkpoint layout in place.
    cleared = 0
    for module in model.modules():
        if id(module) not in skipped_ids and module.__dict__.pop(
            "_already_called_process_weights_after_loading", None
        ):
            cleared += 1
    if cleared:
        logger.info("direct reload: cleared %d one-shot post-load guards", cleared)

    device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.accelerator.current_accelerator(),
    )
    # The storages captured graphs read, taken before the first rerun and kept
    # until a finish succeeds: a retry after a finish that failed midway pins
    # into them, not into the failed rerun's allocations.
    pinned = model.__dict__.get("_direct_side_state")
    if pinned is None:
        pinned = model._direct_side_state = _snapshot_side_state(model, skipped_ids)
    global _rerunning
    _rerunning = True
    try:
        process_weights_after_loading(model, model_config, device)
    finally:
        _rerunning = False
        for module in skipped:
            module.__dict__.pop("_direct_skip_pwal", None)
    unpinned = _pin_side_state(pinned)
    # hand the receive buffers and post-load temporaries back to the driver:
    # the next update's NCCL group allocates outside the caching allocator
    torch.accelerator.empty_cache()
    if skipped:
        logger.info(
            "direct reload: %d excluded modules kept their kernel layout", len(skipped)
        )

    current = _addresses(model)
    moved = sorted(n for n, p in model._direct_addresses.items() if current.get(n) != p)
    moved += sorted(unpinned)
    # kernels and captured graphs would keep reading the old storage
    if moved and os.environ.get("VLLM_DIRECT_RELOAD_ALLOW_MOVED") != "1":
        raise RuntimeError(f"direct reload: {len(moved)} tensors moved: {moved[:8]}")
    del model._direct_side_state


def _snapshot_side_state(model: torch.nn.Module, skipped_ids: Set[int]):
    """Before the rerun: each processed module's derived device tensors
    (`side_state`), as aliases of their storage. CUDA graphs captured at cold
    start read these storages, so the rerun's values have to land in them."""
    from .side_state import SideWalk, snapshot_side_tensors

    side_walk = SideWalk()
    pinned = [
        (name, module, snapshot_side_tensors(module, side_walk))
        for name, module in model.named_modules()
        if id(module) not in skipped_ids
    ]
    _log_truncated(side_walk, "snapshot")
    return pinned


def _pin_side_state(pinned) -> list[str]:
    """After the rerun: copy every derived tensor the rerun reallocated into
    its startup storage and rebind it there. Returns the tensors that cannot
    be pinned (gone, or a different layout)."""
    from .side_state import SideWalk, pin_side_tensors

    side_walk = SideWalk()
    failed = [
        f"{name}.{path}" if name else path
        for name, module, side in pinned
        for path in (p.lstrip(".") for p in pin_side_tensors(module, side, side_walk))
    ]
    _log_truncated(side_walk, "pin")
    return failed


def _log_truncated(side_walk, what: str) -> None:
    from .side_state import _MAX_DEPTH

    if side_walk.truncated:
        logger.warning(
            "direct reload: the side-state %s walk stopped at depth %d on %d "
            "objects; tensors below them are not pinned",
            what,
            _MAX_DEPTH,
            side_walk.truncated,
        )
