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
no re-registration, no post-processing (`_unsynced_modules` has the rule).

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
    "DirectReloadRefused",
    "begin_direct_reload",
    "direct_reload_enabled",
    "finish_direct_reload",
    "record_checkpoint_layout",
    "record_runtime_addresses",
    "unsynced_weights_regex",
]


class DirectReloadRefused(RuntimeError):
    """`begin_direct_reload` refused the reload before modifying the model,
    which still holds the weights it served with."""


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
    model._direct_tied = _guard_tied_writes(model)


def _guard_tied_writes(model: torch.nn.Module) -> list[torch.Tensor]:
    """Cold start decided whether the lm_head is the input embedding's tensor
    (`maybe_retie_word_embeddings`); reloads keep that decision. A cold start
    on an update whose two copies differ would untie them, so a tied tensor
    must receive the same bytes every time an update writes it."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    def weights(cls, exclude=()):
        for m in model.modules():
            w = getattr(m, "weight", None)
            if isinstance(m, cls) and not isinstance(m, exclude) and w is not None:
                yield w

    inputs = {id(w) for w in weights(VocabParallelEmbedding, ParallelLMHead)}
    tied = {id(w): w for w in weights(ParallelLMHead) if id(w) in inputs}
    for w in tied.values():
        loader = getattr(w, "weight_loader", default_weight_loader)

        def load(param, loaded_weight, *args, _load=loader, **kwargs):
            if not getattr(param, "_direct_written", False):
                param._direct_written = True
                return _load(param, loaded_weight, *args, **kwargs)
            first = param.detach().clone()
            _load(param, loaded_weight, *args, **kwargs)
            if not torch.equal(first, param):
                raise ValueError(
                    "direct reload: the update delivers different lm_head and "
                    "input embedding weights, which share one tensor since "
                    "cold start; a cold start on them would untie them. "
                    "Restart the server to serve them."
                )

        w.weight_loader = load
    return list(tied.values())


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


def _unsupported(model: torch.nn.Module, skipped: Set[int]) -> str | None:
    from vllm.model_executor.layers.attention import is_deferred_attention_layer
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    if model._direct_pre_processed:
        return "weights loaded pre-processed (IPC weight cache)"
    for name, module in model.named_modules():
        if id(module) in skipped:
            continue
        method = getattr(module, "quant_method", None)
        if isinstance(method, (OnlineLinearBase, OnlineMoEMethodBase)):
            return f"{name}: online quantization ({type(method).__name__})"
        # post-processing moves host weights to the device and back into new
        # host tensors (`device_loading_context`): their storage would move
        processed_on_device = is_deferred_attention_layer(module) or (
            isinstance(method, QuantizeMethodBase) and method.requires_device_loading
        )
        if processed_on_device and any(
            p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False)
            for p in module.parameters()
        ):
            return f"{name}: CPU-offloaded weights"
    return None


def _unsynced_modules(model: torch.nn.Module) -> set[int]:
    """Ids of the modules a reload leaves alone, from
    ``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX`` matched on canonical names (the
    first name `named_modules()` gives a module; each module once).

    A module is skipped iff it has checkpoint tensors in its subtree and the
    regex matches all of them; its descendants are skipped with it. A module
    that is not skipped raises if
    - it owns a matching tensor: its post-processing would convert the kept
      kernel-layout bytes again;
    - or it has post-load processing (a quantization method, or a deferred
      attention layer) and a skipped module below it: that processing read
      the skipped module at cold start, so it must not rerun on its own
      (e.g. MLA absorbing an excluded kv_b_proj).

    The model-level hook (`finalize_model_weights`) still runs: it belongs
    to the whole model, which a reload never skips. Like at cold start it
    sees the skipped modules in kernel layout, holding their cold-start
    bytes, and any derived tensor it reallocates for them is pinned (their
    side state is snapshotted with every other module's).
    """
    from vllm.model_executor.layers.attention import is_deferred_attention_layer

    excluded = unsynced_weights_regex()
    if excluded is None:
        return set()
    dropped: dict[int, list[str]] = {}
    for module, name, *_ in getattr(model, "_direct_dropped", []):
        dropped.setdefault(id(module), []).append(name)
    modules = list(model.named_modules())
    own: dict[int, list[bool]] = {}  # per checkpoint tensor: does it match
    for name, module in modules:
        prefix = f"{name}." if name else ""
        ckpt = [n for n, t, _ in _own_tensors(module) if hasattr(t, "_ckpt_layout")]
        own[id(module)] = [
            excluded.search(prefix + n) is not None
            for n in ckpt + dropped.get(id(module), [])
        ]
    subtree: dict[int, tuple[bool, bool]] = {}  # any matching, any synced

    def visit(module: torch.nn.Module) -> tuple[bool, bool]:
        if (seen := subtree.get(id(module))) is not None:
            return seen
        hits = own[id(module)]
        unsynced, synced = any(hits), not all(hits)
        for child in module.children():
            u, s = visit(child)
            unsynced, synced = unsynced or u, synced or s
        subtree[id(module)] = (unsynced, synced)
        return unsynced, synced

    skipped: set[int] = set()
    for _, module in modules:
        if id(module) not in skipped and visit(module) == (True, False):
            skipped.update(id(m) for m in module.modules())

    def reads_skipped(module: torch.nn.Module) -> bool:
        method = getattr(module, "quant_method", None)
        return (
            isinstance(method, QuantizeMethodBase)
            or is_deferred_attention_layer(module)
        ) and any(id(m) in skipped for m in module.modules())

    bad = [
        name or "<root>"
        for name, module in modules
        if id(module) not in skipped and (any(own[id(module)]) or reads_skipped(module))
    ]
    if bad:
        raise DirectReloadRefused(
            "direct reload: VLLM_DIRECT_RELOAD_EXCLUDE_REGEX must cover whole "
            "modules, without a module whose post-load processing reads them: "
            f"{bad[:8]}"
        )
    return skipped


def _checkpoint_views(
    model: torch.nn.Module, skipped: Set[int], dropped: list
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(tensor, view of its storage in its checkpoint layout) for every
    tensor post-load processing re-laid out; raises if a storage cannot
    hold its checkpoint layout."""
    tensors = list(_loadable_tensors(model, skipped))
    tensors += [(f"{type(m).__name__}.{n}", t) for m, n, _, t in dropped]
    views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, t in tensors:
        layout = getattr(t, "_ckpt_layout", None)
        if layout is None or id(t) in views:
            continue  # no layout: created by post-load processing, derived
        if (tuple(t.shape), t.stride(), t.dtype, t.storage_offset()) == layout:
            continue
        shape, stride, dtype, offset = layout
        storage = t.untyped_storage()
        span = 1 + sum((s - 1) * st for s, st in zip(shape, stride)) if shape else 1
        need = (offset + span) * dtype.itemsize
        if need > storage.nbytes():
            raise DirectReloadRefused(
                f"direct reload: {name} storage has {storage.nbytes()} bytes, "
                f"its checkpoint layout needs {need}"
            )
        view = torch.empty(0, dtype=dtype, device=t.device)
        view.set_(storage, offset, shape, stride)
        views[id(t)] = (t, view)
    return list(views.values())


@torch.no_grad()
def begin_direct_reload(model: torch.nn.Module) -> None:
    """Make every tensor writable by its checkpoint weight loader.

    Everything that can refuse the reload is checked first and raises
    `DirectReloadRefused` with the model untouched. Once this returns, the
    model holds checkpoint-layout bytes until a finish succeeds."""
    if not getattr(model, "_direct_layout_recorded", False):
        raise DirectReloadRefused("direct reload: no checkpoint layout was recorded")
    # modules the trainer does not sync keep their runtime state untouched
    skipped = _unsynced_modules(model)
    if (reason := _unsupported(model, skipped)) is not None:
        raise DirectReloadRefused(f"direct reload: {reason}")
    # checkpoint names post-load processing deleted or replaced: register the
    # checkpoint tensors again so the model's loader finds them; the rerun at
    # finish unregisters them as cold start did
    dropped = [
        entry
        for entry in getattr(model, "_direct_dropped", [])
        if id(entry[0]) not in skipped
    ]
    views = _checkpoint_views(model, skipped, dropped)

    # nothing above modified the model
    # The derived tensors captured graphs read, as served (before any view
    # back), kept until a finish succeeds: a retry after an update that
    # failed midway pins into them, not into the failed rerun's allocations.
    if "_direct_side_state" not in model.__dict__:
        model._direct_side_state = _snapshot_side_state(model)
    model._direct_skipped = skipped
    for module, name, is_param, t in dropped:
        _register(module, name, is_param, t)
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    for t in getattr(model, "_direct_tied", ()):
        t._direct_written = False
    logger.info(
        "direct reload: %d tensors viewed back to checkpoint layout", len(views)
    )


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
    pinned = model._direct_side_state  # taken by the first begin
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


def _snapshot_side_state(model: torch.nn.Module):
    """Before the update modifies the model: each module's derived device
    tensors (`side_state`), as aliases of their storage and view. CUDA graphs
    captured at cold start read these, so the rerun's values have to land in
    them. Skipped modules too: the model-level hook still runs over them."""
    from .side_state import SideWalk, snapshot_side_tensors

    side_walk = SideWalk()
    pinned = [
        (name, module, snapshot_side_tensors(module, side_walk))
        for name, module in model.named_modules()
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
