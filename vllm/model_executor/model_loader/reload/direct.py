# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload.

Every update is a full sync of checkpoint-format tensors. Instead of moving
layers to meta and rebuilding them, the loader writes straight into the live
tensors and the cold-start post-load processing runs once at the end:

- cold start records each tensor's checkpoint layout (before post-load
  processing); the first ``begin`` records every address the model serves
  with (after the runner wired a draft to the target's modules);
- ``begin`` views tensors whose layout post-load processing changed back to
  their checkpoint layout, on the same storage;
- ``begin`` records the checkpoint names fed to ``model.load_weights`` until
  ``finish``, as cold start records them;
- ``finish`` raises unless the update delivered every name cold start was
  fed, then reruns ``process_weights_after_loading`` for every module, as
  cold start does, and raises if an address moved. Post-load processing
  has to refill derived tensors in place; nothing moves them back.

A reload is exactly as correct as a cold start: no module is skipped because
it "received nothing" (a loader may bypass the parameter's ``weight_loader``,
and post-processing may read other modules' tensors). The only exception is
tensors the trainer deliberately does not sync, named by the regex
``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX`` (e.g. Engram). It has to cover whole
modules with their submodules, which a reload leaves untouched: no view-back,
no re-registration, no post-processing (`_unsynced_modules` has the rule).

Selected with ``VLLM_RELOAD_MODE=direct``.
"""

import functools
import math
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
    "record_delivered",
    "record_dropped_checkpoint_tensors",
    "unsynced_weights_regex",
]


class DirectReloadRefused(RuntimeError):
    """`begin_direct_reload` refused the reload before modifying the model,
    which still holds the weights it served with."""


def direct_reload_enabled() -> bool:
    return os.environ.get("VLLM_RELOAD_MODE", "layerwise") == "direct"


def unsynced_weights_regex() -> "re.Pattern[str] | None":
    """Names the trainer deliberately does not sync (e.g. Engram); weight
    updates do not have to deliver them. A malformed pattern refuses the
    update before anything is modified (`begin_direct_reload` compiles it
    first).

    The regex must name whole vLLM modules (`_unsynced_modules`) or
    checkpoint-only names (MTP layers the served model does not load); a
    tied ``lm_head`` must be delivered. A regex naming only some checkpoint
    parts of a fused module (``gate_proj`` of ``gate_up_proj``) is a
    configuration error that nothing detects: the module is not skipped and
    reconverts its bytes."""
    pattern = os.environ.get("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", "")
    try:
        return re.compile(pattern) if pattern else None
    except re.error as e:
        raise DirectReloadRefused(
            f"VLLM_DIRECT_RELOAD_EXCLUDE_REGEX={pattern!r} is not a regex: {e}"
        ) from e


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


def record_dropped_checkpoint_tensors(model: torch.nn.Module) -> None:
    """Cold start, after post-load processing: keep the checkpoint tensors
    post-load processing unregistered (deleted, or replaced under the same
    name) so each reload can register them again."""
    if hasattr(model, "_direct_dropped"):
        return
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


def _unsupported(model: torch.nn.Module, skipped: Set[int]) -> str | None:
    from vllm.config import get_current_vllm_config_or_none
    from vllm.model_executor.layers.attention import is_deferred_attention_layer
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    if model._direct_pre_processed:
        return "weights loaded pre-processed (IPC weight cache)"
    config = get_current_vllm_config_or_none()
    if config is not None and config.parallel_config.enable_eplb:
        # loading places experts as at cold start, routing keeps EPLB's map
        return "EPLB rearranges experts after cold start"
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


def record_delivered(model: torch.nn.Module, name: str) -> bool:
    """Record that the current direct reload delivered checkpoint tensor
    ``name``, before it is written; True outside one. False for a name
    ``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX`` excludes (a reload leaves those
    tensors in their runtime layout, which checkpoint bytes are not): the
    caller drops it unwritten and `finish_direct_reload` raises. Raising
    here instead would leave a trainer blocked in its broadcast.
    ``model.load_weights`` calls this for every tensor it is fed; an engine
    writing a tensor without it (m2n reshards) calls it."""
    delivery = model.__dict__.get("_direct_delivery")
    if delivery is None:
        return True
    delivered, refused, excluded = delivery
    if excluded is not None and excluded.search(name):
        refused.append(name)
        return False
    delivered.add(name)
    return True


def _track_delivery(model: torch.nn.Module, excluded) -> None:
    """Until finish, feed ``model.load_weights`` through `record_delivered`,
    as `record_checkpoint_weight_names` does at cold start: whoever loads
    the update (an engine, ``reload_weights``) is checked alike."""
    model._direct_delivery = (set(), [], excluded)
    load_weights = getattr(model, "load_weights", None)
    if load_weights is None or getattr(load_weights, "_direct_tracked", False):
        return

    def tracked(weights, *args, **kwargs):
        fed = ((n, t) for n, t in weights if record_delivered(model, n))
        return load_weights(fed, *args, **kwargs)

    tracked._direct_tracked = True  # type: ignore[attr-defined]
    model.load_weights = tracked


def _check_delivered(model: torch.nn.Module) -> set[str] | None:
    """Before any post-load processing, which would convert the runtime
    bytes of a tensor that did not arrive again (they are viewed as
    checkpoint bytes): raise unless the update delivered every checkpoint
    name cold start fed to ``load_weights`` (`record_checkpoint_weight_names`),
    except the excluded ones, none of those, and every part of each
    multi-tensor weight. A
    model cold start fed nothing (dummy weights) adopts the first update's
    names, returned here; an empty first update raises."""
    from vllm.model_executor.model_loader.base_loader import CHECKPOINT_WEIGHT_NAMES
    from vllm.model_executor.model_loader.weight_utils import (
        check_weight_parts_complete,
    )

    delivered, refused, excluded = model.__dict__.pop("_direct_delivery")
    if refused:
        raise ValueError(
            f"weight update delivered {len(refused)} tensors that "
            "VLLM_DIRECT_RELOAD_EXCLUDE_REGEX declares unsynced; they were not "
            f"written: {refused[:16]}"
        )
    required: set[str] | None = getattr(model, CHECKPOINT_WEIGHT_NAMES, None)
    if required is None and not delivered:
        raise ValueError(
            "the first weight update delivered no tensors; with no checkpoint "
            "loaded at cold start it must deliver all of them"
        )
    missing = sorted(
        n
        for n in (required or set()) - delivered
        if excluded is None or not excluded.search(n)
    )
    if missing:
        raise ValueError(
            f"weight update did not deliver {len(missing)} checkpoint tensors "
            f"its cold start loaded: {missing[:16]}. Deliver them, or exclude "
            "them with VLLM_DIRECT_RELOAD_EXCLUDE_REGEX, which must name whole "
            "vLLM modules or checkpoint-only names (MTP layers the served model "
            "does not load); a tied lm_head must be delivered"
        )
    check_weight_parts_complete(model)
    return delivered if required is None else None


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
    bytes, and must refill any derived tensor it keeps for them in place.
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
    # a tensor a synced module shares (a tied lm_head) is written by the update
    synced = {id(t) for _, t in _loadable_tensors(model, skipped)}
    tied = [
        f"{name}.{local}"
        for name, module in modules
        if id(module) in skipped
        for local, t, _ in _own_tensors(module)
        if id(t) in synced
    ]
    if tied:
        raise DirectReloadRefused(
            "direct reload: VLLM_DIRECT_RELOAD_EXCLUDE_REGEX excludes tensors "
            f"a synced module shares, which must be delivered: {tied[:8]}"
        )

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
            "direct reload: VLLM_DIRECT_RELOAD_EXCLUDE_REGEX must name whole "
            "vLLM modules or checkpoint-only names (unused MTP layers), "
            "without a module whose post-load processing reads them: "
            f"{bad[:8]}"
        )
    return skipped


def _refuse_writes(model: torch.nn.Module, skipped: Set[int]) -> None:
    """Skipped modules hold kernel-layout bytes nothing converts again, so a
    write into them is refused, whatever checkpoint name it came under (the
    regex is matched on vLLM names here, on checkpoint names by the engines;
    a fused or renamed tensor matches in one namespace only)."""
    for name, module in model.named_modules():
        if id(module) in skipped:
            for local, t, _ in _own_tensors(module):
                t.weight_loader = functools.partial(_refuse, f"{name}.{local}")


def _refuse(name: str, *args, **kwargs) -> None:
    raise ValueError(
        f"direct reload: the update writes {name}, which "
        "VLLM_DIRECT_RELOAD_EXCLUDE_REGEX excludes (in vLLM's names)"
    )


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
    # Every address as served: a draft shares the target's embedding
    # and lm_head only after its own cold start (llm_base_proposer)
    if "_direct_addresses" not in model.__dict__:
        model._direct_addresses = _addresses(model)
        model._direct_host_scalars = _host_scalars(model)
    model._direct_skipped = skipped
    _track_delivery(model, unsynced_weights_regex())
    _refuse_writes(model, skipped)
    for module, name, is_param, t in dropped:
        _register(module, name, is_param, t)
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    logger.info(
        "direct reload: %d tensors viewed back to checkpoint layout", len(views)
    )


@torch.no_grad()
def finish_direct_reload(model: torch.nn.Module, model_config) -> None:
    """Check the update was complete, rerun cold-start post-load processing,
    then check addresses."""
    from vllm.model_executor.model_loader.base_loader import CHECKPOINT_WEIGHT_NAMES
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    adopted = _check_delivered(model)
    torch.accelerator.synchronize()  # engines may write on other streams

    # Unsynced modules keep their kernel layout and skip post-processing,
    # which would otherwise convert converted bytes again.
    skipped = model.__dict__.pop("_direct_skipped")
    # One-shot "already processed" guards (R7) would make the rerun below a
    # no-op and leave the freshly written checkpoint layout in place.
    cleared = 0
    for module in model.modules():
        if id(module) not in skipped and module.__dict__.pop(
            "_already_called_process_weights_after_loading", None
        ):
            cleared += 1
    if cleared:
        logger.info("direct reload: cleared %d one-shot post-load guards", cleared)

    device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.accelerator.current_accelerator(),
    )
    process_weights_after_loading(model, model_config, device, skipped)
    # hand the receive buffers and post-load temporaries back to the driver:
    # the next update's NCCL group allocates outside the caching allocator
    torch.accelerator.empty_cache()
    if skipped:
        logger.info(
            "direct reload: %d excluded modules kept their kernel layout", len(skipped)
        )

    current = _addresses(model)
    moved = sorted(n for n, p in model._direct_addresses.items() if current.get(n) != p)
    scalars = _host_scalars(model)
    moved += sorted(
        f"{n} (host scalar {v} -> {scalars.get(n)})"
        for n, v in model._direct_host_scalars.items()
        if not _same_float(scalars.get(n), v)
    )
    # kernels and captured graphs would keep reading the old storage
    if moved:
        logger.error("direct reload: tensors moved: %s", moved)
        raise RuntimeError(f"direct reload: {len(moved)} tensors moved: {moved[:8]}")
    if adopted is not None:
        setattr(model, CHECKPOINT_WEIGHT_NAMES, adopted)
        logger.warning(
            "Cold start loaded no checkpoint: the first weight update of %s "
            "must have been complete; later ones must deliver its %d "
            "checkpoint tensors",
            type(model).__name__,
            len(adopted),
        )


def _same_float(a: float | None, b: float) -> bool:
    return a == b or (a is not None and math.isnan(a) and math.isnan(b))


def _host_scalars(model: torch.nn.Module) -> dict[str, float]:
    """Python floats on modules, e.g. the KV-cache scales FlashInfer folds into
    bmm1/bmm2_scale: kernels take them as host arguments, which captured
    graphs (and backends caching the products) keep, so a reload must not
    change them."""
    return {
        f"{name}.{k}": v
        for name, module in model.named_modules()
        for k, v in vars(module).items()
        if type(v) is float
    }
