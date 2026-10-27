# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload.

Instead of moving layers to meta and rebuilding them, the loader writes
checkpoint-format tensors straight into the live tensors and the cold-start
post-load processing runs once at the end:

- cold start records each tensor's checkpoint layout (before post-load
  processing); the first ``begin`` records every address the model serves
  with (after the runner wired a draft to the target's modules);
- ``begin`` views tensors whose layout post-load processing changed back to
  their checkpoint layout, on the same storage, and records the checkpoint
  names fed to ``model.load_weights`` until ``finish``;
- ``finish`` raises unless the update delivered every name cold start was
  fed (`record_checkpoint_weight_names`), except those
  ``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX`` names, then reruns
  ``process_weights_after_loading`` for every module, as
  cold start does, and raises if an address or a host float moved: of
  a parameter, a buffer, or a derived tensor held elsewhere
  (`side_state`).
  Post-load processing has to refill derived tensors in place (captured CUDA
  graphs keep their startup addresses); nothing moves them back.

A reload is exactly as correct as a cold start on the delivered weights: no
module is skipped because it "received nothing" (a loader may bypass the
parameter's ``weight_loader``, and post-processing may read other modules'
tensors).

Delivery contract: an update must deliver every checkpoint tensor of each
module whose post-load processing changes the tensor's layout or is not
idempotent (it would convert the kept runtime bytes again). A tensor without
such processing (e.g. an embedding table the trainer keeps frozen) may be left
out and keeps its bytes; the update names it in
``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX``. ``finish`` checks the names, not which
tensors may be left out.

Post-load processing must keep every checkpoint tensor registered under its
checkpoint name (`replace_parameter`, `retire_parameter`): a reload loads it
by that name.

Selected with ``VLLM_RELOAD_MODE=direct``.
"""

import math
from collections.abc import Iterator

import regex as re
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger

from .side_state import side_tensors

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
    "finish_direct_reload",
    "record_checkpoint_layout",
    "record_delivered",
]


def direct_reload_enabled() -> bool:
    return envs.VLLM_RELOAD_MODE == "direct"


def _own_tensors(module: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """A module's own parameters and persistent buffers."""
    for n, p in module._parameters.items():
        if p is not None:
            yield n, p
    for n, b in module._buffers.items():
        if b is not None and n not in module._non_persistent_buffers_set:
            yield n, b


def _loadable_tensors(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters and persistent buffers, each object once."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for name, tensor in _own_tensors(module):
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
            out[name] = _where(t)
    return out


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


def _same_float(a: float | None, b: float) -> bool:
    return a == b or (a is not None and math.isnan(a) and math.isnan(b))


def _where(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype)


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout. Runs once; reloads reuse the cold-start record."""
    if getattr(model, "_direct_layout_recorded", False):
        return
    model._direct_layout_recorded = True
    config = get_current_vllm_config_or_none()
    spec = config.speculative_config if config is not None else None
    model._direct_draft_lm_head_quantization = (
        spec.draft_lm_head_quantization if spec is not None else None
    )
    for _, t in _loadable_tensors(model):
        if not t.is_meta and not hasattr(t, "_ckpt_layout"):
            t._ckpt_layout = (tuple(t.shape), t.stride(), t.dtype, t.storage_offset())


def _excluded_names() -> "re.Pattern[str] | None":
    """``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX``: checkpoint names an update may
    leave out (e.g. a frozen Engram table)."""
    pattern = envs.VLLM_DIRECT_RELOAD_EXCLUDE_REGEX
    try:
        return re.compile(pattern) if pattern else None
    except re.error as e:
        raise ValueError(
            f"VLLM_DIRECT_RELOAD_EXCLUDE_REGEX={pattern!r} is not a regex: {e}"
        ) from e


def record_delivered(model: torch.nn.Module, name: str) -> None:
    """Record that the current direct reload delivered checkpoint tensor
    ``name``; a no-op outside one. ``model.load_weights`` calls this for
    every tensor it is fed; an engine writing a tensor without it (m2n
    reshards) calls it."""
    delivery = model.__dict__.get("_direct_delivery")
    if delivery is not None:
        delivery[0].add(name)


def _track_delivery(model: torch.nn.Module, excluded) -> None:
    """Until finish, feed ``model.load_weights`` through `record_delivered`,
    as `record_checkpoint_weight_names` does at cold start: whoever loads
    the update (an engine, ``reload_weights``) is checked alike."""
    model._direct_delivery = (set(), excluded)
    load_weights = getattr(model, "load_weights", None)
    if load_weights is None or getattr(load_weights, "_direct_tracked", False):
        return

    def tracked(weights, *args, **kwargs):
        def fed():
            for name, tensor in weights:
                record_delivered(model, name)
                yield name, tensor

        return load_weights(fed(), *args, **kwargs)

    tracked._direct_tracked = True  # type: ignore[attr-defined]
    model.load_weights = tracked


def _check_delivered(model: torch.nn.Module) -> set[str] | None:
    """Before any post-load processing, which would convert the runtime
    bytes of a tensor that did not arrive again (they are viewed as
    checkpoint bytes): raise unless the update delivered every checkpoint
    name cold start fed to ``load_weights``, except the excluded ones, and
    every part of each multi-tensor weight. A model cold start fed nothing
    (dummy weights) adopts the first update's names, returned here; an empty
    first update raises."""
    from vllm.model_executor.model_loader.base_loader import CHECKPOINT_WEIGHT_NAMES
    from vllm.model_executor.model_loader.weight_utils import (
        check_weight_parts_complete,
    )

    delivered, excluded = model.__dict__.pop("_direct_delivery")
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
            f"its cold start loaded: {missing[:16]}. Deliver them, or name them "
            "in VLLM_DIRECT_RELOAD_EXCLUDE_REGEX if their modules have no "
            "layout-changing post-load processing"
        )
    check_weight_parts_complete(model)
    return delivered if required is None else None


def _checkpoint_views(
    model: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(tensor, view of its storage in its checkpoint layout) for every
    tensor post-load processing re-laid out; raises if a storage cannot
    hold its checkpoint layout."""
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, t in _loadable_tensors(model):
        layout = getattr(t, "_ckpt_layout", None)
        if layout is None:
            continue  # no layout: created by post-load processing, derived
        if (tuple(t.shape), t.stride(), t.dtype, t.storage_offset()) == layout:
            continue
        shape, stride, dtype, offset = layout
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
        views.append((t, view))
    return views


@torch.no_grad()
def begin_direct_reload(model: torch.nn.Module) -> None:
    """Make every tensor writable by its checkpoint weight loader.

    Everything that can refuse the reload is checked first and raises with
    the model untouched. Once this returns, the model holds checkpoint-layout
    bytes until a finish succeeds."""
    if not getattr(model, "_direct_layout_recorded", False):
        raise RuntimeError("direct reload: no checkpoint layout was recorded")
    if model._direct_draft_lm_head_quantization is not None:
        # the draft's quantized lm_head is a view of the target lm_head's
        # storage after `begin`: post-load processing would write it there
        raise RuntimeError("direct reload: draft_lm_head_quantization is not supported")
    views = _checkpoint_views(model)
    excluded = _excluded_names()

    # nothing above modified the model
    # The derived tensors captured graphs read, as served (before any view
    # back), kept until a finish succeeds: a retry after an update that
    # failed midway still compares with the served ones.
    if "_direct_side_tensors" not in model.__dict__:
        model._direct_side_tensors = side_tensors(model)
    # Likewise every address as served: a draft shares the target's embedding
    # and lm_head only after its own cold start (llm_base_proposer)
    if "_direct_addresses" not in model.__dict__:
        model._direct_addresses = _addresses(model)
        model._direct_host_scalars = _host_scalars(model)
    _track_delivery(model, excluded)
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    logger.info(
        "direct reload: %d tensors viewed back to checkpoint layout", len(views)
    )


@torch.no_grad()
def finish_direct_reload(model: torch.nn.Module, model_config) -> None:
    """Check the update was complete, rerun cold-start post-load processing,
    then check addresses and host floats."""
    from vllm.model_executor.model_loader.base_loader import CHECKPOINT_WEIGHT_NAMES
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    adopted = _check_delivered(model)
    torch.accelerator.synchronize()  # engines may write on other streams

    # One-shot "already processed" guards (R7) would make the rerun below a
    # no-op and leave the freshly written checkpoint layout in place.
    cleared = 0
    for module in model.modules():
        if module.__dict__.pop("_already_called_process_weights_after_loading", None):
            cleared += 1
    if cleared:
        logger.info("direct reload: cleared %d one-shot post-load guards", cleared)

    device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.accelerator.current_accelerator(),
    )
    process_weights_after_loading(model, model_config, device)

    current = _addresses(model)
    moved = sorted(n for n, p in model._direct_addresses.items() if current.get(n) != p)
    side = {n: _where(t) for n, t in side_tensors(model).items()}
    served = model._direct_side_tensors
    moved += sorted(n for n, t in served.items() if side.get(n) != _where(t))
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
    del model._direct_side_tensors
    if adopted is not None:
        setattr(model, CHECKPOINT_WEIGHT_NAMES, adopted)
        logger.warning(
            "Cold start loaded no checkpoint: the first weight update of %s "
            "must have been complete; later ones must deliver its %d "
            "checkpoint tensors",
            type(model).__name__,
            len(adopted),
        )
