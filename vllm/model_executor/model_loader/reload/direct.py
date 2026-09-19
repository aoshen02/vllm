# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload: checkpoint weights are written into the live parameters.

Layerwise reload moves each layer to the meta device, loads a temporary copy,
re-runs `process_weights_after_loading` and copies the result back into the
storage a CUDA graph captured. For a model whose post-load step leaves the
parameters as the checkpoint has them, that detour is pure overhead, and its
temporary copies span every layer a trainer has started but not finished.

Direct reload skips it: `start` records where every parameter and buffer
lives, `model.load_weights` runs unchanged so each `weight_loader` `copy_`s
into live storage, and `finish` checks that nothing moved or changed layout.
Nothing inspects the model; selecting the mode is the operator's statement
that no post-processing needs redoing. Completeness of the checkpoint is the
sender's contract, as it is for layerwise. There is no rollback: a failure
between `start` and a successful `finish` leaves the model undefined, it
refuses further updates and the engine must be restarted.
"""

import torch

__all__ = ["direct_start", "direct_finish", "direct_abort"]

_UNDEFINED = "_direct_reload_undefined"
_LIVE = "_direct_reload_live"


def direct_start(model: torch.nn.Module) -> None:
    if _LIVE in model.__dict__:
        raise RuntimeError("direct weight reload already in progress")
    if getattr(model, _UNDEFINED, False):
        raise RuntimeError(
            "a previous direct weight reload failed; the weights are undefined "
            "and the engine must be restarted"
        )
    if getattr(model, "_do_torchao_reload", False):
        raise RuntimeError(
            "torchao models re-quantize after loading; use reload_mode=layerwise"
        )
    _unwrap_layerwise_loaders(model)
    model.__dict__[_LIVE] = _tensor_layouts(model)
    setattr(model, _UNDEFINED, True)


def direct_finish(model: torch.nn.Module) -> None:
    before = model.__dict__.pop(_LIVE)
    after = _tensor_layouts(model)
    moved = [n for n, rec in before.items() if after.get(n) != rec]
    if moved:
        raise RuntimeError(
            f"direct weight reload relocated {', '.join(moved[:5])}"
            f"{', ...' if len(moved) > 5 else ''}: a loader replaced storage a "
            "CUDA graph may have captured, or changed its shape, strides or "
            "dtype. The weights are undefined and the engine must be restarted."
        )
    setattr(model, _UNDEFINED, False)


def direct_abort(model: torch.nn.Module) -> None:
    """Whatever was written stays written; the model stays undefined."""
    model.__dict__.pop(_LIVE, None)


def _tensor_layouts(model: torch.nn.Module) -> dict[str, tuple]:
    # Address and layout: a same-storage transpose or dtype view keeps
    # `data_ptr()` but a captured graph would read it with the old layout.
    # Not deduplicated: a name sharing storage with another is still a name a
    # graph may read through, and rebinding it would otherwise go unnoticed.
    return {
        name: (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype)
        for name, t in (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    }


def _unwrap_layerwise_loaders(model: torch.nn.Module) -> None:
    """Drop `online_process_loader` wrappers a layerwise update left behind.

    A layer the checkpoint never completed keeps its wrapper after the layer's
    reload state is reset; that wrapper buffers instead of copying and returns
    early once the state is gone, so a direct load through it would silently
    drop the tensor.
    """
    from .layerwise import LAYERWISE_INFO, _get_original_loader, _get_weight_loader
    from .utils import get_layer_tensors

    for layer in model.modules():
        info = LAYERWISE_INFO.get(layer)
        if info is not None and info.can_load():
            raise RuntimeError(
                f"{type(layer).__name__} is mid-way through a layerwise reload; "
                "finish or abort it before a direct one"
            )
        for param in get_layer_tensors(layer).values():
            # `functools.partial` loaders (RMSNormTP) have no `__name__`.
            loader = _get_weight_loader(param)
            if getattr(loader, "__name__", None) == "online_process_loader":
                param.weight_loader = _get_original_loader(param)
