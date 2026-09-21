# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload: checkpoint weights are written into the live parameters.

`start` records where every parameter and buffer lives, `model.load_weights`
runs unchanged so each `weight_loader` `copy_`s into that storage, and `finish`
rebuilds the model-level derived state and checks nothing moved. Nothing
inspects the model beforehand: choosing the mode is the operator's statement
that no *per-layer* post-load processing needs redoing. There is no rollback --
a failure leaves the model undefined and the engine must restart.
"""

import torch

from .derived import (
    refresh_model_derived_state,
    relocated_names,
    tensor_layouts,
)

__all__ = ["direct_start", "direct_finish"]

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
    model.__dict__[_LIVE] = tensor_layouts(model)
    setattr(model, _UNDEFINED, True)


def direct_finish(model: torch.nn.Module) -> None:
    before = model.__dict__.pop(_LIVE)
    refresh_model_derived_state(model)
    moved = relocated_names(before, tensor_layouts(model))
    if moved:
        raise RuntimeError(
            f"direct weight reload relocated {', '.join(moved[:5])}"
            f"{', ...' if len(moved) > 5 else ''}: a loader replaced storage a "
            "CUDA graph may have captured, or changed its shape, strides or "
            "dtype. The weights are undefined and the engine must be restarted."
        )
    setattr(model, _UNDEFINED, False)
