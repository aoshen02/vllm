# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level derived state, and the storage snapshot both reload modes check.

A root model's ``process_weights_after_loading`` derives state that spans layers
-- Kimi K3 fuses its MegaMoE experts there -- and no reload path dispatched it.
So an updated model kept serving state derived from the previous checkpoint,
silently. Dispatching it is not enough to fix that, because it is a cold-start
contract and re-running one is only correct for a hook written to allow it:

* it must rebuild rather than skip. Every in-tree hook that spans layers guards
  on whether it has *ever* built its state, not on whether that state is stale,
  so re-running it returns immediately and the model stays on the old weights.
* it must write through the tensors the model already holds. ROCm DeepSeek-V4
  calls ``replace_parameter`` here, which swaps storage a CUDA graph captured.
* it must not need an input the cold-start pass consumed. The MegaMoE finalizers
  drop ``w13_weight`` once fused, and nothing restores it.
* it must be idempotent in ways no tensor snapshot can see. Dots3 compiles its
  vision blocks here, so a second call wraps an already-compiled module.

Rather than guess which hook is which, a model is refused unless it declares
``reload_safe_post_load``. Opting in is a per-model review, and the layout check
below then runs as a second line of defence rather than as the only one.
"""

import torch

from .owned import snapshot_owned_tensors

__all__ = [
    "RELOAD_SAFE",
    "check_post_load_is_reload_safe",
    "tensor_layouts",
    "relocated_names",
    "refresh_model_derived_state",
]

# A model sets this to True to declare that its model-level post-load hook meets
# the conditions above. Absent, a reload of that model is refused.
RELOAD_SAFE = "reload_safe_post_load"


def has_model_level_hook(model: torch.nn.Module) -> bool:
    return getattr(model, "process_weights_after_loading", None) is not None


def check_post_load_is_reload_safe(model: torch.nn.Module) -> None:
    """Refuse a reload of a model whose post-load hook has not been reviewed.

    Called from ``start_reload``, before any weight is written. Refusing at the
    end instead would leave the model holding the new checkpoint's ordinary
    weights beside the previous one's derived state: the caller sees the error,
    but nothing on the inference path consults the reload lifecycle, so the
    server would go on serving that mixture.

    Raises:
        RuntimeError: ``model`` has a model-level hook and has not declared
            ``reload_safe_post_load``.

    """
    if not has_model_level_hook(model) or getattr(model, RELOAD_SAFE, False):
        return
    raise RuntimeError(
        f"{type(model).__name__} derives model-level state in "
        "process_weights_after_loading, and that hook has not been declared "
        f"safe to re-run on a weight reload ({RELOAD_SAFE}). Refusing before "
        "any weight is written, rather than serving state derived from the "
        f"previous checkpoint. See {__name__} for what a hook has to satisfy."
    )


def _layout(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype)


def tensor_layouts(model: torch.nn.Module) -> dict[str, tuple]:
    """Where every tensor a module holds lives, and how it is laid out.

    Address alone is not enough: a same-storage transpose or dtype view keeps
    ``data_ptr()`` while a captured graph would read those bytes with the old
    layout. Names sharing storage are kept (``remove_duplicate=False``) because
    each is a name a graph may read through, and rebinding one to the storage
    another already points at would otherwise go unnoticed.

    Plain attribute tensors count too, including ones a module reaches only
    through an object it owns. They are the easier mistake to make --
    ``self.x = torch.stack(...)`` in a post-load hook allocates every time --
    and forward reads them exactly as it reads a buffer, so a captured graph is
    left pointing at freed storage just the same.
    """
    layouts = {
        name: _layout(t)
        for name, t in (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    }
    for name, owned in snapshot_owned_tensors(model).items():
        layouts[name] = _layout(owned.tensor)
    return layouts


def relocated_names(before: dict[str, tuple], after: dict[str, tuple]) -> list[str]:
    """Names that no longer describe the storage they did, including ones gone.

    A vanished name counts: a captured graph holds the tensor, not the
    attribute, so dropping the attribute frees storage the graph still reads.
    Consuming a parameter is only safe at cold start, before capture, which is
    why a model that does it in this hook does not get to opt in.

    """
    return [name for name, layout in before.items() if after.get(name) != layout]


def refresh_model_derived_state(model: torch.nn.Module) -> None:
    """Rebuild ``model``'s model-level derived state from the weights just loaded.

    ``start_reload`` already refused a model that has a hook and has not
    declared ``reload_safe_post_load``, so reaching here without the
    declaration means the caller drove the mechanism directly; leave such a
    model alone rather than run an unreviewed hook.

    Raises:
        RuntimeError: if the hook rebinds a parameter, buffer or attribute
            tensor instead of writing through it, leaving the model undefined.

    """
    if not has_model_level_hook(model) or not getattr(model, RELOAD_SAFE, False):
        return
    hook = model.process_weights_after_loading

    before = tensor_layouts(model)
    hook()
    moved = relocated_names(before, tensor_layouts(model))
    if moved:
        raise RuntimeError(
            f"{type(model).__name__}.process_weights_after_loading relocated "
            f"{', '.join(moved[:5])}{', ...' if len(moved) > 5 else ''} while "
            "rebuilding model-level derived state after a weight reload. It "
            "must write through the tensors the model already holds, since a "
            "CUDA graph may have captured their storage. The model's weights "
            "are in an undefined state and the engine must be restarted."
        )
