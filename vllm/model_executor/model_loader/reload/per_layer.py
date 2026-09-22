# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether a model's layers transform their weights after loading.

`direct` reload skips per-layer post-load processing: its whole premise is that
running it again would change nothing. For a model that requantizes -- GLM-5.3
and DeepSeek V4.1 both do -- the premise is false, and an update in that mode
leaves the layer's kernel-format weights and scales derived from the previous
checkpoint. Nothing on the inference path notices.

The premise is not a judgement call; it is measurable, once, while the cold
start runs the pass anyway. `observe` records whether a layer's post-load
actually changed anything, and `refuse_direct_reload_if_layers_transform`
turns that observation into a refusal.
"""

import torch

__all__ = [
    "observe",
    "layers_that_transform",
    "refuse_direct_reload_if_layers_transform",
]

# Layer names whose post-load pass was seen to change a tensor, recorded on the
# model so it survives to the first reload.
_TRANSFORMING = "_reload_layers_transform_weights"


def _signature(module: torch.nn.Module) -> tuple:
    """Cheap stand-in for the layer's contents: one reduction per tensor.

    A sum catches a requantization, a repack and a rebind alike, and costs one
    kernel per tensor on a pass that is already touching all of them.
    """
    values: list[tuple | None] = []
    for _, tensor in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        if tensor is None or tensor.is_meta or not tensor.numel():
            values.append(None)
            continue
        values.append(
            (tuple(tensor.shape), tensor.dtype, float(tensor.detach().double().sum()))
        )
    return tuple(values)


def observe(model: torch.nn.Module, name: str, module: torch.nn.Module, before: tuple):
    """Record that ``module``'s post-load changed something, if it did."""
    if before == _signature(module):
        return
    model.__dict__.setdefault(_TRANSFORMING, []).append(name or type(module).__name__)


def signature(module: torch.nn.Module) -> tuple:
    return _signature(module)


def layers_that_transform(model: torch.nn.Module) -> list[str]:
    return list(model.__dict__.get(_TRANSFORMING, ()))


def refuse_direct_reload_if_layers_transform(model: torch.nn.Module) -> None:
    """Raise if this model's layers were seen to transform their weights.

    Raises:
        RuntimeError: a layer's cold-start post-load changed a tensor, so
            skipping it on an update would leave that layer derived from the
            previous checkpoint.

    """
    layers = layers_that_transform(model)
    if not layers:
        return
    shown = ", ".join(layers[:5]) + (", ..." if len(layers) > 5 else "")
    raise RuntimeError(
        f"{type(model).__name__} transforms weights when its layers finish "
        f"loading ({shown}), so reload_mode='direct', which does not run that "
        "pass, would leave them derived from the previous checkpoint. Use "
        "reload_mode='layerwise'."
    )
