# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Begin and complete a checkpoint-format weight update in the reload mode
the model's cold start fixed."""

from weakref import WeakSet

import torch

from vllm.config import ModelConfig

from .direct import begin_direct_reload, finish_direct_reload, reloads_directly
from .layerwise import (
    abort_layerwise_reload,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

__all__ = ["abort_reload", "finish_reload", "start_reload", "weight_update_active"]


_active_reloads: WeakSet[torch.nn.Module] = WeakSet()
_weight_update_active = False


def weight_update_active() -> bool:
    """Whether a weight update is running, when one `load_weights` call may hold
    only part of the checkpoint. Process-wide, so a module nested in the model
    passed to `start_reload` sees it too."""
    return _weight_update_active or bool(_active_reloads)


def start_reload(model: torch.nn.Module) -> None:
    """Begin a weight update in the configured mode."""
    if reloads_directly(model):
        begin_direct_reload(model)
    else:
        initialize_layerwise_reload(model)
    _active_reloads.add(model)


def abort_reload(model: torch.nn.Module) -> None:
    """Clear active reload state after a mid-update failure so a failed update
    cannot leave `weight_update_active()` stuck at `True` or leave layerwise
    parameters stranded on `meta`."""
    _active_reloads.discard(model)
    if reloads_directly(model):
        model.__dict__.pop("_direct_reload_begun", None)
        model.__dict__.pop("_loaded_params", None)
    else:
        abort_layerwise_reload(model)


def finish_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Complete the weight update `start_reload` began."""
    _active_reloads.discard(model)
    if reloads_directly(model):
        finish_direct_reload(model, model_config, device)
    else:
        finalize_layerwise_reload(model, model_config)
