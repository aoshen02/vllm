# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public entry points for reloading weights into a live model.

Every caller that streams checkpoint-format weights into an already-built model
goes through these three functions rather than naming the mechanism behind
them, so the mechanism can change in one place::

    start_reload(model)
    try:
        model.load_weights(weights)
        finish_reload(model, model_config)
    except BaseException:
        abort_reload(model)
        raise
"""

import torch

from vllm.config import ModelConfig

from .layerwise import (
    abort_layerwise_reload,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

__all__ = ["start_reload", "finish_reload", "abort_reload"]


def start_reload(model: torch.nn.Module) -> None:
    """Prepare ``model`` to receive checkpoint-format weights."""
    initialize_layerwise_reload(model)


def finish_reload(model: torch.nn.Module, model_config: ModelConfig) -> None:
    """Complete a reload once every weight has been loaded."""
    finalize_layerwise_reload(model, model_config)


def abort_reload(model: torch.nn.Module) -> None:
    """Discard an in-progress reload and leave the model loadable again.

    Not a rollback: layers still waiting for weights get their pre-reload
    tensors back, but anything already written stays written.
    """
    abort_layerwise_reload(model)
