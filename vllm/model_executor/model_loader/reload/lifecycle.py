# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public entry points for reloading weights into a live model.

Every caller that streams checkpoint-format weights into an already-built model
goes through these three functions rather than naming the mechanism behind
them, so the mechanism can change in one place::

    start_reload(model, mode)
    try:
        model.load_weights(weights)
        finish_reload(model, model_config)
    except BaseException:
        abort_reload(model)
        raise

``mode`` is a value of ``WeightTransferConfig.reload_mode``. It is chosen at
``start_reload``; ``finish_reload`` and ``abort_reload`` complete whichever
mode was started.
"""

import torch

from vllm.config import ModelConfig

from .direct import direct_abort, direct_finish, direct_start
from .layerwise import (
    abort_layerwise_reload,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

__all__ = ["start_reload", "finish_reload", "abort_reload"]

_STARTED_MODE = "_reload_started_mode"


def start_reload(model: torch.nn.Module, mode: str = "layerwise") -> None:
    """Prepare ``model`` to receive checkpoint-format weights."""
    if mode == "direct":
        direct_start(model)
    elif mode == "layerwise":
        initialize_layerwise_reload(model)
    else:
        raise ValueError(f"unknown reload mode {mode!r}")
    model.__dict__[_STARTED_MODE] = mode


def finish_reload(model: torch.nn.Module, model_config: ModelConfig) -> None:
    """Complete the reload that ``start_reload`` began."""
    mode = model.__dict__.pop(_STARTED_MODE, None)
    if mode is None:
        raise RuntimeError("finish_reload called without a matching start_reload")
    if mode == "direct":
        direct_finish(model)
    else:
        finalize_layerwise_reload(model, model_config)


def abort_reload(model: torch.nn.Module) -> None:
    """Discard the in-progress reload. Not a rollback: layerwise puts the
    pre-reload tensors back on layers still waiting for weights; direct leaves
    everything written and the model undefined. A no-op when nothing was
    started, so it is safe on any error path."""
    mode = model.__dict__.pop(_STARTED_MODE, None)
    if mode == "direct":
        direct_abort(model)
    elif mode == "layerwise":
        abort_layerwise_reload(model)
