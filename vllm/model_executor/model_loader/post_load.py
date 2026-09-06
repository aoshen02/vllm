# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered traversal of the modules that take part in post-load processing.

Cold-start weight processing and layerwise reload both visit the model in the
same order: one pass over every module, then a second pass over the deferred
attention-like layers (their hook reads weights that sibling layers may have
decompressed or repacked in the first pass). Cold start additionally runs the
model-level hook last; reload skips it. This module is the single place that
order is defined.
"""

from collections.abc import Iterator
from enum import Enum, auto

from torch import nn

from vllm.model_executor.layers.attention import is_deferred_attention_layer


class PostLoadPhase(Enum):
    """Which pass of post-load processing a yielded module belongs to."""

    LAYER = auto()
    """First pass: every module, in ``model.modules()`` order."""

    ATTENTION = auto()
    """Second pass: deferred attention-like layers only, after every ``LAYER``."""

    MODEL = auto()
    """Last: the model itself, only if it defines a zero-argument
    ``process_weights_after_loading`` hook."""


def iter_post_load_modules(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Module, PostLoadPhase]]:
    """Yield ``(name, module, phase)`` in post-load processing order.

    Deferred attention-like layers are yielded twice: once in the ``LAYER``
    pass like every other module, and again in the ``ATTENTION`` pass.
    """
    for name, module in model.named_modules():
        yield name, module, PostLoadPhase.LAYER

    for name, module in model.named_modules():
        if is_deferred_attention_layer(module):
            yield name, module, PostLoadPhase.ATTENTION

    if hasattr(model, "process_weights_after_loading"):
        yield "", model, PostLoadPhase.MODEL
