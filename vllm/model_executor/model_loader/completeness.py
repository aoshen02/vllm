# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether a `load_weights` call may judge the checkpoint's completeness.

A cold start hands `load_weights` the whole checkpoint in one call, so a check
of the form "this name never arrived, so refuse" is sound. A weight reload
streams the checkpoint and calls `load_weights` once per batch, and a name that
is absent from one batch says nothing about the update. Every such check has to
stand down for the duration, and completeness across the whole update is the
sender's contract.

The engines that stream an update hold `streaming_a_checkpoint()` open around
it. This is deliberately a scope rather than a flag on the model: the checks it
governs live in model code, quant methods and loaders that have no reference to
the reload lifecycle, and reach for it from wherever they run.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = ["completeness_checks_enabled", "streaming_a_checkpoint"]

_enabled: ContextVar[bool] = ContextVar(
    "vllm_completeness_checks_enabled", default=True
)


def completeness_checks_enabled() -> bool:
    """Whether this `load_weights` call sees the whole checkpoint."""
    return _enabled.get()


@contextmanager
def streaming_a_checkpoint() -> Iterator[None]:
    """Mark the enclosed `load_weights` calls as batches of one checkpoint."""
    token = _enabled.set(False)
    try:
        yield
    finally:
        _enabled.reset(token)
