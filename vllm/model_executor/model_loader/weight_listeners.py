# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tell listeners when the model's weights stop and start being valid.

A component that exposes the weights' GPU memory outside the forward pass, such
as a loader serving them to peers over RDMA, must stop before that memory is
rewritten or freed and may resume once the weights are final again. Every weight
update and every sleep/wake reports here, so a listener needs no knowledge of
which path changed the weights.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

__all__ = [
    "WeightListener",
    "add_weight_listener",
    "notify_weights_invalidated",
    "notify_weights_ready",
    "weights_updating",
]


@runtime_checkable
class WeightListener(Protocol):
    def weights_invalidated(self) -> None:
        """The weights are about to be rewritten or their memory freed.

        Called before the first byte changes. May repeat without an
        intervening `weights_ready`.
        """
        ...

    def weights_ready(self) -> None:
        """The weights are final (kernel format) and their memory is resident.

        Not called on a wake-up that leaves the weights' content discarded:
        they become ready once the weight update that follows completes. May
        repeat without an intervening `weights_invalidated`.
        """
        ...


_listeners: list[WeightListener] = []


def add_weight_listener(listener: WeightListener) -> None:
    if listener not in _listeners:
        _listeners.append(listener)


def notify_weights_invalidated() -> None:
    for listener in _listeners:
        listener.weights_invalidated()


def notify_weights_ready() -> None:
    for listener in _listeners:
        listener.weights_ready()


@contextmanager
def weights_updating() -> Iterator[None]:
    """Bracket a weight update that does not go through `start_reload`."""
    notify_weights_invalidated()
    yield
    notify_weights_ready()
