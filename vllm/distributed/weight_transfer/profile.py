# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-worker cost breakdown of one weight-update session.

A session is ``start_weight_update`` -> ``update_weights`` x N ->
``finish_weight_update``. The API server already times each of those calls
end to end (``vllm:rl_weight_update_operation_duration_seconds``); this splits
the session on the worker that does the work:

- ``start_s``: engine preparation (e.g. reload setup), device-synchronized.
- ``update_s``: host time inside ``update_weights`` (receive + load), summed
  over ``update_calls`` calls. Device work it enqueues may still be running.
- ``drain_s``: waiting at finish for that device work to complete.
- ``finish_s``: engine finalization (post-load processing), synchronized.
- ``idle_s``: time between calls, i.e. waiting for the sender (reading the
  checkpoint, packing, the next RPC).
- ``wall_s``: first byte of start to last byte of finish.

plus allocated device memory at start and finish, and the session's peak.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_GiB = 1 << 30


def _synchronize() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def _memory() -> tuple[int, int]:
    """(allocated, peak) device bytes."""
    if not torch.accelerator.is_available():
        return 0, 0
    return (
        torch.accelerator.memory_allocated(),
        torch.accelerator.max_memory_allocated(),
    )


def _reset_peak() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.reset_peak_memory_stats()


class WeightUpdateProfiler:
    """Times one worker's weight-update sessions; keeps the last one."""

    def __init__(
        self,
        rank: int = 0,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] = _synchronize,
        memory: Callable[[], tuple[int, int]] = _memory,
        reset_peak: Callable[[], None] = _reset_peak,
    ) -> None:
        self.rank = rank
        self._clock = clock
        self._synchronize = synchronize
        self._memory = memory
        self._reset_peak = reset_peak
        self._session: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None

    @contextmanager
    def start(self, is_draft: bool = False) -> Iterator[None]:
        self._synchronize()
        self._reset_peak()
        t0 = self._clock()
        yield  # an exception leaves no session open
        self._synchronize()
        t1 = self._clock()
        self._session = {
            "is_draft": is_draft,
            "start_s": t1 - t0,
            "update_s": 0.0,
            "update_calls": 0,
            "start_allocated_bytes": self._memory()[0],
            "_t0": t0,
        }

    @contextmanager
    def update(self) -> Iterator[None]:
        t0 = self._clock()
        try:
            yield
        finally:
            if self._session is not None:
                self._session["update_s"] += self._clock() - t0
                self._session["update_calls"] += 1

    @contextmanager
    def finish(self) -> Iterator[None]:
        session, self._session = self._session, None
        t0 = self._clock()
        self._synchronize()
        t1 = self._clock()
        yield
        self._synchronize()
        t2 = self._clock()
        if session is None:
            return
        allocated, peak = self._memory()
        session["drain_s"] = t1 - t0
        session["finish_s"] = t2 - t1
        session["wall_s"] = t2 - session.pop("_t0")
        session["idle_s"] = max(
            0.0,
            session["wall_s"]
            - session["start_s"]
            - session["update_s"]
            - session["drain_s"]
            - session["finish_s"],
        )
        session.update(
            rank=self.rank,
            finish_allocated_bytes=allocated,
            peak_allocated_bytes=peak,
        )
        self.last = session
        logger.info(
            "Weight update profile (rank %d%s): wall %.3fs = start %.3fs + "
            "update %.3fs (%d calls) + drain %.3fs + finish %.3fs + idle %.3fs; "
            "allocated %.2f -> %.2f GiB, peak %.2f GiB",
            self.rank,
            ", draft" if session["is_draft"] else "",
            session["wall_s"],
            session["start_s"],
            session["update_s"],
            session["update_calls"],
            session["drain_s"],
            session["finish_s"],
            session["idle_s"],
            session["start_allocated_bytes"] / _GiB,
            allocated / _GiB,
            peak / _GiB,
        )
