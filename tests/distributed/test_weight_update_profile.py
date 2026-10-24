# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.weight_transfer.profile import WeightUpdateProfiler


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_profiler(clock: FakeClock, sync_seconds: list[float]) -> WeightUpdateProfiler:
    """A profiler whose device synchronizations take ``sync_seconds`` in order."""
    syncs = iter(sync_seconds)
    # (allocated, peak): first session's start and finish, then later sessions
    memory = iter([(10, 10), (12, 30)] + [(0, 0)] * 8)
    return WeightUpdateProfiler(
        rank=3,
        clock=clock,
        synchronize=lambda: clock.advance(next(syncs)),
        memory=lambda: next(memory),
        reset_peak=lambda: None,
    )


class TestWeightUpdateProfiler:
    def test_session_breakdown(self):
        clock = FakeClock()
        # start: before, after; finish: drain, after
        profiler = make_profiler(clock, [0.0, 0.5, 2.0, 0.25])
        with profiler.start(is_draft=False):
            clock.advance(1.0)  # engine preparation
        for _ in range(2):
            clock.advance(3.0)  # waiting for the sender
            with profiler.update():
                clock.advance(4.0)
        with profiler.finish():
            clock.advance(5.0)  # post-load processing

        p = profiler.last
        assert p is not None
        assert p["start_s"] == pytest.approx(1.5)
        assert (p["update_s"], p["update_calls"]) == (pytest.approx(8.0), 2)
        assert p["drain_s"] == pytest.approx(2.0)
        assert p["finish_s"] == pytest.approx(5.25)
        assert p["idle_s"] == pytest.approx(6.0)
        assert p["wall_s"] == pytest.approx(22.75)
        assert (p["rank"], p["is_draft"]) == (3, False)
        assert (
            p["start_allocated_bytes"],
            p["finish_allocated_bytes"],
            p["peak_allocated_bytes"],
        ) == (10, 12, 30)

    def test_failed_start_leaves_no_session(self):
        clock = FakeClock()
        profiler = make_profiler(clock, [0.0, 0.0, 0.0])
        with pytest.raises(RuntimeError), profiler.start():
            raise RuntimeError("engine refused")
        with profiler.update():  # no session: not counted, no error
            pass
        with profiler.finish():
            pass
        assert profiler.last is None

    def test_failed_finish_keeps_previous_profile(self):
        clock = FakeClock()
        profiler = make_profiler(clock, [0.0] * 8)
        with profiler.start():
            pass
        with profiler.finish():
            pass
        first = profiler.last
        with profiler.start():
            pass
        with pytest.raises(RuntimeError), profiler.finish():
            raise RuntimeError("post-load processing failed")
        assert profiler.last is first
