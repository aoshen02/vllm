# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import DraftTokenIds


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return 1, "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output, call_id=0: None
    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[tuple[Any, int]] = []
    worker_proc.handle_output = lambda output, call_id=0: outputs.append(
        (output, call_id)
    )
    worker_proc._execute_worker_rpc((7, "fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0][0], RuntimeError)
    assert str(outputs[0][0]) == "test error"
    assert outputs[0][1] == 7


@pytest.mark.parametrize("stalled", [False, True])
@pytest.mark.parametrize(
    "method,result",
    [
        ("take_draft_token_ids", DraftTokenIds(["req"], [[1, 2, 3]])),
        ("execute_dummy_batch", ["iteration record"]),
    ],
)
def test_model_rpc_uses_execute_model_timeout(monkeypatch, method, result, stalled):
    """A wedged model RPC must not block EngineCore forever. For
    execute_dummy_batch this is an idle DP rank stuck in a collective with a
    hung peer."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7")

    def dequeue(*, timeout=None):
        assert timeout is not None and 0 < timeout <= 7
        if stalled:
            raise TimeoutError
        return 1, WorkerProc.ResponseStatus.SUCCESS, result

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_fault_tolerance=False)
    )
    executor.is_failed = False
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match=f"{method} timed out"):
            getattr(executor, method)()
    else:
        assert getattr(executor, method)() is result


@pytest.mark.parametrize(
    "inherited_fds, local_world_size, is_cpu, numa_bind, expected",
    [
        (None, 8, False, False, True),  # spawn, GPU: start concurrently
        ([], 8, False, False, False),  # fork carries fds between workers
        (None, 1, False, False, False),  # nothing to overlap
        (None, 8, True, False, False),  # CPU backend sets OpenMP envs per worker
        (None, 8, False, True, False),  # NUMA wrapper patches the executable
    ],
)
def test_can_start_workers_concurrently(
    monkeypatch, inherited_fds, local_world_size, is_cpu, numa_bind, expected
):
    import vllm.v1.executor.multiproc_executor as mp_executor

    monkeypatch.setattr(
        mp_executor, "current_platform", SimpleNamespace(is_cpu=lambda: is_cpu)
    )
    executor: Any = SimpleNamespace(
        local_world_size=local_world_size,
        parallel_config=SimpleNamespace(numa_bind=numa_bind),
    )

    assert (
        MultiprocExecutor._can_start_workers_concurrently(executor, inherited_fds)
        is expected
    )


def test_start_workers_concurrently_keeps_started_workers_on_failure():
    """Workers that started must reach cleanup, in rank order, if one fails."""
    executor: Any = SimpleNamespace(local_world_size=4)
    handles = {rank: object() for rank in (0, 1, 3)}

    def start_worker(local_rank):
        if local_rank == 2:
            raise RuntimeError("boom")
        return handles[local_rank]

    unready_workers: list[Any] = []
    with pytest.raises(RuntimeError, match="boom"):
        MultiprocExecutor._start_workers_concurrently(
            executor, start_worker, unready_workers
        )

    assert unready_workers == [handles[0], handles[1], handles[3]]


def test_execute_worker_rpc_feeds_watchdog():
    """Verify the watchdog is fed immediately before and after each RPC."""

    def ok():
        calls.append("rpc")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(ok=ok)
    worker_proc.handle_output = lambda output, call_id=0: None
    calls: list[str] = []
    watchdog = MagicMock()
    watchdog.feed.side_effect = lambda: calls.append("feed")

    with patch(
        "vllm.v1.executor.multiproc_executor.get_watch_dog",
        return_value=watchdog,
    ):
        worker_proc._execute_worker_rpc((1, "ok", (), {}, None))

    assert calls == ["feed", "rpc", "feed"]


def test_collective_rpc_discards_stale_failure_and_timeout_replies():
    """End-to-end TP=2 WorkerProc + MultiprocExecutor test verifying:
    1) When enable_fault_tolerance=False and worker 0 fails, raising early
       without blocking on worker 1 leaves worker 1's stale reply queued, and
       the next RPC discards it via call_id and returns fresh replies.
    2) When a call times out and a worker enqueues its late reply afterwards,
       the next RPC discards stale call_ids and returns fresh replies.
    """
    worker_mqs: list[deque[Any]] = [deque(), deque()]

    class _FakeMQ:
        def __init__(self, idx: int):
            self.idx = idx
            self.fail_next_timeout = False

        def enqueue(self, item: Any) -> None:
            worker_mqs[self.idx].append(item)

        def dequeue(self, timeout: float | None = None) -> Any:
            if self.fail_next_timeout:
                self.fail_next_timeout = False
                raise TimeoutError
            return worker_mqs[self.idx].popleft()

    state = {"fail": True}

    def _make_worker(rank: int) -> Any:
        wp: Any = WorkerProc.__new__(WorkerProc)
        wp.rank = rank
        wp.use_async_scheduling = False
        wp.worker_response_mq = _FakeMQ(rank)

        def update_weights(tag: str) -> str:
            if state["fail"]:
                raise RuntimeError(f"rank {rank} failed on {tag}")
            return f"ok-{rank}-{tag}"

        wp.worker = SimpleNamespace(update_weights=update_weights)
        return wp

    workers = [_make_worker(0), _make_worker(1)]
    response_mqs = [_FakeMQ(0), _FakeMQ(1)]

    def broadcast_enqueue(rpc_request: Any) -> None:
        for wp in workers:
            wp._execute_worker_rpc(rpc_request)

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_fault_tolerance=False)
    )
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=broadcast_enqueue)
    executor.response_mqs = response_mqs

    # 1. Call 1 fails on worker 0 and raises immediately without waiting on
    #    worker 1; worker 1's stale (call_id=1) reply is discarded on retry.
    with pytest.raises(RuntimeError, match="rank 0 failed on attempt1"):
        executor.collective_rpc("update_weights", args=("attempt1",))
    assert len(worker_mqs[0]) == 0
    assert len(worker_mqs[1]) == 1

    # Retry succeeds and gets aligned fresh responses.
    state["fail"] = False
    assert executor.collective_rpc("update_weights", args=("attempt2",)) == [
        "ok-0-attempt2",
        "ok-1-attempt2",
    ]
    assert len(worker_mqs[0]) == 0
    assert len(worker_mqs[1]) == 0

    # 2. Simulate timeout on worker 1: worker 1's reply stays queued in MQ,
    #    then a subsequent retry discards that stale call_id reply.
    response_mqs[1].fail_next_timeout = True
    with pytest.raises(TimeoutError, match="update_weights timed out"):
        executor.collective_rpc("update_weights", timeout=1.0, args=("timed_out",))
    assert len(worker_mqs[1]) == 1

    assert executor.collective_rpc("update_weights", args=("after_timeout",)) == [
        "ok-0-after_timeout",
        "ok-1-after_timeout",
    ]
    assert len(worker_mqs[0]) == 0
    assert len(worker_mqs[1]) == 0


def test_collective_rpc_poll_discards_stale_without_blocking():
    """Verify non-blocking FutureWrapper.poll() does not block when a rank has
    only a stale reply queued while another rank already has its fresh reply.
    """
    worker_mqs: list[deque[Any]] = [deque(), deque()]

    class _PollFakeMQ:
        def __init__(self, idx: int):
            self.idx = idx

        def readable(self) -> bool:
            return len(worker_mqs[self.idx]) > 0

        def enqueue(self, item: Any) -> None:
            worker_mqs[self.idx].append(item)

        def dequeue(self, timeout: float | None = None) -> Any:
            if not worker_mqs[self.idx]:
                if timeout == 0.0:
                    raise TimeoutError
                raise AssertionError(
                    f"poll() unexpectedly called blocking dequeue on rank {self.idx}"
                )
            return worker_mqs[self.idx].popleft()

    state = {"fail_rank0": True, "delay_rank1": False}
    deferred_rank1: list[Any] = []

    def _make_worker(rank: int) -> Any:
        wp: Any = WorkerProc.__new__(WorkerProc)
        wp.rank = rank
        wp.use_async_scheduling = False
        wp.worker_response_mq = _PollFakeMQ(rank)

        def rpc_fn(tag: str) -> str:
            if rank == 0 and state["fail_rank0"]:
                raise RuntimeError(f"rank 0 failed on {tag}")
            return f"ok-{rank}-{tag}"

        wp.worker = SimpleNamespace(rpc_fn=rpc_fn)
        return wp

    workers = [_make_worker(0), _make_worker(1)]
    response_mqs = [_PollFakeMQ(0), _PollFakeMQ(1)]

    def broadcast_enqueue(rpc_request: Any) -> None:
        workers[0]._execute_worker_rpc(rpc_request)
        if state["delay_rank1"]:
            deferred_rank1.append(rpc_request)
        else:
            workers[1]._execute_worker_rpc(rpc_request)

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_fault_tolerance=False)
    )
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=broadcast_enqueue)
    executor.response_mqs = response_mqs

    # 1. Fail first RPC on rank 0 -> rank 1's stale reply (call_id=1) stays queued.
    with pytest.raises(RuntimeError, match="rank 0 failed on attempt1"):
        executor.collective_rpc("rpc_fn", args=("attempt1",))
    assert len(worker_mqs[0]) == 0
    assert len(worker_mqs[1]) == 1

    # 2. Retry with non_block=True while rank 1's fresh reply is delayed.
    state["fail_rank0"] = False
    state["delay_rank1"] = True
    fut = executor.collective_rpc("rpc_fn", args=("retry",), non_block=True)

    # Rank 0's fresh reply and rank 1's stale reply are both readable.
    # poll() must discard rank 1's stale reply, retain rank 0's fresh reply,
    # and return False immediately without blocking.
    assert fut.poll() is False
    assert not fut.done()
    assert len(worker_mqs[0]) == 0
    assert len(worker_mqs[1]) == 0

    # Subsequent poll while rank 1 is still pending also returns False promptly.
    assert fut.poll() is False

    # 3. Once rank 1's fresh reply arrives, poll() completes with both replies.
    workers[1]._execute_worker_rpc(deferred_rank1.pop())
    assert fut.poll() is True
    assert fut.done()
    assert fut.result() == ["ok-0-retry", "ok-1-retry"]
