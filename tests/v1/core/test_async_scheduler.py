# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tests.distributed.aux_output_connector.test_store import (
    _input_batch,
    _make_worker,
)
from tests.distributed.aux_output_connector.utils import fake_store
from vllm.config import AuxOutputConfig, KVTransferConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputSchedulerConnector,
)
from vllm.distributed.aux_output_connector.object_store import read_aux_rows
from vllm.distributed.aux_output_connector.session_objects import (
    SessionObjectWriter,
    salt_aux_session,
)
from vllm.distributed.aux_output_connector.tier_versions import SessionTierVersions
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingManager,
    OffloadingSpec,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar
from vllm.v1.utils import ConstantList

from .utils import (
    EOS_TOKEN_ID,
    create_requests,
    create_scheduler,
    mock_kv,
    new_kv_cache_group,
)

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_stop_by_max_tokens(max_tokens: int):
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=2, max_tokens=max_tokens)
    req0, req1 = requests

    expected_total_num_scheduled_tokens = 0
    sched_outputs: deque[SchedulerOutput] = deque()
    scheduler.add_request(req0)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req0.num_prompt_tokens + max_tokens - 1

    scheduler.add_request(req1)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req1.num_prompt_tokens + max_tokens - 1

    total_num_scheduled_tokens = 0
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        total_num_scheduled_tokens += sched_output.total_num_scheduled_tokens
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req0.num_output_tokens == max_tokens
    assert req1.num_output_tokens == max_tokens
    # Ensure we aren't scheduling more tokens than necessary.
    assert total_num_scheduled_tokens == expected_total_num_scheduled_tokens


def test_no_spec_decode_padding_up_to_max_model_len():
    """Uniform spec-decode padding must leave room for the sampled token.

    Padding a single-token decode up to exactly max_model_len makes the
    running-loop max_model_len cap negative on the next step (async
    scheduling schedules it before the output that would stop the request),
    and a negative num_scheduled_tokens crashes the model runner. The
    request is scheduled un-padded instead.

    Modelled on a disaggregated decode worker, where every request arrives
    with all but the last prompt token already computed.
    """
    max_model_len = 32
    num_spec = 1
    scheduler = create_scheduler(
        async_scheduling=True,
        max_model_len=max_model_len,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )

    # A running decode, so padding's "something already scheduled and no
    # prefill this step" precondition holds for the request below.
    (filler,) = create_requests(num_requests=1, num_tokens=4, req_ids=["filler"])
    scheduler.add_request(filler)
    sched_output = scheduler.schedule()
    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    (request,) = create_requests(
        num_requests=1, num_tokens=max_model_len - num_spec, req_ids=["boundary"]
    )
    request.num_computed_tokens = request.num_tokens - 1
    scheduler.add_request(request)

    sched_output = scheduler.schedule()
    assert sched_output.num_scheduled_tokens["boundary"] == 1
    assert "boundary" not in sched_output.scheduled_spec_decode_tokens
    assert request.num_computed_tokens < max_model_len

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_abort():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_connector_metadata_precedes_async_placeholder_advance(monkeypatch):
    """Mirroring must see earlier unresolved outputs, excluding the current step."""
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    (request,) = create_requests(num_requests=1, num_tokens=4, max_tokens=4)
    scheduler.add_request(request)

    assert scheduler.connector is not None
    build_connector_meta = scheduler.connector.build_connector_meta
    observed_placeholders = []

    def record_placeholders(scheduler_output):
        observed_placeholders.append(request.num_output_placeholders)
        return build_connector_meta(scheduler_output)

    monkeypatch.setattr(
        scheduler.connector, "build_connector_meta", record_placeholders
    )
    scheduler.schedule()
    scheduler.schedule()

    assert observed_placeholders == [0, 1]


def test_prefix_caching_for_prefill_dedup():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=3,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )

    # Two requests with the same prompt.
    req0 = requests.pop(0)
    req1 = requests.pop(0)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)
    # Make sure prefix caching de-duplicates the prompts in the same step,
    # so all the blocks except the last are shared between the two requests.
    assert len(sched_output.num_scheduled_tokens) == 2
    assert sched_output.num_scheduled_tokens[req0.request_id] == num_prompt_tokens
    assert (
        sched_output.num_scheduled_tokens[req1.request_id]
        == num_prompt_tokens % BLOCK_SIZE
    )

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        added_req = None
        if requests:
            added_req = requests.pop(0)
            scheduler.add_request(added_req)
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
            if added_req:
                assert (
                    sched_output.num_scheduled_tokens[added_req.request_id]
                    == num_prompt_tokens % BLOCK_SIZE
                )

    assert scheduler.get_num_unfinished_requests() == 0


def test_prefix_caching_for_multi_turn():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    num_output_tokens = 200
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )

    for req in requests:
        scheduler.add_request(req)
    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    # Process the requests.
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
    assert scheduler.get_num_unfinished_requests() == 0

    # Create next-turn requests whose prompts are the full output of the
    # previous turn.
    next_turn_requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens + num_output_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )
    for i, req in enumerate(next_turn_requests):
        req.prompt_token_ids = requests[i].prompt_token_ids + list(
            requests[i].output_token_ids
        )
        req._all_token_ids = req.prompt_token_ids.copy()
        req.all_token_ids = ConstantList(req._all_token_ids)
        req.block_hashes = []
        req.update_block_hashes()

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert sched_output.num_scheduled_tokens[req.request_id] == (
            req.num_prompt_tokens % BLOCK_SIZE
        )


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(AsyncScheduler)
    request = create_requests(num_requests=1, num_tokens=1)[0]
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.accept_tokens.return_value = False
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = {}
    scheduler.encoder_cache_mismatch_reqs = set()
    scheduler.vllm_config = Mock()
    scheduler.aux_output_connector = None
    scheduler.return_sampling_mask = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.num_spec_tokens = 0
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(output, model_runner_output)

    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running


class PipelinedEngine:
    """Drive a real AsyncScheduler like EngineCore.step_with_batch_queue:
    schedule until the batch queue is full, then process the oldest step's
    output. Async PP runs pp_size+1 concurrent batches, so up to pp_size
    steps are in flight at each schedule() call -- the window in which
    preemption must handle output that has not yet returned. (Single-GPU e2e
    tests can never create this window: at PP=1, exactly one step is in
    flight and it is processed before a preempted request can resume.)

    The model runner is emulated with the V2 runner's own bookkeeping, from
    only what the scheduler serializes to it: slots flushed on
    preempted_req_ids, resumed requests re-added from the NewRequestData
    snapshot, sampling when a step reaches the end of the runner's own view
    of the sequence. This makes preemption races observable: a stale token
    delivered after a resume is scheduled extends the scheduler's sequence
    but not the runner's.

    Every sample emits a globally unique token tagged with its sampled
    position, so tests can assert exact delivery.
    """

    def __init__(
        self,
        scheduler: AsyncScheduler,
        queue_size: int,
        accept_drafts: Callable[[int, str, int], int] | None = None,
    ):
        self.scheduler = scheduler
        self.queue_size = queue_size
        self.accept_drafts = accept_drafts
        # In-flight steps: (scheduler_output, new_reqs snapshot) in FIFO order.
        self.queue: deque[tuple[SchedulerOutput, list[tuple[str, int, int]]]] = deque()
        # Runner-side request state: req_id -> [seq_len, num_computed] as the
        # runner sees them (its own sampled tokens, not the scheduler's).
        self.runner_view: dict[str, list[int]] = {}
        # All tokens the fake runner ever sampled, per request, in order.
        self.emitted: dict[str, list[int]] = defaultdict(list)
        # Sequence position each (globally unique) token was sampled for.
        self.emitted_position: dict[int, int] = {}
        self.step_idx = 0
        self._next_token = 1000

    def _schedule(self) -> bool:
        scheduler_output = self.scheduler.schedule()
        self.step_idx += 1
        # Snapshot what NewRequestData serializes at schedule time (both new
        # and resumed requests for the V2 runner).
        new_reqs = [
            (r.req_id, len(r.prefill_token_ids), r.num_computed_tokens)
            for r in scheduler_output.scheduled_new_reqs
        ]
        # Enqueue empty steps too (the engine executes them), so the runner
        # still observes their preempted/finished request ids in step order.
        self.queue.appendleft((scheduler_output, new_reqs))
        return True

    def _process_oldest_step(self) -> None:
        scheduler_output, new_reqs = self.queue.pop()
        # Worker-side state updates, in step order: flush preempted/finished
        # slots, then (re-)add new/resumed requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            self.runner_view.pop(req_id, None)
        for req_id in scheduler_output.finished_req_ids or ():
            self.runner_view.pop(req_id, None)
        for req_id, seq_len, num_computed in new_reqs:
            self.runner_view[req_id] = [seq_len, num_computed]

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        sampled_token_ids: list[list[int]] = []
        for req_id in req_ids:
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            view = self.runner_view.get(req_id)
            if view is None:
                # Slot already flushed (request finished/aborted mid-flight).
                sampled_token_ids.append([])
                continue
            seq_len, num_computed = view
            end = num_computed + num_scheduled
            if end < seq_len:
                # Partial prefill by the runner's own bookkeeping: no sample.
                view[1] = end
                sampled_token_ids.append([])
                continue
            drafts = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            num_accepted = (
                min(self.accept_drafts(self.step_idx, req_id, len(drafts)), len(drafts))
                if drafts and self.accept_drafts
                else 0
            )
            num_rejected = len(drafts) - num_accepted
            tokens = list(range(self._next_token, self._next_token + 1 + num_accepted))
            self._next_token += 1 + num_accepted
            self.emitted[req_id].extend(tokens)
            sampled_token_ids.append(tokens)
            # Rejected drafts roll back computed; the sampled tokens extend
            # the runner's sequence.
            view[1] = end - num_rejected
            view[0] = view[1] + 1
            for offset, token in enumerate(tokens):
                self.emitted_position[token] = view[0] - len(tokens) + offset
        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        self.scheduler.update_from_output(scheduler_output, model_runner_output)

    def run(
        self,
        max_steps: int = 2000,
        before_step: Callable[[int, "PipelinedEngine"], None] | None = None,
    ) -> None:
        for i in range(max_steps):
            if not self.scheduler.has_requests() and not self.queue:
                return
            if before_step is not None:
                before_step(i, self)
            scheduled = (
                self.scheduler.has_requests()
                and len(self.queue) < self.queue_size
                and self._schedule()
            )
            if scheduled and len(self.queue) < self.queue_size:
                # Queue not yet full: the engine returns without blocking.
                continue
            if self.queue:
                self._process_oldest_step()
        raise AssertionError("engine loop did not converge")


def _create_async_pp_scheduler(
    num_spec: int, pp_size: int = 3, num_blocks: int = 5
) -> AsyncScheduler:
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec or None,
        speculative_method="ngram_gpu" if num_spec else None,
        num_blocks=num_blocks,
        block_size=16,
        max_num_batched_tokens=512,
    )
    # Emulate PP at the scheduler level; constructing with
    # pipeline_parallel_size>1 requires that many visible GPUs. Drive with
    # queue_size=pp_size+1 (V2 async PP runs pp_size+1 concurrent batches).
    scheduler.decode_stagger = pp_size
    scheduler.use_pp = pp_size > 1
    return scheduler


def _assert_ordered_subset(delivered: list[int], emitted: list[int]) -> None:
    """Delivered tokens must be an order-preserving subset of the emitted
    tokens with no duplicates (tokens are globally unique)."""
    it = iter(emitted)
    for token in delivered:
        assert token in it, f"token {token} delivered out of order or twice"


def _assert_positions_consistent(req, engine: PipelinedEngine) -> None:
    """The i-th delivered output token must be one the runner sampled for
    exactly sequence position prompt_len + i: catches a preempted request's
    stale output landing on a position the resumed request resampled (or
    vice versa), which token-stream equality alone cannot see."""
    for i, token in enumerate(req.output_token_ids):
        expected = req.num_prompt_tokens + i
        actual = engine.emitted_position[token]
        assert actual == expected, (
            f"output {i} of {req.request_id}: token sampled for position "
            f"{actual}, delivered as position {expected}"
        )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_kv_pressure_preemption_with_inflight_output(num_spec: int):
    """KV-pressure preemption of requests with in-flight async output.

    PP=3 + async scheduling (batch queue of 4), a block pool small enough
    that decodes contend and preempt mid-flight, and staggered arrivals so
    the batch queue actually pipelines. A preempted request's in-flight steps
    still return: their tokens must be delivered exactly once, their stale
    spec-rejection counts must not corrupt the rolled-back counters, and the
    resume must not resample a position that output later delivers.

    Regression for the num_output_placeholders underflow EngineCore crash:
    with the fix reverted, the num_spec=3 variant fails with exactly
    `assert request.num_output_placeholders >= 0` when a stale spec output
    returns after the preempted request was resumed and sampled.
    """
    max_tokens = 24
    scheduler = _create_async_pp_scheduler(num_spec)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
    )
    pending = list(requests)
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    # Observe that the scenario under test actually occurs.
    preempts_with_inflight_output = 0
    orig_preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal preempts_with_inflight_output
        if request.num_in_flight_tokens > 0:
            preempts_with_inflight_output += 1
        return orig_preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt

    def add_requests(step: int, engine: PipelinedEngine):
        if pending:
            scheduler.add_request(pending.pop(0))

    engine = PipelinedEngine(
        scheduler,
        queue_size=4,
        # Deterministically vary spec acceptance so stale outputs carry
        # nonzero rejection counts.
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    )
    engine.run(before_step=add_requests)

    assert preempts_with_inflight_output > 0, (
        "test did not exercise preemption with in-flight output"
    )
    for req in requests:
        assert req.is_finished()
        assert req.num_output_tokens == max_tokens
        # Lossless: delivered tokens are exactly the sampled tokens, in order
        # (the excluded tail was emitted after the request finished).
        emitted = engine.emitted[req.request_id]
        assert list(req.output_token_ids) == emitted[:max_tokens]
        _assert_positions_consistent(req, engine)


class SessionEngine:
    """A real scheduler and auxiliary output worker over an in-memory store;
    rows carry their position and weight version, and `close` checks every
    key against the KV each request used. It also plays the offloading worker."""

    def __init__(self, scheduler, block_size=16):
        self.scheduler = scheduler
        self.block_size = block_size
        self.connector = scheduler.aux_output_connector or AuxOutputSchedulerConnector(
            scheduler.kv_cache_manager, 0, key_prefix="job"
        )
        scheduler.aux_output_connector = self.connector
        self.client, self.store = fake_store()
        self.worker = _make_worker(16, 256, block_size=16)
        self.worker._object_writer = SessionObjectWriter(
            self.store, key_prefix="job", max_pending_batches=4
        )
        self.worker._capturer = Mock()
        self.set_version(0)
        self.requests = {}
        self.keys = {}
        self.num_hits = 0
        self._kv_versions = {}
        self._group_sizes = [
            manager.block_size
            for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
        ]
        self._draft_only = [
            group.draft_only for group in scheduler.kv_cache_config.kv_cache_groups
        ]
        self._hit_versions = {}
        self._row_versions = {}
        self._in_flight = {}
        self._stopped = []
        self.fail_loads = False
        self.fail_stores = False
        self.num_loads = 0
        self.num_stores = 0
        self._tier = {}
        self._store_jobs = {}
        self._kv_outputs = {}
        self._schedule = scheduler.schedule
        self._update_from_output = scheduler.update_from_output
        scheduler.schedule = self.schedule
        scheduler.update_from_output = self.update_from_output

    def set_version(self, version):
        self.version = version
        self.connector.set_weight_version(f"v{version}")

    def schedule(self):
        scheduler_output = self._schedule()
        metadata = scheduler_output.aux_output_connector_metadata
        self.worker.begin_step(metadata)
        size = self.block_size
        if scheduler_output.kv_connector_metadata is not None:
            self._transfer_kv(scheduler_output)
        block_ids = {}
        request_groups = {}
        starts, counts, rows = [], [], []
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.scheduler.requests[request_id]
            self.requests[request_id] = request
            start = request.num_computed_tokens - num_tokens
            groups = self.scheduler.kv_cache_manager.get_blocks(request_id).blocks
            blocks = groups[self.connector._group_id]
            block_ids[request_id] = [block.block_id for block in blocks]
            request_groups[request_id] = groups
            if request_id in metadata.sessions:
                self._row_versions[request_id] = {}
                self.num_hits += start > 0
            self._row_versions.setdefault(request_id, {})
            step_rows = np.zeros((num_tokens, 3, 2), dtype=np.uint8)
            step_rows[:, 0, 0] = np.arange(start, start + num_tokens)
            step_rows[:, 0, 1] = self.version
            for position in range(start, start + num_tokens):
                for group, group_size in zip(groups, self._group_sizes):
                    block_id = group[position // group_size].block_id
                    versions = self._kv_versions.get(block_id)
                    if versions is None or len(versions) != group_size:
                        versions = self._kv_versions[block_id] = [None] * group_size
                    versions[position % group_size] = self.version
                self._row_versions[request_id][position] = self.version
            starts.append(start)
            counts.append(num_tokens)
            rows.append(step_rows)
        for request_id in metadata.sessions:
            num_hit_blocks = min(self._row_versions[request_id]) // size
            hit_ids = block_ids[request_id][:num_hit_blocks]
            self._hit_versions[request_id] = [
                version
                for block_id in hit_ids
                for version in self._kv_versions[block_id]
            ]
            num_hit = num_hit_blocks * size
            for group, group_size, draft_only in zip(
                request_groups[request_id], self._group_sizes, self._draft_only
            ):
                if draft_only:
                    continue
                for index, block in enumerate(group[: num_hit // group_size]):
                    positions = range(index * group_size, (index + 1) * group_size)
                    assert block.is_null or self._kv_versions[block.block_id] == [
                        self._kv_versions[hit_ids[p // size]][p % size]
                        for p in positions
                    ]
        step = None
        if rows:
            self.worker._capturer.snapshot_routing_data.return_value = torch.from_numpy(
                np.concatenate(rows)
            )
            step = self.worker.prepare_output(
                _input_batch(list(block_ids), np.array(starts), np.cumsum([0] + counts))
            )
        self._in_flight[id(scheduler_output)] = step
        return scheduler_output

    def _transfer_kv(self, scheduler_output):
        metadata = scheduler_output.kv_connector_metadata
        worker_meta = OffloadingWorkerMetadata()
        finished, failed = set(), set()
        for job_id, job in self._store_jobs.items():
            worker_meta.mark_completed(job_id, success=not self.fail_stores)
            if self.fail_stores:
                continue
            self.num_stores += 1
            for block_id, chunk_id in zip(
                job.src_spec.block_ids.tolist(), job.dst_spec.block_ids.tolist()
            ):
                self._tier[chunk_id] = list(self._kv_versions[block_id])
        self._store_jobs = metadata.store_jobs
        for job_id, job in metadata.load_jobs.items():
            worker_meta.mark_completed(job_id, success=not self.fail_loads)
            if not job.sync:
                finished.add(job.req_id)
            if self.fail_loads:
                failed.add(job.req_id)
                continue
            self.num_loads += 1
            for chunk_id, block_id in zip(
                job.src_spec.block_ids.tolist(), job.dst_spec.block_ids.tolist()
            ):
                self._kv_versions[block_id] = list(self._tier[chunk_id])
        self._kv_outputs[id(scheduler_output)] = KVConnectorOutput(
            finished_recving=finished,
            failed_recving=failed,
            kv_connector_worker_meta=worker_meta,
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        kv_output = self._kv_outputs.pop(id(scheduler_output), None)
        if kv_output is not None:
            model_runner_output.kv_connector_output = kv_output
        step = self._in_flight.pop(id(scheduler_output))
        if step is not None:
            num_sampled = np.array(
                [len(tokens) for tokens in model_runner_output.sampled_token_ids]
            )
            drafts = scheduler_output.scheduled_spec_decode_tokens
            num_rejected = np.array(
                [
                    max(len(drafts.get(request_id, ())) + 1 - len(tokens), 0)
                    if tokens
                    else 0
                    for request_id, tokens in zip(
                        model_runner_output.req_ids,
                        model_runner_output.sampled_token_ids,
                    )
                ]
            )
            step.enqueue_cpu_copy(num_sampled, num_rejected)
            model_runner_output.aux_output_connector_output = step.process_output()
        outputs = self._update_from_output(scheduler_output, model_runner_output)
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                if output.aux_output_keys is None:
                    continue
                request_id = output.request_id
                keys = output.aux_output_keys
                row_versions = self._row_versions[request_id]
                num_rows = self.requests[request_id].num_tokens - 1
                versions = self._hit_versions[request_id]
                versions = versions + [
                    row_versions[position]
                    for position in range(len(versions), num_rows)
                ]
                self.keys[request_id] = keys
                self._stopped.append(
                    (request_id, keys, output.routed_experts, versions)
                )
        return outputs

    def step(self, scheduler_output, sampled):
        request_ids = list(scheduler_output.num_scheduled_tokens)
        outputs = self.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=request_ids,
                sampled_token_ids=[sampled.get(r, []) for r in request_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        return {
            output.request_id: output
            for client_outputs in outputs.values()
            for output in client_outputs.outputs
        }

    def run(self):
        outputs = {}
        while self.scheduler.has_requests():
            scheduler_output = self.schedule()
            requests = self.scheduler.requests
            sampled = {
                request_id: [0]
                for request_id in scheduler_output.num_scheduled_tokens
                if requests[request_id].num_computed_tokens
                == requests[request_id].num_tokens
            }
            outputs.update(self.step(scheduler_output, sampled))
        return outputs

    def versions(self, request_id):
        return [key.rpartition("@")[2] for key in self.keys[request_id]]

    def close(self):
        self.worker.close()
        size = self.block_size
        step = size // 16
        writes = Counter(key for batch in self.client.writes for key in batch)
        for request_id, keys, tail, versions in self._stopped:
            request = self.requests[request_id]
            session = request.sampling_params.aux_session
            assert keys == [
                f"job/{session}/{block_hash.hex()}@v{version}"
                for block_hash, version in zip(
                    request.block_hashes[step - 1 :: step],
                    versions[size - 1 : len(keys) * size : size],
                )
            ]
            rows = read_aux_rows(self.store, keys, "routed_experts", tail, timeout_s=0)
            assert rows[:, 0, 0].tolist() == list(range(request.num_tokens - 1))
            row_versions = rows[:, 0, 1].tolist()
            for index, key in enumerate(keys):
                block = slice(index * size, (index + 1) * size)
                if writes[key] == 1:
                    assert row_versions[block] == versions[block]
                assert row_versions[block][-1] == versions[block][-1]
            tail_start = len(keys) * size
            assert row_versions[tail_start:] == versions[tail_start:]


def _session_requests(
    request_ids, num_tokens, max_tokens, lookahead=0, token=0, session="s"
):
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, aux_session=session)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    salted = SimpleNamespace(sampling_params=params, cache_salt=None)
    config = SimpleNamespace(aux_output_config=SimpleNamespace(object_key_prefix="j"))
    salt_aux_session(config, salted)
    return [
        Request(
            request_id=request_id,
            prompt_token_ids=[token] * num_tokens,
            sampling_params=params,
            pooling_params=None,
            cache_salt=salted.cache_salt,
            block_hasher=get_request_block_hasher(16, sha256, lookahead),
        )
        for request_id in request_ids
    ]


@pytest.mark.parametrize(
    "async_scheduling,queue_size", [(False, 1), (True, 2)], ids=["sync", "async"]
)
def test_session_keys_resolve_to_the_rows_of_the_kv_each_request_used(
    async_scheduling: bool, queue_size: int
):
    """Session requests under KV pressure with weight updates between steps.

    Siblings recompute the same blocks across weight updates, later turns hit
    them, preempted requests resume, requests stop exactly on an object
    boundary, and one request is aborted after it was scheduled.
    """
    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        enable_prefix_caching=True,
        num_blocks=20,
        max_num_batched_tokens=64,
    )
    engine = SessionEngine(scheduler)
    pending = [
        *_session_requests([f"0-{i}" for i in range(5)], 70, 27),
        *_session_requests([f"1-{i}" for i in range(3)], 102, 27),
    ]
    request_ids = {request.request_id for request in pending}
    aborted = "0-3"
    num_preemptions = 0
    preempt_request = scheduler._preempt_request

    def preempt(request, timestamp, **kwargs):
        nonlocal num_preemptions
        num_preemptions += 1
        return preempt_request(request, timestamp, **kwargs)

    scheduler._preempt_request = preempt

    def before_step(step: int, _: PipelinedEngine):
        engine.set_version(step // 4)
        if aborted in engine.requests and aborted in scheduler.requests:
            scheduler.finish_requests(aborted, RequestStatus.FINISHED_ABORTED)
        if step % 2 == 0 and pending:
            scheduler.add_request(pending.pop(0))

    scheduler.add_request(pending.pop(0))
    PipelinedEngine(scheduler, queue_size=queue_size).run(before_step=before_step)
    engine.close()

    assert engine.keys.keys() == request_ids - {aborted}
    assert num_preemptions > 0 and engine.num_hits > 0
    assert any(
        len({key.rpartition("@")[2] for key in keys}) > 1
        for keys in engine.keys.values()
    )


@pytest.mark.parametrize(
    "groups", [{}, {"sliding_window": 32}], ids=["one-group", "two-groups"]
)
def test_session_siblings_admitted_in_one_step_share_the_prompt(groups):
    engine = _tier_engine(spec_name="OtherTierSpec", retention=None, **groups)
    scheduler = engine.scheduler
    for request in _session_requests(["first", "second"], 40, 1):
        scheduler.add_request(request)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"first": 40, "second": 8}
    engine.step(scheduler_output, {"first": [1], "second": [2]})
    engine.close()
    assert engine.keys["first"] == engine.keys["second"] and len(engine.keys) == 2
    assert len(engine.keys["first"]) == 2


def test_session_hits_the_blocks_of_a_request_aborted_before_its_output():
    scheduler = create_scheduler(enable_prefix_caching=True)
    engine = SessionEngine(scheduler)
    first, second = _session_requests(["first", "second"], 40, 1)
    scheduler.add_request(first)
    scheduler_output = engine.schedule()
    scheduler.finish_requests("first", RequestStatus.FINISHED_ABORTED)
    engine.step(scheduler_output, {"first": [1]})

    scheduler.add_request(second)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"second": 8}
    engine.step(scheduler_output, {"second": [2]})
    engine.close()
    assert len(engine.keys["second"]) == 2


def test_preempted_session_request_that_stops_on_its_output_in_flight():
    scheduler = create_scheduler(
        async_scheduling=True, enable_prefix_caching=True, num_blocks=3
    )
    engine = SessionEngine(scheduler)
    (first,) = _session_requests(["first"], 15, 5, token=1)
    (second,) = _session_requests(["second"], 16, 1, lookahead=1)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [1]})

    scheduler.add_request(second)
    step = engine.schedule()
    assert step.num_scheduled_tokens == {"first": 1, "second": 16}
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"second"}
    outputs = engine.step(step, {"first": [2], "second": [3]})
    assert outputs["second"].finish_reason is not None
    engine.step(preempting, {"first": [4]})
    engine.close()
    assert engine.keys["second"] == []


def test_preempted_session_request_whose_accepted_drafts_complete_an_object():
    scheduler = create_scheduler(
        async_scheduling=True,
        enable_prefix_caching=True,
        num_speculative_tokens=2,
        speculative_method="ngram_gpu",
        num_blocks=12,
    )
    engine = SessionEngine(scheduler)
    (first,) = _session_requests(["first"], 26, 30, token=1)
    (second,) = _session_requests(["second"], 46, 4)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [1]})

    scheduler.add_request(second)
    prefill = engine.schedule()
    drafts = engine.schedule()
    assert drafts.num_scheduled_tokens == {"first": 3, "second": 3}
    engine.step(prefill, {"first": [2, 3, 4], "second": [5]})
    block_pool = scheduler.kv_cache_manager.block_pool
    block_pool.get_new_blocks(block_pool.get_num_free_blocks())
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"second"}
    outputs = engine.step(drafts, {"first": [6, 7, 8], "second": [9, 10, 11]})
    assert outputs["second"].finish_reason is not None
    engine.step(preempting, {"first": [12]})
    engine.close()
    assert len(engine.keys["second"]) == 2


class HostTierSpec(OffloadingSpec):
    manager_cls = CPUOffloadingManager

    def get_manager(self):
        return self.manager_cls(num_chunks=self.extra_config["num_chunks"])

    def get_worker(self, kv_caches):
        raise NotImplementedError


class OtherManager(CPUOffloadingManager):
    pass


class OtherTierSpec(HostTierSpec):
    manager_cls = OtherManager


@pytest.mark.parametrize(
    "session,expected",
    [("s", 8), ("t", 40), (None, 40)],
    ids=["same-session", "other-session", "no-session"],
)
def test_only_requests_of_a_session_hit_its_blocks(session, expected):
    engine = SessionEngine(create_scheduler(enable_prefix_caching=True))
    _run_turn(engine, 0, "first")
    (second,) = _session_requests(["second"], 40, 1, session=session)
    engine.scheduler.add_request(second)
    assert engine.schedule().num_scheduled_tokens == {"second": expected}


def test_resumed_session_request_names_recomputed_objects_by_the_new_version():
    engine = SessionEngine(create_scheduler(enable_prefix_caching=True))
    (first,) = _session_requests(["first"], 86, 2)
    engine.scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [0]})
    engine.set_version(2)
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    engine.run()
    engine.close()
    assert engine.versions("first") == ["v2"] * 5


def test_resumed_session_request_names_objects_it_hits_by_their_blocks():
    scheduler = create_scheduler(
        enable_prefix_caching=True, scheduling_policy="priority", max_num_seqs=1
    )
    engine = SessionEngine(scheduler)
    block_pool = scheduler.kv_cache_manager.block_pool
    engine.set_version(1)
    (first,) = _session_requests(["first"], 80, 3)
    first.priority = 1
    scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [0]})
    held = block_pool.get_new_blocks(block_pool.get_num_free_blocks())
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"first"}
    engine.step(preempting, {})
    held += block_pool.get_new_blocks(block_pool.get_num_free_blocks())
    block_pool.free_blocks(held)
    engine.set_version(2)
    (second,) = _session_requests(["second"], 80, 1)
    scheduler.add_request(second)
    outputs = engine.step(engine.schedule(), {"second": [0]})
    assert outputs["second"].finish_reason is not None
    engine.set_version(3)
    engine.run()
    engine.close()
    assert engine.versions("first") == engine.versions("second") == ["v2"] * 5


def _tier_engine(
    spec_name="HostTierSpec",
    num_chunks=8,
    key_prefix="job",
    sliding_window=None,
    retention=0,
    block_size=16,
    mamba_align=False,
    draft_window=False,
    **extra_config,
):
    scheduler = create_scheduler(enable_prefix_caching=True)
    config = scheduler.vllm_config
    config.cache_config.prefix_cache_retention_interval = retention
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "spec_name": spec_name,
            "spec_module_path": __name__,
            "num_chunks": num_chunks,
            "offload_prompt_only": False,
            **extra_config,
        },
    )
    config.aux_output_config = AuxOutputConfig(
        enable_return_routed_experts=True, object_key_prefix=key_prefix
    )
    kv_cache_config = scheduler.kv_cache_config
    if sliding_window is not None or mamba_align:
        spec = AttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        full_spec = AttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        other_group = new_kv_cache_group(
            ["window"],
            SlidingWindowManager.layer_config(spec, sliding_window=sliding_window),
        )
        if mamba_align:
            config.cache_config.mamba_cache_mode = "align"
            other_group = new_kv_cache_group(
                ["mamba"],
                MambaSpec(block_size=16, shapes=((1, 1),), dtypes=(torch.float32,)),
                MambaManager.config(mamba_cache_mode="align"),
            )
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                new_kv_cache_group(["full"], full_spec),
                other_group,
            ],
            prefix_cache_retention_interval=retention,
        )
        if draft_window:
            draft_group = new_kv_cache_group(
                ["draft"],
                SlidingWindowManager.layer_config(spec, sliding_window=sliding_window),
            )
            draft_group.draft_only = True
            kv_cache_config.kv_cache_groups.append(draft_group)
    return SessionEngine(
        Scheduler(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            block_size=block_size,
            hash_block_size=16,
            log_stats=True,
            structured_output_manager=scheduler.structured_output_manager,
        ),
        block_size,
    )


def _run_turn(engine, version, request_id, num_tokens=40, max_tokens=1, **kwargs):
    engine.set_version(version)
    (request,) = _session_requests([request_id], num_tokens, max_tokens, **kwargs)
    engine.scheduler.add_request(request)
    return engine.run()


@pytest.mark.parametrize(
    "reset_connector,engine_config,expected_versions,expected_loads",
    [
        (False, {}, ["v1", "v1", "v2", "v2"], 1),
        (False, {"sync_load_budget_bytes": 1 << 30}, ["v1", "v1", "v2", "v2"], 1),
        (False, {"sliding_window": 32}, ["v1", "v1", "v2", "v2"], 1),
        (True, {}, ["v2"] * 4, 0),
        (False, {"spec_name": "OtherTierSpec"}, ["v2"] * 4, 0),
        (False, {"blocks_per_chunk": 2}, ["v2"] * 4, 0),
    ],
    ids=[
        "tier-kept",
        "sync-load",
        "two-groups",
        "tier-reset",
        "other-manager",
        "two-block-chunks",
    ],
)
def test_session_turn_reuses_the_kv_of_the_offload_tier(
    reset_connector, engine_config, expected_versions, expected_loads
):
    engine = _tier_engine(**engine_config)
    _run_turn(engine, 1, "first")
    assert engine.scheduler.reset_prefix_cache(reset_connector=reset_connector)
    _run_turn(engine, 2, "second", 72)
    engine.close()
    assert engine.versions("second") == expected_versions
    assert engine.num_loads == expected_loads
    assert engine.num_stores > 0


def test_session_keys_of_a_tier_load_keep_their_version_across_a_weight_update():
    engine = _tier_engine()
    _run_turn(engine, 1, "first")
    assert engine.scheduler.reset_prefix_cache()
    engine.set_version(2)
    (second,) = _session_requests(["second"], 72, 1)
    engine.scheduler.add_request(second)
    waiting = engine.schedule()
    assert second.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    engine.step(waiting, {})
    engine.set_version(3)
    engine.run()
    engine.close()
    assert engine.versions("second") == ["v1", "v1", "v3", "v3"]
    assert engine.num_loads == 1


@pytest.mark.parametrize(
    "num_chunks,turns,expected",
    [
        (
            8,
            [(1, "first", ""), (2, "second", "reset fail"), (3, "third", "")]
            + [(3, "fourth", "reset")],
            {"second": "v2", "third": "v2", "fourth": "v1"},
        ),
        (
            2,
            [(1, "first", ""), (1, "other", ""), (2, "second", "reset")]
            + [(3, "third", "reset")],
            {"second": "v2", "third": "v2"},
        ),
        (
            2,
            [(1, "first", ""), (1, "other", ""), (3, "third", "")]
            + [(4, "fourth", "reset")],
            {"third": "v1", "fourth": "v1"},
        ),
    ],
    ids=["later-computation", "evicted-and-stored-again", "stored-from-a-hit"],
)
def test_session_keys_carry_the_version_of_the_tier_entry(num_chunks, turns, expected):
    engine = _tier_engine(num_chunks=num_chunks)
    for version, request_id, action in turns:
        if "reset" in action:
            assert engine.scheduler.reset_prefix_cache()
        engine.fail_loads = "fail" in action
        _run_turn(engine, version, request_id, token=int(request_id == "other"))
    engine.close()
    assert {r: engine.versions(r) for r in expected} == {
        r: [v, v] for r, v in expected.items()
    }
    assert engine.num_loads == 1


@pytest.mark.parametrize(
    "event,expected_versions",
    [("evicted", 2), ("failed-store", 0), ("reset", 0)],
    ids=["evicted", "failed-store", "reset"],
)
def test_tier_versions_are_dropped_with_their_entries(event, expected_versions):
    engine = _tier_engine(num_chunks=2)
    engine.fail_stores = event == "failed-store"
    _run_turn(engine, 1, "first")
    if event == "evicted":
        _run_turn(engine, 1, "other", token=1)
    if event == "reset":
        assert engine.scheduler.reset_prefix_cache(reset_connector=True)
    assert len(engine.connector._tier._versions) == expected_versions


@pytest.mark.parametrize(
    "num_tokens,num_external,expected",
    [
        (49, 48, (48, True)),
        (49, 40, (32, True)),
        (48, 48, (32, True)),
        (48, 32, (32, True)),
        (49, 10, (0, False)),
    ],
    ids=[
        "aligned",
        "unaligned",
        "whole-prompt",
        "prompt-but-last-block",
        "below-one-object",
    ],
)
def test_session_tier_hit_ends_on_an_object_below_the_last_token(
    num_tokens, num_external, expected
):
    engine = _tier_engine()
    _run_turn(engine, 1, "first", 48)
    (request,) = _session_requests(["second"], num_tokens, 1)
    engine.scheduler.add_request(request)
    assert engine.connector.cap_external_hit(request, 0, num_external, True) == expected


def test_session_tier_hit_is_cut_to_the_last_consistent_object():
    engine = _tier_engine(
        num_chunks=32, sliding_window=32, retention=None, block_size=32
    )
    _run_turn(engine, 1, "first", 40, max_tokens=10)
    _run_turn(engine, 2, "second", 72)
    assert engine.scheduler.reset_prefix_cache()
    _run_turn(engine, 3, "third", 90)
    engine.close()
    assert engine.versions("third") == ["v1", "v3"]
    assert engine.num_loads == 1


def test_session_tier_hit_is_cut_below_every_disagreeing_window():
    engine = _tier_engine(
        num_chunks=64, sliding_window=32, retention=None, block_size=32
    )
    _run_turn(engine, 1, "first", 8 * 32 + 1)
    (second,) = _session_requests(["second"], 8 * 32 + 1, 1)
    engine.scheduler.add_request(second)
    tier = engine.connector._tier
    for chunk_end in (192, 224, 256):
        key = make_offload_key(second.block_hashes[chunk_end // 16 - 1], 1)
        assert key in tier._versions
        tier._versions[key] = "v2"
    assert engine.connector.cap_external_hit(second, 0, 256, True)[0] == 160


def test_session_tier_hit_check_is_linear_when_every_boundary_disagrees():
    engine = _tier_engine(sliding_window=32, retention=None, block_size=32)
    (request,) = _session_requests(["second"], 2000 * 32 + 1, 1)
    tier = engine.connector._tier
    for block_hash in request.block_hashes:
        tier._versions[make_offload_key(block_hash, 0)] = "v1"
        tier._versions[make_offload_key(block_hash, 1)] = "v2"
    tier._versions = Mock(wraps=tier._versions)
    hit = engine.connector.cap_external_hit(request, 0, 2000 * 32, True)
    assert hit == (0, False)
    assert tier._versions.get.call_count <= 4 * 2000


def test_session_tier_hit_lowered_into_the_sync_budget_loads_in_the_same_step():
    engine = _tier_engine(
        num_chunks=32, sliding_window=32, retention=None, block_size=32
    )
    _run_turn(engine, 1, "first", 40, max_tokens=10)
    _run_turn(engine, 2, "second", 72)
    assert engine.scheduler.reset_prefix_cache()
    offload = engine.connector._kv_connector.scheduler
    offload.config = offload.config._replace(
        sync_load_budget_bytes=48 * offload.config.kv_bytes_per_token
    )
    engine.set_version(3)
    (third,) = _session_requests(["third"], 90, 1)
    engine.scheduler.add_request(third)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"third": 58}
    engine.step(scheduler_output, {"third": [0]})
    engine.close()
    assert engine.versions("third") == ["v1", "v3"]
    assert engine.num_loads == 1


def test_session_tier_hit_lowered_onto_a_window_entry_still_stored_is_refused():
    engine = _tier_engine(
        num_chunks=32, sliding_window=32, retention=None, block_size=32
    )
    _run_turn(engine, 1, "first", 40, max_tokens=10)
    _run_turn(engine, 2, "second", 72)
    assert engine.scheduler.reset_prefix_cache()
    (third,) = _session_requests(["third"], 90, 1)
    engine.scheduler.add_request(third)
    manager = engine.connector._tier._manager
    lookup = manager.lookup
    pending = make_offload_key(third.block_hashes[1], 1)
    with patch.object(
        manager,
        "lookup",
        side_effect=lambda key, context: (
            LookupResult.HIT_PENDING if key == pending else lookup(key, context)
        ),
    ):
        assert engine.connector.cap_external_hit(third, 0, 64, True) == (0, False)


def test_session_does_not_load_a_tier_entry_without_a_weight_version():
    engine = _tier_engine()
    _run_turn(engine, 1, "first")
    assert engine.scheduler.reset_prefix_cache()
    engine.connector._tier._versions.clear()
    _run_turn(engine, 2, "second", 72)
    engine.close()
    assert engine.versions("second") == ["v2"] * 4
    assert engine.num_loads == 0


def test_tier_does_not_store_a_block_whose_object_was_never_written():
    engine = _tier_engine()
    scheduler = engine.scheduler
    engine.set_version(1)
    (first,) = _session_requests(["first"], 31, 2, lookahead=1)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [0]})
    outputs = engine.step(engine.schedule(), {"first": [0]})
    assert outputs["first"].finish_reason is not None
    assert scheduler.reset_prefix_cache()
    engine.run()
    _run_turn(engine, 2, "second", 50, lookahead=1)
    engine.close()
    assert engine.versions("first") == ["v1"]
    assert engine.versions("second") == ["v1", "v2", "v2"]
    assert engine.num_loads == 1


def test_tier_wrapper_changes_nothing_for_requests_without_a_session():
    engines = [_tier_engine(key_prefix=prefix) for prefix in ("job", None)]
    outputs = []
    for engine in engines:
        _run_turn(engine, 1, "first", session=None)
        block_pool = engine.scheduler.kv_cache_manager.block_pool
        block_pool.evict_blocks(set(range(len(block_pool.blocks))))
        outputs.append(_run_turn(engine, 2, "second", 72, session=None)["second"])
    wrapped, plain = engines
    assert wrapped.connector._tier is not None and plain.connector._tier is None
    assert wrapped.num_loads > 0
    np.testing.assert_array_equal(outputs[0].routed_experts, outputs[1].routed_experts)
    assert (
        wrapped.num_stores,
        wrapped.num_loads,
        wrapped._tier,
        wrapped._kv_versions,
    ) == (plain.num_stores, plain.num_loads, plain._tier, plain._kv_versions)


def test_session_tier_versions_delegates_every_manager_method():
    assert all(
        getattr(CPUOffloadingManager, name) is getattr(OffloadingManager, name, None)
        or name in vars(SessionTierVersions)
        for name in dir(CPUOffloadingManager)
        if not name.startswith("_")
    )


@pytest.mark.parametrize(
    "method,args",
    [
        ("lookup", ("key", "context")),
        ("get_load_source", ("key", "context")),
        ("complete_load", (["key"], "context")),
        ("on_new_request", ("context",)),
        ("on_request_finished", ("context",)),
        ("take_events", ()),
        ("get_stats", ()),
        ("config_info", ()),
    ],
)
def test_session_tier_versions_forwards_calls_to_its_manager(method, args):
    manager = Mock(spec=CPUOffloadingManager)
    result = getattr(SessionTierVersions(manager, Mock(), ()), method)(*args)
    assert result is getattr(manager, method).return_value
    getattr(manager, method).assert_called_once_with(*args)


@pytest.mark.parametrize(
    "group,retention,expected_versions",
    [
        ({"sliding_window": 32}, None, ["v1", "v1", "v1", "v2"]),
        ({"sliding_window": 32}, 0, ["v1", "v1", "v2", "v2"]),
        ({"mamba_align": True}, None, ["v1", "v1", "v1", "v2"]),
    ],
    ids=["retention-none", "retention-default", "mamba-align"],
)
def test_session_turn_reuses_earlier_outputs_from_the_tier_with_a_second_kv_group(
    group, retention, expected_versions
):
    engine = _tier_engine(retention=retention, **group)
    _run_turn(engine, 1, "first", max_tokens=10)
    assert engine.scheduler.reset_prefix_cache()
    _run_turn(engine, 2, "second", 72)
    engine.close()
    assert engine.versions("second") == expected_versions
    assert engine.num_loads == 1


@pytest.mark.parametrize(
    "evict_window,expected_versions",
    [(False, ["v1"] * 5 + ["v3"]), (True, ["v1"] * 3 + ["v3"] * 3)],
    ids=["groups-agree", "window-recomputed-at-a-later-version"],
)
def test_session_gpu_hit_ends_where_its_kv_groups_disagree_on_the_version(
    evict_window, expected_versions
):
    engine = _tier_engine(spec_name="OtherTierSpec", sliding_window=32, retention=None)
    _run_turn(engine, 1, "first", 81)
    if evict_window:
        (request,) = _session_requests(["first"], 81, 1)
        block_pool = engine.scheduler.kv_cache_manager.block_pool
        block_pool.evict_blocks(
            {
                block_pool.get_cached_block(request.block_hashes[index], [1])[
                    0
                ].block_id
                for index in (3, 4)
            }
        )
    _run_turn(engine, 2, "second", 82)
    _run_turn(engine, 3, "third", 100)
    engine.close()
    assert engine.versions("third") == expected_versions


def test_session_tier_hit_is_lowered_where_its_kv_groups_disagree_on_the_version():
    engine = _tier_engine(num_chunks=16, sliding_window=32, retention=None)
    _run_turn(engine, 1, "first", 81)
    (request,) = _session_requests(["first"], 81, 1)
    tier = engine.connector._tier._manager
    for index in (3, 4):
        key = make_offload_key(request.block_hashes[index], 1)
        chunk = tier._policy.get(key)
        tier._policy.remove(key)
        tier._free_chunk(chunk)
    assert engine.scheduler.reset_prefix_cache()
    _run_turn(engine, 2, "second", 82)
    assert engine.scheduler.reset_prefix_cache()
    _run_turn(engine, 3, "third", 100)
    engine.close()
    assert engine.versions("second")[3:] == ["v2"] * 2
    assert engine.versions("third") == ["v1"] * 3 + ["v3"] * 3


def test_tier_store_of_a_hit_block_takes_the_version_of_its_own_kv_group():
    engine = _tier_engine(
        num_chunks=32, sliding_window=32, retention=None, block_size=32
    )
    scheduler = engine.scheduler
    scheduler.max_num_scheduled_tokens = 48
    engine.set_version(1)
    (first,) = _session_requests(["first"], 100, 1)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {})
    engine.set_version(2)
    engine.run()
    tier = engine.connector._tier
    key = make_offload_key(first.block_hashes[2], 1)
    chunk = tier._manager._policy.get(key)
    tier._manager._policy.remove(key)
    tier._manager._free_chunk(chunk)
    del tier._versions[key]
    scheduler.max_num_scheduled_tokens = 8192
    _run_turn(engine, 3, "second", 66)
    engine.close()
    assert engine.versions("second") == ["v1", "v2"]
    assert tier._versions[key] == "v1"


@pytest.mark.parametrize("tier_hit", [False, True], ids=["gpu", "tier"])
def test_session_hit_through_a_block_filled_across_a_weight_update(tier_hit):
    engine = _tier_engine(
        num_chunks=32, sliding_window=32, retention=None, block_size=32
    )
    scheduler = engine.scheduler
    scheduler.max_num_scheduled_tokens = 48
    engine.set_version(1)
    (first,) = _session_requests(["first"], 100, 1)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {})
    engine.set_version(2)
    engine.run()
    scheduler.max_num_scheduled_tokens = 8192
    if tier_hit:
        assert scheduler.reset_prefix_cache()
    _run_turn(engine, 3, "second", 66)
    engine.close()
    assert engine.versions("second") == ["v1", "v2"]
    assert engine.num_loads == int(tier_hit)


@pytest.mark.parametrize(
    "disagreeing,expected",
    [((6,), 96), (range(7), 0)],
    ids=["window-below-agrees", "every-window-disagrees"],
)
def test_session_gpu_hit_search_looks_up_the_cache_at_most_twice(disagreeing, expected):
    engine = _tier_engine(spec_name="OtherTierSpec", sliding_window=32, retention=None)
    _run_turn(engine, 1, "first", 113)
    (request,) = _session_requests(["second"], 113, 1)
    manager = engine.scheduler.kv_cache_manager
    block_pool = manager.block_pool
    for index in disagreeing:
        (block,) = block_pool.get_cached_block(request.block_hashes[index], [1])
        engine.connector._block_versions[block.block_id] = f"x{index}"
    assert engine.connector.max_cache_hit(request) == expected


def test_session_hit_check_skips_blocks_of_a_group_past_the_tracked_blocks():
    engine = _tier_engine(spec_name="OtherTierSpec", sliding_window=32, retention=None)
    _run_turn(engine, 1, "first", 113)
    (request,) = _session_requests(["second"], 113, 1)
    block_pool = engine.scheduler.kv_cache_manager.block_pool
    blocks = tuple(
        [block_pool.get_cached_block(h, [group])[0] for h in request.block_hashes[:n]]
        for group, n in ((0, 4), (1, 7))
    )
    assert engine.connector._first_mismatch(blocks) is None


def test_session_hit_is_not_cut_by_a_draft_only_block_of_a_later_version():
    engine = _tier_engine(
        spec_name="OtherTierSpec", sliding_window=32, retention=None, draft_window=True
    )
    _run_turn(engine, 1, "first", 81)
    (request,) = _session_requests(["first"], 81, 1)
    block_pool = engine.scheduler.kv_cache_manager.block_pool
    (draft,) = block_pool.get_cached_block(request.block_hashes[3], [2])
    block_pool.evict_blocks({draft.block_id})
    _run_turn(engine, 2, "second", 82)
    _run_turn(engine, 3, "third", 81)
    engine.close()
    (draft,) = block_pool.get_cached_block(request.block_hashes[3], [2])
    assert engine._kv_versions[draft.block_id] == [2] * 16
    assert engine.versions("third") == ["v1"] * 5


@pytest.mark.parametrize(
    "evicted_window_chunk,expected_versions,expected_loads",
    [(None, ["v1"] * 3, 1), (5, ["v1", "v1", "v2"], 1)],
    ids=["window-complete", "window-chunk-evicted"],
)
def test_session_tier_hit_cut_by_a_window_inside_a_block_is_lowered(
    evicted_window_chunk, expected_versions, expected_loads
):
    engine = _tier_engine(
        num_chunks=16, sliding_window=32, retention=None, block_size=32
    )
    _run_turn(engine, 1, "first", 100)
    if evicted_window_chunk is not None:
        (request,) = _session_requests(["first"], 100, 1)
        tier = engine.connector._tier._manager
        key = make_offload_key(request.block_hashes[evicted_window_chunk], 1)
        chunk = tier._policy.get(key)
        tier._policy.remove(key)
        tier._free_chunk(chunk)
    assert engine.scheduler.reset_prefix_cache()
    _run_turn(engine, 2, "second", 120)
    engine.close()
    assert engine.versions("second") == expected_versions
    assert engine.num_loads == expected_loads


@pytest.mark.parametrize("pp_size", [1, 3])
def test_reset_prefix_cache_with_inflight_output_under_kv_pressure(pp_size: int):
    """reset_prefix_cache(reset_running_requests=True) resumes requests in
    the same step it preempts them, so in-flight output must be dropped (the
    resume resamples those positions).

    pp_size=1: regression for the frame-based discard this fix replaces,
    which with spec decode drained one *token* count per output frame and
    over-discarded, corrupting the fresh frames after the resume.
    pp_size=3: back-to-back resets, so the second re-preempts requests whose
    dropped stale share is still in flight -- it must be recorded once (not
    accumulated) and stay dropped.
    """
    max_tokens = 24
    scheduler = _create_async_pp_scheduler(num_spec=3, pp_size=pp_size)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
    )
    pending = list(requests)
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    # Observe re-preemptions with an undrained stale share (the
    # double-count hazard).
    repreempts_with_stale = 0
    orig_preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal repreempts_with_stale
        if getattr(request, "num_stale_output_tokens", 0) > 0:
            repreempts_with_stale += 1
        return orig_preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt

    resets = 0
    reset_steps = {6, 14} if pp_size == 1 else {6, 7, 18, 19}

    def before_step(step: int, engine: PipelinedEngine):
        nonlocal resets
        if pending:
            scheduler.add_request(pending.pop(0))
        if step in reset_steps and (engine.queue or scheduler.running):
            scheduler.reset_prefix_cache(reset_running_requests=True)
            resets += 1

    engine = PipelinedEngine(
        scheduler,
        queue_size=pp_size + 1,
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    )
    engine.run(before_step=before_step)

    assert resets > 0, "test did not exercise reset_prefix_cache"
    if pp_size > 1:
        # The re-preempt-while-stale-pending window needs pipeline depth.
        assert repreempts_with_stale > 0, (
            "test did not exercise re-preemption with an undrained stale share"
        )
    for req in requests:
        assert req.is_finished()
        assert req.num_output_tokens == max_tokens
        # Dropped tokens are never delivered; order must be preserved with
        # no duplicates.
        _assert_ordered_subset(
            list(req.output_token_ids), engine.emitted[req.request_id]
        )
        _assert_positions_consistent(req, engine)
        # All stale shares fully drained by the end.
        assert getattr(req, "num_stale_output_tokens", 0) == 0


def test_requires_kv_delivery_defaults_to_producer_role():
    # No connector: nothing is handed off, so keep the lossless deliver-stale
    # path on preemption.
    assert create_scheduler(async_scheduling=True).requires_kv_delivery is False
    # Only a producer hands KV off when a request completes.
    for role, expected in (
        ("kv_producer", True),
        ("kv_both", True),
        ("kv_consumer", False),
    ):
        scheduler = create_scheduler(
            async_scheduling=True, use_kv_connector=True, kv_role=role
        )
        assert scheduler.requires_kv_delivery is expected, role


@pytest.mark.parametrize(
    ("kv_role", "defer_free"),
    [
        ("kv_producer", False),
        ("kv_consumer", False),
        ("kv_consumer", True),
    ],
)
def test_kv_pressure_preempt_mid_handoff(kv_role: str, defer_free: bool):
    """P/D race: KV pressure hits while the output of a request's final
    prefill chunk -- the hand-off token that would finish it -- is in flight.

    When the victim's blocks free immediately, the request is preempted. On a
    producer, that output must be dropped so the request recomputes; delivering
    it would finish the request and hand off blocks the preemption already
    freed, so the consumer pulls garbage. A consumer hands nothing off, so it
    keeps the lossless deliver-stale path.

    A consumer with overlapping batches (async scheduling or PP) instead fences
    the victim's free behind its in-flight output, so the allocation retry
    stops instead of preempting; the request then finishes from that output
    once it lands. The gate depends on the platform (async scheduling is
    force-disabled on CPU), so force the flag to cover both paths everywhere.
    """
    is_producer = kv_role == "kv_producer"
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=True,
        kv_role=kv_role,
        num_blocks=5,
        block_size=16,
        max_num_batched_tokens=512,
    )
    assert scheduler.requires_kv_delivery is is_producer
    # The production gate requires overlapping batches, which async
    # scheduling only provides off-CPU.
    scheduler.defer_block_free = defer_free

    # 32-token prompts fill 2 blocks each, exhausting the usable pool, so the
    # next decode allocation targets the tail of the running queue (the
    # handoff request) while its prefill output is still in flight.
    decoder = create_requests(
        num_requests=1, num_tokens=32, max_tokens=8, req_ids=["decoder"]
    )[0]
    handoff = create_requests(
        num_requests=1, num_tokens=32, max_tokens=1, req_ids=["handoff"]
    )[0]
    scheduler.add_request(decoder)
    scheduler.add_request(handoff)
    sched_output = scheduler.schedule()
    assert handoff.status == RequestStatus.RUNNING
    assert handoff.num_output_placeholders == 1

    scheduler.schedule()
    if defer_free:
        # The victim's blocks are fenced behind its in-flight output, so
        # preempting it could not satisfy the allocation; the retry stops
        # instead of preempting.
        assert handoff.status == RequestStatus.RUNNING
        assert handoff.num_stale_output_tokens == 0
    else:
        # Blocks free immediately, so the handoff request is preempted.
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_stale_output_tokens == handoff.num_prompt_tokens
        assert handoff.drop_stale_output is is_producer

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    assert handoff.num_stale_output_tokens == 0
    if is_producer:
        # Dropped: recomputed from the waiting queue, so the hand-off happens
        # against real KV.
        assert not handoff.is_finished()
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_output_tokens == 0
        assert handoff.request_id in scheduler.requests
    else:
        assert handoff.is_finished()
        assert handoff.num_output_tokens == 1


def test_resumable_request_handoff():
    """New streaming input resumes with the correct context and stale results
    are ignored.
    """
    scheduler = create_scheduler(async_scheduling=True)
    engine = PipelinedEngine(scheduler, queue_size=3)
    engine._next_token = EOS_TOKEN_ID

    old_prompt_len = 10
    continuation_len = 12
    request_id = "resumable-race"

    request = create_requests(
        1, num_tokens=old_prompt_len, max_tokens=4, req_ids=[request_id]
    )[0]
    request.resumable = True
    continuation = create_requests(
        1, num_tokens=continuation_len, max_tokens=4, req_ids=[request_id]
    )[0]
    continuation.resumable = True

    scheduler.add_request(request)

    for _ in range(3):
        assert engine._schedule()

    engine._process_oldest_step()
    scheduler.add_request(continuation)
    assert engine._schedule()
    new_step, _ = engine.queue[0]
    assert new_step.scheduled_new_reqs[0].num_computed_tokens == old_prompt_len

    all_token_ids_before_stale = list(request.all_token_ids)
    for _ in range(2):
        engine._process_oldest_step()
    assert list(request.output_token_ids) == []
    assert list(request.all_token_ids) == all_token_ids_before_stale

    engine._process_oldest_step()
    assert len(request.output_token_ids) == 1
