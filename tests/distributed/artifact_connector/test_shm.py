# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import time
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.distributed.artifact_connector import (
    ArtifactCommitRequest,
    ArtifactCommitResult,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactCorruptionError,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
    ArtifactObject,
    ArtifactRequestCore,
    ArtifactSchedulerConnector,
    ArtifactWorkerConnector,
    LocalSharedMemoryArtifactReader,
    LocalSharedMemoryArtifactStore,
    materialize_routed_experts,
)
from vllm.distributed.artifact_connector.fields import ROUTED_EXPERTS
from vllm.distributed.artifact_connector.request_core import (
    decode_artifact_array,
    encode_artifact_array,
)
from vllm.outputs import CompletionOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


class FakeRoutingWriter:
    def __init__(self, slots: np.ndarray) -> None:
        self.slots = slots
        self.dtype = slots.dtype
        self.shape_per_token = slots.shape[1:]

    def read_token_range(
        self,
        block_ids: list[int],
        *,
        token_start: int,
        token_end: int,
        block_size: int,
    ) -> np.ndarray:
        positions = np.arange(token_start, token_end)
        block_ids_array = np.asarray(block_ids)
        slot_ids = (
            block_ids_array[positions // block_size] * block_size
            + positions % block_size
        )
        return self.slots[slot_ids].copy()


def _make_vllm_config(tmp_path):
    return SimpleNamespace(
        artifact_config=SimpleNamespace(shm_dir=str(tmp_path)),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        instance_id="instance",
    )


def _make_worker_connector(tmp_path, *, max_bytes: int = 1 << 20):
    logical = np.arange(12 * 3 * 2, dtype=np.uint8).reshape(12, 3, 2)
    block_ids = [2, 0, 3]
    slots = np.zeros((16, 3, 2), dtype=np.uint8)
    for logical_block, physical_block in enumerate(block_ids):
        logical_start = logical_block * 4
        physical_start = physical_block * 4
        slots[physical_start : physical_start + 4] = logical[
            logical_start : logical_start + 4
        ]

    connector = ArtifactWorkerConnector.__new__(ArtifactWorkerConnector)
    writer = FakeRoutingWriter(slots)
    connector.store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=max_bytes,
        ttl_seconds=60,
    )
    connector.core = ArtifactRequestCore(
        connector.store,
        writer,
        namespace="test-namespace",
    )
    return connector, logical, block_ids


def _request(
    request_id: str,
    request_attempt_id: str,
    block_ids: list[int],
    *,
    token_start: int = 0,
    token_end: int = 10,
    policy_epoch: int = 0,
    cached_blocks=None,
) -> ArtifactFinalizeRequest:
    return ArtifactFinalizeRequest(
        request_id=request_id,
        request_attempt_id=request_attempt_id,
        block_ids=block_ids,
        block_hashes=[bytes([index]) * 32 for index in range(3)],
        token_start=token_start,
        token_end=token_end,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=policy_epoch,
        cached_blocks=[] if cached_blocks is None else cached_blocks,
    )


def _commit_request(request: ArtifactFinalizeRequest) -> ArtifactCommitRequest | None:
    block_end = request.token_end // request.hash_block_size * request.hash_block_size
    if block_end <= 0:
        return None
    return ArtifactCommitRequest(
        operation_id=f"commit-{request.request_id}",
        request_id=request.request_id,
        request_attempt_id=request.request_attempt_id,
        block_ids=request.block_ids,
        block_hashes=request.block_hashes,
        block_start=0,
        block_end=block_end,
        physical_block_size=request.physical_block_size,
        hash_block_size=request.hash_block_size,
        policy_epoch=request.policy_epoch,
    )


def _prepare_and_finalize(
    connector: ArtifactWorkerConnector,
    request: ArtifactFinalizeRequest,
):
    commit = _commit_request(request)
    if commit is not None:
        prepared = connector.core.prepare_commit(commit)
        assert connector.core.publish_commits([prepared])[commit.operation_id] is None
    return connector.core.finalize(request)


def _read_in_child(root: str, store_id: str, keys: list[str], result_queue) -> None:
    reader = LocalSharedMemoryArtifactReader(root, store_id)
    result_queue.put(materialize_routed_experts(reader, keys).tolist())


def test_object_envelope_round_trip():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = encode_artifact_array(
        key="key",
        kind="block",
        field_spec=ROUTED_EXPERTS,
        field_profile_id="profile",
        array=array,
        source_token_start=0,
        valid_len=4,
        kv_block_hash=b"a" * 32,
    )

    decoded, header = decode_artifact_array(payload, expected_key="key")

    np.testing.assert_array_equal(decoded, array)
    assert header["valid_len"] == 4
    assert header["kv_block_hash"] == (b"a" * 32).hex()


def test_object_envelope_rejects_corruption():
    payload = encode_artifact_array(
        key="key",
        kind="tail",
        field_spec=ROUTED_EXPERTS,
        field_profile_id="profile",
        array=np.zeros((2, 1), dtype=np.uint8),
        source_token_start=0,
        valid_len=2,
    )
    corrupted = payload[:-1] + bytes([payload[-1] ^ 1])

    with pytest.raises(ArtifactCorruptionError, match="checksum"):
        decode_artifact_array(corrupted, expected_key="key")


def test_core_returns_ordered_keys_and_inline_value(tmp_path):
    connector, logical, block_ids = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids)

    finalized = _prepare_and_finalize(connector, request)

    assert len(finalized.keys) == 3
    assert "/block/" in finalized.keys[0]
    assert "/block/" in finalized.keys[1]
    assert "/tail/" in finalized.keys[2]
    np.testing.assert_array_equal(finalized.value, logical[:10])
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, finalized.keys),
        logical[:10],
    )
    connector.close()


def test_exact_block_request_has_no_tail(tmp_path):
    connector, logical, block_ids = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids, token_end=8)

    finalized = _prepare_and_finalize(connector, request)

    assert len(finalized.keys) == 2
    assert all("/block/" in key for key in finalized.keys)
    np.testing.assert_array_equal(finalized.value, logical[:8])
    connector.close()


@pytest.mark.parametrize("token_end", [1, 4, 5, 8, 9, 12])
def test_key_count_is_ceiling_of_executed_tokens(tmp_path, token_end):
    connector, _, block_ids = _make_worker_connector(tmp_path)
    request = _request(
        f"request-{token_end}",
        f"{token_end:032x}",
        block_ids,
        token_end=token_end,
    )

    finalized = _prepare_and_finalize(connector, request)

    assert len(finalized.keys) == (token_end + 3) // 4
    connector.close()


def test_nonzero_prompt_start_is_rejected(tmp_path):
    connector, _, block_ids = _make_worker_connector(tmp_path)
    request = _request(
        "request-a",
        "a" * 32,
        block_ids,
        token_start=1,
    )

    with pytest.raises(ValueError, match="routed_experts_prompt_start=0"):
        connector.core.finalize(request)
    connector.close()


def test_worker_batches_commits_and_reports_keys_per_request(tmp_path):
    connector, _, block_ids = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids, token_end=4)
    second = _request("request-b", "b" * 32, block_ids, token_end=8)
    commits = [_commit_request(first), _commit_request(second)]

    output = connector.finalize(
        ArtifactConnectorMetadata(
            commits=[commit for commit in commits if commit is not None]
        )
    )

    assert output is not None
    assert [len(result.block_keys) for result in output.commit_results] == [1, 2]
    assert all(result.error is None for result in output.commit_results)
    connector.close()


def test_cached_blocks_are_reused_without_put(tmp_path):
    connector, logical, block_ids = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids, token_end=8)
    first_finalized = _prepare_and_finalize(connector, first)
    block_refs = [
        SimpleNamespace(block_index=0, block_hash=first.block_hashes[0]),
        SimpleNamespace(block_index=1, block_hash=first.block_hashes[1]),
    ]
    second = _request(
        "request-b",
        "b" * 32,
        block_ids,
        token_end=10,
        cached_blocks=block_refs,
    )
    connector.store.put = Mock(wraps=connector.store.put)

    finalized = connector.core.finalize(second)

    assert connector.store.put.call_count == 1
    assert len(connector.store.put.call_args.args[0]) == 1
    assert finalized.keys[:2] == first_finalized.keys
    np.testing.assert_array_equal(finalized.value, logical[:10])
    connector.close()


def test_missing_cached_block_fails_closed(tmp_path):
    connector, _, block_ids = _make_worker_connector(tmp_path)
    request = _request(
        "request-a",
        "a" * 32,
        block_ids,
        token_end=4,
        cached_blocks=[
            SimpleNamespace(block_index=0, block_hash=b"a" * 32),
        ],
    )

    with pytest.raises(RuntimeError, match="not ready"):
        connector.core.finalize(request)
    connector.close()


def test_policy_epoch_fences_keys(tmp_path):
    connector, _, block_ids = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids, token_end=4)
    first_finalized = _prepare_and_finalize(connector, first)
    connector.core.advance_policy_epoch()
    second = _request(
        "request-b",
        "b" * 32,
        block_ids,
        token_end=4,
        policy_epoch=1,
    )

    second_finalized = _prepare_and_finalize(connector, second)

    assert first_finalized.keys != second_finalized.keys
    connector.close()


def test_reader_materializes_from_another_process(tmp_path):
    connector, logical, block_ids = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids)
    finalized = _prepare_and_finalize(connector, request)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_read_in_child,
        args=(
            str(tmp_path),
            connector.store.store_id,
            finalized.keys,
            result_queue,
        ),
    )

    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)

    assert process.exitcode == 0
    np.testing.assert_array_equal(np.asarray(result_queue.get()), logical[:10])
    connector.close()


def test_store_capacity_failure_is_per_object(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=5,
        ttl_seconds=60,
    )

    results = store.put(
        [
            ArtifactObject("small", b"1234"),
            ArtifactObject("large", b"123456"),
        ]
    )

    assert results[0].error is None
    assert "ArtifactCapacityError" in str(results[1].error)
    assert store.exists(["small", "large"]) == [True, False]
    store.close()


def test_store_rejects_different_value_for_same_key(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )
    assert store.put([ArtifactObject("key", b"a")])[0].error is None

    result = store.put([ArtifactObject("key", b"b")])[0]

    assert "ArtifactCorruptionError" in str(result.error)
    store.close()


def test_gc_removes_expired_objects_and_partial_files(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    assert store.put([ArtifactObject("key", b"value")])[0].error is None
    object_path = store._path("key")
    partial = store.objects_dir / ".orphan.partial"
    partial.write_bytes(b"partial")
    old_time = time.time() - 5
    os.utime(object_path, (old_time, old_time))
    os.utime(partial, (old_time, old_time))

    store.gc()

    assert not object_path.exists()
    assert not partial.exists()
    store.close()


def test_scheduler_connector_checks_shm_existence(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )
    connector = ArtifactSchedulerConnector(_make_vllm_config(tmp_path))
    attempt_id = connector.request_progress(
        request_id="request-a",
        block_ids=[1],
        block_hashes=[b"a" * 32],
        token_start=0,
        accepted_token_end=4,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    metadata = connector.build_connector_metadata()
    assert metadata is not None
    key = "vllm-artifact/test/routed_experts/block/key"
    assert store.put([ArtifactObject(key, b"payload")])[0].error is None
    connector.acknowledge(
        ArtifactConnectorOutput(
            commit_results=[
                ArtifactCommitResult(
                    operation_id=metadata.commits[0].operation_id,
                    request_id="request-a",
                    request_attempt_id=attempt_id,
                    block_end=4,
                    block_keys=[key],
                )
            ]
        )
    )

    assert (
        connector.max_ready_prefix_tokens(
            block_hashes=[b"a" * 32],
            token_start=0,
            max_tokens=4,
            hash_block_size=4,
            policy_epoch=0,
        )
        == 4
    )
    store._path(key).unlink()
    assert (
        connector.max_ready_prefix_tokens(
            block_hashes=[b"a" * 32],
            token_start=0,
            max_tokens=4,
            hash_block_size=4,
            policy_epoch=0,
        )
        == 0
    )
    connector.close()
    store.close()


def test_scheduler_finalizes_only_the_accepted_range():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = Mock()
    scheduler.artifact_connector.has_unacked_commits.return_value = False
    scheduler.artifact_policy_epoch = 0
    scheduler._routed_experts_block_ids = {"request-a": [3, 5, 7]}
    scheduler.routed_experts_manager = SimpleNamespace(block_size=4)
    scheduler.hash_block_size = 4
    scheduler._connector_finished = Mock(return_value=(False, None))
    scheduler.ec_connector = None
    scheduler._inflight_prefills = set()
    scheduler.encoder_cache_manager = Mock()
    scheduler.finished_req_ids = set()
    request = Mock()
    request.request_id = "request-a"
    request.is_finished.return_value = True
    request.sampling_params = SimpleNamespace(routed_experts_prompt_start=None)
    request.num_computed_tokens = 12
    request.num_tokens = 10
    request.block_hashes = [b"a" * 32, b"b" * 32, b"c" * 32]

    scheduler._start_artifact_finalize(request)

    scheduler.artifact_connector.request_finished.assert_called_once_with(
        request_id="request-a",
        block_ids=[3, 5, 7],
        block_hashes=[b"a" * 32, b"b" * 32, b"c" * 32],
        token_start=0,
        token_end=9,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )


def test_async_preemption_retains_slots_until_output_is_consumed():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = Mock()
    scheduler.artifact_connector.has_unacked_commits.return_value = False
    scheduler.kv_cache_manager = Mock()
    scheduler.encoder_cache_manager = Mock()
    scheduler._inflight_prefills = set()
    scheduler.waiting = Mock()
    scheduler.reset_preempted_req_ids = set()
    scheduler.log_stats = False
    scheduler.defer_block_free = True
    scheduler.processed_step_seq = 2
    scheduler.deferred_frees = deque()
    scheduler._artifact_preempted_blocks = {}
    retained_block = Mock()
    scheduler.kv_cache_manager.pop_blocks_for_free.return_value = [retained_block]
    freed_blocks = []
    scheduler.kv_cache_manager.block_pool.free_blocks.side_effect = lambda blocks: (
        freed_blocks.extend(blocks)
    )
    request = Mock()
    request.request_id = "request-a"
    request.status = RequestStatus.RUNNING
    request.num_in_flight_tokens = 1
    request.last_sched_seq = 3
    request.spec_token_ids = []
    request.num_preemptions = 0

    scheduler._preempt_request(request, 0.0)

    assert scheduler._artifact_preempted_blocks[request.request_id] == [
        (3, [retained_block])
    ]
    assert not freed_blocks
    scheduler.processed_step_seq = 3
    scheduler._release_artifact_preempted_blocks(request.request_id)
    assert freed_blocks == [retained_block]


def test_failed_weight_update_stays_fenced_and_reload_recovers():
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = Mock()
    engine_core.model_executor.collective_rpc.side_effect = RuntimeError("failed")
    engine_core.scheduler = Mock()
    engine_core.scheduler.artifact_policy_update_active = False

    with pytest.raises(RuntimeError, match="failed"):
        engine_core.collective_rpc("start_weight_update")

    engine_core.scheduler.begin_artifact_policy_update.assert_called_once_with()
    engine_core.scheduler.advance_artifact_policy_epoch.assert_not_called()

    engine_core.model_executor.collective_rpc.side_effect = None
    engine_core.model_executor.collective_rpc.return_value = [None]
    engine_core.scheduler.artifact_policy_update_active = True
    engine_core.collective_rpc("reload_weights")
    engine_core.scheduler.advance_artifact_policy_epoch.assert_called_once_with()


def test_scheduler_releases_inline_terminal_output_after_ack(tmp_path):
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector(
        _make_vllm_config(tmp_path)
    )
    attempt_id = scheduler.artifact_connector.request_finished(
        request_id="request-a",
        block_ids=[1, 2],
        block_hashes=[b"a" * 32, b"b" * 32],
        token_start=0,
        token_end=6,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    scheduler.artifact_connector.build_connector_metadata()
    request = SimpleNamespace(request_id="request-a", client_index=3)
    terminal_output = EngineCoreOutput(
        request_id="request-a",
        new_token_ids=[7],
        finish_reason=FinishReason.STOP,
    )
    scheduler._pending_artifact_outputs = {"request-a": (request, terminal_output)}
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler._free_blocks = Mock()
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    routed_experts = np.zeros((6, 3, 2), dtype=np.uint8)

    scheduler._release_artifact_outputs(
        ArtifactConnectorOutput(
            [
                ArtifactFinalizeResult(
                    request_id="request-a",
                    request_attempt_id=attempt_id,
                    routed_experts=routed_experts,
                )
            ]
        ),
        outputs,
    )

    assert terminal_output.routed_experts is routed_experts
    assert terminal_output.artifact_keys is None
    scheduler._free_blocks.assert_called_once_with(request)
    scheduler.artifact_connector.close()


def test_scheduler_releases_external_keys_after_ack(tmp_path):
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector(
        _make_vllm_config(tmp_path)
    )
    attempt_id = scheduler.artifact_connector.request_finished(
        request_id="request-a",
        block_ids=[1],
        block_hashes=[b"a" * 32],
        token_start=0,
        token_end=2,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    scheduler.artifact_connector.build_connector_metadata()
    request = SimpleNamespace(request_id="request-a", client_index=0)
    terminal_output = EngineCoreOutput(
        request_id="request-a",
        new_token_ids=[7],
        finish_reason=FinishReason.STOP,
    )
    scheduler._pending_artifact_outputs = {"request-a": (request, terminal_output)}
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler._free_blocks = Mock()
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    scheduler._release_artifact_outputs(
        ArtifactConnectorOutput(
            [
                ArtifactFinalizeResult(
                    request_id="request-a",
                    request_attempt_id=attempt_id,
                    artifact_keys=["key"],
                )
            ]
        ),
        outputs,
    )

    assert terminal_output.routed_experts is None
    assert terminal_output.artifact_keys == ["key"]
    scheduler.artifact_connector.close()


def test_public_output_exposes_ordered_artifact_keys():
    output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        artifact_keys=["key-0", "key-1"],
    )

    assert output.artifact_keys == ["key-0", "key-1"]
