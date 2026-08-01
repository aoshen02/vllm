# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import multiprocessing
import os
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.connector import (
    ArtifactSchedulerConnector,
    ArtifactWorkerConnector,
)
from vllm.distributed.artifact_connector.protocol import (
    ArtifactCommitRequest,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
)
from vllm.distributed.artifact_connector.request_core import (
    ArtifactKeySpace,
    decode_artifact_array,
    encode_artifact_array,
    materialize_routed_experts,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactReader,
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _make_vllm_config(tmp_path):
    return SimpleNamespace(
        artifact_config=SimpleNamespace(
            shm_dir=str(tmp_path),
            max_shm_bytes=1 << 20,
            shm_ttl_seconds=60,
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_hidden_layers=3,
                num_experts_per_tok=2,
                num_experts=256,
            ),
        ),
        instance_id="instance",
    )


def _make_scheduler_connector(tmp_path):
    return ArtifactSchedulerConnector(_make_vllm_config(tmp_path))


def _make_worker_connector(tmp_path, *, max_bytes: int = 1 << 20):
    logical = np.arange(12 * 3 * 2, dtype=np.uint8).reshape(12, 3, 2)
    config = _make_vllm_config(tmp_path)
    config.artifact_config.max_shm_bytes = max_bytes
    connector = ArtifactWorkerConnector(config, Mock())
    return connector, logical


def _block_key(
    block_hash: bytes,
    weight_version: str = "default",
) -> str:
    key_space = ArtifactKeySpace(
        np.dtype("uint8"),
        (3, 2),
    )
    return key_space.block_key(block_hash, 4, weight_version)


def _request(
    request_id: str,
    request_attempt_id: str,
    *,
    token_end: int = 10,
) -> ArtifactFinalizeRequest:
    return ArtifactFinalizeRequest(
        request_id=request_id,
        request_attempt_id=request_attempt_id,
        weight_version="default",
        block_hashes=[bytes([index]) * 32 for index in range(3)],
        token_end=token_end,
        hash_block_size=4,
    )


def _commit_request(request: ArtifactFinalizeRequest) -> ArtifactCommitRequest | None:
    block_end = request.token_end // request.hash_block_size * request.hash_block_size
    if block_end <= 0:
        return None
    return ArtifactCommitRequest(
        request_id=request.request_id,
        weight_version=request.weight_version,
        block_hashes=request.block_hashes[: block_end // request.hash_block_size],
        block_start=0,
        hash_block_size=request.hash_block_size,
    )


def _prepare_and_finalize(
    connector: ArtifactWorkerConnector,
    request: ArtifactFinalizeRequest,
    logical: np.ndarray,
):
    connector.buffer.capture(request.request_id, 0, logical[: request.token_end])
    commit = _commit_request(request)
    if commit is not None:
        prepared = connector.request_core.prepare_commit(commit)
        connector.request_core.publish_commits([prepared])
    return connector.request_core.finalize(request)


def _read_in_child(root: str, store_id: str, keys: list[str], result_queue) -> None:
    reader = LocalSharedMemoryArtifactReader(root, store_id)
    result_queue.put(materialize_routed_experts(reader, keys).tolist())


def test_object_envelope_round_trip():
    array = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    payload = encode_artifact_array(
        key="key",
        kind="block",
        field_profile_id="profile",
        array=array,
        source_token_start=0,
    )

    decoded, header = decode_artifact_array(payload, expected_key="key")

    np.testing.assert_array_equal(decoded, array)
    assert header["shape"] == [4, 3, 2]


def test_object_envelope_rejects_corruption():
    payload = encode_artifact_array(
        key="key",
        kind="tail",
        field_profile_id="profile",
        array=np.zeros((2, 1), dtype=np.uint8),
        source_token_start=0,
    )
    corrupted = payload[:-1] + bytes([payload[-1] ^ 1])

    with pytest.raises(ArtifactCorruptionError, match="checksum"):
        decode_artifact_array(corrupted, expected_key="key")


def test_logical_buffer_survives_recompute_and_release():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,))
    buffer.capture("request", 4, np.arange(4, 8, dtype=np.uint8).reshape(-1, 1))
    buffer.capture("request", 6, np.array([[60], [70], [80]], dtype=np.uint8))

    np.testing.assert_array_equal(
        buffer.read("request", 4, 9).ravel(),
        [4, 5, 60, 70, 80],
    )
    buffer.release_through("request", 8)
    np.testing.assert_array_equal(buffer.read("request", 8, 9).ravel(), [80])
    buffer.capture("request", 0, np.arange(9, dtype=np.uint8).reshape(-1, 1))
    np.testing.assert_array_equal(buffer.read("request", 8, 9).ravel(), [8])


def test_logical_buffer_encodes_router_ids_to_artifact_dtype():
    buffer = RoutedExpertsArtifactBuffer(np.dtype("uint8"), (1,))

    buffer.capture("request", 0, np.array([[1], [2]], dtype=np.int32))

    encoded = buffer.read("request", 0, 2)
    assert encoded.dtype == np.uint8
    np.testing.assert_array_equal(encoded.ravel(), [1, 2])


def test_core_returns_ordered_keys_and_shm_value(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32)

    keys = _prepare_and_finalize(connector, request, logical)

    assert len(keys) == 3
    assert "/block/" in keys[0]
    assert "/block/" in keys[1]
    assert "/tail/" in keys[2]
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, keys),
        logical[:10],
    )
    connector.shutdown()


def test_exact_block_request_has_no_tail(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, token_end=8)

    keys = _prepare_and_finalize(connector, request, logical)

    assert len(keys) == 2
    assert all("/block/" in key for key in keys)
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, keys), logical[:8]
    )
    connector.shutdown()


def test_finalize_recovers_tail_from_released_full_block(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    connector.buffer.capture("request-a", 0, logical[:8])
    commit = ArtifactCommitRequest(
        request_id="request-a",
        weight_version="default",
        block_hashes=[bytes([index]) * 32 for index in range(2)],
        block_start=0,
        hash_block_size=4,
    )
    prepared = connector.request_core.prepare_commit(commit)
    connector.request_core.publish_commits([prepared])

    request = _request("request-a", "attempt-a", token_end=6)
    keys = connector.request_core.finalize(request)

    assert len(keys) == 2
    assert "/block/" in keys[0]
    assert "/tail/" in keys[1]
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, keys), logical[:6]
    )
    connector.shutdown()


def test_same_metadata_commit_then_earlier_finalize(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    connector.buffer.capture("request-a", 0, logical[:8])
    finalize = _request("request-a", "attempt-a", token_end=6)
    commit = ArtifactCommitRequest(
        request_id="request-a",
        weight_version="default",
        block_hashes=finalize.block_hashes[:2],
        block_start=0,
        hash_block_size=4,
    )

    output = connector.start_step(
        ArtifactConnectorMetadata(commits=[commit], finalizes=[finalize]),
        finished_req_ids={"request-a"},
    )

    assert output is not None
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, output.results[0].keys),
        logical[:6],
    )
    connector.shutdown()


@pytest.mark.parametrize("token_end", [1, 4, 5, 8, 9, 12])
def test_key_count_is_ceiling_of_executed_tokens(tmp_path, token_end):
    connector, logical = _make_worker_connector(tmp_path)
    request = _request(
        f"request-{token_end}",
        f"{token_end:032x}",
        token_end=token_end,
    )

    keys = _prepare_and_finalize(connector, request, logical)

    assert len(keys) == (token_end + 3) // 4
    connector.shutdown()


def test_worker_batches_commits_and_reports_keys_per_request(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, token_end=4)
    second = _request("request-b", "b" * 32, token_end=8)
    connector.buffer.capture(first.request_id, 0, logical[:4])
    connector.buffer.capture(second.request_id, 0, logical[:8])
    commits = [_commit_request(first), _commit_request(second)]

    output = connector.start_step(
        ArtifactConnectorMetadata(
            commits=[commit for commit in commits if commit is not None]
        )
    )

    assert output is None
    connector.shutdown()


def test_cached_blocks_are_reused_without_put(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, token_end=8)
    first_keys = _prepare_and_finalize(connector, first, logical)
    second = _request(
        "request-b",
        "b" * 32,
        token_end=10,
    )
    connector.buffer.capture(second.request_id, 8, logical[8:10])
    connector.store.put = Mock(wraps=connector.store.put)

    keys = connector.request_core.finalize(second)

    assert connector.store.put.call_count == 1
    assert len(connector.store.put.call_args.args[0]) == 1
    assert keys[:2] == first_keys
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, keys), logical[:10]
    )
    connector.shutdown()


def test_missing_cached_block_fails_closed(tmp_path):
    connector, _ = _make_worker_connector(tmp_path)
    request = _request(
        "request-a",
        "a" * 32,
        token_end=4,
    )

    keys = connector.request_core.finalize(request)
    with pytest.raises(RuntimeError, match="does not exist"):
        materialize_routed_experts(connector.store, keys)
    connector.shutdown()


def test_reader_materializes_from_another_process(tmp_path):
    connector, logical = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32)
    keys = _prepare_and_finalize(connector, request, logical)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_read_in_child,
        args=(
            str(tmp_path),
            connector.store.store_id,
            keys,
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
    connector.shutdown()


def test_store_capacity_failure_raises(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=5,
        ttl_seconds=60,
    )

    with pytest.raises(ArtifactCapacityError):
        store.put(
            [
                ArtifactObject("small", b"1234"),
                ArtifactObject("large", b"123456"),
            ]
        )

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
    store.put([ArtifactObject("key", b"a")])

    with pytest.raises(ArtifactCorruptionError):
        store.put([ArtifactObject("key", b"b")])
    store.close()


def test_live_store_retains_objects_older_than_ttl(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    store.put([ArtifactObject("key", b"value")])
    object_path = store._path("key")
    partial = store.objects_dir / ".orphan.partial"
    partial.write_bytes(b"partial")
    old_time = time.time() - 5
    os.utime(object_path, (old_time, old_time))
    os.utime(partial, (old_time, old_time))

    store.put([ArtifactObject("second", b"value")])

    assert object_path.exists()
    store.close()

    reopened = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    assert reopened.get(["key"]) == [b"value"]
    assert not partial.exists()
    reopened.close()


def test_ttl_removes_inactive_engine_store(tmp_path):
    stale = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "stale-instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    stale.put([ArtifactObject("key", b"value")])
    stale_root = stale.root
    stale.close()

    old_time = time.time() - 5
    for path in [
        *stale.objects_dir.iterdir(),
        stale.objects_dir,
        stale_root / ".writer.lock",
        stale_root,
    ]:
        os.utime(path, (old_time, old_time))

    fresh = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "fresh-instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )

    assert not stale_root.exists()
    fresh.close()


def test_ttl_keeps_expired_store_with_live_writer(tmp_path):
    live = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "live-instance",
        0,
        max_bytes=100,
        ttl_seconds=1,
    )
    live.put([ArtifactObject("key", b"value")])
    live_root = live.root

    old_time = time.time() - 5
    for path in [
        *live.objects_dir.iterdir(),
        live.objects_dir,
        live_root / ".writer.lock",
        live_root,
    ]:
        os.utime(path, (old_time, old_time))

    collector = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "collector-instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )

    assert live_root.exists()
    assert live.get(["key"]) == [b"value"]
    collector.close()
    live.close()


def test_writer_lock_retries_if_collector_unlinks_open_inode(tmp_path, monkeypatch):
    store = object.__new__(LocalSharedMemoryArtifactStore)
    store.root = tmp_path / "store"
    store.ttl_seconds = 60

    real_flock = fcntl.flock
    first_call = True

    def unlink_on_first_flock(fd, operation):
        nonlocal first_call
        real_flock(fd, operation)
        if first_call:
            first_call = False
            (store.root / ".writer.lock").unlink()
            (store.root / ".writer.lock").touch(mode=0o600)

    monkeypatch.setattr(fcntl, "flock", unlink_on_first_flock)
    fd = store._acquire_writer_lock()
    try:
        opened = os.fstat(fd)
        current = (store.root / ".writer.lock").stat()
        assert (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
    finally:
        os.close(fd)


def test_scheduler_connector_checks_shm_existence(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )
    connector = _make_scheduler_connector(tmp_path)
    key = _block_key(b"a" * 32)
    store.put([ArtifactObject(key, b"payload")])

    connector.ensure_prefix_ready(
        block_hashes=[b"a" * 32],
        cached_token_end=4,
        hash_block_size=4,
        weight_version="default",
    )
    store._path(key).unlink()
    with pytest.raises(RuntimeError, match="missing a routed-experts artifact"):
        connector.ensure_prefix_ready(
            block_hashes=[b"a" * 32],
            cached_token_end=4,
            hash_block_size=4,
            weight_version="default",
        )
    connector.shutdown()
    store.close()


def test_partial_kv_hit_requires_covering_full_artifact_block(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )
    connector = _make_scheduler_connector(tmp_path)
    block_hash = b"a" * 32
    with pytest.raises(RuntimeError, match="missing a routed-experts artifact"):
        connector.request_started(
            request_id="missing",
            block_hashes=[block_hash],
            cached_token_end=1,
            hash_block_size=4,
        )

    key = _block_key(block_hash)
    store.put([ArtifactObject(key, b"payload")])
    connector.request_started(
        request_id="ready",
        block_hashes=[block_hash],
        cached_token_end=1,
        hash_block_size=4,
    )
    connector.request_progress(
        request_id="ready",
        block_hashes=[block_hash],
        accepted_token_end=4,
        hash_block_size=4,
    )

    assert connector.build_connector_meta() is None
    connector.shutdown()
    store.close()


def test_scheduler_and_worker_derive_the_same_block_key(tmp_path):
    scheduler = _make_scheduler_connector(tmp_path)
    worker = ArtifactWorkerConnector(_make_vllm_config(tmp_path), Mock())
    block_hash = b"a" * 32

    assert _block_key(block_hash) == worker.request_core.key_space.block_key(
        block_hash,
        4,
        "default",
    )
    scheduler.shutdown()
    worker.shutdown()


def test_logical_key_space_has_no_store_identity():
    first = ArtifactKeySpace(
        dtype=np.dtype("uint8"),
        shape_per_token=(3, 2),
    )
    second = ArtifactKeySpace(
        dtype=np.dtype("uint8"),
        shape_per_token=(3, 2),
    )

    assert first.block_key(b"a" * 32, 4, "default") == second.block_key(
        b"a" * 32, 4, "default"
    )


def test_abort_keeps_completed_blocks_and_cancels_finalize(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    connector.request_started(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        cached_token_end=0,
        hash_block_size=4,
    )
    connector.request_progress(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        accepted_token_end=4,
        hash_block_size=4,
    )
    connector.request_finished(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        token_end=4,
        hash_block_size=4,
    )

    connector.request_aborted("request-a")

    metadata = connector.build_connector_meta()
    assert metadata is not None
    assert len(metadata.commits) == 1
    assert metadata.finalizes == []
    connector.shutdown()


def test_abort_does_not_wait_for_already_sent_block_commit(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    connector.request_started(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        cached_token_end=0,
        hash_block_size=4,
    )
    connector.request_progress(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        accepted_token_end=4,
        hash_block_size=4,
    )
    metadata = connector.build_connector_meta()
    assert metadata is not None
    assert len(metadata.commits) == 1

    connector.request_aborted("request-a")

    assert connector.build_connector_meta() is None
    connector.shutdown()


def test_reconstructed_scheduler_discovers_ready_blocks(tmp_path):
    worker = ArtifactWorkerConnector(_make_vllm_config(tmp_path), Mock())
    logical = np.arange(12 * 3 * 2, dtype=np.uint8).reshape(12, 3, 2)
    request = _request("request-a", "a" * 32, token_end=4)
    _prepare_and_finalize(worker, request, logical)
    scheduler = _make_scheduler_connector(tmp_path)

    scheduler.ensure_prefix_ready(
        block_hashes=request.block_hashes,
        cached_token_end=4,
        hash_block_size=4,
        weight_version="default",
    )
    scheduler.shutdown()
    worker.shutdown()


def test_scheduler_finalizes_only_the_accepted_range():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = Mock()
    scheduler.hash_block_size = 4
    scheduler._connector_finished = Mock(return_value=(False, None))
    scheduler.ec_connector = None
    scheduler._inflight_prefills = set()
    scheduler.encoder_cache_manager = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_blocks = Mock()
    request = Mock()
    request.request_id = "request-a"
    request.is_finished.return_value = True
    request.num_computed_tokens = 12
    request.num_tokens = 10
    request.block_hashes = [b"a" * 32, b"b" * 32, b"c" * 32]

    scheduler._free_request(request, artifact_token_end=9)

    scheduler.artifact_connector.request_finished.assert_called_once_with(
        request_id="request-a",
        block_hashes=[b"a" * 32, b"b" * 32, b"c" * 32],
        token_end=9,
        hash_block_size=4,
    )
    scheduler._free_blocks.assert_called_once_with(request)


def test_frontend_artifact_finalize_uses_standard_finish_path():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = Mock()
    scheduler._pending_artifact_outputs = {}
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.waiting = Mock()
    scheduler.skipped_waiting = Mock()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler._free_request = Mock(  # type: ignore[method-assign]
        return_value=(None, None)
    )
    request = Mock()
    request.request_id = "request-a"
    request.status = RequestStatus.RUNNING
    request.is_finished.side_effect = lambda: RequestStatus.is_finished(request.status)
    request.num_tokens = 10
    request.client_index = 0
    request.take_events.return_value = None
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]

    scheduler.finalize_artifact_requests([(request.request_id, 8, "stop")])

    assert scheduler.running == []
    assert request.status == RequestStatus.FINISHED_STOPPED
    scheduler._free_request.assert_called_once_with(
        request,
        delay_free_blocks=False,
        artifact_token_end=8,
    )
    assert request.request_id in scheduler._pending_artifact_outputs


def test_async_preemption_uses_standard_kv_lifecycle():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = Mock()
    scheduler._free_request_blocks = Mock()
    scheduler.encoder_cache_manager = Mock()
    scheduler._inflight_prefills = set()
    scheduler.waiting = Mock()
    scheduler.reset_preempted_req_ids = set()
    scheduler.log_stats = False
    request = Mock()
    request.request_id = "request-a"
    request.status = RequestStatus.RUNNING
    request.num_in_flight_tokens = 1
    request.last_sched_seq = 3
    request.spec_token_ids = []
    request.num_preemptions = 0
    request.drop_stale_output = False
    request.num_stale_output_tokens = 0

    scheduler._preempt_request(request, 0.0)

    scheduler._free_request_blocks.assert_called_once_with(request)


def test_weight_version_selects_distinct_artifact_keys(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    block_hash = b"a" * 32
    old_key = _block_key(block_hash)

    connector.set_weight_version("step-42")

    new_key = _block_key(block_hash, "step-42")
    assert new_key != old_key
    connector.shutdown()


def test_weight_version_prevents_cross_version_prefix_hit(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=100,
        ttl_seconds=60,
    )
    block_hash = b"a" * 32
    old_key = _block_key(block_hash)
    store.put([ArtifactObject(old_key, b"payload")])
    connector.ensure_prefix_ready(
        block_hashes=[block_hash],
        cached_token_end=4,
        hash_block_size=4,
        weight_version="default",
    )

    connector.set_weight_version("step-42")

    with pytest.raises(RuntimeError, match="weight_version=step-42"):
        connector.ensure_prefix_ready(
            block_hashes=[block_hash],
            cached_token_end=4,
            hash_block_size=4,
            weight_version="step-42",
        )
    connector.shutdown()
    store.close()


def test_inflight_request_keeps_admission_weight_version(tmp_path):
    connector = _make_scheduler_connector(tmp_path)
    connector.request_started(
        request_id="old-request",
        block_hashes=[b"a" * 32],
        cached_token_end=0,
        hash_block_size=4,
    )
    connector.request_progress(
        request_id="old-request",
        block_hashes=[b"a" * 32],
        accepted_token_end=4,
        hash_block_size=4,
    )

    connector.set_weight_version("step-42")
    metadata = connector.build_connector_meta()

    assert metadata is not None
    assert metadata.commits[0].weight_version == "default"
    connector.shutdown()


def test_engine_core_propagates_weight_version_to_artifact_connector():
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = Mock()

    engine_core.set_weight_version("step-42")

    assert engine_core.get_weight_version() == "step-42"
    engine_core.scheduler.set_weight_version.assert_called_once_with("step-42")


def test_scheduler_releases_inline_terminal_output_after_ack(tmp_path):
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = _make_scheduler_connector(tmp_path)
    scheduler.artifact_connector.request_started(
        request_id="request-a",
        block_hashes=[b"a" * 32, b"b" * 32],
        cached_token_end=0,
        hash_block_size=4,
    )
    scheduler.artifact_connector.request_finished(
        request_id="request-a",
        block_hashes=[b"a" * 32, b"b" * 32],
        token_end=6,
        hash_block_size=4,
    )
    metadata = scheduler.artifact_connector.build_connector_meta()
    assert metadata is not None
    attempt_id = metadata.finalizes[0].request_attempt_id
    terminal_output = EngineCoreOutput(
        request_id="request-a",
        new_token_ids=[7],
        finish_reason=FinishReason.STOP,
    )
    scheduler._pending_artifact_outputs = {"request-a": (3, terminal_output)}
    scheduler.finished_req_ids_dict = defaultdict(set)
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    routed_experts = np.zeros((6, 3, 2), dtype=np.uint8)
    scheduler.artifact_connector.materialize = Mock(return_value=routed_experts)

    scheduler._release_artifact_outputs(
        ArtifactConnectorOutput(
            [
                ArtifactFinalizeResult(
                    request_id="request-a",
                    request_attempt_id=attempt_id,
                    keys=["block", "tail"],
                )
            ]
        ),
        outputs,
    )

    assert terminal_output.routed_experts is routed_experts
    assert terminal_output.artifact_finalized
    assert scheduler.finished_req_ids_dict[3] == {"request-a"}
    scheduler.artifact_connector.shutdown()


def test_scheduler_fails_on_materialization_error(tmp_path):
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = _make_scheduler_connector(tmp_path)
    scheduler.artifact_connector.request_started(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        cached_token_end=0,
        hash_block_size=4,
    )
    scheduler.artifact_connector.request_finished(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        token_end=4,
        hash_block_size=4,
    )
    metadata = scheduler.artifact_connector.build_connector_meta()
    assert metadata is not None
    terminal_output = EngineCoreOutput(
        request_id="request-a",
        new_token_ids=[7],
        finish_reason=FinishReason.STOP,
    )
    scheduler._pending_artifact_outputs = {"request-a": (3, terminal_output)}
    scheduler.finished_req_ids_dict = defaultdict(set)
    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

    with pytest.raises(ArtifactNotFoundError):
        scheduler._release_artifact_outputs(
            ArtifactConnectorOutput(
                [
                    ArtifactFinalizeResult(
                        request_id="request-a",
                        request_attempt_id=metadata.finalizes[0].request_attempt_id,
                        keys=["missing"],
                    )
                ]
            ),
            outputs,
        )

    assert not terminal_output.artifact_finalized
    assert outputs[3] == []
    scheduler.artifact_connector.shutdown()
