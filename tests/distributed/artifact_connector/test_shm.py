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
    ArtifactArray,
    ArtifactCapacityError,
    ArtifactCommitRequest,
    ArtifactCommitResult,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactCorruptionError,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
    ArtifactRequestCore,
    ArtifactSchedulerConnector,
    ArtifactWorkerConnector,
    LocalSharedMemoryArtifactReader,
    LocalSharedMemoryArtifactStore,
    materialize_routed_experts,
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


def _read_in_child(root: str, store_id: str, sample_id: str, result_queue) -> None:
    reader = LocalSharedMemoryArtifactReader(root, store_id)
    result_queue.put(materialize_routed_experts(reader, sample_id).tolist())


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
        inline_value=True,
    )
    return connector, logical, block_ids, slots


def _request(
    request_id: str,
    sample_id: str,
    block_ids: list[int],
    *,
    token_start: int = 0,
    token_end: int = 10,
    policy_epoch: int = 0,
) -> ArtifactFinalizeRequest:
    return ArtifactFinalizeRequest(
        request_id=request_id,
        artifact_sample_id=sample_id,
        block_ids=block_ids,
        block_hashes=[bytes([index]) * 32 for index in range(3)],
        token_start=token_start,
        token_end=token_end,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=policy_epoch,
    )


def _commit_request(request: ArtifactFinalizeRequest) -> ArtifactCommitRequest | None:
    block_start = (
        (request.token_start + request.hash_block_size - 1)
        // request.hash_block_size
        * request.hash_block_size
    )
    block_end = request.token_end // request.hash_block_size * request.hash_block_size
    if block_end <= block_start:
        return None
    return ArtifactCommitRequest(
        operation_id=f"commit-{request.request_id}",
        request_id=request.request_id,
        artifact_sample_id=request.artifact_sample_id,
        block_ids=request.block_ids,
        block_hashes=request.block_hashes,
        block_start=block_start,
        block_end=block_end,
        physical_block_size=request.physical_block_size,
        hash_block_size=request.hash_block_size,
        policy_epoch=request.policy_epoch,
    )


def _finalize_request(
    connector: ArtifactWorkerConnector, request: ArtifactFinalizeRequest
) -> ArtifactConnectorOutput | None:
    commit = _commit_request(request)
    return connector.finalize(
        ArtifactConnectorMetadata(
            requests=[request], commits=[] if commit is None else [commit]
        )
    )


def test_worker_finalizes_kv_keyed_blocks_and_request_tail(tmp_path):
    connector, logical, block_ids, slots = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids)

    commit = _commit_request(request)
    assert commit is not None
    commit_output = connector.finalize(ArtifactConnectorMetadata(commits=[commit]))

    assert commit_output is not None
    assert commit_output.commit_results[0].error is None
    assert len(list(connector.store.blocks_dir.glob("*.bin"))) == 2
    assert not list(connector.store.manifests_dir.glob("*.json"))
    assert not list(connector.store.tails_dir.glob("*.bin"))

    output = connector.finalize(ArtifactConnectorMetadata(requests=[request]))

    assert output is not None
    assert output.results[0].error is None
    np.testing.assert_array_equal(output.results[0].routed_experts, logical[:10])
    assert output.results[0].delivery == "inline"
    reader = LocalSharedMemoryArtifactReader(str(tmp_path), connector.store.store_id)
    np.testing.assert_array_equal(
        materialize_routed_experts(reader, request.artifact_sample_id),
        logical[:10],
    )
    manifest = reader.read_manifest(request.artifact_sample_id)
    segments = manifest["fields"]["routed_experts"]["segments"]
    assert [segment["kind"] for segment in segments] == [
        "block",
        "block",
        "tail",
    ]
    assert [segment.get("kv_block_hash") for segment in segments[:2]] == [
        request.block_hashes[0].hex(),
        request.block_hashes[1].hex(),
    ]

    slots.fill(255)
    np.testing.assert_array_equal(
        materialize_routed_experts(reader, request.artifact_sample_id),
        logical[:10],
    )
    connector.close()


def test_worker_terminal_finalize_requires_prior_full_block_commit(tmp_path):
    connector, _, block_ids, _ = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids)

    output = connector.finalize(ArtifactConnectorMetadata(requests=[request]))

    assert output is not None
    assert output.results[0].delivery is None
    assert "missing a reusable full block" in (output.results[0].error or "")
    connector.close()


def test_worker_batches_completed_blocks_from_multiple_requests(tmp_path):
    connector, _, block_ids, _ = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids)
    second = _request("request-b", "b" * 32, block_ids)
    first_commit = _commit_request(first)
    second_commit = _commit_request(second)
    assert first_commit is not None and second_commit is not None
    connector.store.put_blocks = Mock(wraps=connector.store.put_blocks)

    output = connector.finalize(
        ArtifactConnectorMetadata(commits=[first_commit, second_commit])
    )

    assert output is not None
    assert all(result.error is None for result in output.commit_results)
    connector.store.put_blocks.assert_called_once()
    assert len(connector.store.put_blocks.call_args.args[0]) == 4
    connector.close()


def test_scheduler_progress_drives_consecutive_incremental_worker_commits(tmp_path):
    scheduler_connector = ArtifactSchedulerConnector()
    worker_connector, logical, block_ids, _ = _make_worker_connector(tmp_path)
    block_hashes = [bytes([index]) * 32 for index in range(3)]

    sample_id = scheduler_connector.request_progress(
        request_id="request-a",
        block_ids=block_ids,
        block_hashes=block_hashes,
        token_start=0,
        accepted_token_end=7,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    assert (
        scheduler_connector.request_progress(
            request_id="request-a",
            block_ids=block_ids,
            block_hashes=block_hashes,
            token_start=0,
            accepted_token_end=10,
            physical_block_size=4,
            hash_block_size=4,
            policy_epoch=0,
        )
        == sample_id
    )
    metadata = scheduler_connector.build_connector_metadata()
    assert metadata is not None
    assert [(commit.block_start, commit.block_end) for commit in metadata.commits] == [
        (0, 4),
        (4, 8),
    ]

    commit_output = worker_connector.finalize(metadata)
    assert commit_output is not None
    assert all(result.error is None for result in commit_output.commit_results)
    scheduler_connector.acknowledge(commit_output)
    assert not scheduler_connector.has_unacked_commits("request-a")

    scheduler_connector.request_finished(
        request_id="request-a",
        block_ids=block_ids,
        block_hashes=block_hashes,
        token_start=0,
        token_end=10,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    final_metadata = scheduler_connector.build_connector_metadata()
    assert final_metadata is not None
    final_output = worker_connector.finalize(final_metadata)
    assert final_output is not None
    assert final_output.results[0].error is None
    np.testing.assert_array_equal(final_output.results[0].routed_experts, logical[:10])
    worker_connector.close()


def test_ready_prefix_binds_existing_blocks_without_republishing(tmp_path):
    scheduler_connector = ArtifactSchedulerConnector()
    worker_connector, logical, block_ids, _ = _make_worker_connector(tmp_path)
    block_hashes = [bytes([index]) * 32 for index in range(3)]

    scheduler_connector.request_progress(
        request_id="request-a",
        block_ids=block_ids,
        block_hashes=block_hashes,
        token_start=0,
        accepted_token_end=8,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    first_metadata = scheduler_connector.build_connector_metadata()
    assert first_metadata is not None
    first_output = worker_connector.finalize(first_metadata)
    assert first_output is not None
    scheduler_connector.acknowledge(first_output)

    assert (
        scheduler_connector.max_ready_prefix_tokens(
            block_hashes=block_hashes,
            token_start=0,
            max_tokens=10,
            hash_block_size=4,
            policy_epoch=0,
        )
        == 8
    )
    scheduler_connector.request_started(
        request_id="request-b",
        block_hashes=block_hashes,
        token_start=0,
        cached_token_end=8,
        hash_block_size=4,
        policy_epoch=0,
    )
    scheduler_connector.request_finished(
        request_id="request-b",
        block_ids=block_ids,
        block_hashes=block_hashes,
        token_start=0,
        token_end=10,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    second_metadata = scheduler_connector.build_connector_metadata()
    assert second_metadata is not None
    assert not second_metadata.commits
    assert len(second_metadata.requests[0].cached_blocks) == 2
    worker_connector.store.put_blocks = Mock(wraps=worker_connector.store.put_blocks)

    second_output = worker_connector.finalize(second_metadata)

    assert second_output is not None
    assert second_output.results[0].error is None
    np.testing.assert_array_equal(second_output.results[0].routed_experts, logical[:10])
    worker_connector.store.put_blocks.assert_not_called()
    worker_connector.close()


def test_worker_excludes_rejected_speculative_rows_after_accepted_end(tmp_path):
    connector, logical, block_ids, slots = _make_worker_connector(tmp_path)
    accepted_end = 7
    rejected_positions = np.arange(accepted_end, 10)
    block_ids_array = np.asarray(block_ids)
    rejected_slots = (
        block_ids_array[rejected_positions // 4] * 4 + rejected_positions % 4
    )
    slots[rejected_slots] = 255
    request = _request(
        "request-a",
        "a" * 32,
        block_ids,
        token_end=accepted_end,
    )

    output = _finalize_request(connector, request)

    assert output is not None and output.results[0].delivery == "inline"
    materialized = materialize_routed_experts(
        connector.store, request.artifact_sample_id
    )
    np.testing.assert_array_equal(materialized, logical[:accepted_end])
    assert materialized.shape[0] == accepted_end
    connector.close()


def test_prefix_blocks_deduplicate_across_samples(tmp_path):
    connector, logical, block_ids, _ = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids)
    second = _request("request-b", "b" * 32, block_ids)

    first_output = _finalize_request(connector, first)
    second_output = _finalize_request(connector, second)

    assert first_output is not None and second_output is not None
    assert first_output.results[0].delivery == "inline"
    assert second_output.results[0].delivery == "inline"
    reader = LocalSharedMemoryArtifactReader(str(tmp_path), connector.store.store_id)
    np.testing.assert_array_equal(
        materialize_routed_experts(reader, second.artifact_sample_id),
        logical[:10],
    )
    first_manifest = reader.read_manifest(first.artifact_sample_id)
    second_manifest = reader.read_manifest(second.artifact_sample_id)
    assert (
        first_manifest["fields"]["routed_experts"]["segments"][:2]
        == second_manifest["fields"]["routed_experts"]["segments"][:2]
    )
    assert len(list(connector.store.blocks_dir.glob("*.bin"))) == 2
    assert len(list(connector.store.tails_dir.glob("*.bin"))) == 2
    connector.close()


def test_full_blocks_use_canonical_value_across_execution_shapes(tmp_path):
    connector, logical, block_ids, slots = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids)
    first_output = _finalize_request(connector, first)
    assert first_output is not None and first_output.results[0].delivery == "inline"

    logical_positions = np.arange(10)
    physical_slots = (
        np.asarray(block_ids)[logical_positions // 4] * 4 + logical_positions % 4
    )
    slots[physical_slots] += 1
    second = _request("request-b", "b" * 32, block_ids)
    second_output = _finalize_request(connector, second)

    assert second_output is not None and second_output.results[0].delivery == "inline"
    expected = logical[:10].copy()
    expected[8:10] += 1
    np.testing.assert_array_equal(second_output.results[0].routed_experts, expected)
    np.testing.assert_array_equal(
        materialize_routed_experts(connector.store, second.artifact_sample_id),
        expected,
    )
    assert len(list(connector.store.blocks_dir.glob("*.bin"))) == 2
    assert len(list(connector.store.tails_dir.glob("*.bin"))) == 2
    connector.close()


def test_request_scoped_tail_rejects_different_value_for_same_id(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=60,
    )
    object_id = "a" * 64
    store.put_array("tail", object_id, np.zeros((2, 1), dtype=np.uint8), {})

    with pytest.raises(ArtifactCorruptionError, match="object id collision"):
        store.put_array("tail", object_id, np.ones((2, 1), dtype=np.uint8), {})
    store.close()


def test_weight_update_epoch_prevents_stale_block_reuse(tmp_path):
    connector, _, block_ids, slots = _make_worker_connector(tmp_path)
    first = _request("request-a", "a" * 32, block_ids, token_end=8)
    first_output = _finalize_request(connector, first)
    assert first_output is not None and first_output.results[0].delivery == "inline"
    first_manifest = connector.store.read_manifest(first.artifact_sample_id)

    connector.advance_policy_epoch()
    slots += 1
    second = _request(
        "request-b",
        "b" * 32,
        block_ids,
        token_end=8,
        policy_epoch=1,
    )
    second_output = _finalize_request(connector, second)

    assert second_output is not None and second_output.results[0].delivery == "inline"
    second_manifest = connector.store.read_manifest(second.artifact_sample_id)
    assert first_manifest["policy_epoch"] == 0
    assert second_manifest["policy_epoch"] == 1
    assert (
        first_manifest["fields"]["routed_experts"]["segments"]
        != second_manifest["fields"]["routed_experts"]["segments"]
    )
    assert len(list(connector.store.blocks_dir.glob("*.bin"))) == 4
    connector.close()


def test_worker_rejects_unsynchronized_policy_epoch(tmp_path):
    connector, _, block_ids, _ = _make_worker_connector(tmp_path)
    request = _request(
        "request-a",
        "a" * 32,
        block_ids,
        policy_epoch=1,
    )

    output = _finalize_request(connector, request)

    assert output is not None
    assert output.results[0].delivery is None
    assert "policy epoch mismatch" in (output.results[0].error or "")
    connector.close()


def test_unaligned_prompt_start_uses_request_local_head(tmp_path):
    connector, logical, block_ids, _ = _make_worker_connector(tmp_path)
    request = _request(
        "request-a",
        "a" * 32,
        block_ids,
        token_start=3,
        token_end=10,
    )

    output = _finalize_request(connector, request)

    assert output is not None and output.results[0].delivery == "inline"
    reader = LocalSharedMemoryArtifactReader(str(tmp_path), connector.store.store_id)
    np.testing.assert_array_equal(
        materialize_routed_experts(reader, request.artifact_sample_id),
        logical[3:10],
    )
    manifest = reader.read_manifest(request.artifact_sample_id)
    segments = manifest["fields"]["routed_experts"]["segments"]
    assert [segment["kind"] for segment in segments] == [
        "tail",
        "block",
        "tail",
    ]
    assert segments[1]["kv_block_hash"] == request.block_hashes[1].hex()
    connector.close()


def test_reader_materializes_from_another_process(tmp_path):
    connector, logical, block_ids, _ = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids)
    output = _finalize_request(connector, request)
    assert output is not None and output.results[0].delivery == "inline"

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_read_in_child,
        args=(
            str(tmp_path),
            connector.store.store_id,
            request.artifact_sample_id,
            result_queue,
        ),
    )
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 0
    np.testing.assert_array_equal(np.asarray(result_queue.get()), logical[:10])
    connector.close()


def test_capacity_exhaustion_is_returned_as_an_explicit_error(tmp_path):
    connector, _, block_ids, _ = _make_worker_connector(tmp_path, max_bytes=4096)
    request = _request("request-a", "a" * 32, block_ids)

    output = _finalize_request(connector, request)

    assert output is not None
    assert output.results[0].delivery is None
    assert "ArtifactCapacityError" in (output.results[0].error or "")
    connector.close()


def test_corrupt_payload_is_rejected(tmp_path):
    connector, _, block_ids, _ = _make_worker_connector(tmp_path)
    request = _request("request-a", "a" * 32, block_ids)
    output = _finalize_request(connector, request)
    assert output is not None and output.results[0].delivery == "inline"
    manifest = connector.store.read_manifest(request.artifact_sample_id)
    block_path = connector.store._path(
        "block",
        manifest["fields"]["routed_experts"]["segments"][0]["object_id"],
    )
    with block_path.open("r+b") as artifact_file:
        artifact_file.seek(-1, os.SEEK_END)
        artifact_file.write(b"\xff")

    with pytest.raises(ArtifactCorruptionError, match="checksum"):
        materialize_routed_experts(connector.store, request.artifact_sample_id)
    connector.close()


def test_store_capacity_error_is_not_an_overwrite(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=4096,
        ttl_seconds=60,
    )
    with pytest.raises(ArtifactCapacityError, match="capacity exceeded"):
        store.put_array(
            "block",
            "a" * 64,
            np.zeros((4, 3, 2), dtype=np.uint8),
            {"kind": "vllm.artifact_block"},
        )
    assert not list(store.blocks_dir.glob("*.bin"))
    store.close()


def test_store_reports_filesystem_exhaustion_before_mmap_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=60,
    )
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096),
    )

    with pytest.raises(ArtifactCapacityError, match="filesystem is full"):
        store.put_array(
            "block",
            "a" * 64,
            np.zeros((4, 3, 2), dtype=np.uint8),
            {},
        )

    assert not list(store.blocks_dir.glob("*.bin"))
    store.close()


def test_new_store_collects_expired_inactive_store(tmp_path):
    stale = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "stale-instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )
    stale.put_array(
        "tail",
        "a" * 64,
        np.ones((1, 1), dtype=np.uint8),
        {},
    )
    stale_root = stale.root
    stale.close()
    old_time = time.time() - 5
    for directory, _, filenames in os.walk(stale_root, topdown=False):
        for filename in filenames:
            os.utime(os.path.join(directory, filename), (old_time, old_time))
        os.utime(directory, (old_time, old_time))

    current = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "current-instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )

    assert not stale_root.exists()
    current.close()


def test_new_store_does_not_collect_active_store(tmp_path):
    active = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "active-instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )
    old_time = time.time() - 5
    for directory, _, filenames in os.walk(active.root, topdown=False):
        for filename in filenames:
            os.utime(os.path.join(directory, filename), (old_time, old_time))
        os.utime(directory, (old_time, old_time))

    current = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "current-instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )

    assert active.root.exists()
    current.close()
    active.close()


def test_stale_store_uses_its_own_ttl(tmp_path):
    retained = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "retained-instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=60,
    )
    retained_root = retained.root
    retained.close()
    old_time = time.time() - 5
    for directory, _, filenames in os.walk(retained_root, topdown=False):
        for filename in filenames:
            os.utime(os.path.join(directory, filename), (old_time, old_time))
        os.utime(directory, (old_time, old_time))

    collector = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "collector-instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )

    assert retained_root.exists()
    collector.close()


def test_store_does_not_rescan_all_objects_on_every_put(tmp_path, monkeypatch):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=60,
    )
    usage_bytes = Mock(wraps=store._usage_bytes)
    monkeypatch.setattr(store, "_usage_bytes", usage_bytes)

    store.put_array(
        "tail",
        "a" * 64,
        np.ones((1, 1), dtype=np.uint8),
        {},
    )
    store.put_manifest("b" * 32, {"segments": []})

    usage_bytes.assert_not_called()
    store.close()


def test_gc_removes_expired_partial_files(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )
    partial = store.tails_dir / ".orphan.bin.deadbeef.partial"
    partial.touch(mode=0o600)
    old_time = time.time() - 5
    os.utime(partial, (old_time, old_time))

    store.gc()

    assert not partial.exists()
    store.close()


def test_gc_retains_incremental_blocks_until_request_releases_them(tmp_path):
    store = LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=1 << 20,
        ttl_seconds=1,
    )
    object_id = "a" * 64
    store.put_blocks(
        [
            ArtifactArray(
                object_id=object_id,
                array=np.ones((4, 1), dtype=np.uint8),
                metadata={},
            )
        ]
    )
    path = store._path("block", object_id)
    old_time = time.time() - 5
    os.utime(path, (old_time, old_time))

    store.gc()
    assert path.exists()

    store.release_blocks([object_id])
    store.gc()
    assert not path.exists()
    store.close()


def test_scheduler_connector_tracks_metadata_and_acknowledgements():
    connector = ArtifactSchedulerConnector()
    sample_id = connector.request_finished(
        request_id="request-a",
        block_ids=[1, 2],
        block_hashes=[b"a" * 32, b"b" * 32],
        token_start=0,
        token_end=6,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    assert connector.has_pending_work()

    metadata = connector.build_connector_metadata()

    assert metadata is not None
    assert metadata.requests[0].artifact_sample_id == sample_id
    result = ArtifactFinalizeResult(
        request_id="request-a",
        artifact_sample_id=sample_id,
        delivery="sample_id",
    )
    acknowledged = connector.acknowledge(ArtifactConnectorOutput([result]))

    assert acknowledged.results == [result]
    assert not connector.has_pending_work()


def test_scheduler_finalizes_only_the_accepted_speculative_range():
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
    # num_computed_tokens includes an optimistic async/speculative future,
    # while num_tokens contains only outputs committed to the request.
    request.num_computed_tokens = 12
    request.num_tokens = 10
    request.num_prompt_tokens = 6
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
    scheduler.kv_cache_manager.free.assert_not_called()
    assert not freed_blocks

    scheduler.processed_step_seq = 3
    scheduler._release_artifact_preempted_blocks(request.request_id)

    assert freed_blocks == [retained_block]
    assert request.request_id not in scheduler._artifact_preempted_blocks


def test_scheduler_policy_transition_requires_idle_and_resets_all_kv_caches():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector()
    scheduler.artifact_policy_epoch = 2
    scheduler.artifact_policy_update_active = True
    scheduler.has_requests = Mock(return_value=False)
    scheduler.reset_prefix_cache = Mock(return_value=True)

    scheduler.advance_artifact_policy_epoch()

    assert scheduler.artifact_policy_epoch == 3
    assert not scheduler.artifact_policy_update_active
    scheduler.reset_prefix_cache.assert_called_once_with(reset_connector=True)


def test_scheduler_policy_transition_rejects_active_requests():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector()
    scheduler.artifact_policy_epoch = 2
    scheduler.artifact_policy_update_active = True
    scheduler.has_requests = Mock(return_value=True)
    scheduler.reset_prefix_cache = Mock(return_value=True)

    with pytest.raises(RuntimeError, match="requires all requests to be drained"):
        scheduler.advance_artifact_policy_epoch()

    assert scheduler.artifact_policy_epoch == 2
    scheduler.reset_prefix_cache.assert_not_called()


def test_scheduler_policy_update_fences_new_requests_before_weight_change():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector()
    scheduler.artifact_policy_update_active = False
    scheduler.has_requests = Mock(return_value=False)

    scheduler.begin_artifact_policy_update()

    assert scheduler.artifact_policy_update_active
    with pytest.raises(RuntimeError, match="cannot admit requests"):
        scheduler.add_request(SimpleNamespace(request_id="request-a"))


@pytest.mark.parametrize("method", ["finish_weight_update", "reload_weights"])
def test_engine_core_synchronizes_artifact_policy_after_weight_change(method):
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = Mock()
    engine_core.model_executor.collective_rpc.return_value = [None]
    engine_core.scheduler = Mock()
    engine_core.scheduler.artifact_policy_update_active = False

    result = engine_core.collective_rpc(method)

    assert result == [None]
    engine_core.scheduler.advance_artifact_policy_epoch.assert_called_once_with()
    if method == "reload_weights":
        engine_core.scheduler.begin_artifact_policy_update.assert_called_once_with()
    else:
        engine_core.scheduler.begin_artifact_policy_update.assert_not_called()


@pytest.mark.parametrize("method", ["start_weight_update", "start_draft_weight_update"])
def test_engine_core_fences_requests_before_weight_update_starts(method):
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = Mock()
    engine_core.model_executor.collective_rpc.return_value = [None]
    engine_core.scheduler = Mock()

    engine_core.collective_rpc(method)

    engine_core.scheduler.begin_artifact_policy_update.assert_called_once_with()
    engine_core.scheduler.advance_artifact_policy_epoch.assert_not_called()


def test_failed_weight_update_keeps_artifact_admission_fenced():
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = Mock()
    engine_core.model_executor.collective_rpc.side_effect = RuntimeError("failed")
    engine_core.scheduler = Mock()
    engine_core.scheduler.artifact_policy_update_active = False

    with pytest.raises(RuntimeError, match="failed"):
        engine_core.collective_rpc("start_weight_update")

    engine_core.scheduler.begin_artifact_policy_update.assert_called_once_with()
    engine_core.scheduler.advance_artifact_policy_epoch.assert_not_called()


def test_successful_reload_recovers_a_failed_artifact_policy_update():
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = Mock()
    engine_core.model_executor.collective_rpc.return_value = [None]
    engine_core.scheduler = Mock()
    engine_core.scheduler.artifact_policy_update_active = True

    engine_core.collective_rpc("reload_weights")

    engine_core.scheduler.begin_artifact_policy_update.assert_not_called()
    engine_core.scheduler.advance_artifact_policy_epoch.assert_called_once_with()


def test_scheduler_releases_terminal_output_only_after_artifact_ack():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector()
    sample_id = scheduler.artifact_connector.request_finished(
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
                    artifact_sample_id=sample_id,
                    delivery="inline",
                    routed_experts=routed_experts,
                )
            ]
        ),
        outputs,
    )

    assert outputs[3] == [terminal_output]
    assert terminal_output.artifact_sample_id is None
    assert terminal_output.routed_experts is routed_experts
    assert scheduler.finished_req_ids_dict[3] == {"request-a"}
    scheduler._free_blocks.assert_called_once_with(request)


def test_scheduler_defers_commit_ack_block_release_until_outputs_are_processed():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector()
    sample_id = scheduler.artifact_connector.request_progress(
        request_id="request-a",
        block_ids=[1],
        block_hashes=[b"a" * 32],
        token_start=0,
        accepted_token_end=4,
        physical_block_size=4,
        hash_block_size=4,
        policy_epoch=0,
    )
    metadata = scheduler.artifact_connector.build_connector_metadata()
    assert metadata is not None
    operation_id = metadata.commits[0].operation_id
    scheduler._release_artifact_preempted_blocks = Mock()

    candidates = scheduler._release_artifact_outputs(
        ArtifactConnectorOutput(
            commit_results=[
                ArtifactCommitResult(
                    operation_id=operation_id,
                    request_id="request-a",
                    artifact_sample_id=sample_id,
                    block_end=4,
                )
            ]
        ),
        defaultdict(list),
    )

    assert candidates == {"request-a"}
    scheduler._release_artifact_preempted_blocks.assert_not_called()


def test_scheduler_fails_request_closed_when_artifact_finalize_fails():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector()
    sample_id = scheduler.artifact_connector.request_finished(
        request_id="request-a",
        block_ids=[1],
        block_hashes=[b"a" * 32],
        token_start=0,
        token_end=4,
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

    scheduler._release_artifact_outputs(
        ArtifactConnectorOutput(
            [
                ArtifactFinalizeResult(
                    request_id="request-a",
                    artifact_sample_id=sample_id,
                    error="ArtifactCapacityError: capacity exceeded",
                )
            ]
        ),
        outputs,
    )

    assert outputs[3] == [terminal_output]
    assert terminal_output.finish_reason == FinishReason.ERROR
    assert terminal_output.stop_reason == "ArtifactCapacityError: capacity exceeded"
    assert terminal_output.artifact_sample_id is None
    scheduler._free_blocks.assert_called_once_with(request)


def test_public_output_exposes_only_the_artifact_sample_identity():
    output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        artifact_sample_id="a" * 32,
    )

    assert output.artifact_sample_id == "a" * 32
