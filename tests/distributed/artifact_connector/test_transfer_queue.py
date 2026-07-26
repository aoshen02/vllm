# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
import uuid
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.distributed.artifact_connector import (
    ArtifactArray,
    ArtifactConnectorOutput,
    ArtifactFinalizeResult,
    ArtifactSchedulerConnector,
    TransferQueueArtifactStore,
    materialize_routed_experts,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, FinishReason

pytestmark = pytest.mark.cpu_test


def test_store_only_connects_to_existing_transfer_queue(
    monkeypatch: pytest.MonkeyPatch,
):
    ray = types.ModuleType("ray")
    ray_is_initialized = Mock(return_value=False)
    ray_init = Mock()
    ray_shutdown = Mock()
    ray_module = cast(Any, ray)
    ray_module.is_initialized = ray_is_initialized
    ray_module.init = ray_init
    ray_module.shutdown = ray_shutdown
    tq = types.ModuleType("transfer_queue")
    tq_connect = Mock()
    tq_disconnect = Mock()
    tq_init = Mock()
    tq_module = cast(Any, tq)
    tq_module.connect = tq_connect
    tq_module.disconnect = tq_disconnect
    tq_module.init = tq_init
    tq_module.TransferQueueObjectNotFoundError = type(
        "TransferQueueObjectNotFoundError", (ValueError,), {}
    )
    tq_module.TransferQueueObjectNotReadyError = type(
        "TransferQueueObjectNotReadyError", (ValueError,), {}
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "transfer_queue", tq)

    store = TransferQueueArtifactStore(
        ray_address="ray://controller:10001",
        store_id="test-transfer-queue",
        data_partition="artifact-data",
        request_partition="artifact-requests",
        connect_timeout_seconds=7,
    )

    ray_init.assert_called_once_with(
        address="ray://controller:10001",
        namespace="transfer_queue",
        logging_level="ERROR",
    )
    tq_connect.assert_called_once_with(timeout=7)
    tq_init.assert_not_called()
    store.close()
    store.close()
    tq_disconnect.assert_called_once_with()
    ray_shutdown.assert_called_once_with()


def test_transfer_queue_terminal_ack_exposes_only_sample_id():
    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = ArtifactSchedulerConnector("transfer_queue")
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
            results=[
                ArtifactFinalizeResult(
                    request_id="request-a",
                    artifact_sample_id=sample_id,
                    delivery="sample_id",
                    manifest_sha256="b" * 64,
                )
            ]
        ),
        outputs,
    )

    assert outputs[3] == [terminal_output]
    assert terminal_output.artifact_sample_id == sample_id
    assert terminal_output.routed_experts is None


@pytest.fixture(scope="module")
def transfer_queue_simple_storage():
    ray = pytest.importorskip("ray")
    tq = pytest.importorskip("transfer_queue")
    omegaconf = pytest.importorskip("omegaconf")
    if not ray.is_initialized():
        ray.init(namespace="transfer_queue", include_dashboard=False, num_cpus=4)
    config = omegaconf.OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": 64,
                    "num_data_storage_units": 2,
                },
            },
        }
    )
    tq.init(config)
    yield
    tq.close()
    ray.shutdown()


def test_simple_storage_round_trip_uses_common_manifest(
    transfer_queue_simple_storage,
):
    suffix = uuid.uuid4().hex
    sample_id = uuid.uuid4().hex
    object_ids = [uuid.uuid4().hex for _ in range(3)]
    blocks = [
        np.arange(24, dtype=np.uint16).reshape(4, 3, 2),
        np.arange(24, 48, dtype=np.uint16).reshape(4, 3, 2),
    ]
    tail = np.arange(48, 60, dtype=np.uint16).reshape(2, 3, 2)
    store = TransferQueueArtifactStore(
        ray_address="auto",
        store_id="test-transfer-queue",
        data_partition=f"artifact-data-{suffix}",
        request_partition=f"artifact-requests-{suffix}",
    )
    store.put_blocks(
        [
            ArtifactArray(
                object_id=object_id,
                array=array,
                metadata={"block_index": index},
            )
            for index, (object_id, array) in enumerate(
                zip(object_ids[:2], blocks, strict=True)
            )
        ]
    )
    store.put_array("tail", object_ids[2], tail, {"valid_len": 2})
    segments = [
        {
            "kind": "block",
            "object_id": object_ids[0],
            "output_start": 0,
            "source_token_start": 0,
            "valid_len": 4,
        },
        {
            "kind": "block",
            "object_id": object_ids[1],
            "output_start": 4,
            "source_token_start": 4,
            "valid_len": 4,
        },
        {
            "kind": "tail",
            "object_id": object_ids[2],
            "output_start": 8,
            "source_token_start": 8,
            "valid_len": 2,
        },
    ]
    manifest = {
        "schema_version": 2,
        "store_id": store.store_id,
        "artifact_sample_id": sample_id,
        "namespace": "test",
        "policy_epoch": 0,
        "request_id": "request-a",
        "terminal_boundary": {
            "coordinate": "executed_token",
            "start": 0,
            "end": 10,
        },
        "fields": {
            "routed_experts": {
                "reuse_policy": "prefix_block",
                "logical_coordinate": "executed_token",
                "field_profile_id": "test-profile",
                "dtype": np.dtype(np.uint16).str,
                "shape": [10, 3, 2],
                "source_token_start": 0,
                "source_token_end": 10,
                "segments": segments,
            }
        },
    }

    manifest_sha256 = store.put_manifest(sample_id, manifest)

    assert store.read_manifest(sample_id)["manifest_sha256"] == manifest_sha256
    np.testing.assert_array_equal(
        materialize_routed_experts(store, sample_id),
        np.concatenate([*blocks, tail]),
    )
    store.close()
