# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vllm.distributed.artifact_connector import (
    ArtifactCommitRequest,
    ArtifactFinalizeRequest,
    ArtifactObject,
    ArtifactRequestCore,
    ArtifactSchedulerConnector,
    ArtifactWorkerConnector,
    MooncakeArtifactPublisher,
    MooncakeArtifactReader,
    MooncakeArtifactStore,
    materialize_routed_experts,
)
from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.store import (
    ArtifactNotFoundError,
    ArtifactStoreError,
)
from vllm.outputs import CompletionOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, FinishReason

pytestmark = pytest.mark.cpu_test


class FakeMooncakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.registered: tuple[int, int] | None = None
        self.unregistered: int | None = None
        self.put_batches: list[list[str]] = []
        self.get_batches: list[tuple[list[str], list[list[int]]]] = []
        self.exists_override: Any = None
        self.put_override: Any = None
        self.get_override: Any = None
        self.closed = False

    def register_buffer(self, address, size):
        self.registered = (address, size)
        return 0

    def unregister_buffer(self, address):
        self.unregistered = address
        return 0

    def batch_is_exist(self, keys):
        if self.exists_override is not None:
            return self.exists_override(keys)
        return [int(key in self.objects) for key in keys]

    def get_size(self, key):
        value = self.objects.get(key)
        return len(value) if value is not None else -1

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
        self.put_batches.append(list(keys))
        if self.put_override is not None:
            return self.put_override(keys, addresses, sizes)
        for key, address, size in zip(keys, addresses, sizes, strict=True):
            self.objects[key] = ctypes.string_at(address[0], size[0])
        return [size[0] for size in sizes]

    def batch_get_into_multi_buffers(self, keys, addresses, capacities):
        self.get_batches.append((list(keys), capacities))
        if self.get_override is not None:
            return self.get_override(keys, addresses, capacities)
        results = []
        for key, address, capacity in zip(keys, addresses, capacities, strict=True):
            value = self.objects.get(key)
            if value is None:
                results.append(-1)
                continue
            assert len(value) == capacity[0]
            ctypes.memmove(address[0], value, len(value))
            results.append(len(value))
        return results

    def close(self):
        self.closed = True


class FakeRoutingWriter:
    dtype = np.dtype(np.uint8)
    shape_per_token = (3, 2)

    def __init__(self) -> None:
        self.value = np.arange(10 * 3 * 2, dtype=np.uint8).reshape(10, 3, 2)

    def read_token_range(
        self,
        block_ids,
        *,
        token_start,
        token_end,
        block_size,
    ):
        return self.value[token_start:token_end].copy()


def _make_store(fake, *, staging_bytes=16, object_bytes=8):
    return MooncakeArtifactStore(
        "deployment",
        staging_buffer_bytes=staging_bytes,
        max_object_bytes=object_bytes,
        store=fake,
        replicate_config=object(),
    )


def _make_config(tmp_path, *, backend="mooncake"):
    return SimpleNamespace(
        instance_id="instance",
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        artifact_config=SimpleNamespace(
            backend=backend,
            shm_dir=str(tmp_path),
            max_shm_bytes=1 << 20,
            shm_ttl_seconds=60,
            mooncake_store_id="deployment",
            mooncake_staging_buffer_bytes=8192,
        ),
        cache_config=SimpleNamespace(prefix_match_unit=4, block_size=16),
        model_config=SimpleNamespace(
            model="model",
            revision="revision",
            tokenizer_revision="tokenizer-revision",
        ),
    )


def test_mooncake_reader_does_not_register_a_staging_buffer():
    fake = FakeMooncakeStore()
    reader = MooncakeArtifactReader("deployment", store=fake)

    assert reader.exists(["missing"]) == [False]
    assert fake.registered is None
    reader.close()
    assert fake.closed


def test_mooncake_store_rejects_unsafe_store_id():
    with pytest.raises(ValueError, match="store id"):
        MooncakeArtifactStore(
            "../deployment",
            staging_buffer_bytes=16,
            max_object_bytes=8,
            store=FakeMooncakeStore(),
        )


def test_mooncake_store_batches_exact_size_registered_buffer_io():
    fake = FakeMooncakeStore()
    store = _make_store(fake)
    results = store.put(
        [
            ArtifactObject("a", b"aaaaaa"),
            ArtifactObject("b", b"bbbbbb"),
            ArtifactObject("c", b"cccccc"),
        ]
    )

    assert all(result.error is None for result in results)
    assert fake.put_batches == [["a", "b"], ["c"]]
    assert store.exists(["a", "missing", "c"]) == [True, False, True]
    assert store.get(["c", "a", "b"]) == [b"cccccc", b"aaaaaa", b"bbbbbb"]
    assert fake.get_batches == [
        (["c", "a"], [[6], [6]]),
        (["b"], [[6]]),
    ]

    store.close()
    assert fake.registered is not None
    assert fake.unregistered == fake.registered[0]
    assert fake.closed


def test_mooncake_store_reports_per_object_failures():
    fake = FakeMooncakeStore()
    fake.put_override = lambda keys, addresses, sizes: [-7, sizes[1][0]]
    store = _make_store(fake)

    results = store.put(
        [
            ArtifactObject("failed", b"x" * 4),
            ArtifactObject("ready", b"y" * 4),
            ArtifactObject("large", b"z" * 9),
        ]
    )

    assert "code=-7" in str(results[0].error)
    assert results[1].error is None
    assert "ArtifactCapacityError" in str(results[2].error)
    store.close()


def test_mooncake_store_fails_closed_on_lookup_and_get_errors():
    fake = FakeMooncakeStore()
    store = _make_store(fake)
    fake.exists_override = lambda keys: [-7] * len(keys)
    with pytest.raises(ArtifactStoreError, match="code=-7"):
        store.exists(["key"])

    fake.exists_override = None
    with pytest.raises(ArtifactNotFoundError, match="does not exist"):
        store.get(["key"])

    fake.objects["key"] = b"value"
    fake.get_override = lambda keys, addresses, capacities: [-9]
    with pytest.raises(ArtifactStoreError, match="code=-9"):
        store.get(["key"])
    store.close()


def test_mooncake_publisher_returns_after_put_start():
    fake = FakeMooncakeStore()
    started = threading.Event()
    release = threading.Event()

    def blocking_put(keys, addresses, sizes):
        started.set()
        assert release.wait(10)
        return [size[0] for size in sizes]

    fake.put_override = blocking_put
    store = _make_store(fake)
    publisher = MooncakeArtifactPublisher(store)

    results = publisher.put([ArtifactObject("key", b"value")])

    assert results[0].error is None
    assert started.wait(10)
    assert publisher._thread.is_alive()
    release.set()
    publisher.close()
    assert fake.put_batches == [["key"]]


def test_mooncake_core_returns_ordered_keys_without_materializing():
    fake = FakeMooncakeStore()
    store = _make_store(fake, staging_bytes=32768, object_bytes=8192)
    writer = FakeRoutingWriter()
    buffer = RoutedExpertsArtifactBuffer(writer.dtype, writer.shape_per_token)
    buffer.capture("request", 0, writer.value)
    core = ArtifactRequestCore(
        store,
        buffer,
        namespace="model",
        materialize=False,
    )
    hashes = [b"a" * 32, b"b" * 32, b"c" * 32]
    commit = ArtifactCommitRequest(
        operation_id="commit",
        request_id="request",
        request_attempt_id="attempt",
        block_hashes=hashes,
        block_start=0,
        block_end=8,
        hash_block_size=4,
    )
    prepared = core.prepare_commit(commit)
    assert core.publish_commits([prepared]) == {"commit": None}
    finalized = core.finalize(
        ArtifactFinalizeRequest(
            request_id="request",
            request_attempt_id="attempt",
            block_hashes=hashes,
            token_end=10,
            hash_block_size=4,
        )
    )

    assert finalized.value is None
    assert len(finalized.keys) == 3
    np.testing.assert_array_equal(
        materialize_routed_experts(store, finalized.keys),
        writer.value,
    )
    core.close()


def test_scheduler_queries_mooncake_with_deterministic_keys(tmp_path):
    config = _make_config(tmp_path)
    reader = Mock()
    reader.exists.return_value = [True, True, False]
    with patch(
        "vllm.distributed.artifact_connector.connector.MooncakeArtifactReader",
        return_value=reader,
    ):
        connector = ArtifactSchedulerConnector(
            config,
            dtype=np.dtype(np.uint8),
            shape_per_token=(3, 2),
        )

    assert (
        connector.max_ready_prefix_tokens(
            block_hashes=[b"a", b"b", b"c"],
            max_tokens=12,
            hash_block_size=4,
        )
        == 8
    )
    queried_keys = reader.exists.call_args.args[0]
    assert len(queried_keys) == 3
    assert all(key.startswith("vllm-artifact/deployment/") for key in queried_keys)
    connector.close()


def test_worker_constructs_mooncake_backend_without_lookup_server(tmp_path):
    config = _make_config(tmp_path)
    store = Mock(store_id="deployment", backend_name="mooncake")
    store.max_object_bytes = 8192
    publisher = Mock(store_id="deployment", backend_name="mooncake")
    with (
        patch(
            "vllm.distributed.artifact_connector.connector.MooncakeArtifactStore",
            return_value=store,
        ) as store_class,
        patch(
            "vllm.distributed.artifact_connector.connector.MooncakeArtifactPublisher",
            return_value=publisher,
        ) as publisher_class,
    ):
        connector = ArtifactWorkerConnector(
            config,
            dtype=np.dtype(np.uint8),
            shape_per_token=(3, 2),
        )

    assert store_class.call_args.args == ("deployment",)
    assert store_class.call_args.kwargs["staging_buffer_bytes"] == 8192
    assert store_class.call_args.kwargs["max_object_bytes"] == 4 + 4096 + 24
    publisher_class.assert_called_once_with(store)
    connector.close()
    publisher.close.assert_called_once_with()


def test_scheduler_releases_ordered_external_keys(tmp_path):
    config = _make_config(tmp_path)
    reader = Mock()
    with patch(
        "vllm.distributed.artifact_connector.connector.MooncakeArtifactReader",
        return_value=reader,
    ):
        connector = ArtifactSchedulerConnector(
            config,
            dtype=np.dtype(np.uint8),
            shape_per_token=(3, 2),
        )
    attempt_id = connector.request_finished(
        request_id="request-a",
        block_hashes=[b"a" * 32],
        token_end=2,
        hash_block_size=4,
    )
    connector.build_connector_metadata()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.artifact_connector = connector
    scheduler.requests = {
        "request-a": SimpleNamespace(request_id="request-a", client_index=0)
    }
    terminal = EngineCoreOutput(
        request_id="request-a",
        new_token_ids=[7],
        finish_reason=FinishReason.STOP,
    )
    scheduler._pending_artifact_outputs = {"request-a": terminal}
    scheduler.finished_req_ids_dict = None
    scheduler._free_blocks = Mock()
    outputs: dict[int, list[EngineCoreOutput]] = {0: []}

    from vllm.distributed.artifact_connector import (
        ArtifactConnectorOutput,
        ArtifactFinalizeResult,
    )

    scheduler._release_artifact_outputs(
        ArtifactConnectorOutput(
            [
                ArtifactFinalizeResult(
                    request_id="request-a",
                    request_attempt_id=attempt_id,
                    artifact_keys=["key-0"],
                )
            ]
        ),
        outputs,
    )

    assert terminal.artifact_keys == ["key-0"]
    connector.close()


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
