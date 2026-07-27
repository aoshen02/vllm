# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct Mooncake backend for immutable execution-artifact objects."""

from __future__ import annotations

import ctypes
import queue
import threading
from typing import Any

import regex as re

from vllm.distributed.artifact_connector.store import (
    ArtifactCapacityError,
    ArtifactCorruptionError,
    ArtifactNotFoundError,
    ArtifactObject,
    ArtifactPutResult,
    ArtifactStoreError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreConfig,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

_SAFE_STORE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _create_store() -> tuple[Any, Any]:
    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    except ImportError as error:
        raise ImportError(
            "Install Mooncake to use the execution-artifact Mooncake backend."
        ) from error

    config = MooncakeStoreConfig.load_from_config()
    store = MooncakeDistributedStore()
    local_ip = get_ip()
    local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
    try:
        result = store.setup(
            local_hostname,
            config.metadata_server,
            config.global_segment_size,
            config.local_buffer_size,
            config.protocol,
            config.device_name,
            config.master_server_address,
        )
    except Exception:
        store.close()
        raise
    if result != 0:
        store.close()
        raise RuntimeError(f"Initialize Mooncake store failed: code={result}")
    logger.info(
        "Initialized artifact Mooncake client in %s mode "
        "(global_segment_size=%d, local_buffer_size=%d)",
        config.mode,
        config.global_segment_size,
        config.local_buffer_size,
    )
    return store, ReplicateConfig()


class MooncakeArtifactReader:
    """Lightweight Mooncake client used for scheduler-side existence checks."""

    backend_name = "mooncake"

    def __init__(self, store_id: str, *, store: Any | None = None) -> None:
        if not _SAFE_STORE_ID.fullmatch(store_id):
            raise ValueError(f"invalid Mooncake artifact store id: {store_id!r}")
        self.store_id = store_id
        self._lock = threading.Lock()
        self._closed = False
        self._store = _create_store()[0] if store is None else store

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Mooncake artifact store is closed")

    @staticmethod
    def _validate_exists(keys: list[str], results: Any) -> list[bool]:
        if not isinstance(results, list) or len(results) != len(keys):
            raise ArtifactStoreError(
                "Mooncake returned the wrong batch_is_exist result count"
            )
        exists: list[bool] = []
        for key, result in zip(keys, results, strict=True):
            if result == 1:
                exists.append(True)
            elif result == 0:
                exists.append(False)
            else:
                raise ArtifactStoreError(
                    f"Mooncake existence lookup failed for {key}: code={result}"
                )
        return exists

    def exists(self, keys: list[str]) -> list[bool]:
        if not keys:
            return []
        with self._lock:
            self._check_open()
            results = self._store.batch_is_exist(keys)
        return self._validate_exists(keys, results)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            store = self._store
            self._store = None
            try:
                store.close()
            except Exception:
                logger.warning("Error closing artifact Mooncake client", exc_info=True)


class MooncakeArtifactStore(MooncakeArtifactReader):
    """Mooncake object I/O over one registered, bounded staging region."""

    def __init__(
        self,
        store_id: str,
        *,
        staging_buffer_bytes: int,
        max_object_bytes: int,
        store: Any | None = None,
        replicate_config: Any | None = None,
    ) -> None:
        if staging_buffer_bytes <= 0:
            raise ValueError("Mooncake staging buffer size must be positive")
        if max_object_bytes <= 0:
            raise ValueError("Mooncake object capacity must be positive")
        if max_object_bytes > staging_buffer_bytes:
            raise ValueError(
                "Mooncake staging buffer must fit at least one artifact object"
            )
        if store is None:
            store, default_replicate_config = _create_store()
            if replicate_config is None:
                replicate_config = default_replicate_config
        super().__init__(store_id, store=store)
        self.max_object_bytes = max_object_bytes
        self._replicate_config = replicate_config
        self._staging = bytearray(staging_buffer_bytes)
        self._staging_address = ctypes.addressof(
            ctypes.c_char.from_buffer(self._staging)
        )
        try:
            result = self._store.register_buffer(
                self._staging_address, len(self._staging)
            )
        except Exception:
            super().close()
            raise
        if result != 0:
            super().close()
            raise RuntimeError(
                "Mooncake failed to register the artifact staging buffer: "
                f"code={result}"
            )

    def _put_batch(
        self,
        objects: list[ArtifactObject],
        results: list[ArtifactPutResult],
        result_indices: list[int],
    ) -> None:
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        offset = 0
        for obj in objects:
            size = len(obj.payload)
            self._staging[offset : offset + size] = obj.payload
            addresses.append([self._staging_address + offset])
            sizes.append([size])
            offset += size
        put_results = self._store.batch_put_from_multi_buffers(
            [obj.key for obj in objects],
            addresses,
            sizes,
            self._replicate_config,
        )
        if not isinstance(put_results, list) or len(put_results) != len(objects):
            message = "ArtifactStoreError: wrong Mooncake batch put result count"
            for index in result_indices:
                results[index] = ArtifactPutResult(results[index].key, message)
            return
        for obj, index, result in zip(
            objects, result_indices, put_results, strict=True
        ):
            if result < 0:
                results[index] = ArtifactPutResult(
                    obj.key,
                    f"ArtifactStoreError: Mooncake put failed: code={result}",
                )

    def put(self, objects: list[ArtifactObject]) -> list[ArtifactPutResult]:
        if not objects:
            return []
        results = [ArtifactPutResult(obj.key) for obj in objects]
        with self._lock:
            self._check_open()
            try:
                exists = self._validate_exists(
                    [obj.key for obj in objects],
                    self._store.batch_is_exist([obj.key for obj in objects]),
                )
            except Exception as error:
                message = _error_text(error)
                return [ArtifactPutResult(obj.key, message) for obj in objects]

            batch: list[ArtifactObject] = []
            batch_indices: list[int] = []
            batch_bytes = 0

            def flush() -> None:
                nonlocal batch, batch_indices, batch_bytes
                if not batch:
                    return
                try:
                    self._put_batch(batch, results, batch_indices)
                except Exception as error:
                    message = _error_text(error)
                    for index in batch_indices:
                        results[index] = ArtifactPutResult(results[index].key, message)
                batch = []
                batch_indices = []
                batch_bytes = 0

            for index, (obj, already_exists) in enumerate(
                zip(objects, exists, strict=True)
            ):
                if already_exists:
                    continue
                size = len(obj.payload)
                if size > self.max_object_bytes:
                    results[index] = ArtifactPutResult(
                        obj.key,
                        _error_text(
                            ArtifactCapacityError(
                                f"artifact object requires {size} bytes; "
                                f"capacity={self.max_object_bytes}"
                            )
                        ),
                    )
                    continue
                if batch and batch_bytes + size > len(self._staging):
                    flush()
                batch.append(obj)
                batch_indices.append(index)
                batch_bytes += size
            flush()
        return results

    def get(self, keys: list[str]) -> list[bytes]:
        if not keys:
            return []
        with self._lock:
            self._check_open()
            object_sizes: list[int] = []
            for key in keys:
                size = self._store.get_size(key)
                if size < 0:
                    raise ArtifactNotFoundError(
                        f"Mooncake artifact object does not exist: {key}"
                    )
                if size > self.max_object_bytes:
                    raise ArtifactCorruptionError(
                        f"Mooncake object exceeds registered capacity: {key}"
                    )
                object_sizes.append(size)

            payloads: list[bytes] = []
            start = 0
            while start < len(keys):
                end = start
                batch_bytes = 0
                while end < len(keys):
                    size = object_sizes[end]
                    if end > start and batch_bytes + size > len(self._staging):
                        break
                    batch_bytes += size
                    end += 1

                batch_keys = keys[start:end]
                batch_sizes = object_sizes[start:end]
                offsets: list[int] = []
                offset = 0
                for size in batch_sizes:
                    offsets.append(offset)
                    offset += size
                addresses = [[self._staging_address + offset] for offset in offsets]
                capacities = [[size] for size in batch_sizes]
                get_results = self._store.batch_get_into_multi_buffers(
                    batch_keys, addresses, capacities
                )
                if not isinstance(get_results, list) or len(get_results) != len(
                    batch_keys
                ):
                    raise ArtifactStoreError(
                        "Mooncake returned the wrong batch get result count"
                    )
                for key, expected, actual, offset in zip(
                    batch_keys,
                    batch_sizes,
                    get_results,
                    offsets,
                    strict=True,
                ):
                    if actual < 0:
                        raise ArtifactStoreError(
                            f"Mooncake get failed for {key}: code={actual}"
                        )
                    if actual != expected:
                        raise ArtifactCorruptionError(
                            f"Mooncake object size changed while reading: {key}"
                        )
                    payloads.append(bytes(self._staging[offset : offset + actual]))
                start = end
        return payloads

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                result = self._store.unregister_buffer(self._staging_address)
                if result != 0:
                    logger.warning(
                        "Mooncake failed to unregister artifact staging buffer: "
                        "code=%s",
                        result,
                    )
            except Exception:
                logger.warning(
                    "Error unregistering artifact Mooncake buffer", exc_info=True
                )
        super().close()


class MooncakeArtifactPublisher:
    """Start bounded Mooncake puts without waiting for remote readiness."""

    backend_name = "mooncake"

    def __init__(
        self,
        store: MooncakeArtifactStore,
        *,
        max_pending_batches: int = 1,
    ) -> None:
        if max_pending_batches <= 0:
            raise ValueError("Mooncake pending batch count must be positive")
        self.store_id = store.store_id
        self._store = store
        self._queue: queue.Queue[list[ArtifactObject] | None] = queue.Queue(
            max_pending_batches
        )
        self._state_lock = threading.Lock()
        self._closed = False
        self._fatal_error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-artifact-mooncake",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            objects = self._queue.get()
            try:
                if objects is None:
                    return
                results = self._store.put(objects)
                error = next((result.error for result in results if result.error), None)
                if error is not None:
                    with self._state_lock:
                        self._fatal_error = error
                    logger.error("Artifact Mooncake publication failed: %s", error)
            except Exception as error:
                message = _error_text(error)
                with self._state_lock:
                    self._fatal_error = message
                logger.exception("Artifact Mooncake publication failed")
            finally:
                self._queue.task_done()

    def put(self, objects: list[ArtifactObject]) -> list[ArtifactPutResult]:
        if not objects:
            return []
        results = [ArtifactPutResult(obj.key) for obj in objects]
        accepted: list[ArtifactObject] = []
        accepted_indices: list[int] = []
        for index, obj in enumerate(objects):
            if len(obj.payload) > self._store.max_object_bytes:
                results[index] = ArtifactPutResult(
                    obj.key,
                    _error_text(
                        ArtifactCapacityError(
                            f"artifact object requires {len(obj.payload)} bytes; "
                            f"capacity={self._store.max_object_bytes}"
                        )
                    ),
                )
            else:
                accepted.append(obj)
                accepted_indices.append(index)
        if accepted:
            with self._state_lock:
                message = (
                    "RuntimeError: Mooncake artifact publisher is closed"
                    if self._closed
                    else self._fatal_error
                )
                if message is None:
                    self._queue.put(accepted)
                else:
                    for index in accepted_indices:
                        results[index] = ArtifactPutResult(results[index].key, message)
        return results

    def exists(self, keys: list[str]) -> list[bool]:
        return self._store.exists(keys)

    def get(self, keys: list[str]) -> list[bytes]:
        return self._store.get(keys)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._store.close()
