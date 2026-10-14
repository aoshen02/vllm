# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable auxiliary objects and completion manifests in Mooncake."""

from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass, field
from math import prod
from threading import RLock
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import msgspec
from typing_extensions import Buffer

from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObject,
    BlockObjectStoreError,
    iter_object_batches,
)
from vllm.distributed.aux_output_connector.types import (
    DEFAULT_STAGING_BUDGET_BYTES,
    NUMERIC_OBJECT_BYTES,
    AuxChunkRef,
    AuxOutputManifest,
    AuxOutputRef,
    PayloadRef,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store
from vllm.utils.network_utils import get_ip

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
    from vllm.v1.request import Request


class MooncakeClient(Protocol):
    def put_batch(self, keys: list[str], values: list[bytes]) -> int: ...

    def batch_get_buffer(self, keys: list[str]) -> list[Buffer | None]: ...

    def close(self) -> None: ...


class MooncakeBlockObjectStore:
    def __init__(
        self,
        store: MooncakeClient,
        *,
        object_nbytes: int,
        max_batch_bytes: int,
        local_buffer_bytes: int | None = None,
    ) -> None:
        max_batch_bytes = min(
            max_batch_bytes, 64 * 1024**2, local_buffer_bytes or max_batch_bytes
        )
        if not 0 < object_nbytes <= min(NUMERIC_OBJECT_BYTES, max_batch_bytes):
            raise ValueError("Mooncake batch must fit at least one auxiliary block")
        self._store = store
        self.object_nbytes = object_nbytes
        self._max_batch_bytes = max_batch_bytes
        self._local_buffer_bytes = local_buffer_bytes or max_batch_bytes
        self._batch_size = max_batch_bytes // object_nbytes

    def put(
        self,
        objects: list[BlockObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        unique = {obj.key: obj for obj in objects}
        for batch in iter_object_batches(unique.values(), self._max_batch_bytes):
            self._put_batch(batch)

    def _put_batch(self, objects: list[BlockObject]) -> None:
        result = self._store.put_batch(
            [obj.key for obj in objects], [obj.payload for obj in objects]
        )
        if result != 0:
            raise BlockObjectStoreError(f"Mooncake publication failed: {result}")

    def put_manifest(self, obj: BlockObject) -> None:
        for batch in iter_object_batches([obj], self._local_buffer_bytes):
            self._put_batch(batch)

    def get_concatenated(self, keys: list[str]) -> bytes:
        chunks: list[bytes] = []
        for start in range(0, len(keys), self._batch_size):
            batch_keys = keys[start : start + self._batch_size]
            buffers = self._store.batch_get_buffer(batch_keys)
            for key, buffer in zip(batch_keys, buffers, strict=True):
                if buffer is None:
                    raise BlockObjectStoreError(f"Mooncake object unavailable: {key}")
                chunks.append(bytes(buffer))
            del buffer, buffers
        return b"".join(chunks)

    def close(self) -> None:
        self._store.close()


@dataclass
class CompletionState:
    prompt_tokens: int
    token_end: int
    payload: PayloadRef
    manifest_key: str = field(
        default_factory=lambda: f"vllm-aux/{uuid4().hex}/manifest"
    )
    tail: bytearray = field(default_factory=bytearray)
    final_count: int | None = None
    result: Future[AuxOutputRef] = field(default_factory=Future)
    cancelled: bool = False


class MooncakeOutputPublisher:
    def __init__(
        self,
        output: "AuxRequestOutput",
        block_size: int,
        *,
        staging_budget_bytes: int = DEFAULT_STAGING_BUDGET_BYTES,
        max_snapshot_bytes: int = 0,
    ) -> None:
        self._block_size = block_size
        self._row_nbytes = prod(output.rows.shape[1:]) * output.rows.dtype.itemsize
        block_nbytes = block_size * self._row_nbytes
        if staging_budget_bytes < max(8 * block_nbytes, 4 * max_snapshot_bytes):
            raise ValueError("Auxiliary staging budget is too small")
        self._staging_budget_bytes = staging_budget_bytes
        self._step_bytes = 2 * max_snapshot_bytes
        self._store = create_mooncake_block_store(object_nbytes=block_nbytes)
        self._chunk_bytes = min(
            NUMERIC_OBJECT_BYTES,
            staging_budget_bytes // 8,
            self._store._max_batch_bytes,
        )
        self._chunk_bytes -= self._chunk_bytes % self._row_nbytes
        self._reserved_bytes = 0
        self._pending_updates = 0
        self._writer = BackgroundBlockObjectStore(self._store, max_pending_batches=0)
        self._requests: dict[str, CompletionState] = {}
        self._queued_bytes = 0
        self._staged_bytes = 0
        self._lock = RLock()

    def try_reserve_step(self) -> bool:
        with self._lock:
            if (
                self._queued_bytes + self._reserved_bytes + self._step_bytes
                > self._staging_budget_bytes // 2
                or self._pending_updates >= 1024
            ):
                return False
            self._reserved_bytes += self._step_bytes
            return True

    def release_step(self) -> None:
        with self._lock:
            self._reserved_bytes -= self._step_bytes

    def take_output(self, request: "Request", output: "AuxRequestOutput") -> None:
        start, end = output.token_start, request.num_tokens - 1
        if end < start:
            if request.is_finished():
                raise ValueError("Invalid final routing range")
            return
        if output.block_keys is None:
            raise ValueError("Mooncake routing output requires block keys")
        if output.routed_layer_ids is None:
            raise ValueError("Mooncake routing output requires routed layer IDs")
        with self._lock:
            state = self._requests.get(request.request_id)
            if state is None:
                state = CompletionState(
                    request.num_tokens - request.num_output_tokens,
                    start,
                    PayloadRef(
                        start,
                        0,
                        {
                            "dtype": output.rows.dtype.str,
                            "num_layers": output.rows.shape[1],
                            "experts_per_token": output.rows.shape[2],
                            "routed_layer_ids": output.routed_layer_ids,
                        },
                        [],
                    ),
                )
                self._requests[request.request_id] = state
            if state.cancelled or state.final_count is not None:
                return
            if start != state.token_end:
                raise ValueError("Noncontiguous routed-experts output")
            state.token_end = end
            size = output.rows.nbytes
            if self._queued_bytes + 2 * size > self._staging_budget_bytes // 2:
                raise ValueError("Routing snapshot exceeds auxiliary staging budget")
            self._queued_bytes += size
            self._pending_updates += 1
        rows = output.rows.tobytes(order="C")
        keys = list(output.block_keys)

        def append() -> None:
            if not state.cancelled:
                self._append(state, start, end, keys, rows)

        def complete(future: Future[None]) -> None:
            with self._lock:
                self._queued_bytes -= size
                self._pending_updates -= 1
                error = future.exception()
                if error is not None and not state.result.done():
                    state.result.set_exception(error)

        self._writer.submit(append).add_done_callback(complete)

    def _append(
        self, state: CompletionState, start: int, end: int, keys: list[str], rows: bytes
    ) -> None:
        block_size = self._block_size
        stored_start = start // block_size * block_size
        stored_end = stored_start + len(keys) * block_size if keys else start
        if end > stored_end + len(rows) // self._row_nbytes:
            raise ValueError("Routing snapshot does not cover accepted rows")
        for index, key in enumerate(keys):
            block_start = stored_start + index * block_size
            lo, hi = max(start, block_start), min(end, block_start + block_size)
            if hi <= lo:
                break
            if lo == block_start and hi == block_start + block_size:
                self._seal_tail(state)
                state.payload.chunks.append(
                    AuxChunkRef(block_size, {"expert_ids": key})
                )
            else:
                data = self._store.get_concatenated([key])
                if len(data) != block_size * self._row_nbytes:
                    raise BlockObjectStoreError("Routing block has an invalid length")
                self._append_tail(
                    state,
                    data[
                        (lo - block_start) * self._row_nbytes : (hi - block_start)
                        * self._row_nbytes
                    ],
                )
        if end > stored_end:
            self._append_tail(state, rows[: (end - stored_end) * self._row_nbytes])
        with self._lock:
            pressure = self._staged_bytes >= self._staging_budget_bytes // 4
            states = list(self._requests.values()) if pressure else []
        for pending in states:
            self._seal_tail(pending)

    def _append_tail(self, state: CompletionState, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            count = min(self._chunk_bytes - len(state.tail), len(data) - offset)
            state.tail.extend(data[offset : offset + count])
            with self._lock:
                self._staged_bytes += count
            offset += count
            if len(state.tail) == self._chunk_bytes:
                self._seal_tail(state)

    def _seal_tail(self, state: CompletionState) -> None:
        if not state.tail:
            return
        size = len(state.tail)
        if not state.cancelled:
            key = f"r3-tail:{uuid4().hex}"
            self._store.put([BlockObject(key, bytes(state.tail))])
            state.payload.chunks.append(
                AuxChunkRef(size // self._row_nbytes, {"expert_ids": key})
            )
        state.tail.clear()
        with self._lock:
            self._staged_bytes -= size

    def finalize(self, request_id: str, num_tokens: int) -> Future[AuxOutputRef]:
        with self._lock:
            state = self._requests[request_id]
            if not 0 <= num_tokens <= state.token_end + 1 - state.prompt_tokens:
                raise ValueError("Invalid final generated-token count")
            if state.final_count is not None:
                if state.final_count != num_tokens:
                    raise ValueError("Conflicting auxiliary finalization counts")
                return state.result
            state.final_count = num_tokens

        def publish() -> None:
            if state.cancelled:
                return
            visible = max(
                0, state.prompt_tokens + num_tokens - 1 - state.payload.token_start
            )
            sealed = sum(chunk.num_rows for chunk in state.payload.chunks)
            keep = max(0, visible - sealed) * self._row_nbytes
            removed = len(state.tail) - min(keep, len(state.tail))
            del state.tail[keep:]
            with self._lock:
                self._staged_bytes -= removed
            self._seal_tail(state)
            state.payload.num_rows = visible
            if visible == 0:
                state.payload.chunks.clear()
            manifest = AuxOutputManifest({"routed_experts": state.payload})
            self._store.put_manifest(
                BlockObject(state.manifest_key, msgspec.json.encode(manifest))
            )
            with self._lock:
                if not state.result.done():
                    state.result.set_result(
                        AuxOutputRef("mooncake", state.manifest_key)
                    )

        def complete(future: Future[None]) -> None:
            with self._lock:
                error = future.exception()
                if error is not None and not state.result.done():
                    state.result.set_exception(error)
            self._release(request_id, state)

        self._writer.submit(publish).add_done_callback(complete)
        return state.result

    def _release(self, request_id: str, state: CompletionState) -> None:
        with self._lock:
            self._staged_bytes -= len(state.tail)
            state.tail.clear()
            state.payload.chunks.clear()
            if self._requests.get(request_id) is state:
                del self._requests[request_id]

    def discard(self, request_id: str, *, unless_finalizing: bool = False) -> None:
        with self._lock:
            state = self._requests.get(request_id)
            if state is None or (unless_finalizing and state.final_count is not None):
                return
            state.cancelled = True
            if not state.result.done():
                state.result.set_exception(
                    BlockObjectStoreError("Auxiliary output cancelled")
                )
        self._writer.submit(lambda: None).add_done_callback(
            lambda _: self._release(request_id, state)
        )

    def close(self) -> None:
        for request_id in list(self._requests):
            self.discard(request_id)
        self._writer.close()


def create_mooncake_block_store(
    *, object_nbytes: int, max_batch_bytes: int = 64 * 1024**2
) -> MooncakeBlockObjectStore:
    from mooncake.store import (
        MooncakeDistributedStore,  # type: ignore[import-not-found]
    )

    # Pool capacity is user-sized, not derived from the GPU KV cache. Each
    # embedded client contributes its configured segment, including the
    # EngineCore publisher.
    config = MooncakeStoreConfig.load_from_config(
        "VLLM_AUX_OUTPUT_MOONCAKE_CONFIG_PATH"
    )
    store: MooncakeClient = MooncakeDistributedStore()
    try:
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        return MooncakeBlockObjectStore(
            store,
            object_nbytes=object_nbytes,
            max_batch_bytes=max_batch_bytes,
            local_buffer_bytes=config.local_buffer_size,
        )
    except Exception:
        store.close()
        raise
