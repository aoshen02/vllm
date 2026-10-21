# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable auxiliary-output blocks in a deployment-managed Mooncake Store."""

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import groupby
from math import prod
from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

import numpy as np
from typing_extensions import Buffer

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
    from vllm.v1.request import Request

from vllm.distributed.aux_output_connector.codec import (
    HEADER,
    MAX_OBJECT_BYTES,
    OUTPUT_KEY_PREFIX,
    AuxBlock,
    RoutingArray,
    RoutingArrayDtype,
    decode_block,
    encode_parts,
)
from vllm.distributed.aux_output_connector.store import (
    BlockObjectStoreError,
    BlockPartsObject,
)


class MooncakeClient(Protocol):
    def put_parts(self, key: str, *parts: bytes) -> int: ...
    def batch_get_buffer(self, keys: list[str]) -> Sequence[Buffer | None]: ...
    def close(self) -> None: ...


class MooncakeBlockObjectStore:
    """Publish immutable objects; retention belongs to the deployment."""

    def __init__(
        self, store: MooncakeClient, *, object_nbytes: int, max_batch_bytes: int
    ) -> None:
        encoded_nbytes = HEADER.size + object_nbytes
        if not object_nbytes > 0 or encoded_nbytes > min(
            max_batch_bytes, MAX_OBJECT_BYTES
        ):
            raise ValueError(
                "Mooncake batch must fit at least one encoded auxiliary block"
            )
        if not callable(getattr(store, "put_parts", None)):
            raise RuntimeError("Mooncake auxiliary output requires put_parts support")
        self._store = store
        self.object_nbytes = object_nbytes
        self._encoded_nbytes = encoded_nbytes
        self._batch_size = max_batch_bytes // encoded_nbytes
        self._parallel_puts = min(8, self._batch_size)
        self._writer = ThreadPoolExecutor(
            max_workers=self._parallel_puts, thread_name_prefix="mooncake-aux-put"
        )

    def put(
        self,
        objects: Sequence[BlockPartsObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        unique = {obj.key: obj.parts for obj in objects}
        for parts in unique.values():
            if not HEADER.size < sum(map(len, parts)) <= self._encoded_nbytes:
                raise ValueError("Mooncake object exceeds auxiliary block capacity")
        entries = list(unique.items())
        if len(entries) <= 1 or self._parallel_puts == 1:
            self._put_parts(entries)
            return
        futures = [
            self._writer.submit(self._put_parts, entries[index :: self._parallel_puts])
            for index in range(min(len(entries), self._parallel_puts))
        ]
        wait(futures)
        for future in futures:
            future.result()

    def _put_parts(self, entries: list[tuple[str, tuple[bytes, ...]]]) -> None:
        for key, parts in entries:
            result = self._store.put_parts(key, *parts)
            if result != 0:
                raise BlockObjectStoreError(f"Mooncake publication failed: {result}")

    def get_concatenated(
        self,
        keys: list[str],
        *,
        shape_per_token: tuple[int, ...] | None = None,
        dtype: RoutingArrayDtype | None = None,
        token_start: int | None = None,
    ) -> bytes:
        chunks: list[bytes] = []
        cursor = token_start
        for start in range(0, len(keys), self._batch_size):
            batch_keys = keys[start : start + self._batch_size]
            buffers = self._store.batch_get_buffer(batch_keys)
            for key, buffer in zip(batch_keys, buffers, strict=True):
                if buffer is None:
                    raise BlockObjectStoreError(f"Mooncake object unavailable: {key}")
                view = memoryview(buffer)
                if view.nbytes > self._encoded_nbytes:
                    raise ValueError("Mooncake object exceeds auxiliary block capacity")
                block = decode_block(bytes(view))
                del view
                rows = block.routed_experts
                if shape_per_token is None:
                    shape_per_token = rows.shape[1:]
                if dtype is None:
                    dtype = rows.dtype
                if rows.shape[1:] != shape_per_token or rows.dtype != dtype:
                    raise ValueError("Mooncake routing profile does not match")
                if cursor is not None and block.token_start != cursor:
                    raise ValueError(
                        "Mooncake auxiliary token ranges are not contiguous"
                    )
                cursor = block.token_start + block.num_rows
                chunks.append(rows.tobytes())
            del buffer, buffers
        return b"".join(chunks)

    def close(self) -> None:
        self._writer.shutdown(wait=True)
        self._store.close()


@dataclass(frozen=True)
class _RoutingFragment:
    token_start: int
    payload: bytes


class MooncakeOutputPublisher:
    """Finalize accepted boundaries without reading complete output blocks."""

    def __init__(self, output: "AuxRequestOutput", block_size: int) -> None:
        self._block_size = block_size
        self._shape = output.rows.shape[1:]
        self._dtype = output.rows.dtype
        self._row_nbytes = prod(self._shape) * self._dtype.itemsize
        self._store = create_mooncake_block_store(
            object_nbytes=block_size * self._row_nbytes
        )
        self._chunks: dict[str, list[str | _RoutingFragment]] = {}
        self._completion_ids: dict[str, str] = {}

    def take_output(
        self, request: "Request", output: "AuxRequestOutput"
    ) -> list[str] | None:
        start, end = int(output.token_start), request.num_tokens - 1
        if end < start:
            assert not request.is_finished()
            return None
        keys = output.block_keys
        assert keys is not None
        block_size = self._block_size
        stored_start = start // block_size * block_size
        stored_end = stored_start + len(keys) * block_size if keys else start
        assert end <= stored_end + len(output.rows)
        chunks = self._chunks.setdefault(request.request_id, [])
        for index, key in enumerate(keys):
            block_start = stored_start + index * block_size
            lo, hi = max(start, block_start), min(end, block_start + block_size)
            if hi <= lo:
                break
            if lo == block_start and hi == block_start + block_size:
                chunks.append(key)
            else:
                payload = self._store.get_concatenated(
                    [key],
                    shape_per_token=self._shape,
                    dtype=self._dtype,
                    token_start=block_start,
                )
                chunks.append(
                    _RoutingFragment(
                        lo,
                        payload[
                            (lo - block_start) * self._row_nbytes : (hi - block_start)
                            * self._row_nbytes
                        ],
                    )
                )
        if end > stored_end:
            chunks.append(
                _RoutingFragment(stored_end, output.rows[: end - stored_end].tobytes())
            )
        # String stops need keys before the API process aborts the core request.
        assert request.sampling_params is not None
        if not request.is_finished() and not request.sampling_params.stop:
            return None
        del self._chunks[request.request_id]
        completion_id = self._completion_ids.setdefault(request.request_id, uuid4().hex)
        result: list[str] = []
        objects: list[BlockPartsObject] = []
        for is_key, group in groupby(chunks, lambda chunk: isinstance(chunk, str)):
            if is_key:
                result.extend(cast(Iterable[str], group))
            else:
                fragments = list(cast(Iterable[_RoutingFragment], group))
                position = fragments[0].token_start
                cursor = position
                for fragment in fragments:
                    if fragment.token_start != cursor:
                        raise ValueError(
                            "Auxiliary output fragments are not contiguous"
                        )
                    cursor += len(fragment.payload) // self._row_nbytes
                payload = b"".join(fragment.payload for fragment in fragments)
                rows = cast(
                    RoutingArray,
                    np.frombuffer(payload, dtype=self._dtype).reshape(-1, *self._shape),
                )
                offset = 0
                while offset < len(rows):
                    count = min(block_size - position % block_size, len(rows) - offset)
                    key = f"{OUTPUT_KEY_PREFIX}/{completion_id}/{position}"
                    objects.append(
                        BlockPartsObject(
                            key,
                            encode_parts(
                                AuxBlock(position, rows[offset : offset + count])
                            ),
                        )
                    )
                    result.append(key)
                    offset += count
                    position += count
        self._store.put(objects)
        if request.is_finished():
            self._completion_ids.pop(request.request_id, None)
        return result

    def discard(self, request_id: str) -> None:
        self._chunks.pop(request_id, None)
        self._completion_ids.pop(request_id, None)

    def close(self) -> None:
        self._chunks.clear()
        self._completion_ids.clear()
        self._store.close()


def create_mooncake_block_store(
    *, object_nbytes: int, max_batch_bytes: int = 64 * 1024**2
) -> MooncakeBlockObjectStore:
    from mooncake.store import (
        MooncakeDistributedStore,  # type: ignore[import-not-found]
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
    from vllm.distributed.mooncake_store import (
        MooncakeStoreConfig,
        setup_mooncake_store,
    )
    from vllm.utils.network_utils import get_ip

    # Pool capacity is user-sized, not derived from the GPU KV cache. Each
    # embedded client contributes its configured segment, including the
    # EngineCore publisher.
    config = MooncakeStoreConfig.load_from_config(
        "VLLM_AUX_OUTPUT_MOONCAKE_CONFIG_PATH"
    )
    store = MooncakeDistributedStore()
    try:
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        return MooncakeBlockObjectStore(
            store,
            object_nbytes=object_nbytes,
            max_batch_bytes=min(max_batch_bytes, config.local_buffer_size),
        )
    except Exception:
        store.close()
        raise
