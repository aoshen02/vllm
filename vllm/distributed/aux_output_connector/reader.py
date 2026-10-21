# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read unified auxiliary blocks without owning the Mooncake client."""

from collections.abc import Iterable, Iterator
from typing import Protocol

from typing_extensions import Buffer

from vllm.distributed.aux_output_connector.codec import (
    MAX_OBJECT_BYTES,
    AuxBlock,
    RoutingArray,
    decode_block,
)
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError


class AuxStoreReader(Protocol):
    def get_buffer(self, key: str) -> Buffer | None: ...


def read_aux_block(store: AuxStoreReader, key: str) -> AuxBlock:
    handle = store.get_buffer(key)
    if handle is None:
        raise BlockObjectStoreError(f"Auxiliary object unavailable: {key}")
    view = memoryview(handle)
    if view.nbytes > MAX_OBJECT_BYTES:
        raise ValueError("Auxiliary object exceeds size limit")
    raw = bytes(view)
    del view, handle
    return decode_block(raw)


def get_aux_output(store: AuxStoreReader, key: str) -> dict[str, RoutingArray]:
    return read_aux_block(store, key).fields


def iter_aux_blocks(
    store: AuxStoreReader,
    keys: Iterable[str],
    *,
    start: int,
    stop: int,
) -> Iterator[AuxBlock]:
    if type(start) is not int or type(stop) is not int or not 0 <= start <= stop:
        raise ValueError("Invalid auxiliary read range")
    cursor = start
    if cursor == stop:
        return
    for key in keys:
        block = read_aux_block(store, key)
        end = block.token_start + block.num_rows
        if end <= start:
            continue
        lo, hi = max(start, block.token_start), min(stop, end)
        if lo != cursor or hi <= lo:
            raise ValueError("Auxiliary blocks do not cover a contiguous range")
        yield AuxBlock(
            lo, block.routed_experts[lo - block.token_start : hi - block.token_start]
        )
        cursor = hi
        if cursor == stop:
            return
    raise ValueError("Auxiliary blocks do not cover the requested range")
