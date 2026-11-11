# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write auxiliary outputs of session requests to a shared object store."""

from __future__ import annotations

import queue
import threading
from collections.abc import Sequence

import numpy as np

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStore,
    AuxObjectStoreError,
)
from vllm.distributed.aux_output_connector.objects import encode_aux_object

ROUTED_EXPERTS_FIELD = "routed_experts"
INDEXER_TOPK_FIELD = "indexer_topk"
HIDDEN_STATES_FIELD = "hidden_states"
TOPK_TOKEN_IDS_FIELD = "topk_token_ids"
TOPK_LOGPROBS_FIELD = "topk_logprobs"


def aux_object_key(
    key_prefix: str, session: str, block_hash_hex: str, weight_version: str
) -> str:
    """Key of the object that ends with the hash block `block_hash_hex` and
    whose last row was computed with weights `weight_version`."""
    return f"{key_prefix}/{session}/{block_hash_hex}@{weight_version}"


def record_weight_version(
    versions: list[tuple[int, str]], start: int, weight_version: str
) -> None:
    """Record that positions from `start` on are computed with `weight_version`."""
    while versions and versions[-1][0] >= start:
        versions.pop()
    if not versions or versions[-1][1] != weight_version:
        versions.append((start, weight_version))


def weight_version_at(versions: Sequence[tuple[int, str]], position: int) -> str:
    """Weight version that computed `position`."""
    for start, weight_version in reversed(versions):
        if start <= position:
            return weight_version
    raise ValueError(f"no weight version recorded for position {position}")


class SessionObjectWriter:
    """Assemble full objects from hash blocks and write them in the background.

    An object holds `object_size` positions and is named by the hash of its
    last hash block and the weight version of its last row within its
    session. Each key is written once; a failed write fails every later call.
    """

    def __init__(
        self,
        store: AuxObjectStore,
        *,
        key_prefix: str,
        object_size: int,
        hash_block_size: int,
        max_pending_batches: int,
    ) -> None:
        if object_size % hash_block_size:
            raise ValueError(
                f"object size {object_size} is not a multiple of the hash block "
                f"size {hash_block_size}"
            )
        self._store = store
        self._key_prefix = key_prefix
        self.object_size = object_size
        self._hash_block_size = hash_block_size
        self._pieces_per_object = object_size // hash_block_size
        self._staged: dict[str, dict[int, dict[int, np.ndarray]]] = {}
        # Request -> field -> object index -> (rows, positions written).
        self._rows: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {}
        self._written: dict[str, set[str]] = {}
        self._queue: queue.Queue[tuple[list[str], list[bytes]] | None] = queue.Queue(
            maxsize=max_pending_batches
        )
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def _run(self) -> None:
        while (batch := self._queue.get()) is not None:
            try:
                if self._error is None:
                    self._store.put_batch(*batch)
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()
        self._queue.task_done()

    def check(self) -> None:
        if self._error is not None:
            raise AuxObjectStoreError(
                "auxiliary output object write failed"
            ) from self._error

    def add_blocks(
        self,
        request_id: str,
        session: str,
        block_hash_hexes: Sequence[str],
        blocks: Sequence[tuple[int, np.ndarray]],
        weight_versions: Sequence[tuple[int, str]],
    ) -> None:
        """Stage computed hash blocks; write the objects they complete.

        `blocks` are `(start, rows)` pairs of full hash blocks,
        `block_hash_hexes` are the request's hash-block hashes so far and
        `weight_versions` are `(start, version)` runs of its positions.
        """
        self.check()
        staged = self._staged.setdefault(request_id, {})
        written = self._written.setdefault(session, set())
        keys: list[str] = []
        values: list[bytes] = []
        for start, rows in blocks:
            index, offset = divmod(start, self.object_size)
            pieces = staged.setdefault(index, {})
            pieces[offset // self._hash_block_size] = np.array(rows)
            if len(pieces) < self._pieces_per_object:
                continue
            del staged[index]
            fields = {
                ROUTED_EXPERTS_FIELD: np.concatenate(
                    [pieces[i] for i in range(self._pieces_per_object)]
                )
            }
            for field, objects in self._rows.get(request_id, {}).items():
                field_rows, written_mask = objects.pop(index, (None, None))
                if field_rows is None or not written_mask.all():
                    raise AssertionError(
                        f"{field} rows of object {index} of request {request_id} "
                        "are incomplete"
                    )
                fields[field] = field_rows
            last = (index + 1) * self._pieces_per_object - 1
            key = aux_object_key(
                self._key_prefix,
                session,
                block_hash_hexes[last],
                weight_version_at(weight_versions, (index + 1) * self.object_size - 1),
            )
            if key in written:
                continue
            written.add(key)
            keys.append(key)
            values.append(encode_aux_object(fields))
        if keys:
            self._queue.put((keys, values))

    def add_rows(
        self, request_id: str, field: str, start: int, rows: np.ndarray
    ) -> None:
        """Stage rows of another field for positions from `start` on; each
        object the request completes later carries them."""
        objects = self._rows.setdefault(request_id, {}).setdefault(field, {})
        end = start + len(rows)
        for index in range(start // self.object_size, -(-end // self.object_size)):
            object_start = index * self.object_size
            if index not in objects:
                objects[index] = (
                    np.empty((self.object_size, *rows.shape[1:]), dtype=rows.dtype),
                    np.zeros(self.object_size, dtype=bool),
                )
            field_rows, written_mask = objects[index]
            lo = max(start, object_start)
            hi = min(end, object_start + self.object_size)
            field_rows[lo - object_start : hi - object_start] = rows[
                lo - start : hi - start
            ]
            written_mask[lo - object_start : hi - object_start] = True

    def read_rows(
        self, request_id: str, field: str, start: int, end: int
    ) -> tuple[int, np.ndarray] | None:
        """Staged rows of `field` up to `end`, from `start` or the first
        position not yet in a completed object, whichever is later."""
        objects = self._rows.get(request_id, {}).get(field)
        if not objects:
            return None
        first = max(start, min(objects) * self.object_size)
        if first >= end:
            return None
        chunks = []
        for index in range(first // self.object_size, -(-end // self.object_size)):
            object_start = index * self.object_size
            chunks.append(
                objects[index][0][
                    max(first, object_start) - object_start : min(
                        end, object_start + self.object_size
                    )
                    - object_start
                ]
            )
        return first, np.concatenate(chunks)

    def discard(self, request_id: str) -> None:
        self._staged.pop(request_id, None)
        self._rows.pop(request_id, None)

    def forget_session(self, session: str) -> None:
        """Drop what was written for `session`; later blocks are written again."""
        self._written.pop(session, None)

    def flush(self) -> None:
        self._queue.join()
        self.check()

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        try:
            self.check()
        finally:
            self._store.close()
