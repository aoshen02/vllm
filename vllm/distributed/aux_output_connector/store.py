# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared auxiliary-output objects and background publication."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.codec import RoutingArrayDtype
    from vllm.distributed.aux_output_connector.mooncake import MooncakeBlockObjectStore
    from vllm.distributed.aux_output_connector.shm import ShmBlockObjectStore


@dataclass(frozen=True)
class BlockObject:
    """One immutable auxiliary output object."""

    key: str
    payload: bytes


@dataclass(frozen=True)
class BlockPartsObject:
    key: str
    parts: tuple[bytes, ...]


class BlockObjectStoreError(RuntimeError):
    """AuxOutput storage or retrieval failed."""


class BackgroundBlockObjectStore:
    """Serialize store mutations on a background thread."""

    def __init__(
        self,
        store: ShmBlockObjectStore | MooncakeBlockObjectStore,
        *,
        max_pending_batches: int,
    ) -> None:
        self._store = store
        self._queue: queue.Queue[
            tuple[
                list[BlockObject | BlockPartsObject], tuple[str, ...], tuple[str, ...]
            ]
            | None
        ] = queue.Queue(maxsize=max_pending_batches)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-aux-output-writer",
        )
        self._thread.start()

    def _run(self) -> None:
        from vllm.distributed.aux_output_connector.mooncake import (
            MooncakeBlockObjectStore,
        )

        while True:
            update = self._queue.get()
            try:
                if update is None:
                    return
                if self._error is None:
                    objects, retain_keys, release_keys = update
                    if isinstance(self._store, MooncakeBlockObjectStore):
                        parts = [
                            obj for obj in objects if isinstance(obj, BlockPartsObject)
                        ]
                        if len(parts) != len(objects):
                            raise TypeError(
                                "Mooncake requires encoded auxiliary objects"
                            )
                        self._store.put(
                            parts, retain_keys=retain_keys, release_keys=release_keys
                        )
                    else:
                        values = [
                            obj for obj in objects if isinstance(obj, BlockObject)
                        ]
                        if len(values) != len(objects):
                            raise TypeError("SHM requires raw auxiliary objects")
                        self._store.put(
                            values, retain_keys=retain_keys, release_keys=release_keys
                        )
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise BlockObjectStoreError(
                "auxiliary output publication failed"
            ) from self._error

    def put(
        self,
        objects: Sequence[BlockObject | BlockPartsObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        retains = tuple(retain_keys)
        releases = tuple(release_keys)
        if not objects and not retains and not releases:
            return
        if self._closed:
            raise RuntimeError("auxiliary output store is closed")
        self._queue.put((list(objects), retains, releases))
        self._raise_if_failed()

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def get_concatenated(
        self,
        keys: list[str],
        *,
        shape_per_token: tuple[int, ...] | None = None,
        dtype: RoutingArrayDtype | None = None,
        token_start: int | None = None,
    ) -> bytes:
        from vllm.distributed.aux_output_connector.mooncake import (
            MooncakeBlockObjectStore,
        )

        self.flush()
        if isinstance(self._store, MooncakeBlockObjectStore):
            return self._store.get_concatenated(
                keys,
                shape_per_token=shape_per_token,
                dtype=dtype,
                token_start=token_start,
            )
        return self._store.get_concatenated(keys)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        self._store.close()
        self._raise_if_failed()
