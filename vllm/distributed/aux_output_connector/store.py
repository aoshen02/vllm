# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared auxiliary-output objects and background publication."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.mooncake import MooncakeBlockObjectStore
    from vllm.distributed.aux_output_connector.shm import ShmBlockObjectStore


@dataclass(frozen=True)
class BlockObject:
    """One immutable auxiliary output object."""

    key: str
    payload: bytes


class BlockObjectStoreError(RuntimeError):
    """AuxOutput storage or retrieval failed."""


def iter_object_batches(
    objects: Iterable[BlockObject], max_batch_bytes: int
) -> Iterator[list[BlockObject]]:
    batch: list[BlockObject] = []
    batch_bytes = 0
    for obj in objects:
        size = len(obj.payload)
        if not 0 < size <= max_batch_bytes:
            raise ValueError("Auxiliary object does not fit the byte budget")
        if batch and batch_bytes + size > max_batch_bytes:
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(obj)
        batch_bytes += size
    if batch:
        yield batch


class BackgroundBlockObjectStore:
    """Serialize store mutations on a background thread."""

    def __init__(
        self,
        store: ShmBlockObjectStore | MooncakeBlockObjectStore,
        *,
        max_pending_batches: int,
    ) -> None:
        self._store = store
        self._queue: queue.Queue[tuple[Callable[[], None], Future[None]] | None] = (
            queue.Queue(maxsize=max_pending_batches)
        )
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm-aux-output-writer",
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            update = self._queue.get()
            if update is None:
                self._queue.task_done()
                return
            operation, future = update
            del update
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                self._raise_if_failed()
                operation()
            except BaseException as error:
                if self._error is None:
                    self._error = error
                future.set_exception(error)
            else:
                future.set_result(None)
            finally:
                del operation, future
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise BlockObjectStoreError(
                "auxiliary output publication failed"
            ) from self._error

    def put(
        self,
        objects: list[BlockObject],
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
        self.submit(
            lambda: self._store.put(objects, retain_keys=retains, release_keys=releases)
        )
        self._raise_if_failed()

    def submit(self, operation: Callable[[], None]) -> Future[None]:
        future: Future[None] = Future()
        try:
            if self._closed:
                raise RuntimeError("auxiliary output store is closed")
            self._raise_if_failed()
            self._queue.put((operation, future))
        except Exception as error:
            future.set_exception(error)
        return future

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def get_concatenated(self, keys: list[str]) -> bytes:
        self.flush()
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
