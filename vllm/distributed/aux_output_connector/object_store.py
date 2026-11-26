# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

import json
import queue
import threading
import time
import zlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import regex as re
import torch
from safetensors.numpy import load, save

from vllm.distributed.aux_output_connector.store import BlockObjectStoreError
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class AuxObjectStore(Protocol):
    """A store of immutable objects that outlive the processes writing and
    reading them. Keys are opaque strings. Writing a key that exists succeeds
    and keeps the object written first. A refusal that may succeed later
    raises `vllm.distributed.aux_output_connector.store.BlockObjectStoreError`
    and is retried until the batch's deadline;
    any other exception drops the batch. The class connects with the
    classmethod `create(**object_store_extra_config)`; the engine calls
    `put_batch` and `close` from one writer thread."""

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None: ...

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist (yet)."""

    def remove_prefix(self, prefix: str) -> None:
        """Remove the objects whose keys start with `prefix`, as built by
        `aux_prefix` or `aux_namespace_prefix`."""

    def close(self) -> None: ...


def create_aux_object_store(name: str, extra_config: dict[str, Any]) -> AuxObjectStore:
    """Connect the object store `name`: `"mooncake"`, or the qualified name of
    a class that implements `AuxObjectStore`, with `extra_config` as the
    keyword arguments of its `create`."""
    if name == "mooncake":
        return MooncakeAuxObjectStore.create(**extra_config)
    if not all(name.split(".")) or "." not in name:
        raise ValueError(
            f"object_store {name!r} is neither 'mooncake' nor a qualified class name."
        )
    return resolve_obj_by_qualname(name).create(**extra_config)


class MooncakeAuxObjectStore:
    """`AuxObjectStore` on a standalone Mooncake store."""

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(
        cls, config_path: str | None = None, hard_pin: bool = False
    ) -> MooncakeAuxObjectStore:
        """Connect with the Mooncake configuration file `config_path`, or the
        one at `MOONCAKE_CONFIG_PATH` if it is `None`, with a master and pool
        separate from any Mooncake KV store. Objects are written hard-pinned if
        `hard_pin`, for a memory-only store; by default they are not pinned, for
        a store that offloads to SSD on eviction."""
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
            rdma_utils,
        )
        from vllm.distributed.mooncake_store import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )
        from vllm.utils.network_utils import get_ip

        config = (
            MooncakeStoreConfig.load_from_config()
            if config_path is None
            else MooncakeStoreConfig.from_file(config_path)
        )
        if config.mode != "standalone-store":
            raise ValueError(
                "Auxiliary-output objects must outlive the clients that write "
                "and read them: set mode to 'standalone-store' in the Mooncake "
                "configuration."
            )
        store = MooncakeDistributedStore()
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        replicate_config = ReplicateConfig()
        replicate_config.with_hard_pin = hard_pin
        return cls(store, replicate_config)

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        """Write every object, or raise `BlockObjectStoreError`."""
        code = self._store.put_batch(list(keys), list(values), self._replicate_config)
        if code:
            raise BlockObjectStoreError(
                f"Mooncake failed to write {len(keys)} auxiliary-output "
                f"objects from {keys[0]!r}: {code}"
            )

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""
        return self._store.get(key) or None

    def remove_prefix(self, prefix: str) -> None:
        self._store.remove_by_regex(f"^{_escape(prefix)}", True)

    def close(self) -> None:
        self._store.close()


_REGEX_SYNTAX = re.compile(r"[\\^$.*+?()[\]{}|]")


def _escape(text: str) -> str:
    return _REGEX_SYNTAX.sub(r"\\\g<0>", text)


def aux_prefix(key_prefix: str) -> str:
    """Key prefix of every object under `key_prefix`."""
    return f"{key_prefix}/"


def aux_namespace_prefix(key_prefix: str, namespace: str) -> str:
    """Key prefix of every object of a namespace and of the namespaces nested
    in it."""
    return f"{key_prefix}/{namespace}/"


def aux_namespace_cache_salt(
    key_prefix: str, namespace: str, cache_salt: str | None
) -> str:
    """The cache salt of a request with `aux_namespace`: only requests of the
    same namespace and key prefix share KV blocks, as they share objects."""
    return json.dumps([key_prefix, namespace, cache_salt])


class AuxObjectCorrupted(ValueError):
    """An object that does not parse or does not match its checksum."""


class AuxRowsMisaligned(ValueError):
    """Rows that are not those of the tokens they were placed next to.

    `index` is the first misaligned row, `position` the position the trainer
    placed it at, and `offset` the position it was computed for minus
    `position`: nonzero for rows shifted against their tokens, 0 for rows of
    other tokens at the same position, such as another sample's."""

    def __init__(self, message: str, index: int, position: int, offset: int):
        super().__init__(message)
        self.index = index
        self.position = position
        self.offset = offset


def _checksum(fields: dict[str, np.ndarray], metadata: dict[str, str]) -> str:
    """CRC-32 of the metadata other than the checksum, and of the arrays with
    their names, dtypes and shapes."""
    rest = sorted((k, v) for k, v in metadata.items() if k != "crc32")
    crc = zlib.crc32(json.dumps(rest).encode())
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        crc = zlib.crc32(f"{name}:{array.dtype.str}:{array.shape}".encode(), crc)
        crc = zlib.crc32(array, crc)
    return f"{crc:08x}"


def _load(key: str, data: bytes) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The arrays and metadata of an object, checked against its checksum.
    Objects written without metadata are not checked."""
    try:
        size = int.from_bytes(data[:8], "little")
        metadata = json.loads(data[8 : 8 + size]).get("__metadata__") or {}
        fields = load(data)
    except Exception as e:
        raise AuxObjectCorrupted(
            f"auxiliary-output object {key!r} does not parse"
        ) from e
    if metadata and metadata.get("crc32") != _checksum(fields, metadata):
        raise AuxObjectCorrupted(
            f"auxiliary-output object {key!r} does not match its checksum"
        )
    return fields, metadata


def _get(store: AuxObjectStore, key: str, deadline: float, timeout_s: float) -> bytes:
    while (data := store.get(key)) is None:
        if time.monotonic() >= deadline:
            raise BlockObjectStoreError(
                f"auxiliary-output object {key!r} is missing after {timeout_s}s"
            )
        time.sleep(0.05)
    return data


def read_aux_rows(
    store: AuxObjectStore,
    keys: Sequence[str],
    field: str,
    timeout_s: float = 600.0,
) -> np.ndarray:
    """Concatenate `field` of the objects `keys`; wait up to `timeout_s` in
    total for objects that are still being written."""
    deadline = time.monotonic() + timeout_s
    parts = [
        _load(key, _get(store, key, deadline, timeout_s))[0][field] for key in keys
    ]
    return np.concatenate(parts) if parts else np.empty(0)


def read_aux_spans(
    store: AuxObjectStore,
    spans: Sequence[tuple[str, int, int]],
    field: str = "routed_experts",
    timeout_s: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read positions `[start, end)` of object `key` for each `(key, start,
    end)`, in order: an `aux_objects` listing as returned, or as spliced onto
    a trajectory. The spans must abut: a gap masks the sample. Return the rows
    of `field`, the token ids they were computed for, and their positions, for
    `verify_aux_rows`. Waits up to `timeout_s` in total for objects that are
    still being written."""
    deadline = time.monotonic() + timeout_s
    rows, token_ids, positions = [], [], []
    for index, (key, start, end) in enumerate(spans):
        if index and start != spans[index - 1][2]:
            raise ValueError(
                f"auxiliary-output span {key!r} starts at {start}, not at the "
                f"end of the span before it, {spans[index - 1][2]}"
            )
        fields, metadata = _load(key, _get(store, key, deadline, timeout_s))
        if "crc32" not in metadata or "start" not in metadata:
            raise AuxObjectCorrupted(
                f"auxiliary-output object {key!r} has no token ids or checksum"
            )
        first = int(metadata["start"])
        last = first + len(fields["token_ids"])
        if not first <= start <= end <= last:
            raise ValueError(
                f"positions [{start}, {end}) are not in auxiliary-output object "
                f"{key!r}, which holds [{first}, {last})"
            )
        rows.append(fields[field][start - first : end - first])
        token_ids.append(fields["token_ids"][start - first : end - first])
        positions.append(np.arange(start, end, dtype=np.int64))
    if not rows:
        return np.empty(0), np.empty(0, np.int32), np.empty(0, np.int64)
    return np.concatenate(rows), np.concatenate(token_ids), np.concatenate(positions)


def verify_aux_rows(
    token_ids: torch.Tensor | np.ndarray,
    positions: torch.Tensor | np.ndarray,
    row_token_ids: torch.Tensor | np.ndarray,
    row_positions: torch.Tensor | np.ndarray,
    *,
    mask: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Check that the rows placed at `token_ids` and `positions` (a trainer
    rank's inputs, after splicing, packing or context-parallel sharding) were
    computed for these tokens at these positions. `row_token_ids` and
    `row_positions` come from `read_aux_spans` and must go through every step
    that moves the rows, as the rows do. Positions are indices into the token
    sequence the engine was sent. Only the rows where `mask` is true are
    checked, such as the replay mask; all by default. Raises
    `AuxRowsMisaligned`."""
    expected = torch.as_tensor(token_ids).reshape(-1)
    device = expected.device
    position = torch.as_tensor(positions, device=device).reshape(-1)
    row_token = torch.as_tensor(row_token_ids, device=device).reshape(-1)
    row_position = torch.as_tensor(row_positions, device=device).reshape(-1)
    selected = (
        torch.ones_like(expected, dtype=torch.bool)
        if mask is None
        else torch.as_tensor(mask, device=device).reshape(-1).bool()
    )
    tensors = (expected, position, row_token, row_position, selected)
    if len({t.numel() for t in tensors}) > 1:
        raise ValueError(
            f"{expected.numel()} token ids, {position.numel()} positions, "
            f"{row_token.numel()} rows bound to {row_position.numel()} positions "
            f"and a mask of {selected.numel()}"
        )
    bad = selected & ((row_token != expected) | (row_position != position))
    if not bool(bad.any()):
        return
    num_mismatched = int(bad.sum())
    index = int(bad.nonzero()[0, 0])
    at = int(position[index])
    offset = int(row_position[index]) - at
    raise AuxRowsMisaligned(
        f"{num_mismatched} auxiliary-output rows are misaligned; the first, "
        f"row {index} at position {at}, was computed for token "
        f"{int(row_token[index])} at position {at + offset} (offset {offset}), "
        f"not token {int(expected[index])}",
        index,
        at,
        offset,
    )


_RETRY_INTERVAL_S = 0.2
_MAX_QUEUED_BATCHES = 256

AuxObjects = list[tuple[str, dict[str, np.ndarray], dict[str, str]]]
"""`(key, fields, metadata)`; the writer adds the checksum to the metadata."""


@dataclass(slots=True)
class _Batch:
    # Dropped when the batch misses its deadline, so that a put that does not
    # return holds no more than the batches queued before their deadline.
    objects: AuxObjects | None
    deadline: float


class AuxObjectWriter:
    """Write objects in the background, as safetensors with one array per
    field and the CRC-32 of the arrays in the metadata. A batch not written
    within `timeout_s` of being queued is dropped: the store refused it until
    then (refusals are retried every
    `_RETRY_INTERVAL_S`), its put failed, or it was still queued behind a put
    that had not returned. Its keys are missing; nothing else changes.
    Queuing waits while `_MAX_QUEUED_BATCHES` batches are queued, at most
    until the first of them is written or misses its deadline. While a batch
    past its deadline is still queued, the store is not taking writes:
    queuing drops the batch without waiting."""

    def __init__(self, store: AuxObjectStore, *, timeout_s: float) -> None:
        self._store = store
        self._timeout_s = timeout_s
        self._queue: queue.SimpleQueue[_Batch | None] = queue.SimpleQueue()
        self._cond = threading.Condition()
        # The batches not written yet, in queue order.
        self._pending: deque[_Batch] = deque()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def put(self, objects: AuxObjects) -> None:
        """Queue `(key, fields, metadata)` objects as one batch: a caller that
        puts once per step waits at most once per step."""
        with self._cond:
            self._expire(time.monotonic())
            while len(self._pending) >= _MAX_QUEUED_BATCHES and not self._stuck():
                self._cond.wait(self._pending[0].deadline - time.monotonic())
                self._expire(time.monotonic())
            if self._stuck():
                _drop(objects, "an earlier write missed its deadline")
                return
            batch = _Batch(objects, time.monotonic() + self._timeout_s)
            self._pending.append(batch)
            self._queue.put(batch)

    def _stuck(self) -> bool:
        # Deadlines follow queue order, so dropped batches come first.
        return bool(self._pending) and self._pending[0].objects is None

    def _expire(self, now: float) -> None:
        """Drop the pending batches past their deadline."""
        for batch in self._pending:
            if batch.deadline > now:
                break
            if batch.objects is not None:
                _drop(batch.objects, f"not written within {self._timeout_s:g} s")
                batch.objects = None

    def _run(self) -> None:
        try:
            while (batch := self._queue.get()) is not None:
                error = None
                if (objects := batch.objects) is not None:
                    try:
                        error = self._write(batch.deadline, objects)
                    except Exception as e:
                        error = e
                with self._cond:
                    if error is not None and batch.objects is not None:
                        _drop(batch.objects, error)
                        batch.objects = None
                    self._expire(time.monotonic())
                    self._pending.popleft()
                    self._cond.notify_all()
        finally:
            self._store.close()

    def _write(self, deadline: float, objects: AuxObjects) -> object:
        keys = [key for key, _, _ in objects]
        values = []
        for _, fields, metadata in objects:
            # safetensors writes the buffer of a strided array, not its values.
            fields = {name: np.ascontiguousarray(a) for name, a in fields.items()}
            crc32 = _checksum(fields, metadata)
            values.append(save(fields, metadata={**metadata, "crc32": crc32}))
        error: object = "its deadline passed"
        while (wait := deadline - time.monotonic()) > 0:
            try:
                self._store.put_batch(keys, values)
                return None
            except BlockObjectStoreError as e:
                error = e
                time.sleep(min(_RETRY_INTERVAL_S, wait))
        return error

    def close(self) -> None:
        """Write what is queued, waiting until the last deadline at most; the
        writer thread closes the store once it is done."""
        with self._cond:
            pending = self._pending[-1].deadline if self._pending else None
        deadline = pending or time.monotonic() + self._timeout_s
        self._queue.put(None)
        self._thread.join(max(deadline - time.monotonic(), 0.0))


def _drop(objects: AuxObjects, reason: object) -> None:
    logger.error(
        "%d auxiliary-output objects from %r are not written: %s",
        len(objects),
        objects[0][0],
        reason,
    )
