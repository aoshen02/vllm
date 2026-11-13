# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any

import regex as re

if TYPE_CHECKING:
    from vllm.distributed.mooncake_store import MooncakeStoreConfig


class AuxObjectStoreError(RuntimeError):
    """A shared auxiliary-output object could not be written."""


class AuxObjectStore(ABC):
    """Immutable named objects shared by engines and their consumers.

    A key is written once and never changes. Writing a key that already
    exists succeeds and keeps the stored bytes, so retried writes are safe.
    Objects stay until they are removed by key.
    """

    @abstractmethod
    def put_batch(
        self, keys: Sequence[str], values: Sequence[bytes | memoryview]
    ) -> None:
        """Write every object, or raise `AuxObjectStoreError`."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""

    @abstractmethod
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """Return whether each object exists."""

    @abstractmethod
    def remove(self, keys: Sequence[str]) -> None:
        """Remove the objects that exist."""

    @abstractmethod
    def remove_prefix(self, prefix: str) -> int:
        """Remove every object whose key starts with `prefix`; return how many."""

    def close(self) -> None:
        """Release the client's resources."""
        return None


class LocalAuxObjectStore(AuxObjectStore):
    """Objects held in this process."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = Lock()

    def put_batch(
        self, keys: Sequence[str], values: Sequence[bytes | memoryview]
    ) -> None:
        if len(keys) != len(values):
            raise ValueError("key and value counts differ")
        with self._lock:
            for key, value in zip(keys, values):
                self._objects.setdefault(key, bytes(value))

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._objects.get(key)

    def exists(self, keys: Sequence[str]) -> list[bool]:
        with self._lock:
            return [key in self._objects for key in keys]

    def remove(self, keys: Sequence[str]) -> None:
        with self._lock:
            for key in keys:
                self._objects.pop(key, None)

    def remove_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [key for key in self._objects if key.startswith(prefix)]
            for key in keys:
                del self._objects[key]
            return len(keys)


class MooncakeAuxObjectStore(AuxObjectStore):
    """Objects in a Mooncake distributed store.

    Objects are hard-pinned: the store never evicts them, so a consumer can
    read every key it was given until it removes them. A full pool fails the
    write instead of dropping objects that were already delivered.
    """

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(
        cls,
        config: MooncakeStoreConfig | None = None,
        local_hostname: str | None = None,
    ) -> MooncakeAuxObjectStore:
        """Connect with `config`, by default read from `MOONCAKE_CONFIG_PATH`."""
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to deliver auxiliary outputs through Mooncake."
            ) from e
        from vllm.distributed.mooncake_store import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )
        from vllm.utils.network_utils import get_ip

        store = MooncakeDistributedStore()
        setup_mooncake_store(
            store,
            config or MooncakeStoreConfig.load_from_config(),
            local_hostname or get_ip(),
        )
        replicate_config = ReplicateConfig()
        replicate_config.with_hard_pin = True
        return cls(store, replicate_config)

    def put_batch(
        self, keys: Sequence[str], values: Sequence[bytes | memoryview]
    ) -> None:
        if len(keys) != len(values):
            raise ValueError("key and value counts differ")
        if not keys:
            return
        if self._store.put_batch(list(keys), list(values), self._replicate_config):
            missing = [key for key, ok in zip(keys, self.exists(keys)) if not ok]
            if missing:
                raise AuxObjectStoreError(
                    f"Mooncake failed to write {len(missing)} of {len(keys)} "
                    f"auxiliary-output objects, first {missing[0]!r}"
                )

    def get(self, key: str) -> bytes | None:
        return self._store.get(key) or None

    def exists(self, keys: Sequence[str]) -> list[bool]:
        if not keys:
            return []
        return [code == 1 for code in self._store.batch_is_exist(list(keys))]

    def remove(self, keys: Sequence[str]) -> None:
        if keys:
            self._store.batch_remove(list(keys), True)

    def remove_prefix(self, prefix: str) -> int:
        removed = self._store.remove_by_regex("^" + re.escape(prefix), True)
        if removed < 0:
            raise AuxObjectStoreError(
                f"Mooncake failed to remove objects under {prefix!r}: {removed}"
            )
        return removed

    def close(self) -> None:
        self._store.close()


def create_aux_object_store(config: dict[str, Any]) -> AuxObjectStore:
    """Create the store named by `config["backend"]`."""
    if config["backend"] == "mooncake":
        return MooncakeAuxObjectStore.create()
    if config["backend"] == "local":
        return LocalAuxObjectStore()
    raise ValueError(f"Unknown auxiliary-output store backend {config['backend']!r}")
