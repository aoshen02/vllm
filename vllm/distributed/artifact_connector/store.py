# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend boundary for immutable execution-artifact objects."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ArtifactObject:
    """One immutable, self-describing object."""

    key: str
    payload: bytes


@dataclass(frozen=True)
class ArtifactPutResult:
    """Per-object result from a batch publication."""

    key: str
    error: str | None = None


class ArtifactStoreError(RuntimeError):
    """Base class for artifact-store failures."""


class ArtifactCapacityError(ArtifactStoreError):
    """The artifact store cannot retain another object."""


class ArtifactCorruptionError(ArtifactStoreError):
    """An artifact object failed structural or checksum validation."""


class ArtifactNotFoundError(ArtifactStoreError):
    """A requested artifact object is not present."""


class ArtifactStore(Protocol):
    """Opaque byte-object operations required by the request core."""

    backend_name: str
    store_id: str

    def put(self, objects: list[ArtifactObject]) -> list[ArtifactPutResult]: ...

    def exists(self, keys: list[str]) -> list[bool]: ...

    def get(self, keys: list[str]) -> list[bytes]: ...

    def close(self) -> None: ...
