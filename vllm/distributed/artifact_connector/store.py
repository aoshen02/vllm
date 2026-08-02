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


class ArtifactStoreError(RuntimeError):
    """Base class for artifact-store failures."""


class ArtifactCapacityError(ArtifactStoreError):
    """The artifact store cannot retain another object."""


class ArtifactCorruptionError(ArtifactStoreError):
    """An artifact object failed structural or checksum validation."""


class ArtifactNotFoundError(ArtifactStoreError):
    """A requested artifact object is not present."""


class ArtifactReader(Protocol):
    """Opaque byte-object reads used to materialize terminal artifacts."""

    def get(self, keys: list[str]) -> list[bytes]: ...

    def close(self) -> None: ...


class ArtifactStore(ArtifactReader, Protocol):
    """Artifact reader that can publish immutable objects."""

    def put(self, objects: list[ArtifactObject]) -> None: ...
