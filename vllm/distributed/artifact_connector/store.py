# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend boundary for immutable execution-artifact objects."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ArtifactArray:
    """One immutable array object prepared by the common assembly path."""

    object_id: str
    array: np.ndarray
    metadata: dict[str, Any]


class ArtifactStore(Protocol):
    """Storage operations required by the common artifact assembler."""

    store_id: str

    def put_blocks(self, blocks: list[ArtifactArray]) -> None: ...

    def retain_blocks(self, object_ids: list[str]) -> None: ...

    def release_blocks(self, object_ids: list[str]) -> None: ...

    def put_array(
        self,
        kind: str,
        object_id: str,
        array: np.ndarray,
        metadata: dict[str, Any],
    ) -> None: ...

    def put_manifest(self, sample_id: str, manifest: dict[str, Any]) -> str: ...

    def read_array(self, kind: str, object_id: str) -> np.ndarray: ...

    def read_manifest(self, sample_id: str) -> dict[str, Any]: ...

    def materialize(self, handle: dict[str, Any]) -> np.ndarray: ...

    def close(self) -> None: ...
