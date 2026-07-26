# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler/worker protocol for execution-artifact publication."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ArtifactCommitRequest:
    """Newly completed full blocks that can be published before finalize."""

    operation_id: str
    request_id: str
    request_attempt_id: str
    block_hashes: list[bytes]
    block_start: int
    block_end: int
    hash_block_size: int


@dataclass(frozen=True)
class ArtifactFinalizeRequest:
    """One terminal request whose staged routing must be made immutable."""

    request_id: str
    request_attempt_id: str
    block_hashes: list[bytes]
    token_end: int
    hash_block_size: int


@dataclass
class ArtifactConnectorMetadata:
    """Artifact work sent from the scheduler to the authoritative worker."""

    finalizes: list[ArtifactFinalizeRequest] = field(default_factory=list)
    commits: list[ArtifactCommitRequest] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactCommitResult:
    """Worker acknowledgement that one full-block publication was started."""

    operation_id: str
    request_id: str
    request_attempt_id: str
    block_end: int
    block_keys: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ArtifactFinalizeResult:
    """Worker acknowledgement for one terminal artifact."""

    request_id: str
    request_attempt_id: str
    artifact_keys: list[str] | None = None
    value: np.ndarray | None = None
    error: str | None = None


@dataclass
class ArtifactConnectorOutput:
    """Artifact acknowledgements sent from the worker to the scheduler."""

    results: list[ArtifactFinalizeResult] = field(default_factory=list)
    commit_results: list[ArtifactCommitResult] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.results or self.commit_results)
