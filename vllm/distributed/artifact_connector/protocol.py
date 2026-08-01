# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler/worker protocol for execution-artifact publication."""

from dataclasses import dataclass, field
from typing import Protocol


class ArtifactWriteTask(Protocol):
    """Asynchronous device-to-host artifact publication."""

    def start_copy(self) -> None: ...

    def finalize(self) -> None: ...


@dataclass(frozen=True)
class ArtifactCommitRequest:
    """Newly completed full blocks that can be published before finalize."""

    request_id: str
    weight_version: str
    block_hashes: list[bytes]
    block_start: int
    hash_block_size: int


@dataclass(frozen=True)
class ArtifactFinalizeRequest:
    """One terminal request whose staged routing must be made immutable."""

    request_id: str
    request_attempt_id: str
    weight_version: str
    block_hashes: list[bytes]
    token_end: int
    hash_block_size: int


@dataclass
class ArtifactConnectorMetadata:
    """Artifact work sent from the scheduler to the authoritative worker."""

    commits: list[ArtifactCommitRequest] = field(default_factory=list)
    finalizes: list[ArtifactFinalizeRequest] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactFinalizeResult:
    """Worker acknowledgement for one terminal artifact."""

    request_id: str
    request_attempt_id: str
    keys: list[str]


@dataclass
class ArtifactConnectorOutput:
    """Artifact acknowledgements sent from the worker to the scheduler."""

    results: list[ArtifactFinalizeResult] = field(default_factory=list)
