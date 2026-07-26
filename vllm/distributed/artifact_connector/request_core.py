# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-independent request lifecycle for execution artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.protocol import (
    ArtifactBlockRef,
    ArtifactCommitRequest,
    ArtifactFinalizeRequest,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactArray,
    ArtifactCorruptionError,
    ArtifactStore,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        RoutedExpertsWorkerWriter,
    )

_FIELD_NAME = "routed_experts"
_REUSE_POLICY = "prefix_block"
_LOGICAL_COORDINATE = "executed_token"


def _digest(prefix: bytes, *values: bytes) -> str:
    digest = hashlib.sha256(prefix)
    for value in values:
        digest.update(b"\0")
        digest.update(value)
    return digest.hexdigest()


@dataclass
class _RequestState:
    sample_id: str
    profile_id: str
    next_full_start: int | None = None
    segments: dict[int, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class PreparedCommit:
    """One validated commit whose payload has not been published yet."""

    request: ArtifactCommitRequest
    blocks: list[ArtifactArray]
    segments: list[dict[str, Any]]


@dataclass(frozen=True)
class FinalizedArtifact:
    """Terminal publication result returned to the worker connector."""

    sample_id: str
    manifest_sha256: str
    value: np.ndarray | None


def materialize_routed_experts(store: ArtifactStore, sample_id: str) -> np.ndarray:
    """Materialize the routed-experts field using the common manifest."""
    manifest = store.read_manifest(sample_id)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("store_id") != store.store_id
        or manifest.get("artifact_sample_id") != sample_id
    ):
        raise ArtifactCorruptionError("invalid artifact manifest identity")
    fields = manifest.get("fields")
    field_manifest = fields.get(_FIELD_NAME) if isinstance(fields, dict) else None
    if not isinstance(field_manifest, dict):
        raise ArtifactCorruptionError("routed-experts field is missing")
    try:
        dtype = np.dtype(field_manifest["dtype"])
        shape = tuple(int(value) for value in field_manifest["shape"])
        segments = field_manifest["segments"]
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactCorruptionError(
            "invalid routed-experts field manifest"
        ) from error
    if not isinstance(segments, list):
        raise ArtifactCorruptionError("artifact segments must be a list")
    output = np.empty(shape, dtype=dtype)
    covered = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ArtifactCorruptionError("invalid artifact segment")
        kind = segment.get("kind")
        object_id = segment.get("object_id")
        output_start = segment.get("output_start")
        valid_len = segment.get("valid_len")
        if (
            kind not in ("block", "tail")
            or not isinstance(object_id, str)
            or not isinstance(output_start, int)
            or not isinstance(valid_len, int)
            or output_start != covered
            or valid_len <= 0
        ):
            raise ArtifactCorruptionError("invalid artifact segment layout")
        array = store.read_array(kind, object_id)
        if array.dtype != dtype or array.shape[0] != valid_len:
            raise ArtifactCorruptionError("artifact segment shape mismatch")
        output[output_start : output_start + valid_len] = array
        covered += valid_len
    if covered != shape[0]:
        raise ArtifactCorruptionError(
            "artifact segments do not cover the terminal output"
        )
    return output


class ArtifactRequestCore:
    """Own the single request state machine shared by all stores."""

    def __init__(
        self,
        store: ArtifactStore,
        writer: RoutedExpertsWorkerWriter,
        *,
        namespace: str,
        inline_value: bool,
    ) -> None:
        self.store = store
        self.writer = writer
        self.namespace = namespace
        self.inline_value = inline_value
        self.policy_epoch = 0
        self._states: dict[str, _RequestState] = {}

    def advance_policy_epoch(self) -> None:
        """Prevent block reuse after an in-place model weight update."""
        self.policy_epoch += 1

    def _profile_id(self, hash_block_size: int, policy_epoch: int) -> str:
        profile = {
            "schema_version": 2,
            "field": _FIELD_NAME,
            "namespace": self.namespace,
            "dtype": self.writer.dtype.str,
            "shape_per_token": list(self.writer.shape_per_token),
            "reuse_policy": _REUSE_POLICY,
            "logical_coordinate": _LOGICAL_COORDINATE,
            "hash_block_size": hash_block_size,
            "policy_epoch": policy_epoch,
        }
        return hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _state(
        self,
        request_id: str,
        sample_id: str,
        *,
        hash_block_size: int,
        policy_epoch: int,
    ) -> _RequestState:
        if policy_epoch != self.policy_epoch:
            raise RuntimeError(
                "artifact policy epoch mismatch: "
                f"scheduler={policy_epoch}, worker={self.policy_epoch}"
            )
        profile_id = self._profile_id(hash_block_size, policy_epoch)
        state = self._states.get(request_id)
        if state is None:
            state = _RequestState(sample_id=sample_id, profile_id=profile_id)
            self._states[request_id] = state
        elif state.sample_id != sample_id:
            raise RuntimeError(
                f"artifact sample changed for request {request_id}: "
                f"{state.sample_id} != {sample_id}"
            )
        elif state.profile_id != profile_id:
            raise RuntimeError(
                f"artifact field profile changed for request {request_id}"
            )
        return state

    def _block_object_id(self, profile_id: str, block_hash: bytes) -> str:
        return _digest(
            b"vllm.artifact.field-block.v2",
            _FIELD_NAME.encode(),
            profile_id.encode(),
            block_hash,
        )

    def _tail_object_id(
        self, sample_id: str, source_start: int, source_end: int
    ) -> str:
        return _digest(
            b"vllm.artifact.field-tail.v2",
            _FIELD_NAME.encode(),
            sample_id.encode(),
            source_start.to_bytes(8, "big"),
            source_end.to_bytes(8, "big"),
        )

    def prepare_commit(self, request: ArtifactCommitRequest) -> PreparedCommit:
        """Validate and assemble newly completed full blocks."""
        if (
            request.block_start < 0
            or request.block_end <= request.block_start
            or request.block_start % request.hash_block_size
            or request.block_end % request.hash_block_size
        ):
            raise ValueError(
                "invalid artifact full-block range: "
                f"[{request.block_start}, {request.block_end})"
            )
        state = self._state(
            request.request_id,
            request.artifact_sample_id,
            hash_block_size=request.hash_block_size,
            policy_epoch=request.policy_epoch,
        )
        if state.error is not None:
            raise RuntimeError(state.error)
        if state.next_full_start is None:
            state.next_full_start = request.block_start
        if request.block_start != state.next_full_start:
            raise RuntimeError(
                "non-contiguous artifact block commit: "
                f"expected={state.next_full_start}, got={request.block_start}"
            )
        array = self.writer.read_token_range(
            request.block_ids,
            token_start=request.block_start,
            token_end=request.block_end,
            block_size=request.physical_block_size,
        )
        if (
            array.dtype != self.writer.dtype
            or array.shape[1:] != self.writer.shape_per_token
        ):
            raise RuntimeError("routed-experts capture profile changed")

        blocks: list[ArtifactArray] = []
        segments: list[dict[str, Any]] = []
        for source_start in range(
            request.block_start, request.block_end, request.hash_block_size
        ):
            block_index = source_start // request.hash_block_size
            if block_index >= len(request.block_hashes):
                raise RuntimeError(
                    "missing KV-compatible block hash for routed-experts artifact: "
                    f"request={request.request_id}, block_index={block_index}, "
                    f"num_hashes={len(request.block_hashes)}"
                )
            block_hash = request.block_hashes[block_index]
            object_id = self._block_object_id(state.profile_id, block_hash)
            local_start = source_start - request.block_start
            block_array = array[local_start : local_start + request.hash_block_size]
            blocks.append(
                ArtifactArray(
                    object_id=object_id,
                    array=block_array,
                    metadata={
                        "namespace": self.namespace,
                        "field": _FIELD_NAME,
                        "field_profile_id": state.profile_id,
                        "kv_block_hash": block_hash.hex(),
                        "source_token_start": source_start,
                        "valid_len": request.hash_block_size,
                    },
                )
            )
            segments.append(
                {
                    "kind": "block",
                    "object_id": object_id,
                    "kv_block_hash": block_hash.hex(),
                    "source_token_start": source_start,
                    "valid_len": request.hash_block_size,
                }
            )
        state.next_full_start = request.block_end
        return PreparedCommit(request=request, blocks=blocks, segments=segments)

    def publish_commits(self, commits: list[PreparedCommit]) -> None:
        """Publish a worker-step batch and attach its segments to requests."""
        self.store.put_blocks([block for commit in commits for block in commit.blocks])
        for commit in commits:
            state = self._states[commit.request.request_id]
            state.segments.update(
                (segment["source_token_start"], segment) for segment in commit.segments
            )

    def mark_error(self, request_id: str, message: str) -> None:
        state = self._states.get(request_id)
        if state is not None:
            state.error = message

    def _bind_cached_blocks(
        self,
        state: _RequestState,
        cached_blocks: list[ArtifactBlockRef],
        hash_block_size: int,
    ) -> None:
        object_ids: list[str] = []
        new_segments: list[tuple[int, dict[str, Any]]] = []
        for cached in cached_blocks:
            source_start = cached.block_index * hash_block_size
            if source_start in state.segments:
                continue
            object_id = self._block_object_id(state.profile_id, cached.block_hash)
            object_ids.append(object_id)
            new_segments.append(
                (
                    source_start,
                    {
                        "kind": "block",
                        "object_id": object_id,
                        "kv_block_hash": cached.block_hash.hex(),
                        "source_token_start": source_start,
                        "valid_len": hash_block_size,
                    },
                )
            )
        self.store.retain_blocks(object_ids)
        state.segments.update(new_segments)

    def _put_tail(
        self,
        request: ArtifactFinalizeRequest,
        state: _RequestState,
        *,
        source_start: int,
        source_end: int,
        output_start: int,
    ) -> dict[str, Any]:
        array = self.writer.read_token_range(
            request.block_ids,
            token_start=source_start,
            token_end=source_end,
            block_size=request.physical_block_size,
        )
        object_id = self._tail_object_id(
            request.artifact_sample_id, source_start, source_end
        )
        self.store.put_array(
            "tail",
            object_id,
            array,
            {
                "namespace": self.namespace,
                "field": _FIELD_NAME,
                "field_profile_id": state.profile_id,
                "artifact_sample_id": request.artifact_sample_id,
                "source_token_start": source_start,
                "valid_len": source_end - source_start,
            },
        )
        return {
            "kind": "tail",
            "object_id": object_id,
            "output_start": output_start,
            "source_token_start": source_start,
            "valid_len": source_end - source_start,
        }

    def finalize(self, request: ArtifactFinalizeRequest) -> FinalizedArtifact:
        """Seal one manifest and optionally materialize its inline value."""
        if request.token_start < 0 or request.token_end <= request.token_start:
            raise ValueError(
                "invalid artifact token range: "
                f"[{request.token_start}, {request.token_end})"
            )
        state = self._state(
            request.request_id,
            request.artifact_sample_id,
            hash_block_size=request.hash_block_size,
            policy_epoch=request.policy_epoch,
        )
        if state.error is not None:
            raise RuntimeError(state.error)
        self._bind_cached_blocks(state, request.cached_blocks, request.hash_block_size)

        block_size = request.hash_block_size
        first_full_start = (
            (request.token_start + block_size - 1) // block_size
        ) * block_size
        full_end = request.token_end // block_size * block_size
        segments: list[dict[str, Any]] = []
        output_start = 0

        if request.token_start < min(first_full_start, request.token_end):
            head_end = min(first_full_start, request.token_end)
            segments.append(
                self._put_tail(
                    request,
                    state,
                    source_start=request.token_start,
                    source_end=head_end,
                    output_start=output_start,
                )
            )
            output_start += head_end - request.token_start

        for source_start in range(first_full_start, full_end, block_size):
            segment = state.segments.get(source_start)
            if segment is None:
                raise RuntimeError(
                    "terminal artifact is missing a reusable full block: "
                    f"request={request.request_id}, block_start={source_start}"
                )
            segments.append({**segment, "output_start": output_start})
            output_start += block_size

        tail_start = max(full_end, request.token_start)
        if tail_start < request.token_end:
            segments.append(
                self._put_tail(
                    request,
                    state,
                    source_start=tail_start,
                    source_end=request.token_end,
                    output_start=output_start,
                )
            )
            output_start += request.token_end - tail_start

        expected_len = request.token_end - request.token_start
        if output_start != expected_len:
            raise RuntimeError("artifact segments do not cover the terminal range")
        field_manifest = {
            "reuse_policy": _REUSE_POLICY,
            "logical_coordinate": _LOGICAL_COORDINATE,
            "field_profile_id": state.profile_id,
            "dtype": self.writer.dtype.str,
            "shape": [expected_len, *self.writer.shape_per_token],
            "source_token_start": request.token_start,
            "source_token_end": request.token_end,
            "segments": segments,
        }
        manifest = {
            "schema_version": 2,
            "store_id": self.store.store_id,
            "artifact_sample_id": request.artifact_sample_id,
            "namespace": self.namespace,
            "policy_epoch": request.policy_epoch,
            "request_id": request.request_id,
            "terminal_boundary": {
                "coordinate": _LOGICAL_COORDINATE,
                "start": request.token_start,
                "end": request.token_end,
            },
            "fields": {_FIELD_NAME: field_manifest},
            "created_at_unix_ns": time.time_ns(),
        }
        manifest_sha256 = self.store.put_manifest(request.artifact_sample_id, manifest)
        value = (
            self.materialize(request.artifact_sample_id) if self.inline_value else None
        )
        self.discard(request.request_id)
        return FinalizedArtifact(
            sample_id=request.artifact_sample_id,
            manifest_sha256=manifest_sha256,
            value=value,
        )

    def materialize(self, sample_id: str) -> np.ndarray:
        """Materialize the routed-experts field using the common manifest."""
        return materialize_routed_experts(self.store, sample_id)

    def discard(self, request_id: str) -> None:
        """Drop one request state while keeping immutable published blocks."""
        state = self._states.pop(request_id, None)
        if state is not None:
            self.store.release_blocks(
                [segment["object_id"] for segment in state.segments.values()]
            )

    def close(self) -> None:
        for request_id in list(self._states):
            self.discard(request_id)
        self.store.close()
