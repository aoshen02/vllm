# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt-logprobs artifacts aligned to KV-compatible token blocks."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from vllm.distributed.artifact_connector.protocol import (
    PromptLogprobsArtifactRequest,
)
from vllm.distributed.artifact_connector.store import ArtifactArray, ArtifactStore


@dataclass(frozen=True)
class PromptLogprobsArrays:
    """Owned CPU arrays representing vLLM prompt-logprobs tensors."""

    token_ids: np.ndarray
    logprobs: np.ndarray
    ranks: np.ndarray
    boundary_hidden: np.ndarray | None = None


@dataclass
class _PromptLogprobsState:
    spec: PromptLogprobsArtifactRequest
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    segments: dict[int, dict[str, object]] = field(default_factory=dict)


class PromptLogprobsArtifactManager:
    """Publish and restore mandatory prompt-logprobs block bundles."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        namespace: str,
        logprobs_mode: str,
    ) -> None:
        self.store = store
        self.namespace = namespace
        self.logprobs_mode = logprobs_mode
        self._states: dict[str, _PromptLogprobsState] = {}

    @staticmethod
    def _digest(prefix: bytes, *values: bytes) -> str:
        digest = hashlib.sha256(prefix)
        for value in values:
            digest.update(b"\0")
            digest.update(value)
        return digest.hexdigest()

    def _profile_id(self, spec: PromptLogprobsArtifactRequest) -> str:
        profile = {
            "schema_version": 1,
            "field": "prompt_logprobs",
            "namespace": self.namespace,
            "logprobs_mode": self.logprobs_mode,
            "num_prompt_logprobs": spec.num_prompt_logprobs,
            "hash_block_size": spec.hash_block_size,
            "policy_epoch": spec.policy_epoch,
            "token_ids_dtype": "int32",
            "logprobs_dtype": "float32",
            "ranks_dtype": "int32",
            "boundary_hidden_dtype": "float32",
        }
        encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _block_object_id(
        self, spec: PromptLogprobsArtifactRequest, block_index: int
    ) -> str:
        if block_index >= len(spec.block_hashes):
            raise RuntimeError(
                "missing KV-compatible hash for prompt-logprobs block: "
                f"request={spec.request_id}, block_index={block_index}"
            )
        return self._digest(
            b"vllm.artifact.prompt_logprobs.block.v1",
            self._profile_id(spec).encode(),
            spec.block_hashes[block_index],
        )

    def _state(self, spec: PromptLogprobsArtifactRequest) -> _PromptLogprobsState:
        state = self._states.get(spec.request_id)
        if state is None:
            state = _PromptLogprobsState(spec=spec)
            self._states[spec.request_id] = state
        elif state.spec != spec:
            raise RuntimeError(
                "prompt-logprobs artifact profile changed within request "
                f"{spec.request_id}"
            )
        return state

    @staticmethod
    def _row_slice(block_index: int, block_size: int) -> slice:
        if block_index == 0:
            return slice(0, block_size - 1)
        start = block_index * block_size - 1
        return slice(start, start + block_size)

    @staticmethod
    def _pack(
        token_ids: np.ndarray,
        logprobs: np.ndarray,
        ranks: np.ndarray,
        boundary_hidden: np.ndarray | None,
    ) -> np.ndarray:
        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        logprobs = np.ascontiguousarray(logprobs, dtype=np.float32)
        ranks = np.ascontiguousarray(ranks, dtype=np.int32)
        if token_ids.shape != logprobs.shape:
            raise ValueError("prompt-logprobs token/value shapes differ")
        if ranks.shape != (token_ids.shape[0],):
            raise ValueError("prompt-logprobs rank shape differs")
        parts = [
            token_ids.view(np.uint8).reshape(-1),
            logprobs.view(np.uint8).reshape(-1),
            ranks.view(np.uint8).reshape(-1),
        ]
        if boundary_hidden is not None:
            hidden = np.ascontiguousarray(boundary_hidden, dtype=np.float32)
            if hidden.ndim != 1:
                raise ValueError("boundary hidden state must be one-dimensional")
            parts.append(hidden.view(np.uint8).reshape(-1))
        return np.concatenate(parts)

    @staticmethod
    def _decode(
        payload: np.ndarray,
        *,
        num_rows: int,
        width: int,
        has_boundary_hidden: bool,
    ) -> PromptLogprobsArrays:
        if payload.dtype != np.uint8 or payload.ndim != 1:
            raise RuntimeError("invalid prompt-logprobs bundle payload")
        matrix_bytes = num_rows * width * 4
        ranks_bytes = num_rows * 4
        fixed_bytes = matrix_bytes * 2 + ranks_bytes
        if payload.nbytes < fixed_bytes:
            raise RuntimeError("truncated prompt-logprobs bundle")
        if not has_boundary_hidden and payload.nbytes != fixed_bytes:
            raise RuntimeError("unexpected prompt-logprobs tail payload")
        hidden_bytes = payload.nbytes - fixed_bytes
        if has_boundary_hidden and (hidden_bytes <= 0 or hidden_bytes % 4):
            raise RuntimeError("invalid prompt-logprobs boundary hidden state")

        offset = 0
        token_ids = payload[offset : offset + matrix_bytes].view(np.int32)
        token_ids = token_ids.reshape(num_rows, width).copy()
        offset += matrix_bytes
        logprobs = payload[offset : offset + matrix_bytes].view(np.float32)
        logprobs = logprobs.reshape(num_rows, width).copy()
        offset += matrix_bytes
        ranks = payload[offset : offset + ranks_bytes].view(np.int32).copy()
        offset += ranks_bytes
        boundary_hidden = None
        if has_boundary_hidden:
            boundary_hidden = payload[offset:].view(np.float32).copy()
        return PromptLogprobsArrays(
            token_ids=token_ids,
            logprobs=logprobs,
            ranks=ranks,
            boundary_hidden=boundary_hidden,
        )

    def restore_cached_prefix(
        self, spec: PromptLogprobsArtifactRequest
    ) -> PromptLogprobsArrays | None:
        """Restore every mandatory artifact for a KV prefix-cache hit."""
        state = self._state(spec)
        cached_tokens = spec.num_cached_tokens
        if cached_tokens == 0:
            return None
        block_size = spec.hash_block_size
        if cached_tokens < 0 or cached_tokens % block_size:
            raise RuntimeError(
                "prompt-logprobs KV hit is not hash-block aligned: "
                f"request={spec.request_id}, cached_tokens={cached_tokens}, "
                f"block_size={block_size}"
            )
        num_blocks = cached_tokens // block_size
        if num_blocks > len(spec.block_hashes):
            raise RuntimeError(
                "prompt-logprobs KV hit exceeds available block hashes: "
                f"request={spec.request_id}, num_blocks={num_blocks}, "
                f"num_hashes={len(spec.block_hashes)}"
            )

        restored: list[PromptLogprobsArrays] = []
        object_ids: list[str] = []
        width = spec.num_prompt_logprobs + 1
        try:
            for block_index in range(num_blocks):
                object_id = self._block_object_id(spec, block_index)
                payload = self.store.read_array("block", object_id)
                row_slice = self._row_slice(block_index, block_size)
                num_rows = row_slice.stop - row_slice.start
                restored.append(
                    self._decode(
                        payload,
                        num_rows=num_rows,
                        width=width,
                        has_boundary_hidden=True,
                    )
                )
                object_ids.append(object_id)
        except (FileNotFoundError, OSError) as error:
            self.discard(spec.request_id)
            raise RuntimeError(
                "mandatory prompt-logprobs artifact is missing for a KV "
                f"prefix-cache hit: request={spec.request_id}"
            ) from error

        self.store.retain_blocks(object_ids)
        for block_index, object_id in enumerate(object_ids):
            row_slice = self._row_slice(block_index, block_size)
            state.segments[block_index] = {
                "kind": "block",
                "object_id": object_id,
                "token_block_index": block_index,
                "valid_len": row_slice.stop - row_slice.start,
            }
        return PromptLogprobsArrays(
            token_ids=np.concatenate([item.token_ids for item in restored]),
            logprobs=np.concatenate([item.logprobs for item in restored]),
            ranks=np.concatenate([item.ranks for item in restored]),
            boundary_hidden=restored[-1].boundary_hidden,
        )

    def pending_block_indices(
        self, spec: PromptLogprobsArtifactRequest, completed_token_end: int
    ) -> list[int]:
        state = self._state(spec)
        num_completed_blocks = completed_token_end // spec.hash_block_size
        return [
            block_index
            for block_index in range(num_completed_blocks)
            if block_index not in state.segments
        ]

    def store_completed_blocks(
        self,
        spec: PromptLogprobsArtifactRequest,
        arrays: PromptLogprobsArrays,
        completed_token_end: int,
        boundary_hidden: dict[int, np.ndarray],
    ) -> None:
        """Publish all newly completed token blocks in one store batch."""
        state = self._state(spec)
        block_size = spec.hash_block_size
        width = spec.num_prompt_logprobs + 1
        pending = self.pending_block_indices(spec, completed_token_end)
        blocks: list[ArtifactArray] = []
        segments: list[tuple[int, dict[str, object]]] = []
        profile_id = self._profile_id(spec)
        for block_index in pending:
            hidden = boundary_hidden.get(block_index)
            if hidden is None:
                raise RuntimeError(
                    "missing boundary hidden state for completed prompt-logprobs "
                    f"block: request={spec.request_id}, block={block_index}"
                )
            row_slice = self._row_slice(block_index, block_size)
            if row_slice.stop > arrays.token_ids.shape[0]:
                raise RuntimeError(
                    "prompt-logprobs rows are incomplete for block: "
                    f"request={spec.request_id}, block={block_index}"
                )
            object_id = self._block_object_id(spec, block_index)
            payload = self._pack(
                arrays.token_ids[row_slice],
                arrays.logprobs[row_slice],
                arrays.ranks[row_slice],
                hidden,
            )
            blocks.append(
                ArtifactArray(
                    object_id=object_id,
                    array=payload,
                    metadata={
                        "namespace": self.namespace,
                        "profile_id": profile_id,
                        "field": "prompt_logprobs",
                        "kv_block_hash": spec.block_hashes[block_index].hex(),
                        "token_block_index": block_index,
                        "valid_len": row_slice.stop - row_slice.start,
                        "width": width,
                    },
                )
            )
            segments.append(
                (
                    block_index,
                    {
                        "kind": "block",
                        "object_id": object_id,
                        "token_block_index": block_index,
                        "valid_len": row_slice.stop - row_slice.start,
                    },
                )
            )
        if blocks:
            self.store.put_blocks(blocks)
            state.segments.update(segments)

    def _materialize(
        self, state: _PromptLogprobsState, manifest_sha256: str
    ) -> PromptLogprobsArrays:
        manifest = self.store.read_manifest(state.sample_id)
        if manifest.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("prompt-logprobs manifest checksum mismatch")
        width = state.spec.num_prompt_logprobs + 1
        arrays: list[PromptLogprobsArrays] = []
        for segment in manifest["segments"]:
            kind = segment["kind"]
            payload = self.store.read_array(kind, segment["object_id"])
            arrays.append(
                self._decode(
                    payload,
                    num_rows=segment["valid_len"],
                    width=width,
                    has_boundary_hidden=kind == "block",
                )
            )
        expected_rows = state.spec.num_prompt_tokens - 1
        if arrays:
            result = PromptLogprobsArrays(
                token_ids=np.concatenate([item.token_ids for item in arrays]),
                logprobs=np.concatenate([item.logprobs for item in arrays]),
                ranks=np.concatenate([item.ranks for item in arrays]),
            )
        else:
            width = state.spec.num_prompt_logprobs + 1
            result = PromptLogprobsArrays(
                token_ids=np.empty((0, width), dtype=np.int32),
                logprobs=np.empty((0, width), dtype=np.float32),
                ranks=np.empty((0,), dtype=np.int32),
            )
        if result.token_ids.shape != (expected_rows, width):
            raise RuntimeError(
                "prompt-logprobs manifest does not cover the prompt: "
                f"expected={(expected_rows, width)}, "
                f"actual={result.token_ids.shape}"
            )
        return result

    def finalize(
        self,
        spec: PromptLogprobsArtifactRequest,
        arrays: PromptLogprobsArrays,
    ) -> PromptLogprobsArrays:
        """Publish the request-local tail and materialize the SHM result."""
        state = self._state(spec)
        block_size = spec.hash_block_size
        num_full_blocks = spec.num_prompt_tokens // block_size
        missing = [
            block_index
            for block_index in range(num_full_blocks)
            if block_index not in state.segments
        ]
        if missing:
            raise RuntimeError(
                "prompt-logprobs manifest is missing full blocks: "
                f"request={spec.request_id}, blocks={missing}"
            )

        segments = [state.segments[index] for index in range(num_full_blocks)]
        tail_start = max(num_full_blocks * block_size - 1, 0)
        tail_end = spec.num_prompt_tokens - 1
        if tail_start < tail_end:
            tail_id = self._digest(
                b"vllm.artifact.prompt_logprobs.tail.v1",
                state.sample_id.encode(),
                tail_start.to_bytes(8, "big"),
                tail_end.to_bytes(8, "big"),
            )
            payload = self._pack(
                arrays.token_ids[tail_start:tail_end],
                arrays.logprobs[tail_start:tail_end],
                arrays.ranks[tail_start:tail_end],
                None,
            )
            self.store.put_array(
                "tail",
                tail_id,
                payload,
                {
                    "namespace": self.namespace,
                    "profile_id": self._profile_id(spec),
                    "field": "prompt_logprobs",
                    "artifact_sample_id": state.sample_id,
                    "valid_len": tail_end - tail_start,
                    "width": spec.num_prompt_logprobs + 1,
                },
            )
            segments.append(
                {
                    "kind": "tail",
                    "object_id": tail_id,
                    "valid_len": tail_end - tail_start,
                }
            )

        manifest = {
            "schema_version": 1,
            "backend": "shm",
            "store_id": self.store.store_id,
            "artifact_sample_id": state.sample_id,
            "profile_id": self._profile_id(spec),
            "namespace": self.namespace,
            "policy_epoch": spec.policy_epoch,
            "request_id": spec.request_id,
            "field": "prompt_logprobs",
            "num_prompt_tokens": spec.num_prompt_tokens,
            "num_prompt_logprobs": spec.num_prompt_logprobs,
            "num_cached_tokens": spec.num_cached_tokens,
            "segments": segments,
            "created_at_unix_ns": time.time_ns(),
        }
        manifest_sha256 = self.store.put_manifest(state.sample_id, manifest)
        result = self._materialize(state, manifest_sha256)
        self.store.release_blocks(
            [cast(str, segment["object_id"]) for segment in state.segments.values()]
        )
        self._states.pop(spec.request_id, None)
        return result

    def discard(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state is not None:
            self.store.release_blocks(
                [cast(str, segment["object_id"]) for segment in state.segments.values()]
            )
