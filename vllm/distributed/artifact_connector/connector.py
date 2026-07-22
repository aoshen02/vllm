# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-specific connectors for routed-experts execution artifacts."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.prompt_logprobs import (
    PromptLogprobsArrays,
    PromptLogprobsArtifactManager,
)
from vllm.distributed.artifact_connector.protocol import (
    ArtifactCommitRequest,
    ArtifactCommitResult,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactDiscardRequest,
    ArtifactDiscardResult,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
    PromptLogprobsArtifactRequest,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactStore,
)
from vllm.distributed.artifact_connector.store import ArtifactArray, ArtifactStore
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        RoutedExpertsWorkerWriter,
    )

logger = init_logger(__name__)


@dataclass
class _SchedulerArtifactState:
    sample_id: str
    token_start: int
    next_full_end: int


class ArtifactSchedulerConnector:
    """Turn accepted-token progress into worker block commits."""

    def __init__(self) -> None:
        self._states: dict[str, _SchedulerArtifactState] = {}
        self._pending_commits: OrderedDict[str, ArtifactCommitRequest] = OrderedDict()
        self._inflight_commits: dict[str, ArtifactCommitRequest] = {}
        self._pending_finalizes: OrderedDict[str, ArtifactFinalizeRequest] = (
            OrderedDict()
        )
        self._inflight_finalizes: dict[str, ArtifactFinalizeRequest] = {}
        self._pending_discards: OrderedDict[str, ArtifactDiscardRequest] = OrderedDict()
        self._inflight_discards: dict[str, ArtifactDiscardRequest] = {}

    def _state(
        self, request_id: str, token_start: int, hash_block_size: int
    ) -> _SchedulerArtifactState:
        state = self._states.get(request_id)
        if state is None:
            first_full_start = (
                (token_start + hash_block_size - 1) // hash_block_size
            ) * hash_block_size
            state = _SchedulerArtifactState(
                sample_id=uuid.uuid4().hex,
                token_start=token_start,
                next_full_end=first_full_start,
            )
            self._states[request_id] = state
        elif state.token_start != token_start:
            raise RuntimeError(
                f"artifact token start changed for request {request_id}: "
                f"{state.token_start} != {token_start}"
            )
        return state

    @staticmethod
    def make_prompt_logprobs_request(
        *,
        request_id: str,
        block_hashes: list[bytes],
        num_prompt_tokens: int,
        num_prompt_logprobs: int,
        num_cached_tokens: int,
        hash_block_size: int,
        policy_epoch: int,
    ) -> PromptLogprobsArtifactRequest:
        """Describe the mandatory fixed-profile artifacts for one prompt."""
        return PromptLogprobsArtifactRequest(
            request_id=request_id,
            block_hashes=list(block_hashes),
            num_prompt_tokens=num_prompt_tokens,
            num_prompt_logprobs=num_prompt_logprobs,
            num_cached_tokens=num_cached_tokens,
            hash_block_size=hash_block_size,
            policy_epoch=policy_epoch,
        )

    def request_progress(
        self,
        *,
        request_id: str,
        block_ids: list[int],
        block_hashes: list[bytes],
        token_start: int,
        accepted_token_end: int,
        physical_block_size: int,
        hash_block_size: int,
        policy_epoch: int,
    ) -> str:
        """Queue all newly completed hash blocks in the accepted prefix."""
        if hash_block_size <= 0:
            raise ValueError("artifact hash block size must be positive")
        state = self._state(request_id, token_start, hash_block_size)
        full_end = accepted_token_end // hash_block_size * hash_block_size
        if full_end <= state.next_full_end:
            return state.sample_id
        operation_id = uuid.uuid4().hex
        self._pending_commits[operation_id] = ArtifactCommitRequest(
            operation_id=operation_id,
            request_id=request_id,
            artifact_sample_id=state.sample_id,
            block_ids=list(block_ids),
            block_hashes=list(block_hashes),
            block_start=state.next_full_end,
            block_end=full_end,
            physical_block_size=physical_block_size,
            hash_block_size=hash_block_size,
            policy_epoch=policy_epoch,
        )
        state.next_full_end = full_end
        return state.sample_id

    def request_finished(
        self,
        *,
        request_id: str,
        block_ids: list[int],
        block_hashes: list[bytes],
        token_start: int,
        token_end: int,
        physical_block_size: int,
        hash_block_size: int,
        policy_epoch: int,
    ) -> str:
        if (
            request_id in self._pending_finalizes
            or request_id in self._inflight_finalizes
        ):
            raise RuntimeError(f"artifact request is already pending: {request_id}")
        state = self._state(request_id, token_start, hash_block_size)
        self._pending_finalizes[request_id] = ArtifactFinalizeRequest(
            request_id=request_id,
            artifact_sample_id=state.sample_id,
            block_ids=list(block_ids),
            block_hashes=list(block_hashes),
            token_start=token_start,
            token_end=token_end,
            physical_block_size=physical_block_size,
            hash_block_size=hash_block_size,
            policy_epoch=policy_epoch,
        )
        return state.sample_id

    def request_discarded(self, request_id: str) -> None:
        state = self._states.get(request_id)
        if state is None or request_id in self._pending_discards:
            return
        operation_id = uuid.uuid4().hex
        self._pending_discards[request_id] = ArtifactDiscardRequest(
            operation_id=operation_id,
            request_id=request_id,
            artifact_sample_id=state.sample_id,
        )

    def build_connector_metadata(self) -> ArtifactConnectorMetadata | None:
        if not (
            self._pending_commits or self._pending_finalizes or self._pending_discards
        ):
            return None
        commits = list(self._pending_commits.values())
        requests = list(self._pending_finalizes.values())
        discards = list(self._pending_discards.values())
        self._pending_commits.clear()
        self._pending_finalizes.clear()
        self._pending_discards.clear()
        self._inflight_commits.update(
            (request.operation_id, request) for request in commits
        )
        self._inflight_finalizes.update(
            (request.request_id, request) for request in requests
        )
        self._inflight_discards.update(
            (request.operation_id, request) for request in discards
        )
        return ArtifactConnectorMetadata(
            requests=requests, commits=commits, discards=discards
        )

    def acknowledge(
        self, output: ArtifactConnectorOutput | None
    ) -> ArtifactConnectorOutput:
        if output is None:
            return ArtifactConnectorOutput()
        for commit_result in output.commit_results:
            commit_request = self._inflight_commits.pop(
                commit_result.operation_id, None
            )
            if commit_request is None or (
                commit_request.request_id != commit_result.request_id
                or commit_request.artifact_sample_id != commit_result.artifact_sample_id
                or commit_request.block_end != commit_result.block_end
            ):
                raise RuntimeError(
                    "worker acknowledged an unknown artifact block commit: "
                    f"{commit_result.operation_id}"
                )
        for finalize_result in output.results:
            finalize_request = self._inflight_finalizes.pop(
                finalize_result.request_id, None
            )
            if finalize_request is None:
                raise RuntimeError(
                    "worker acknowledged an unknown artifact request: "
                    f"{finalize_result.request_id}"
                )
            if (
                finalize_request.artifact_sample_id
                != finalize_result.artifact_sample_id
            ):
                raise RuntimeError(
                    "artifact acknowledgement sample mismatch for request "
                    f"{finalize_result.request_id}"
                )
            self._states.pop(finalize_result.request_id, None)
        for discard_result in output.discard_results:
            discard_request = self._inflight_discards.pop(
                discard_result.operation_id, None
            )
            if discard_request is None or (
                discard_request.request_id != discard_result.request_id
                or discard_request.artifact_sample_id
                != discard_result.artifact_sample_id
            ):
                raise RuntimeError(
                    "worker acknowledged an unknown artifact discard: "
                    f"{discard_result.operation_id}"
                )
            self._states.pop(discard_result.request_id, None)
        return output

    def has_unacked_commits(self, request_id: str) -> bool:
        return any(
            request.request_id == request_id
            for request in (
                *self._pending_commits.values(),
                *self._inflight_commits.values(),
            )
        )

    def has_pending_work(self) -> bool:
        return bool(
            self._pending_commits
            or self._inflight_commits
            or self._pending_finalizes
            or self._inflight_finalizes
            or self._pending_discards
            or self._inflight_discards
        )


@dataclass
class _WorkerArtifactState:
    sample_id: str
    profile_id: str | None = None
    dtype: str | None = None
    shape_per_token: tuple[int, ...] | None = None
    next_full_start: int | None = None
    segments: dict[int, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


class ArtifactWorkerConnector:
    """Assemble accepted routed-experts blocks and publish them incrementally."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        writer: RoutedExpertsWorkerWriter,
    ) -> None:
        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        self.writer = writer
        self.store: ArtifactStore = LocalSharedMemoryArtifactStore(
            config.shm_dir,
            vllm_config.instance_id,
            parallel_config.data_parallel_rank,
            max_bytes=config.max_shm_bytes,
            ttl_seconds=config.shm_ttl_seconds,
        )
        model_config = vllm_config.model_config
        namespace_input = {
            "model": model_config.model,
            "revision": model_config.revision,
            "tokenizer_revision": model_config.tokenizer_revision,
        }
        self.namespace = hashlib.sha256(
            json.dumps(
                namespace_input,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        self.policy_epoch = 0
        self._states: dict[str, _WorkerArtifactState] = {}
        self.prompt_logprobs = PromptLogprobsArtifactManager(
            self.store,
            namespace=self.namespace,
            logprobs_mode=model_config.logprobs_mode,
        )

    def advance_policy_epoch(self) -> None:
        """Prevent block-key reuse after an in-place model weight update."""
        self.policy_epoch += 1

    def restore_prompt_logprobs(
        self, request: PromptLogprobsArtifactRequest
    ) -> PromptLogprobsArrays | None:
        self._validate_prompt_logprobs_epoch(request)
        return self.prompt_logprobs.restore_cached_prefix(request)

    def pending_prompt_logprobs_blocks(
        self,
        request: PromptLogprobsArtifactRequest,
        completed_token_end: int,
    ) -> list[int]:
        self._validate_prompt_logprobs_epoch(request)
        return self.prompt_logprobs.pending_block_indices(request, completed_token_end)

    def store_prompt_logprobs_blocks(
        self,
        request: PromptLogprobsArtifactRequest,
        arrays: PromptLogprobsArrays,
        completed_token_end: int,
        boundary_hidden: dict[int, np.ndarray],
    ) -> None:
        self._validate_prompt_logprobs_epoch(request)
        self.prompt_logprobs.store_completed_blocks(
            request, arrays, completed_token_end, boundary_hidden
        )

    def finalize_prompt_logprobs(
        self,
        request: PromptLogprobsArtifactRequest,
        arrays: PromptLogprobsArrays,
    ) -> PromptLogprobsArrays:
        self._validate_prompt_logprobs_epoch(request)
        return self.prompt_logprobs.finalize(request, arrays)

    def discard_prompt_logprobs(self, request_id: str) -> None:
        self.prompt_logprobs.discard(request_id)

    def _validate_prompt_logprobs_epoch(
        self, request: PromptLogprobsArtifactRequest
    ) -> None:
        if request.policy_epoch != self.policy_epoch:
            raise RuntimeError(
                "prompt-logprobs artifact policy epoch mismatch: "
                f"scheduler={request.policy_epoch}, worker={self.policy_epoch}"
            )

    @staticmethod
    def _object_id(prefix: bytes, *values: bytes) -> str:
        digest = hashlib.sha256(prefix)
        for value in values:
            digest.update(b"\0")
            digest.update(value)
        return digest.hexdigest()

    def _state(self, request_id: str, sample_id: str) -> _WorkerArtifactState:
        state = self._states.get(request_id)
        if state is None:
            state = _WorkerArtifactState(sample_id=sample_id)
            self._states[request_id] = state
        elif state.sample_id != sample_id:
            raise RuntimeError(
                f"artifact sample changed for request {request_id}: "
                f"{state.sample_id} != {sample_id}"
            )
        return state

    def _drop_state(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state is not None:
            self.store.release_blocks(
                [segment["object_id"] for segment in state.segments.values()]
            )

    def _profile_id(
        self,
        *,
        dtype: np.dtype,
        shape_per_token: tuple[int, ...],
        hash_block_size: int,
        policy_epoch: int,
    ) -> str:
        profile = {
            "schema_version": 1,
            "field": "routed_experts",
            "namespace": self.namespace,
            "dtype": dtype.str,
            "shape_per_token": list(shape_per_token),
            "hash_block_size": hash_block_size,
            "policy_epoch": policy_epoch,
        }
        return hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _set_profile(
        self,
        state: _WorkerArtifactState,
        array: np.ndarray,
        *,
        hash_block_size: int,
        policy_epoch: int,
    ) -> str:
        profile_id = self._profile_id(
            dtype=array.dtype,
            shape_per_token=array.shape[1:],
            hash_block_size=hash_block_size,
            policy_epoch=policy_epoch,
        )
        if state.profile_id is None:
            state.profile_id = profile_id
            state.dtype = array.dtype.str
            state.shape_per_token = array.shape[1:]
        elif state.profile_id != profile_id:
            raise RuntimeError("artifact payload profile changed within a request")
        return profile_id

    def _prepare_commit(
        self, request: ArtifactCommitRequest
    ) -> tuple[_WorkerArtifactState, list[ArtifactArray], list[dict[str, Any]]]:
        if request.policy_epoch != self.policy_epoch:
            raise RuntimeError(
                "artifact policy epoch mismatch: "
                f"scheduler={request.policy_epoch}, worker={self.policy_epoch}"
            )
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
        state = self._state(request.request_id, request.artifact_sample_id)
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
        profile_id = self._set_profile(
            state,
            array,
            hash_block_size=request.hash_block_size,
            policy_epoch=request.policy_epoch,
        )
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
            object_id = self._object_id(
                b"vllm.artifact.block.v1", profile_id.encode(), block_hash
            )
            local_start = source_start - request.block_start
            block_array = array[local_start : local_start + request.hash_block_size]
            blocks.append(
                ArtifactArray(
                    object_id=object_id,
                    array=block_array,
                    metadata={
                        "namespace": self.namespace,
                        "profile_id": profile_id,
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
        # Advance the worker-local assembler while preparing this metadata
        # batch so consecutive commits for one request can share one put.
        state.next_full_start = request.block_end
        return state, blocks, segments

    def _put_tail(
        self,
        *,
        request: ArtifactFinalizeRequest,
        state: _WorkerArtifactState,
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
        profile_id = self._set_profile(
            state,
            array,
            hash_block_size=request.hash_block_size,
            policy_epoch=request.policy_epoch,
        )
        object_id = self._object_id(
            b"vllm.artifact.tail.v1",
            request.artifact_sample_id.encode(),
            source_start.to_bytes(8, "big"),
            source_end.to_bytes(8, "big"),
        )
        self.store.put_array(
            "tail",
            object_id,
            array,
            {
                "namespace": self.namespace,
                "profile_id": profile_id,
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

    def _finalize(
        self, request: ArtifactFinalizeRequest
    ) -> tuple[dict[str, Any], np.ndarray]:
        if request.policy_epoch != self.policy_epoch:
            raise RuntimeError(
                "artifact policy epoch mismatch: "
                f"scheduler={request.policy_epoch}, worker={self.policy_epoch}"
            )
        if request.token_start < 0 or request.token_end <= request.token_start:
            raise ValueError(
                "invalid artifact token range: "
                f"[{request.token_start}, {request.token_end})"
            )
        state = self._state(request.request_id, request.artifact_sample_id)
        if state.error is not None:
            raise RuntimeError(state.error)
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
                    request=request,
                    state=state,
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
                    "terminal artifact is missing an incremental full-block commit: "
                    f"request={request.request_id}, block_start={source_start}"
                )
            segment = {**segment, "output_start": output_start}
            segments.append(segment)
            output_start += block_size

        tail_start = max(full_end, request.token_start)
        if tail_start < request.token_end:
            segments.append(
                self._put_tail(
                    request=request,
                    state=state,
                    source_start=tail_start,
                    source_end=request.token_end,
                    output_start=output_start,
                )
            )
            output_start += request.token_end - tail_start

        if output_start != request.token_end - request.token_start:
            raise RuntimeError("artifact segments do not cover the terminal range")
        if (
            state.profile_id is None
            or state.dtype is None
            or state.shape_per_token is None
        ):
            raise RuntimeError("artifact payload profile is unavailable")
        shape = [request.token_end - request.token_start, *state.shape_per_token]
        manifest = {
            "schema_version": 1,
            "backend": "shm",
            "store_id": self.store.store_id,
            "artifact_sample_id": request.artifact_sample_id,
            "profile_id": state.profile_id,
            "namespace": self.namespace,
            "policy_epoch": request.policy_epoch,
            "request_id": request.request_id,
            "field": "routed_experts",
            "dtype": state.dtype,
            "shape": shape,
            "source_token_start": request.token_start,
            "source_token_end": request.token_end,
            "segments": segments,
            "created_at_unix_ns": time.time_ns(),
        }
        manifest_sha256 = self.store.put_manifest(request.artifact_sample_id, manifest)
        handle = {
            "backend": "shm",
            "schema_version": 1,
            "store_id": self.store.store_id,
            "artifact_sample_id": request.artifact_sample_id,
            "profile_id": state.profile_id,
            "policy_epoch": request.policy_epoch,
            "manifest_sha256": manifest_sha256,
            "field": "routed_experts",
            "dtype": state.dtype,
            "shape": shape,
        }
        value = self.store.materialize(handle)
        self._drop_state(request.request_id)
        return handle, value

    def finalize(
        self, metadata: ArtifactConnectorMetadata | None
    ) -> ArtifactConnectorOutput | None:
        if metadata is None:
            return None
        commit_results: list[ArtifactCommitResult] = []
        prepared: list[
            tuple[
                ArtifactCommitRequest,
                _WorkerArtifactState,
                list[ArtifactArray],
                list[dict[str, Any]],
            ]
        ] = []
        for commit_request in metadata.commits:
            try:
                state, blocks, segments = self._prepare_commit(commit_request)
                prepared.append((commit_request, state, blocks, segments))
            except Exception as error:
                logger.exception(
                    "Failed to prepare artifact blocks for request %s",
                    commit_request.request_id,
                )
                failed_state = self._states.get(commit_request.request_id)
                message = f"{type(error).__name__}: {error}"
                if failed_state is not None:
                    failed_state.error = message
                commit_results.append(
                    ArtifactCommitResult(
                        operation_id=commit_request.operation_id,
                        request_id=commit_request.request_id,
                        artifact_sample_id=commit_request.artifact_sample_id,
                        block_end=commit_request.block_end,
                        error=message,
                    )
                )
        if prepared:
            try:
                self.store.put_blocks(
                    [block for _, _, blocks, _ in prepared for block in blocks]
                )
                for prepared_request, state, _, segments in prepared:
                    state.segments.update(
                        (segment["source_token_start"], segment) for segment in segments
                    )
                    commit_results.append(
                        ArtifactCommitResult(
                            operation_id=prepared_request.operation_id,
                            request_id=prepared_request.request_id,
                            artifact_sample_id=prepared_request.artifact_sample_id,
                            block_end=prepared_request.block_end,
                        )
                    )
            except Exception as error:
                logger.exception("Failed to publish artifact block batch")
                message = f"{type(error).__name__}: {error}"
                for prepared_request, state, _, _ in prepared:
                    state.error = message
                    commit_results.append(
                        ArtifactCommitResult(
                            operation_id=prepared_request.operation_id,
                            request_id=prepared_request.request_id,
                            artifact_sample_id=prepared_request.artifact_sample_id,
                            block_end=prepared_request.block_end,
                            error=message,
                        )
                    )

        results: list[ArtifactFinalizeResult] = []
        for finalize_request in metadata.requests:
            try:
                handle, routed_experts = self._finalize(finalize_request)
                result = ArtifactFinalizeResult(
                    request_id=finalize_request.request_id,
                    artifact_sample_id=finalize_request.artifact_sample_id,
                    handle=handle,
                    routed_experts=routed_experts,
                )
            except Exception as error:
                logger.exception(
                    "Failed to finalize routed-experts artifact for request %s",
                    finalize_request.request_id,
                )
                self._drop_state(finalize_request.request_id)
                result = ArtifactFinalizeResult(
                    request_id=finalize_request.request_id,
                    artifact_sample_id=finalize_request.artifact_sample_id,
                    error=f"{type(error).__name__}: {error}",
                )
            results.append(result)

        discard_results: list[ArtifactDiscardResult] = []
        for discard_request in metadata.discards:
            self.prompt_logprobs.discard(discard_request.request_id)
            discard_state = self._states.get(discard_request.request_id)
            if (
                discard_state is not None
                and discard_state.sample_id == discard_request.artifact_sample_id
            ):
                self._drop_state(discard_request.request_id)
            discard_results.append(
                ArtifactDiscardResult(
                    operation_id=discard_request.operation_id,
                    request_id=discard_request.request_id,
                    artifact_sample_id=discard_request.artifact_sample_id,
                )
            )
        output = ArtifactConnectorOutput(
            results=results,
            commit_results=commit_results,
            discard_results=discard_results,
        )
        return None if output.is_empty() else output

    def close(self) -> None:
        self.store.close()
