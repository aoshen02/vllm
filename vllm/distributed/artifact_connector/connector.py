# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-specific connectors for routed-experts execution artifacts."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.mooncake import (
    MooncakeArtifactPublisher,
    MooncakeArtifactReader,
    MooncakeArtifactStore,
)
from vllm.distributed.artifact_connector.protocol import (
    ArtifactCommitRequest,
    ArtifactCommitResult,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
)
from vllm.distributed.artifact_connector.request_core import (
    ArtifactKeySpace,
    ArtifactRequestCore,
    PreparedCommit,
    max_artifact_object_bytes,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactReader,
    LocalSharedMemoryArtifactStore,
    make_shm_store_id,
)
from vllm.distributed.artifact_connector.store import ArtifactStore
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _artifact_namespace(vllm_config: VllmConfig) -> str:
    model_config = vllm_config.model_config
    namespace_input = {
        "model": model_config.model,
        "revision": model_config.revision,
        "tokenizer_revision": model_config.tokenizer_revision,
    }
    return hashlib.sha256(
        json.dumps(
            namespace_input,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


@dataclass
class _SchedulerArtifactState:
    request_attempt_id: str
    next_full_end: int
    started: bool = False


class ArtifactSchedulerConnector:
    """Turn accepted-token progress into worker block commits."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
    ) -> None:
        self._states: dict[str, _SchedulerArtifactState] = {}
        self._pending_commits: OrderedDict[str, ArtifactCommitRequest] = OrderedDict()
        self._inflight_commits: dict[str, ArtifactCommitRequest] = {}
        self._pending_finalizes: OrderedDict[str, ArtifactFinalizeRequest] = (
            OrderedDict()
        )
        self._inflight_finalizes: dict[str, ArtifactFinalizeRequest] = {}

        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        if config.backend == "shm":
            store_id = make_shm_store_id(
                vllm_config.instance_id,
                parallel_config.data_parallel_rank,
            )
            self._reader: LocalSharedMemoryArtifactReader | MooncakeArtifactReader = (
                LocalSharedMemoryArtifactReader(
                    config.shm_dir,
                    store_id,
                )
            )
        else:
            store_id = config.mooncake_store_id
            self._reader = MooncakeArtifactReader(store_id)
        self._key_space = ArtifactKeySpace(
            store_id,
            _artifact_namespace(vllm_config),
            dtype,
            shape_per_token,
        )

    def _block_keys(self, block_hashes: list[bytes], hash_block_size: int) -> list[str]:
        return [
            self._key_space.block_key(block_hash, hash_block_size)
            for block_hash in block_hashes
        ]

    def _state(self, request_id: str) -> _SchedulerArtifactState:
        state = self._states.get(request_id)
        if state is None:
            state = _SchedulerArtifactState(
                request_attempt_id=uuid.uuid4().hex,
                next_full_end=0,
            )
            self._states[request_id] = state
        return state

    def max_ready_prefix_tokens(
        self,
        *,
        block_hashes: list[bytes],
        max_tokens: int,
        hash_block_size: int,
    ) -> int:
        """Return the longest prefix jointly reusable by KV and R3."""
        if max_tokens <= 0:
            return max_tokens
        num_candidates = (
            min(max_tokens, len(block_hashes) * hash_block_size) // hash_block_size
        )
        candidate_keys = self._block_keys(
            block_hashes[:num_candidates], hash_block_size
        )
        ready = self._reader.exists(candidate_keys)
        ready_blocks = 0
        for exists in ready:
            if not exists:
                break
            ready_blocks += 1
        return min(ready_blocks * hash_block_size, max_tokens)

    def request_started(
        self,
        *,
        request_id: str,
        block_hashes: list[bytes],
        cached_token_end: int,
        hash_block_size: int,
    ) -> str:
        state = self._state(request_id)
        if state.started:
            return state.request_attempt_id
        state.started = True
        cached_full_end = cached_token_end // hash_block_size * hash_block_size
        if cached_full_end > 0:
            keys = self._block_keys(
                block_hashes[: cached_full_end // hash_block_size],
                hash_block_size,
            )
            if not all(self._reader.exists(keys)):
                raise RuntimeError("scheduler admitted an R3 block that is not ready")
            state.next_full_end = cached_full_end
        return state.request_attempt_id

    def request_progress(
        self,
        *,
        request_id: str,
        block_hashes: list[bytes],
        accepted_token_end: int,
        hash_block_size: int,
    ) -> str:
        state = self._state(request_id)
        full_end = accepted_token_end // hash_block_size * hash_block_size
        if full_end <= state.next_full_end:
            return state.request_attempt_id
        operation_id = uuid.uuid4().hex
        self._pending_commits[operation_id] = ArtifactCommitRequest(
            operation_id=operation_id,
            request_id=request_id,
            request_attempt_id=state.request_attempt_id,
            block_hashes=list(block_hashes),
            block_start=state.next_full_end,
            block_end=full_end,
            hash_block_size=hash_block_size,
        )
        state.next_full_end = full_end
        return state.request_attempt_id

    def request_finished(
        self,
        *,
        request_id: str,
        block_hashes: list[bytes],
        token_end: int,
        hash_block_size: int,
    ) -> str:
        if (
            request_id in self._pending_finalizes
            or request_id in self._inflight_finalizes
        ):
            raise RuntimeError(f"artifact request is already pending: {request_id}")
        state = self._state(request_id)
        self._pending_finalizes[request_id] = ArtifactFinalizeRequest(
            request_id=request_id,
            request_attempt_id=state.request_attempt_id,
            block_hashes=list(block_hashes),
            token_end=token_end,
            hash_block_size=hash_block_size,
        )
        return state.request_attempt_id

    def request_aborted(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def build_connector_metadata(self) -> ArtifactConnectorMetadata | None:
        if not (self._pending_commits or self._pending_finalizes):
            return None
        commits = list(self._pending_commits.values())
        finalizes = list(self._pending_finalizes.values())
        self._pending_commits.clear()
        self._pending_finalizes.clear()
        self._inflight_commits.update(
            (request.operation_id, request) for request in commits
        )
        self._inflight_finalizes.update(
            (request.request_id, request) for request in finalizes
        )
        return ArtifactConnectorMetadata(
            finalizes=finalizes,
            commits=commits,
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
                or commit_request.request_attempt_id != commit_result.request_attempt_id
                or commit_request.block_end != commit_result.block_end
            ):
                raise RuntimeError(
                    "worker acknowledged an unknown artifact block commit: "
                    f"{commit_result.operation_id}"
                )
            expected_blocks = (
                commit_request.block_end - commit_request.block_start
            ) // commit_request.hash_block_size
            if (
                commit_result.error is None
                and len(commit_result.block_keys) != expected_blocks
            ):
                raise RuntimeError("worker returned the wrong artifact key count")
        for finalize_result in output.results:
            finalize_request = self._inflight_finalizes.pop(
                finalize_result.request_id, None
            )
            if finalize_request is None or (
                finalize_request.request_attempt_id
                != finalize_result.request_attempt_id
            ):
                raise RuntimeError(
                    "worker acknowledged an unknown artifact request: "
                    f"{finalize_result.request_id}"
                )
            self._states.pop(finalize_result.request_id, None)
        return output

    def has_pending_work(self) -> bool:
        return bool(
            self._pending_commits
            or self._inflight_commits
            or self._pending_finalizes
            or self._inflight_finalizes
        )

    def reset(self) -> None:
        if self.has_pending_work() or self._states:
            raise RuntimeError("cannot reset Artifact Connector with active requests")

    def close(self) -> None:
        self._reader.close()


class ArtifactWorkerConnector:
    """Translate worker metadata into request-core operations."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
    ) -> None:
        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        self.store: ArtifactStore
        if config.backend == "shm":
            self.store = LocalSharedMemoryArtifactStore(
                config.shm_dir,
                vllm_config.instance_id,
                parallel_config.data_parallel_rank,
                max_bytes=config.max_shm_bytes,
                ttl_seconds=config.shm_ttl_seconds,
            )
        else:
            max_object_bytes = max_artifact_object_bytes(
                dtype,
                shape_per_token,
                vllm_config.cache_config.prefix_match_unit
                or vllm_config.cache_config.block_size,
            )
            mooncake_store = MooncakeArtifactStore(
                config.mooncake_store_id,
                staging_buffer_bytes=config.mooncake_staging_buffer_bytes,
                max_object_bytes=max_object_bytes,
            )
            self.store = MooncakeArtifactPublisher(mooncake_store)
        self.buffer = RoutedExpertsArtifactBuffer(
            dtype,
            shape_per_token,
        )
        self.core = ArtifactRequestCore(
            self.store,
            self.buffer,
            namespace=_artifact_namespace(vllm_config),
            materialize=config.backend == "shm",
        )

    def capture_step(
        self,
        request_id: str,
        token_start: int,
        rows: np.ndarray,
    ) -> None:
        self.buffer.capture(request_id, token_start, rows)

    def process(
        self,
        metadata: ArtifactConnectorMetadata | None,
        finished_req_ids: set[str] | None = None,
    ) -> ArtifactConnectorOutput | None:
        if metadata is None:
            if finished_req_ids:
                for request_id in finished_req_ids:
                    self.buffer.discard(request_id)
            return None
        commit_results: list[ArtifactCommitResult] = []
        prepared: list[PreparedCommit] = []
        for commit_request in metadata.commits:
            try:
                prepared.append(self.core.prepare_commit(commit_request))
            except Exception as exc:
                logger.exception(
                    "Failed to prepare artifact blocks for request %s",
                    commit_request.request_id,
                )
                message = f"{type(exc).__name__}: {exc}"
                commit_results.append(
                    ArtifactCommitResult(
                        operation_id=commit_request.operation_id,
                        request_id=commit_request.request_id,
                        request_attempt_id=commit_request.request_attempt_id,
                        block_end=commit_request.block_end,
                        error=message,
                    )
                )
        if prepared:
            errors = self.core.publish_commits(prepared)
            for commit in prepared:
                commit_request = commit.request
                commit_error = errors[commit_request.operation_id]
                commit_results.append(
                    ArtifactCommitResult(
                        operation_id=commit_request.operation_id,
                        request_id=commit_request.request_id,
                        request_attempt_id=commit_request.request_attempt_id,
                        block_end=commit_request.block_end,
                        block_keys=(commit.keys if commit_error is None else []),
                        error=commit_error,
                    )
                )

        results: list[ArtifactFinalizeResult] = []
        for finalize_request in metadata.finalizes:
            try:
                finalized = self.core.finalize(finalize_request)
                result = ArtifactFinalizeResult(
                    request_id=finalize_request.request_id,
                    request_attempt_id=finalize_request.request_attempt_id,
                    artifact_keys=(finalized.keys if finalized.value is None else None),
                    value=finalized.value,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to finalize routed-experts artifact for request %s",
                    finalize_request.request_id,
                )
                result = ArtifactFinalizeResult(
                    request_id=finalize_request.request_id,
                    request_attempt_id=finalize_request.request_attempt_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.buffer.discard(finalize_request.request_id)
            results.append(result)

        output = ArtifactConnectorOutput(
            results=results,
            commit_results=commit_results,
        )
        if finished_req_ids:
            for request_id in finished_req_ids:
                self.buffer.discard(request_id)
        return None if output.is_empty() else output

    def close(self) -> None:
        self.core.close()
