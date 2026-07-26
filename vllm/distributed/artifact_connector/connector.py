# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-specific connectors for routed-experts execution artifacts."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.distributed.artifact_connector.protocol import (
    ArtifactBlockRef,
    ArtifactCommitRequest,
    ArtifactCommitResult,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactDiscardRequest,
    ArtifactDiscardResult,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
)
from vllm.distributed.artifact_connector.request_core import (
    ArtifactRequestCore,
    PreparedCommit,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactStore,
)
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
    cached_blocks: list[ArtifactBlockRef] = field(default_factory=list)


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
        self._ready_blocks: set[tuple[int, bytes]] = set()

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

    def max_ready_prefix_tokens(
        self,
        *,
        block_hashes: list[bytes],
        token_start: int,
        max_tokens: int,
        hash_block_size: int,
        policy_epoch: int,
    ) -> int:
        """Return the longest prefix jointly reusable by KV and R3."""
        if max_tokens <= token_start:
            return max_tokens
        first_full_start = (
            (token_start + hash_block_size - 1) // hash_block_size
        ) * hash_block_size
        if first_full_start != token_start:
            return token_start
        ready_end = token_start
        for block_index in range(
            first_full_start // hash_block_size,
            min(max_tokens, len(block_hashes) * hash_block_size) // hash_block_size,
        ):
            if (policy_epoch, block_hashes[block_index]) not in self._ready_blocks:
                break
            ready_end = (block_index + 1) * hash_block_size
        return min(ready_end, max_tokens)

    def request_started(
        self,
        *,
        request_id: str,
        block_hashes: list[bytes],
        token_start: int,
        cached_token_end: int,
        hash_block_size: int,
        policy_epoch: int,
    ) -> str:
        """Bind jointly cached R3 blocks to a new request state."""
        state = self._state(request_id, token_start, hash_block_size)
        if state.cached_blocks:
            return state.sample_id
        first_full_start = (
            (token_start + hash_block_size - 1) // hash_block_size
        ) * hash_block_size
        cached_full_end = cached_token_end // hash_block_size * hash_block_size
        if cached_full_end > first_full_start:
            state.cached_blocks = [
                ArtifactBlockRef(
                    block_index=block_index,
                    block_hash=block_hashes[block_index],
                )
                for block_index in range(
                    first_full_start // hash_block_size,
                    cached_full_end // hash_block_size,
                )
            ]
            for ref in state.cached_blocks:
                if (policy_epoch, ref.block_hash) not in self._ready_blocks:
                    raise RuntimeError(
                        "scheduler admitted an R3 block that is not ready"
                    )
            state.next_full_end = cached_full_end
        return state.sample_id

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
            cached_blocks=list(state.cached_blocks),
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
            if commit_result.error is None:
                block_size = commit_request.hash_block_size
                for block_start in range(
                    commit_request.block_start,
                    commit_request.block_end,
                    block_size,
                ):
                    block_index = block_start // block_size
                    self._ready_blocks.add(
                        (
                            commit_request.policy_epoch,
                            commit_request.block_hashes[block_index],
                        )
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


class ArtifactWorkerConnector:
    """Translate worker metadata into request-core operations."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        writer: RoutedExpertsWorkerWriter,
    ) -> None:
        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        self.store = LocalSharedMemoryArtifactStore(
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
        namespace = hashlib.sha256(
            json.dumps(
                namespace_input,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        self.core = ArtifactRequestCore(
            self.store,
            writer,
            namespace=namespace,
            inline_value=True,
        )

    def advance_policy_epoch(self) -> None:
        """Prevent block-key reuse after an in-place model weight update."""
        self.core.advance_policy_epoch()

    def finalize(
        self, metadata: ArtifactConnectorMetadata | None
    ) -> ArtifactConnectorOutput | None:
        if metadata is None:
            return None
        commit_results: list[ArtifactCommitResult] = []
        prepared: list[PreparedCommit] = []
        for commit_request in metadata.commits:
            try:
                prepared.append(self.core.prepare_commit(commit_request))
            except Exception as error:
                logger.exception(
                    "Failed to prepare artifact blocks for request %s",
                    commit_request.request_id,
                )
                message = f"{type(error).__name__}: {error}"
                self.core.mark_error(commit_request.request_id, message)
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
                self.core.publish_commits(prepared)
                for commit in prepared:
                    prepared_request = commit.request
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
                for commit in prepared:
                    prepared_request = commit.request
                    self.core.mark_error(prepared_request.request_id, message)
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
                finalized = self.core.finalize(finalize_request)
                result = ArtifactFinalizeResult(
                    request_id=finalize_request.request_id,
                    artifact_sample_id=finalize_request.artifact_sample_id,
                    delivery="inline",
                    manifest_sha256=finalized.manifest_sha256,
                    routed_experts=finalized.value,
                )
            except Exception as error:
                logger.exception(
                    "Failed to finalize routed-experts artifact for request %s",
                    finalize_request.request_id,
                )
                self.core.discard(finalize_request.request_id)
                result = ArtifactFinalizeResult(
                    request_id=finalize_request.request_id,
                    artifact_sample_id=finalize_request.artifact_sample_id,
                    error=f"{type(error).__name__}: {error}",
                )
            results.append(result)

        discard_results: list[ArtifactDiscardResult] = []
        for discard_request in metadata.discards:
            self.core.discard(discard_request.request_id)
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
        self.core.close()
