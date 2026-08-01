# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-specific connectors for routed-experts execution artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.artifact_connector.buffer import RoutedExpertsArtifactBuffer
from vllm.distributed.artifact_connector.protocol import (
    ArtifactCommitRequest,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactFinalizeRequest,
    ArtifactFinalizeResult,
    ArtifactWriteTask,
)
from vllm.distributed.artifact_connector.request_core import (
    ArtifactKeySpace,
    ArtifactRequestCore,
    PreparedCommit,
    materialize_routed_experts,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactReader,
    LocalSharedMemoryArtifactStore,
    make_shm_store_id,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_routing_shape_and_dtype,
)

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        RoutedExpertsCaptureState,
    )


@dataclass
class _SchedulerArtifactState:
    request_attempt_id: str
    next_full_end: int
    weight_version: str


class ArtifactSchedulerConnector:
    """Turn accepted-token progress into worker block commits."""

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self._states: dict[str, _SchedulerArtifactState] = {}
        self._pending_commits: list[ArtifactCommitRequest] = []
        self._pending_finalizes: dict[str, ArtifactFinalizeRequest] = {}
        self._inflight_finalizes: dict[str, str] = {}
        self._weight_version = "default"

        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        store_id = make_shm_store_id(
            vllm_config.instance_id,
            parallel_config.data_parallel_rank,
        )
        self._reader = LocalSharedMemoryArtifactReader(
            config.shm_dir,
            store_id,
        )
        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._key_space = ArtifactKeySpace(
            np.dtype(dtype),
            shape_per_token,
        )

    def _state(self, request_id: str) -> _SchedulerArtifactState:
        try:
            return self._states[request_id]
        except KeyError as error:
            raise RuntimeError(
                f"artifact request has not started: {request_id}"
            ) from error

    def ensure_prefix_ready(
        self,
        *,
        block_hashes: Sequence[bytes],
        cached_token_end: int,
        hash_block_size: int,
        weight_version: str,
    ) -> None:
        """Fail if a KV-reused prefix lacks any corresponding artifact block."""
        if cached_token_end <= 0:
            return
        num_required = (cached_token_end + hash_block_size - 1) // hash_block_size
        if num_required > len(block_hashes):
            raise RuntimeError(
                "KV prefix hit has no matching artifact block hash: "
                f"cached_token_end={cached_token_end}, "
                f"num_block_hashes={len(block_hashes)}"
            )
        keys = [
            self._key_space.block_key(block_hash, hash_block_size, weight_version)
            for block_hash in block_hashes[:num_required]
        ]
        ready = self._reader.exists(keys)
        if len(ready) != len(keys) or not all(ready):
            missing = next(
                (index for index, exists in enumerate(ready) if not exists),
                len(ready),
            )
            raise RuntimeError(
                "KV prefix hit is missing a routed-experts artifact block: "
                f"block_index={missing}, weight_version={weight_version}"
            )

    def request_started(
        self,
        *,
        request_id: str,
        block_hashes: Sequence[bytes],
        cached_token_end: int,
        hash_block_size: int,
    ) -> None:
        if request_id in self._states:
            return
        weight_version = self._weight_version
        self.ensure_prefix_ready(
            block_hashes=block_hashes,
            cached_token_end=cached_token_end,
            hash_block_size=hash_block_size,
            weight_version=weight_version,
        )
        self._states[request_id] = _SchedulerArtifactState(
            request_attempt_id=uuid.uuid4().hex,
            next_full_end=(
                (cached_token_end + hash_block_size - 1)
                // hash_block_size
                * hash_block_size
            ),
            weight_version=weight_version,
        )

    def request_progress(
        self,
        *,
        request_id: str,
        block_hashes: Sequence[bytes],
        accepted_token_end: int,
        hash_block_size: int,
    ) -> None:
        state = self._state(request_id)
        full_end = accepted_token_end // hash_block_size * hash_block_size
        if full_end <= state.next_full_end:
            return
        first_block = state.next_full_end // hash_block_size
        last_block = full_end // hash_block_size
        commit_hashes = list(block_hashes[first_block:last_block])
        if len(commit_hashes) != last_block - first_block:
            raise RuntimeError(
                "missing KV-compatible hashes for completed artifact blocks"
            )
        self._pending_commits.append(
            ArtifactCommitRequest(
                request_id=request_id,
                weight_version=state.weight_version,
                block_hashes=commit_hashes,
                block_start=state.next_full_end,
                hash_block_size=hash_block_size,
            )
        )
        state.next_full_end = full_end

    def request_finished(
        self,
        *,
        request_id: str,
        block_hashes: Sequence[bytes],
        token_end: int,
        hash_block_size: int,
    ) -> None:
        if (
            request_id in self._pending_finalizes
            or request_id in self._inflight_finalizes
        ):
            raise RuntimeError(f"artifact request is already pending: {request_id}")
        state = self._state(request_id)
        self._pending_finalizes[request_id] = ArtifactFinalizeRequest(
            request_id=request_id,
            request_attempt_id=state.request_attempt_id,
            weight_version=state.weight_version,
            block_hashes=list(block_hashes),
            token_end=token_end,
            hash_block_size=hash_block_size,
        )

    def request_aborted(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self._pending_finalizes.pop(request_id, None)

    def build_connector_meta(self) -> ArtifactConnectorMetadata | None:
        if not (self._pending_commits or self._pending_finalizes):
            return None
        commits = self._pending_commits
        finalizes = list(self._pending_finalizes.values())
        self._pending_commits = []
        self._pending_finalizes.clear()
        self._inflight_finalizes.update(
            (request.request_id, request.request_attempt_id) for request in finalizes
        )
        return ArtifactConnectorMetadata(
            commits=commits,
            finalizes=finalizes,
        )

    def update_connector_output(self, output: ArtifactConnectorOutput) -> None:
        for finalize_result in output.results:
            request_attempt_id = self._inflight_finalizes.pop(
                finalize_result.request_id, None
            )
            if request_attempt_id != finalize_result.request_attempt_id:
                raise RuntimeError(
                    "worker acknowledged an unknown artifact request: "
                    f"{finalize_result.request_id}"
                )
            self._states.pop(finalize_result.request_id, None)

    def materialize(self, keys: list[str]) -> np.ndarray:
        """Materialize one terminal artifact through the configured reader."""
        return materialize_routed_experts(self._reader, keys)

    def set_weight_version(self, weight_version: str) -> None:
        """Use a new immutable key namespace for subsequently admitted requests."""
        self._weight_version = weight_version

    def shutdown(self) -> None:
        self._reader.close()


class ArtifactWorkerConnector:
    """Translate worker metadata into request-core operations."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        capture_state: RoutedExpertsCaptureState,
    ) -> None:
        self.capture_state = capture_state
        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        self.store = LocalSharedMemoryArtifactStore(
            config.shm_dir,
            vllm_config.instance_id,
            parallel_config.data_parallel_rank,
            max_bytes=config.max_shm_bytes,
            ttl_seconds=config.shm_ttl_seconds,
        )
        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self.buffer = RoutedExpertsArtifactBuffer(np.dtype(dtype), shape_per_token)
        self.request_core = ArtifactRequestCore(
            self.store,
            self.buffer,
        )

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        max_num_batched_tokens: int,
    ) -> ArtifactWorkerConnector:
        """Create and bind all worker-side artifact resources."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
            RoutedExpertsCaptureState,
        )

        capture_state = RoutedExpertsCaptureState.create(
            model=model,
            vllm_config=vllm_config,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        return cls(vllm_config, capture_state)

    def start_step(
        self,
        metadata: ArtifactConnectorMetadata | None,
        finished_req_ids: set[str] | None = None,
    ) -> ArtifactConnectorOutput | None:
        """Process control work and reset capture for the next forward."""
        self.capture_state.clear()
        if metadata is None:
            if finished_req_ids:
                for request_id in finished_req_ids:
                    self.buffer.discard(request_id)
            return None
        prepared: list[PreparedCommit] = [
            self.request_core.prepare_commit(commit_request)
            for commit_request in metadata.commits
        ]
        if prepared:
            self.request_core.publish_commits(prepared)

        results = [
            ArtifactFinalizeResult(
                request_id=finalize_request.request_id,
                request_attempt_id=finalize_request.request_attempt_id,
                keys=self.request_core.finalize(finalize_request),
            )
            for finalize_request in metadata.finalizes
        ]

        if finished_req_ids:
            for request_id in finished_req_ids:
                self.buffer.discard(request_id)
        return ArtifactConnectorOutput(results=results) if results else None

    def make_write_task(
        self,
        num_tokens: int,
        *,
        request_ids: tuple[str, ...],
        query_start_locs: np.ndarray,
        token_starts: np.ndarray,
    ) -> ArtifactWriteTask:
        """Snapshot the current forward for asynchronous publication."""
        return self.capture_state.make_write_task(
            num_tokens,
            request_ids=request_ids,
            query_start_locs=query_start_locs,
            token_starts=token_starts,
            artifact_sink=self._capture_step,
        )

    def _capture_step(
        self,
        request_id: str,
        token_start: int,
        rows: np.ndarray,
    ) -> None:
        self.buffer.capture(request_id, token_start, rows)

    def shutdown(self) -> None:
        self.store.close()
