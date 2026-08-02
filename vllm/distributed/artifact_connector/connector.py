# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-specific connectors for routed-experts execution artifacts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    PreparedCommit,
    RoutedExpertsRequestCore,
    get_routing_shape_and_dtype,
    materialize_routed_experts,
)
from vllm.distributed.artifact_connector.shm import (
    LocalSharedMemoryArtifactReader,
    LocalSharedMemoryArtifactStore,
    make_shm_store_id,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    hash_block_tokens,
)

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        RoutedExpertsCaptureState,
    )
    from vllm.v1.request import Request


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
        self._inflight_finalizes: dict[str, ArtifactFinalizeRequest] = {}
        self._weight_version = "default"
        self._hash_fn = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )

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
        self._shape_per_token = shape_per_token
        self._dtype: np.dtype[Any] = np.dtype(dtype)

    def _state(self, request_id: str) -> _SchedulerArtifactState:
        try:
            return self._states[request_id]
        except KeyError as error:
            raise RuntimeError(
                f"artifact request has not started: {request_id}"
            ) from error

    def request_started(
        self,
        *,
        request: Request,
        cached_token_end: int,
        hash_block_size: int,
    ) -> None:
        request_id = request.request_id
        if request_id in self._states:
            return
        weight_version = self._weight_version
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
        request: Request,
        accepted_token_end: int,
        hash_block_size: int,
    ) -> None:
        request_id = request.request_id
        state = self._state(request_id)
        full_end = accepted_token_end // hash_block_size * hash_block_size
        if full_end <= state.next_full_end:
            return
        first_block = state.next_full_end // hash_block_size
        last_block = full_end // hash_block_size
        commit_hashes = list(request.block_hashes[first_block:last_block])
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
        request: Request,
        token_end: int,
        hash_block_size: int,
    ) -> None:
        request_id = request.request_id
        if (
            request_id in self._pending_finalizes
            or request_id in self._inflight_finalizes
        ):
            raise RuntimeError(f"artifact request is already pending: {request_id}")
        state = self._state(request_id)
        full_end = token_end // hash_block_size * hash_block_size
        tail_block_hash = None
        if full_end < token_end:
            if token_end > request.num_tokens:
                raise RuntimeError(
                    "artifact token boundary exceeds the request token count"
                )
            parent_hash = (
                request.block_hashes[full_end // hash_block_size - 1]
                if full_end
                else None
            )
            extra_keys, _ = generate_block_hash_extra_keys(
                request,
                full_end,
                token_end,
                -1 if full_end else 0,
            )
            tail_block_hash = hash_block_tokens(
                self._hash_fn,
                parent_hash,
                request.all_token_ids[full_end:token_end],
                extra_keys,
            )
        self._pending_finalizes[request_id] = ArtifactFinalizeRequest(
            request_id=request_id,
            request_attempt_id=state.request_attempt_id,
            weight_version=state.weight_version,
            block_hashes=list(request.block_hashes),
            tail_block_hash=tail_block_hash,
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
            (request.request_id, request) for request in finalizes
        )
        return ArtifactConnectorMetadata(
            commits=commits,
            finalizes=finalizes,
        )

    def consume_output(
        self, output: ArtifactConnectorOutput
    ) -> list[tuple[str, np.ndarray]]:
        """Validate worker ACKs and materialize their terminal R3 values."""
        materialized: list[tuple[str, np.ndarray]] = []
        for finalize_result in output.results:
            request = self._inflight_finalizes.pop(finalize_result.request_id, None)
            if (
                request is None
                or request.request_attempt_id != finalize_result.request_attempt_id
            ):
                raise RuntimeError(
                    "worker acknowledged an unknown artifact request: "
                    f"{finalize_result.request_id}"
                )
            self._states.pop(finalize_result.request_id, None)
            materialized.append(
                (
                    finalize_result.request_id,
                    materialize_routed_experts(
                        self._reader,
                        finalize_result.keys,
                        expected_shape_per_token=self._shape_per_token,
                        expected_dtype=self._dtype,
                        expected_token_end=request.token_end,
                        hash_block_size=request.hash_block_size,
                    ),
                )
            )
        return materialized

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
        self._capture_state = capture_state
        self._store: LocalSharedMemoryArtifactStore | None = None
        self._buffer: RoutedExpertsArtifactBuffer | None = None
        self._request_core: RoutedExpertsRequestCore | None = None
        if vllm_config.parallel_config.rank != 0:
            return

        config = vllm_config.artifact_config
        parallel_config = vllm_config.parallel_config
        self._store = LocalSharedMemoryArtifactStore(
            config.shm_dir,
            vllm_config.instance_id,
            parallel_config.data_parallel_rank,
            max_bytes=config.max_shm_bytes,
            ttl_seconds=config.shm_ttl_seconds,
        )
        shape_per_token, dtype = get_routing_shape_and_dtype(vllm_config)
        self._buffer = RoutedExpertsArtifactBuffer(np.dtype(dtype), shape_per_token)
        self._request_core = RoutedExpertsRequestCore(
            self._store,
            self._buffer,
        )

    @classmethod
    def create(
        cls,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        max_num_batched_tokens: int,
    ) -> ArtifactWorkerConnector | None:
        """Create and bind all worker-side artifact resources."""
        if not vllm_config.artifact_config.enabled:
            return None
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
        self._capture_state.clear()
        request_core = self._request_core
        buffer = self._buffer
        if request_core is None:
            return None
        assert buffer is not None
        if metadata is None:
            if finished_req_ids:
                for request_id in finished_req_ids:
                    buffer.discard(request_id)
            return None
        prepared: list[PreparedCommit] = [
            request_core.prepare_commit(commit_request)
            for commit_request in metadata.commits
        ]
        if prepared:
            request_core.publish_commits(prepared)

        results = [
            ArtifactFinalizeResult(
                request_id=finalize_request.request_id,
                request_attempt_id=finalize_request.request_attempt_id,
                keys=request_core.finalize(finalize_request),
            )
            for finalize_request in metadata.finalizes
        ]

        if finished_req_ids:
            for request_id in finished_req_ids:
                buffer.discard(request_id)
        return ArtifactConnectorOutput(results=results) if results else None

    def make_write_task(
        self,
        num_tokens: int,
        *,
        request_ids: tuple[str, ...],
        query_start_locs: np.ndarray,
        token_starts: np.ndarray,
    ) -> ArtifactWriteTask | None:
        """Snapshot the current forward for asynchronous publication."""
        if self._buffer is None:
            return None
        return self._capture_state.make_write_task(
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
        assert self._buffer is not None
        self._buffer.capture(request_id, token_start, rows)

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()
