# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side auxiliary-output data plane."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
)
from vllm.distributed.aux_output_connector.routed_experts import (
    RoutedExpertsCacheLayer,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class _WorkerRequestState:
    capture_cursor: int | None = None
    scheduled_cursor: int = 0
    emit_cursor: int = 0
    pending_outputs: int = 0
    # Terminal event received; teardown waits for in-flight step outputs.
    finished: bool = False


@dataclass
class PendingAuxOutput:
    """Own one step's R3 rows until their asynchronous copy is consumed."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    # Rows of earlier tokens read back from the cache, by batch index.
    cached_ranges: dict[int, tuple[int, int, int]]
    # GPU rows taken on the main stream: this step's rows, then cached rows.
    routed_experts_gpu: torch.Tensor
    # Filled by enqueue_cpu_copy on the output copy stream.
    routed_experts: np.ndarray | None = None
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None

    def enqueue_cpu_copy(
        self, num_sampled: np.ndarray, num_rejected: np.ndarray
    ) -> None:
        """Enqueue asynchronous D2H copies; call on the output copy stream."""
        self.num_sampled = num_sampled
        self.num_rejected = num_rejected
        self.routed_experts = self.routed_experts_gpu.to(
            "cpu", non_blocking=True
        ).numpy()

    def process_output(self) -> dict[str, AuxRequestOutput]:
        return self.connector.process_output(self)


class AuxOutputWorkerConnector:
    """Own capture, the routed-experts cache layer, and request cursors."""

    def __init__(self, *, vllm_config: VllmConfig, model: torch.nn.Module) -> None:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=max_num_batched_tokens, vllm_config=vllm_config
        )
        bind_routed_experts_capturer(model, capturer)
        self._capturer = capturer
        self._cache = RoutedExpertsCacheLayer(
            vllm_config, capturer.shape_per_token, capturer.output_dtype
        )
        self._block_tables: BlockTables | None = None
        self._group_id = -1
        self._step_rows: torch.Tensor | None = None
        self._requests: dict[str, _WorkerRequestState] = {}
        self._step_metadata: AuxOutputConnectorMetadata | None = None
        # Steps whose asynchronous copy has not been consumed yet. The engine
        # consumes step outputs in order, so at most max_concurrent_batches
        # are outstanding; guarded by _lock against the output thread.
        self._pending_outputs: list[PendingAuxOutput] = []
        self._lock = Lock()
        self._max_concurrent_batches = vllm_config.max_concurrent_batches
        # Every TP rank caches routed experts with its KV, but only the
        # executor output rank emits them.
        self._is_output_rank = get_tp_group().is_first_rank

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        self._group_id = next(
            i
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if self._cache.layer_name in group.layer_names
        )
        self._block_tables = block_tables

    def write_cache(
        self, num_tokens: int, slot_mappings_by_layer: dict[str, torch.Tensor] | None
    ) -> None:
        """Store this step's rows in the KV cache before KV is saved."""
        assert slot_mappings_by_layer is not None, (
            "routed-experts caching requires per-layer slot mappings"
        )
        rows = self._capturer.snapshot_routing_data(num_tokens)
        self._cache.write(rows, slot_mappings_by_layer[self._cache.layer_name])
        self._step_rows = rows

    def prepare_output(self, input_batch: InputBatch) -> PendingAuxOutput | None:
        """Collect one step's R3 rows for asynchronous CPU transfer."""
        if not self._is_output_rank or self._step_metadata is None:
            return None
        rows = self._step_rows
        assert rows is not None, "routed experts were not cached this step"
        self._step_rows = None

        request_ids = list(input_batch.req_ids)
        num_reqs = len(request_ids)
        query_start_loc = input_batch.query_start_loc_np[: num_reqs + 1]
        token_starts = input_batch.num_computed_tokens_np[:num_reqs]
        # Prefill chunks that end the prompt emit earlier tokens' rows, which
        # were computed in earlier steps or hit in the prefix cache.
        ends_prefill = input_batch.is_prefilling_np[:num_reqs] & (
            token_starts + np.diff(query_start_loc)
            >= input_batch.prefill_len_np[:num_reqs]
        )
        cached_ranges: dict[int, tuple[int, int, int]] = {}
        offset = len(rows)
        for i in np.flatnonzero(ends_prefill):
            emit_start = self._requests[request_ids[i]].emit_cursor
            token_start = int(token_starts[i])
            if emit_start < token_start:
                cached_ranges[int(i)] = (emit_start, token_start, offset)
                offset += token_start - emit_start
        if cached_ranges:
            rows = torch.cat((rows, self._read_cache(input_batch, cached_ranges)))

        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            token_starts=token_starts,
            query_start_loc=query_start_loc,
            cached_ranges=cached_ranges,
            routed_experts_gpu=rows,
        )
        with self._lock:
            assert len(self._pending_outputs) < self._max_concurrent_batches, (
                "auxiliary output step outputs are not consumed in order"
            )
            self._pending_outputs.append(pending_output)
            for request_id in pending_output.request_ids:
                self._requests[request_id].pending_outputs += 1
        return pending_output

    def _read_cache(
        self, input_batch: InputBatch, ranges: dict[int, tuple[int, int, int]]
    ) -> torch.Tensor:
        block_tables = self._block_tables
        assert block_tables is not None
        device = input_batch.idx_mapping.device
        lengths = [end - start for start, end, _ in ranges.values()]
        num_tokens = sum(lengths)
        query_start_loc = torch.tensor(
            np.cumsum([0, *lengths]), dtype=torch.int32, device=device
        )
        positions = torch.from_numpy(
            np.concatenate([np.arange(start, end) for start, end, _ in ranges.values()])
        ).to(device=device, dtype=input_batch.positions.dtype)
        idx_mapping = input_batch.idx_mapping[
            torch.tensor(list(ranges), dtype=torch.int64, device=device)
        ]
        slot_mappings = torch.empty(
            (block_tables.num_kv_cache_groups, num_tokens),
            dtype=block_tables.slot_mappings.dtype,
            device=device,
        )
        block_tables.compute_slot_mappings(
            idx_mapping, query_start_loc, positions, num_tokens, out=slot_mappings
        )
        return self._cache.read(slot_mappings[self._group_id])

    def process_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        """Build request outputs from one consumed R3 copy.

        Request teardown deferred by begin_step is completed here once no
        in-flight step covers the request.
        """
        with self._lock:
            try:
                return self._commit_output(pending)
            finally:
                self._pending_outputs.remove(pending)
                for request_id in pending.request_ids:
                    state = self._requests[request_id]
                    state.pending_outputs -= 1
                    if state.pending_outputs == 0 and state.finished:
                        del self._requests[request_id]

    def _commit_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        routed_experts = pending.routed_experts
        num_sampled = pending.num_sampled
        num_rejected = pending.num_rejected
        assert (
            routed_experts is not None
            and num_sampled is not None
            and num_rejected is not None
        ), "auxiliary output CPU copy was not enqueued"

        outputs: dict[str, AuxRequestOutput] = {}
        # Use the ModelRunner's actual batch boundaries rather than rebuilding them.
        for i, (request_id, token_start, start, end, sampled, rejected) in enumerate(
            zip(
                pending.request_ids,
                pending.token_starts,
                pending.query_start_loc[:-1],
                pending.query_start_loc[1:],
                num_sampled,
                num_rejected,
                strict=True,
            )
        ):
            request_num_tokens = end - start
            assert request_num_tokens > 0, (
                "auxiliary output request token count must be positive"
            )
            state = self._requests[request_id]

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            assert 0 <= rejected <= request_num_tokens, (
                "auxiliary output rejected-token count is invalid"
            )
            rows = routed_experts[start : end - rejected]

            capture_start = int(token_start)
            capture_cursor = state.capture_cursor
            if capture_cursor is not None and capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                assert capture_cursor < state.scheduled_cursor, (
                    "auxiliary output capture has an unbacked token gap"
                )
                capture_start = capture_cursor
            token_end = capture_start + len(rows)
            state.capture_cursor = token_end
            state.scheduled_cursor = int(token_start) + request_num_tokens

            emit_start = state.emit_cursor
            if sampled == 0 or emit_start > token_end:
                continue
            if emit_start < capture_start:
                cached = pending.cached_ranges.get(i)
                assert cached is not None and cached[:2] == (
                    emit_start,
                    capture_start,
                ), "auxiliary output rows before this step were not read back"
                cached_rows = routed_experts[
                    cached[2] : cached[2] + capture_start - emit_start
                ]
                rows = np.concatenate((cached_rows, rows))
            else:
                rows = rows[emit_start - capture_start :]
            outputs[request_id] = AuxRequestOutput(emit_start, rows)
            state.emit_cursor = token_end
        return outputs

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        """Apply one scheduler step's request updates.

        While an unconsumed step covers a finished request, its teardown is
        deferred to that step's process_output.
        """
        self._step_metadata = metadata
        if not self._is_output_rank or metadata is None:
            return
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "auxiliary output request cannot run and finish in one step"
        )
        with self._lock:
            for request_id, emit_start in metadata.requests.items():
                state = self._requests.setdefault(
                    request_id, _WorkerRequestState(emit_cursor=emit_start)
                )
                if state.pending_outputs == 0:
                    # In-flight steps leave the worker cursor behind the
                    # scheduler's optimistic view; only check settled requests.
                    assert emit_start <= state.emit_cursor, (
                        "auxiliary output Scheduler emit cursor moved ahead"
                    )
            for request_id in metadata.finished_requests:
                state = self._requests[request_id]
                state.finished = True
                if state.pending_outputs == 0:
                    del self._requests[request_id]


def get_aux_output_connector(
    model: torch.nn.Module, vllm_config: VllmConfig
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(model=model, vllm_config=vllm_config)
