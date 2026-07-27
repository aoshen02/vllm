# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side routed-experts capture state."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capture.async_output import (
    RoutedExpertsTensors,
    RoutedExpertsWriteTask,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_routed_experts_output_rank,
    get_routing_slot_shape_and_dtype,
    require_full_attn_group_id,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsWorkerWriter,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


class RoutedExpertsCaptureState:
    """Own worker capture resources shared by both model runners."""

    def __init__(
        self,
        capturer: RoutedExpertsCapturer,
        writer: RoutedExpertsWorkerWriter | None,
        full_attn_group_id: int,
    ) -> None:
        self.capturer: RoutedExpertsCapturer | None = capturer
        self.writer = writer
        self.full_attn_group_id = full_attn_group_id

    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        max_num_batched_tokens: int,
    ) -> RoutedExpertsCaptureState:
        full_attn_group_id = require_full_attn_group_id(kv_cache_config)
        capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=max_num_batched_tokens,
            vllm_config=vllm_config,
        )
        bind_routed_experts_capturer(model, capturer)

        writer = None
        parallel_config = vllm_config.parallel_config
        if parallel_config.rank == get_routed_experts_output_rank():
            slot_shape, slot_dtype = get_routing_slot_shape_and_dtype(
                vllm_config, kv_cache_config
            )
            writer = RoutedExpertsWorkerWriter(
                instance_id=vllm_config.instance_id,
                dp_rank=parallel_config.data_parallel_rank,
                slot_shape=slot_shape,
                dtype=slot_dtype,
            )
        return cls(capturer, writer, full_attn_group_id)

    @property
    def can_write(self) -> bool:
        return self.writer is not None

    def clear(self) -> None:
        if self.capturer is not None:
            self.capturer.clear_buffer()

    def get_device_buffer(self) -> torch.Tensor:
        assert self.capturer is not None, "routed-experts capture state is closed"
        return self.capturer.get_device_buffer()

    def make_write_task(
        self,
        slot_mapping: torch.Tensor,
        num_tokens: int,
        *,
        request_ids: tuple[str, ...] = (),
        query_start_locs: np.ndarray | None = None,
        token_starts: np.ndarray | None = None,
        artifact_sink: Callable[[str, int, np.ndarray], None] | None = None,
    ) -> RoutedExpertsWriteTask | None:
        writer = self.writer
        if writer is None:
            return None
        tensors = RoutedExpertsTensors(
            routing_data=self.get_device_buffer()[:num_tokens].clone(),
            slot_mapping=slot_mapping[:num_tokens].clone(),
        )
        return RoutedExpertsWriteTask(
            routed_experts_tensors=tensors,
            writer=writer,
            request_ids=request_ids,
            query_start_locs=query_start_locs,
            token_starts=token_starts,
            artifact_sink=artifact_sink,
        )

    def store_batch(
        self,
        routing_data: np.ndarray,
        slot_mapping: np.ndarray,
    ) -> None:
        assert self.writer is not None, "routed-experts writer is unavailable"
        self.writer.store_batch(routing_data, slot_mapping)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.capturer = None
