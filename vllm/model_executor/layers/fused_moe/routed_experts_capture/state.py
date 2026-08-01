# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side routed-experts capture state."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capture.async_output import (
    RoutedExpertsWriteTask,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class RoutedExpertsCaptureState:
    """Own worker capture resources shared by both model runners."""

    def __init__(
        self,
        capturer: RoutedExpertsCapturer,
    ) -> None:
        self.capturer = capturer

    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        vllm_config: VllmConfig,
        max_num_batched_tokens: int,
    ) -> RoutedExpertsCaptureState:
        capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=max_num_batched_tokens,
            vllm_config=vllm_config,
        )
        bind_routed_experts_capturer(model, capturer)

        return cls(capturer)

    def clear(self) -> None:
        self.capturer.clear_buffer()

    def get_device_buffer(self) -> torch.Tensor:
        return self.capturer.get_device_buffer()

    def make_write_task(
        self,
        num_tokens: int,
        *,
        request_ids: tuple[str, ...],
        query_start_locs: np.ndarray,
        token_starts: np.ndarray,
        artifact_sink: Callable[[str, int, np.ndarray], None],
    ) -> RoutedExpertsWriteTask:
        return RoutedExpertsWriteTask(
            routing_data=self.get_device_buffer()[:num_tokens].clone(),
            request_ids=request_ids,
            query_start_locs=query_start_locs,
            token_starts=token_starts,
            artifact_sink=artifact_sink,
        )
