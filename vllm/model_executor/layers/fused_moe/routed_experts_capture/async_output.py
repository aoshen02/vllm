# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous routed-experts output handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsWorkerWriter,
)


class RoutedExpertsTensors(NamedTuple):
    """Store one step of routed-experts tensors pending async D2H."""

    # (num_scheduled_tokens, num_layers, moe_top_k)
    routing_data: torch.Tensor
    # (num_scheduled_tokens,)
    slot_mapping: torch.Tensor | None

    def to_cpu_nonblocking(self) -> RoutedExpertsTensors:
        """Copy the tensors to CPU without blocking the current stream."""
        if self.routing_data.device.type == "cpu":
            return self
        return RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            (
                self.slot_mapping.to("cpu", non_blocking=True)
                if self.slot_mapping is not None
                else None
            ),
        )

    def tolists(self) -> RoutedExpertsLists:
        """Convert the tensors to the numpy-backed worker representation."""
        return RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            (
                self.slot_mapping.cpu().numpy()
                if self.slot_mapping is not None
                else None
            ),
        )


class RoutedExpertsLists(NamedTuple):
    """Store one step of CPU routed-experts data and slot indices."""

    # (num_scheduled_tokens, num_layers, moe_top_k)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    slot_mapping: np.ndarray | None


@dataclass
class RoutedExpertsWriteTask:
    """Copy and publish one step of routed-experts output."""

    routed_experts_tensors: RoutedExpertsTensors
    writer: RoutedExpertsWorkerWriter | None
    request_ids: tuple[str, ...] = ()
    query_start_locs: np.ndarray | None = None
    token_starts: np.ndarray | None = None
    artifact_sink: Callable[[str, int, np.ndarray], None] | None = None
    _routed_experts_tensors_cpu: RoutedExpertsTensors | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        if (self.writer is None) == (self.artifact_sink is None):
            raise ValueError(
                "routed-experts write task requires exactly one destination"
            )
        if self.writer is not None and self.routed_experts_tensors.slot_mapping is None:
            raise ValueError("shared-slot writer requires a slot mapping")

    def start_copy(self) -> None:
        """Start copying the routed-experts tensors on the current stream."""
        self._routed_experts_tensors_cpu = (
            self.routed_experts_tensors.to_cpu_nonblocking()
        )

    def finalize(self, num_rejected_tokens: np.ndarray | None = None) -> None:
        """Publish the copied routing data."""
        assert self._routed_experts_tensors_cpu is not None, (
            "routed-experts CPU tensors are unavailable; call start_copy first"
        )
        routed_experts = self._routed_experts_tensors_cpu.tolists()
        if self.writer is not None:
            assert routed_experts.slot_mapping is not None
            self.writer.store_batch(
                routed_experts.routing_data,
                routed_experts.slot_mapping,
            )
        if self.artifact_sink is None:
            return
        assert self.query_start_locs is not None
        assert self.token_starts is not None
        if num_rejected_tokens is None:
            num_rejected_tokens = np.zeros(len(self.request_ids), dtype=np.int32)
        for index, request_id in enumerate(self.request_ids):
            row_start = int(self.query_start_locs[index])
            row_end = int(self.query_start_locs[index + 1])
            accepted_end = row_end - int(num_rejected_tokens[index])
            self.artifact_sink(
                request_id,
                int(self.token_starts[index]),
                routed_experts.routing_data[row_start:accepted_end],
            )
