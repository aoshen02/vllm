# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous routed-experts output handling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RoutedExpertsWriteTask:
    """Copy and publish one step of routed-experts output."""

    # (num_scheduled_tokens, num_layers, moe_top_k)
    routing_data: torch.Tensor
    request_ids: tuple[str, ...]
    query_start_locs: np.ndarray
    token_starts: np.ndarray
    artifact_sink: Callable[[str, int, np.ndarray], None]
    _routing_data_cpu: torch.Tensor | None = field(init=False, default=None)

    def start_copy(self) -> None:
        """Start copying the routed-experts tensors on the current stream."""
        self._routing_data_cpu = (
            self.routing_data
            if self.routing_data.device.type == "cpu"
            else self.routing_data.to("cpu", non_blocking=True)
        )

    def finalize(self) -> None:
        """Publish the copied routing data by logical request coordinates."""
        assert self._routing_data_cpu is not None, (
            "routed-experts CPU tensors are unavailable; call start_copy first"
        )
        routing_data = self._routing_data_cpu.numpy()
        num_requests = len(self.request_ids)
        if (
            self.query_start_locs.shape != (num_requests + 1,)
            or self.token_starts.shape != (num_requests,)
            or self.query_start_locs[0] != 0
            or self.query_start_locs[-1] != len(routing_data)
            or np.any(self.query_start_locs[1:] < self.query_start_locs[:-1])
        ):
            raise RuntimeError("invalid routed-experts logical batch metadata")
        for index, request_id in enumerate(self.request_ids):
            row_start = int(self.query_start_locs[index])
            row_end = int(self.query_start_locs[index + 1])
            self.artifact_sink(
                request_id,
                int(self.token_starts[index]),
                routing_data[row_start:row_end],
            )
