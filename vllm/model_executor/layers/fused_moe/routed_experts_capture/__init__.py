# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts capture and KV-connector sidecars.

The worker captures logical expert IDs for enable_return_routed_experts and
writes them into a scheduler-visible slot buffer. Supported KV connectors keep
the routing rows with their KV blocks when blocks move off GPU.
"""

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
from vllm.model_executor.layers.fused_moe.routed_experts_capture.manager import (
    RoutedExpertsManager,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsShmWriter,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.state import (
    RoutedExpertsCaptureState,
)

__all__ = [
    "RoutedExpertsCaptureState",
    "RoutedExpertsCapturer",
    "RoutedExpertsManager",
    "RoutedExpertsTensors",
    "RoutedExpertsShmWriter",
    "RoutedExpertsWriteTask",
    "bind_routed_experts_capturer",
    "get_routed_experts_output_rank",
    "get_routing_slot_shape_and_dtype",
    "require_full_attn_group_id",
]
