# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Duration and in-flight metrics for RL weight-transfer operations."""

from threading import Lock
from typing import Literal

from prometheus_client import REGISTRY, CollectorRegistry

from vllm.v1.metrics.utils import OperationMetrics

Operation = Literal["init", "shutdown", "start", "start_draft", "update", "finish"]


class WeightOperationMetrics(OperationMetrics[Operation]):
    def __init__(self, registry: CollectorRegistry | None = None):
        super().__init__(
            "vllm:rl_weight_update_operation_duration_seconds",
            "Duration of one weight-transfer operation.",
            "vllm:rl_weight_update_operations_in_flight",
            "Weight-transfer operations currently awaited.",
            registry=REGISTRY if registry is None else registry,
        )


_metrics: WeightOperationMetrics | None = None
_metrics_lock = Lock()


def weight_operation_metrics() -> WeightOperationMetrics:
    # Lazy, so the collectors exist only after multiprocess Prometheus setup.
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = WeightOperationMetrics()
    return _metrics
