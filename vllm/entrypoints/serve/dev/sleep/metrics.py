# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics for dispatched sleep-mode frontend operations."""

from threading import Lock
from typing import Literal

from prometheus_client import REGISTRY, CollectorRegistry

from vllm.v1.metrics.utils import OPERATION_DURATION_BUCKETS, OperationMetrics

Operation = Literal["sleep", "release_kv_cache_memory", "wake"]
BUCKETS = OPERATION_DURATION_BUCKETS


class SleepModeOperationMetrics(OperationMetrics[Operation]):
    def __init__(self, registry: CollectorRegistry | None = None):
        super().__init__(
            "vllm:sleep_mode_operation_duration_seconds",
            "Duration of one sleep-mode operation.",
            "vllm:sleep_mode_operations_in_flight",
            "Sleep-mode operations currently awaited.",
            buckets=BUCKETS,
            registry=REGISTRY if registry is None else registry,
        )


_metrics: SleepModeOperationMetrics | None = None
_metrics_lock = Lock()


def sleep_mode_operation_metrics() -> SleepModeOperationMetrics:
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = SleepModeOperationMetrics()
    return _metrics
