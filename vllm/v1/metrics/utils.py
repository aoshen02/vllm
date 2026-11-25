# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from time import perf_counter
from typing import Generic, TypeAlias, TypeVar

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

PromMetric: TypeAlias = Gauge | Counter | Histogram

OPERATION_DURATION_BUCKETS = (0.01, 0.1, 1, 10, 30, 60, 120, 300, 600)

_OpT = TypeVar("_OpT", bound=str)


class OperationMetrics(Generic[_OpT]):
    """Duration histogram and in-flight gauge for labeled frontend operations."""

    def __init__(
        self,
        duration_name: str,
        duration_documentation: str,
        in_flight_name: str,
        in_flight_documentation: str,
        *,
        buckets: Sequence[float] = OPERATION_DURATION_BUCKETS,
        registry: CollectorRegistry | None = None,
    ) -> None:
        registry = REGISTRY if registry is None else registry
        self.duration = Histogram(
            duration_name,
            duration_documentation,
            ["operation"],
            buckets=buckets,
            registry=registry,
        )
        self.in_flight = Gauge(
            in_flight_name,
            in_flight_documentation,
            ["operation"],
            multiprocess_mode="livesum",
            registry=registry,
        )

    @contextmanager
    def record(self, operation: _OpT) -> Iterator[None]:
        in_flight = self.in_flight.labels(operation)
        in_flight.inc()
        started = perf_counter()
        try:
            yield
        finally:
            in_flight.dec()
            self.duration.labels(operation).observe(perf_counter() - started)


def create_metric_per_engine(
    metric: PromMetric,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, PromMetric]:
    """Create a labeled metric child for each engine index."""
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
