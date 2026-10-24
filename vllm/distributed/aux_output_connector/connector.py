# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class AuxOutputConnectorMetadata:
    requests: dict[str, int]
    finished_requests: tuple[str, ...]


@dataclass
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads."""

    def __init__(self) -> None:
        self._active_requests: set[str] = set()
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: list[str] = []

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's worker metadata."""
        scheduled_requests: dict[str, int] = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            self._active_requests.add(request_id)
            request = requests[request_id]
            assert request.sampling_params is not None
            # A P/D prefiller may cut the prompt short of the requested start.
            scheduled_requests[request_id] = max(
                min(
                    request.sampling_params.routed_experts_prompt_start,
                    request.num_prompt_tokens,
                ),
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
        finished_requests = tuple(self._finished_requests)
        self._finished_requests = []
        return AuxOutputConnectorMetadata(scheduled_requests, finished_requests)

    def take_output(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request."""
        request_id = request.request_id
        assert output is not None and request_id in output, (
            f"auxiliary output worker output is missing {request_id}"
        )
        request_output = output[request_id]
        token_end = request.num_tokens - 1
        local_end = token_end - request_output.token_start
        if local_end < 0:
            assert not request.is_finished(), (
                "finished auxiliary output output has no accepted token range: "
                f"request={request_id}, token_end={token_end}, "
                f"output_start={request_output.token_start}, "
                "output_end="
                f"{request_output.token_start + len(request_output.rows)}"
            )
            return None
        assert local_end <= len(request_output.rows), (
            "auxiliary output worker output has an invalid token range: "
            f"request={request_id}, token_end={token_end}, "
            f"output_start={request_output.token_start}, "
            f"output_end={request_output.token_start + len(request_output.rows)}"
        )
        return request_output.rows[:local_end]

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event."""
        if request.request_id in self._active_requests:
            self._active_requests.remove(request.request_id)
            self._finished_requests.append(request.request_id)
