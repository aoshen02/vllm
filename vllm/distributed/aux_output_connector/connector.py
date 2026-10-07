# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class PackedBlockHashes:
    """Contiguous, self-contained block hashes for scheduler-worker IPC."""

    data: bytes
    item_size: int

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.item_size):
            yield self.data[start : start + self.item_size]


@dataclass
class AuxOutputConnectorMetadata:
    generation: int
    requests: dict[str, int]
    block_hashes: dict[str, PackedBlockHashes]
    finished_requests: tuple[str, ...]


@dataclass
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


@dataclass(eq=False)
class AuxStepOutput(Mapping[str, AuxRequestOutput]):
    """One worker step's R3 rows for all requests, keyed by request ID.

    Rows captured in this step are packed into one array so the worker-to-engine
    IPC carries a single buffer instead of one array per request.

    Attributes:
        rows: Packed rows captured in this step.
        spans: Request ID to ``(token_start, lo, hi)``, where ``rows[lo:hi]``
            are the request's rows starting at token ``token_start``.
        materialized: Outputs rebuilt from the store or capture buffer.

    """

    rows: np.ndarray
    spans: dict[str, tuple[int, int, int]]
    materialized: dict[str, AuxRequestOutput]

    def __getitem__(self, request_id: str) -> AuxRequestOutput:
        span = self.spans.get(request_id)
        if span is None:
            return self.materialized[request_id]
        token_start, lo, hi = span
        return AuxRequestOutput(token_start, self.rows[lo:hi])

    def __iter__(self) -> Iterator[str]:
        yield from self.spans
        yield from self.materialized

    def __len__(self) -> int:
        return len(self.spans) + len(self.materialized)


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(self) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        # Accepted rows held until request finish: (buffer, num_rows).
        self._held_rows: dict[str, tuple[np.ndarray, int]] = {}
        self._generation = 0

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        for request_id in scheduler_output.num_scheduled_tokens:
            num_sent = self._sent_hash_counts.setdefault(request_id, 0)
            request = requests[request_id]
            packed = self._pack_new_hashes(request.block_hashes, num_sent)
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._sent_hash_counts[request_id] = len(request.block_hashes)
            assert request.sampling_params is not None
            scheduled_requests[request_id] = max(
                request.sampling_params.routed_experts_prompt_start,
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
        # A settled token can complete a hash block after the next async schedule
        # was built. Send a hash-only update if the request was not rescheduled.
        if len(self._sent_hash_counts) != len(scheduled_requests):
            for request_id, num_sent in self._sent_hash_counts.items():
                if request_id in scheduled_requests:
                    continue
                request = requests[request_id]
                packed = self._pack_new_hashes(request.block_hashes, num_sent)
                if packed is None:
                    continue
                block_hashes_by_request[request_id] = packed
                self._sent_hash_counts[request_id] = len(request.block_hashes)
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        self._finished_requests = {}
        return AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
        )

    def take_output(
        self,
        request: Request,
        output: Mapping[str, AuxRequestOutput] | None,
    ) -> np.ndarray | None:
        """Return the accepted R3 rows for one scheduled request.

        The frontend only exposes routed experts on the final output, so rows
        are held here and returned once when the request finishes. Requests
        with stop strings may be finished by the frontend and get their rows
        every step instead.
        """
        request_id = request.request_id
        request_output = output.get(request_id) if output is not None else None
        assert request_output is not None, (
            f"auxiliary output worker output is missing {request_id}"
        )
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
        rows = request_output.rows[:local_end]
        assert request.sampling_params is not None
        if request.sampling_params.stop:
            return rows
        return self._hold_rows(request, rows)

    def _hold_rows(self, request: Request, rows: np.ndarray) -> np.ndarray | None:
        held = self._held_rows.pop(request.request_id, None)
        if held is None:
            if request.is_finished():
                return rows
            if len(rows) == 0:
                return None
            held = (np.empty((max(512, 2 * len(rows)), *rows.shape[1:]), rows.dtype), 0)
        buffer, num_rows = held
        end = num_rows + len(rows)
        if end > len(buffer):
            grown = np.empty(
                (max(end, 2 * len(buffer)), *buffer.shape[1:]), buffer.dtype
            )
            grown[:num_rows] = buffer[:num_rows]
            buffer = grown
        buffer[num_rows:end] = rows
        if request.is_finished():
            return buffer[:end]
        self._held_rows[request.request_id] = (buffer, end)
        return None

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        request_id = request.request_id
        if request.is_finished():
            # Preemption also calls this; a preempted request keeps its rows.
            self._held_rows.pop(request_id, None)
        num_sent = self._sent_hash_counts.pop(request_id, None)
        if num_sent is None:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._pack_new_hashes(
            request.block_hashes, num_sent
        )

    @staticmethod
    def _pack_new_hashes(
        block_hashes: Sequence[bytes], num_sent: int
    ) -> PackedBlockHashes | None:
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        new_hashes = block_hashes[num_sent:]
        if not new_hashes:
            return None
        return PackedBlockHashes(b"".join(new_hashes), len(new_hashes[0]))

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
