# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.aux_output_connector.session_objects import (
    aux_object_key,
    record_weight_version,
    weight_version_at,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_bytes_per_token,
)
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import ExternalBlockHash
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def retained_objects(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, int, int]:
    """Number of hash-block objects whose routed experts the scheduler retains
    and the worker stores, the KV cache block size and the hash block size."""
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    max_bytes = vllm_config.aux_output_config.max_bytes
    if max_bytes is not None:
        object_nbytes = hash_block_size * get_routed_experts_bytes_per_token(
            vllm_config
        )
        return max(max_bytes // object_nbytes, 1), block_size, hash_block_size
    num_blocks = kv_cache_config.num_blocks
    if vllm_config.kv_transfer_config is not None:
        num_blocks *= 2
    num_objects = num_blocks * (block_size // hash_block_size)
    return num_objects, block_size, hash_block_size


@dataclass(slots=True)
class PackedBlockHashes:
    """Contiguous, self-contained block hashes for scheduler-worker IPC."""

    data: bytes
    item_size: int

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), self.item_size):
            yield self.data[start : start + self.item_size]


@dataclass(slots=True)
class AuxOutputConnectorMetadata:
    generation: int
    requests: dict[str, int]
    block_hashes: dict[str, PackedBlockHashes]
    finished_requests: tuple[str, ...]
    retained_block_hashes: PackedBlockHashes | None = None
    released_block_hashes: PackedBlockHashes | None = None
    sessions: dict[str, str] | None = None
    released_sessions: tuple[str, ...] = ()
    weight_version: str = "default"


@dataclass(slots=True)
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output payloads or stores."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        num_objects: int,
        object_store: bool = False,
        key_prefix: str = "",
        session_idle_timeout_s: float = 3600.0,
    ) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0
        coordinator = kv_cache_manager.coordinator
        self._group_id = getattr(coordinator, "full_attention_group_id", None) or 0
        self._block_pool = kv_cache_manager.block_pool
        self._manager = coordinator.single_type_managers[self._group_id]
        self._block_size = self._manager.block_size
        self._hash_block_size = self._block_pool.hash_block_size
        self._kv_cached: OrderedDict[ExternalBlockHash, Sequence[BlockHash]] = (
            OrderedDict()
        )
        self._loading: dict[str, set[ExternalBlockHash]] = {}
        self._capacity = num_objects // (self._block_size // self._hash_block_size)
        self._retained_hashes: list[bytes] = []
        self._released_hashes: list[bytes] = []
        self._object_size = self._block_size if object_store else None
        self._key_prefix = key_prefix
        self._weight_version = "default"
        # Weight version of each object a session published, and of the
        # latest commit of each object by any session.
        self._published: dict[str, dict[bytes, str]] = {}
        self._commit_versions: dict[bytes, str] = {}
        self._weight_versions: dict[str, list[tuple[int, str]]] = {}
        self._session_idle_timeout_s = session_idle_timeout_s
        self._session_requests: Counter[str] = Counter()
        self._session_last_used: dict[str, float] = {}
        self._next_session_sweep = 0.0
        self._tails: dict[str, tuple[int, np.ndarray]] = {}

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, Request],
    ) -> AuxOutputConnectorMetadata:
        """Build one step's incremental worker metadata."""
        scheduled_requests: dict[str, int] = {}
        block_hashes_by_request: dict[str, PackedBlockHashes] = {}
        sessions: dict[str, str] = {}
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = requests[request_id]
            if request_id not in self._sent_hash_counts:
                self._sent_hash_counts[request_id] = 0
                if (session := self._session(request)) is not None:
                    sessions[request_id] = session
                    self._session_requests[session] += 1
            if self._object_size is not None:
                record_weight_version(
                    self._weight_versions.setdefault(request_id, []),
                    request.num_computed_tokens,
                    self._weight_version,
                )
            packed = self._take_new_hashes(
                request, request.num_computed_tokens + num_tokens
            )
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._loading.pop(request_id, None)
            assert request.sampling_params is not None
            # A P/D prefiller may cut the prompt short of the requested start.
            scheduled_requests[request_id] = max(
                min(
                    request.sampling_params.routed_experts_prompt_start,
                    request.num_prompt_tokens,
                ),
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
        # A settled token can complete a hash block after the next async schedule
        # was built. Send a hash-only update if the request was not rescheduled.
        if len(self._sent_hash_counts) != len(scheduled_requests):
            for request_id in self._sent_hash_counts:
                if request_id in scheduled_requests:
                    continue
                request = requests[request_id]
                packed = self._take_new_hashes(request, request.num_computed_tokens)
                if packed is not None:
                    block_hashes_by_request[request_id] = packed
        # Sending transfers ownership of these one-shot events.
        finished_requests = tuple(self._finished_requests)
        block_hashes_by_request.update(
            (request_id, block_hashes)
            for request_id, block_hashes in self._finished_requests.items()
            if block_hashes is not None
        )
        self._finished_requests = {}
        metadata = AuxOutputConnectorMetadata(
            self._generation,
            scheduled_requests,
            block_hashes_by_request,
            finished_requests,
            self._pack_hashes(self._retained_hashes),
            self._pack_hashes(self._released_hashes),
            sessions or None,
            self._forget_idle_sessions(),
            self._weight_version,
        )
        self._retained_hashes, self._released_hashes = [], []
        return metadata

    def _forget_idle_sessions(self) -> tuple[str, ...]:
        now = time.monotonic()
        if now < self._next_session_sweep:
            return ()
        self._next_session_sweep = now + 1.0
        deadline = now - self._session_idle_timeout_s
        idle = tuple(
            session
            for session, last_used in self._session_last_used.items()
            if last_used < deadline and not self._session_requests[session]
        )
        for session in idle:
            del self._session_last_used[session]
            del self._session_requests[session]
            self._published.pop(session, None)
        if idle:
            live = set(chain.from_iterable(self._published.values()))
            self._commit_versions = {
                object_hash: weight_version
                for object_hash, weight_version in self._commit_versions.items()
                if object_hash in live
            }
        return idle

    def set_weight_version(self, weight_version: str) -> None:
        """Set the weight version of later steps; call while no step runs."""
        self._weight_version = weight_version

    def _session(self, request: Request) -> str | None:
        if self._object_size is None or request.sampling_params is None:
            return None
        return request.sampling_params.aux_session

    def _object_hashes(self, hashes: Sequence[BlockHash]) -> Iterator[bytes]:
        assert self._object_size is not None
        step = self._object_size // self._hash_block_size
        for end in range(step, len(hashes) + 1, step):
            yield bytes(hashes[end - 1])

    def max_cache_hit(self, request: Request) -> int:
        if (session := self._session(request)) is not None:
            assert self._object_size is not None
            published = self._published.get(session, {})
            num_objects = 0
            for object_hash in self._object_hashes(request.block_hashes):
                if (
                    object_hash not in published
                    or published[object_hash] != self._commit_versions[object_hash]
                ):
                    break
                num_objects += 1
            last = (request.num_tokens - 1) // self._object_size
            return min(num_objects, last) * self._object_size
        num_blocks = 0
        for block, keys in self._kv_blocks(request.block_hashes):
            if block in self._kv_cached:
                self._kv_cached.move_to_end(block)
            elif not self._gpu_cached(keys[-1]):
                break
            num_blocks += 1
        return min(num_blocks * self._block_size, request.num_tokens - 1)

    def cap_external_hit(
        self, request: Request, num_local: int, num_external: int
    ) -> int:
        num_hit = num_local + num_external
        if num_hit % self._block_size == 0 and num_hit <= self.max_cache_hit(request):
            return num_external
        return 0

    def request_loading(self, request: Request, num_tokens: int) -> None:
        blocks = islice(
            self._kv_blocks(request.block_hashes), num_tokens // self._block_size
        )
        self._loading[request.request_id] = {block for block, _ in blocks}

    def _release_unheld(self) -> None:
        excess = len(self._kv_cached) - self._capacity
        if excess <= 0:
            return
        loading = set().union(*self._loading.values())
        victims = []
        for block, keys in self._kv_cached.items():
            if block not in loading and not self._gpu_cached(keys[-1]):
                victims.append(block)
                if len(victims) == excess:
                    break
        for block in victims:
            self._released_hashes.extend(self._kv_cached.pop(block))

    def _gpu_cached(self, block_hash: BlockHash) -> list[KVCacheBlock] | None:
        return self._manager.block_pool.get_cached_block(block_hash, [self._group_id])

    def _kv_blocks(
        self, hashes: Sequence[BlockHash]
    ) -> Iterator[tuple[ExternalBlockHash, Sequence[BlockHash]]]:
        step = self._block_size // self._hash_block_size
        for end in range(step, len(hashes) + 1, step):
            yield maybe_convert_block_hash(hashes[end - 1]), hashes[end - step : end]

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
        rows = request_output.rows[:local_end]
        if (session := self._session(request)) is None:
            if self._object_size is not None:
                self._record_commits(request, request_output.token_start, token_end)
            return rows
        assert self._object_size is not None
        tail_start, tail = self._tails.get(
            request_id, (request_output.token_start, rows[:0])
        )
        assert request_output.token_start == tail_start + len(tail), (
            "auxiliary output rows of a session request are not contiguous"
        )
        weight_versions = self._weight_versions[request_id]
        first_object = max(
            (tail_start + len(tail)) // self._object_size,
            -(-weight_versions[0][0] // self._object_size),
        )
        tail = np.concatenate((tail, rows))
        num_objects = (tail_start + len(tail)) // self._object_size
        published = self._published.setdefault(session, {})
        object_hashes = self._object_hashes(request.block_hashes)
        for index, object_hash in enumerate(
            islice(object_hashes, first_object, num_objects), first_object
        ):
            weight_version = weight_version_at(
                weight_versions, (index + 1) * self._object_size - 1
            )
            published[object_hash] = weight_version
            self._commit_versions[object_hash] = weight_version
        boundary = token_end // self._object_size * self._object_size
        if boundary > tail_start:
            tail = tail[boundary - tail_start :]
            tail_start = boundary
        self._tails[request_id] = (tail_start, tail)
        return tail if request.is_finished() else None

    def _record_commits(self, request: Request, start: int, end: int) -> None:
        assert self._object_size is not None
        weight_versions = self._weight_versions[request.request_id]
        first = max(
            start // self._object_size,
            -(-weight_versions[0][0] // self._object_size),
        )
        object_hashes = self._object_hashes(request.block_hashes)
        for index, object_hash in enumerate(
            islice(object_hashes, first, end // self._object_size), first
        ):
            if object_hash in self._commit_versions:
                self._commit_versions[object_hash] = weight_version_at(
                    weight_versions, (index + 1) * self._object_size - 1
                )

    def take_object_keys(self, request: Request) -> list[str] | None:
        """Keys of a finished session request's objects before its tail."""
        session = self._session(request)
        if session is None or request.request_id not in self._tails:
            return None
        assert self._object_size is not None
        tail_start, _ = self._tails.pop(request.request_id)
        published = self._published[session]
        object_hashes = self._object_hashes(request.block_hashes)
        return [
            aux_object_key(
                self._key_prefix, session, object_hash.hex(), published[object_hash]
            )
            for object_hash in islice(object_hashes, tail_start // self._object_size)
        ]

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        request_id = request.request_id
        self._loading.pop(request_id, None)
        self._weight_versions.pop(request_id, None)
        if (session := self._session(request)) is not None:
            if request.is_finished():
                self._tails.pop(request_id, None)
            if request_id in self._sent_hash_counts:
                self._session_requests[session] -= 1
            self._session_last_used[session] = time.monotonic()
        num_blocks = request.num_computed_tokens // self._block_size
        for block, keys in islice(self._kv_blocks(request.block_hashes), num_blocks):
            if block not in self._kv_cached and self._gpu_cached(keys[-1]):
                self._kv_cached[block] = keys
                self._retained_hashes.extend(keys)
        self._release_unheld()
        if request_id not in self._sent_hash_counts:
            return
        # The next metadata delivers this terminal event to the worker.
        self._finished_requests[request_id] = self._take_new_hashes(
            request, request.num_computed_tokens
        )
        del self._sent_hash_counts[request_id]

    @staticmethod
    def _pack_hashes(hashes: Sequence[bytes]) -> PackedBlockHashes | None:
        if not hashes:
            return None
        return PackedBlockHashes(b"".join(hashes), len(hashes[0]))

    def _take_new_hashes(
        self, request: Request, num_computed_tokens: int
    ) -> PackedBlockHashes | None:
        block_hashes = request.block_hashes
        num_sent = self._sent_hash_counts[request.request_id]
        assert num_sent <= len(block_hashes), "KV block-hash history shrank"
        num_keyed = min(len(block_hashes), num_computed_tokens // self._hash_block_size)
        if num_keyed <= num_sent:
            return None
        self._sent_hash_counts[request.request_id] = num_keyed
        return self._pack_hashes(block_hashes[num_sent:num_keyed])

    def reset(self) -> None:
        """Start a new auxiliary output namespace after a prefix-cache reset."""
        # The worker drops temporary state on generation changes; resend hashes.
        self._sent_hash_counts.clear()
        self._finished_requests.clear()
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
