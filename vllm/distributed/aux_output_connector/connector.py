# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side control plane for execution auxiliary outputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.aux_output_connector.session_objects import (
    aux_object_key,
    record_weight_version,
    weight_version_at,
)
from vllm.distributed.aux_output_connector.tier_versions import SessionTierVersions
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_bytes_per_token,
)
from vllm.v1.core.kv_cache_utils import (
    maybe_convert_block_hash,
    resolve_kv_cache_block_sizes,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import ExternalBlockHash
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

    # The weight version of a block's rows: one version, or runs of
    # `(first position, version)` for a block filled across a weight update.
    Label: TypeAlias = str | tuple[tuple[int, str], ...]


def _label(versions: Sequence[tuple[int, str]], start: int, end: int) -> Label:
    """Label of positions `[start, end)`, each with the version that computed
    it."""
    runs = [(start, weight_version_at(versions, start))]
    for position, version in versions:
        if start < position < end and version != runs[-1][1]:
            runs.append((position, version))
    return runs[0][1] if len(runs) == 1 else tuple(runs)


def _runs(label: Label, start: int, end: int) -> list[tuple[int, str]]:
    if isinstance(label, str):
        return [(start, label)]
    runs: list[tuple[int, str]] = []
    for position, version in label:
        if position >= end:
            break
        if position <= start:
            runs = [(start, version)]
        else:
            runs.append((position, version))
    return runs


def last_version(label: Label) -> str:
    return label if isinstance(label, str) else label[-1][1]


def retained_objects(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, int]:
    """Number of hash-block objects whose routed experts the scheduler retains
    and the worker stores, and the tokens per object."""
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    max_bytes = vllm_config.aux_output_config.max_bytes
    if max_bytes is not None:
        object_nbytes = hash_block_size * get_routed_experts_bytes_per_token(
            vllm_config
        )
        return max(max_bytes // object_nbytes, 1), hash_block_size
    num_blocks = kv_cache_config.num_blocks
    if vllm_config.kv_transfer_config is not None:
        num_blocks *= 2
    return num_blocks * (block_size // hash_block_size), hash_block_size


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
    sessions: dict[str, str] = field(default_factory=dict)
    weight_version: str = "default"
    session_object_size: int = 0


@dataclass(slots=True)
class AuxRequestOutput:
    token_start: int
    rows: np.ndarray


@dataclass(slots=True)
class _SessionRequest:
    session: str
    weight_versions: list[tuple[int, str]] = field(default_factory=list)
    hit_versions: dict[int, dict[int, Label]] = field(default_factory=dict)
    tail: np.ndarray | None = None
    num_sent_hashes: int = 0
    stopped: bool = False


class AuxOutputSchedulerConnector:
    """Build worker metadata without owning auxiliary output stores."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        num_objects: int,
        key_prefix: str | None = None,
        kv_connector: object = None,
    ) -> None:
        # Number of hashes already sent to the worker for each active request.
        self._sent_hash_counts: dict[str, int] = {}
        # Terminal events are delivered with the next connector metadata.
        self._finished_requests: dict[str, PackedBlockHashes | None] = {}
        self._generation = 0
        coordinator = kv_cache_manager.coordinator
        self._coordinator = coordinator
        self._group_id = getattr(coordinator, "full_attention_group_id", None) or 0
        groups = kv_cache_manager.kv_cache_config.kv_cache_groups
        self._groups = {
            group_id: manager
            for group_id, manager in enumerate(coordinator.single_type_managers)
            if manager.prefix_cacheable(manager.manager_config)
            and not groups[group_id].draft_only
        }
        self._block_pool = kv_cache_manager.block_pool
        self._manager = coordinator.single_type_managers[self._group_id]
        self._block_size = self._manager.block_size
        self._hash_block_size = self._block_pool.hash_block_size
        self._step = self._block_size // self._hash_block_size
        self._kv_cached: OrderedDict[ExternalBlockHash, Sequence[BlockHash]] = (
            OrderedDict()
        )
        self._loading: dict[str, set[ExternalBlockHash]] = {}
        self._capacity = num_objects // self._step
        self._retained_hashes: list[bytes] = []
        self._released_hashes: list[bytes] = []
        self._tier = None
        if key_prefix is not None:
            from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

            if (
                coordinator.enable_partial_hash_hits
                or self._block_pool.num_hash_lookahead_tokens > 1
                or type(self._manager) is not FullAttentionManager
                or groups[self._group_id].draft_only
                or any(group.prefix_replay_tokens for group in groups)
                or any(self._block_size % m.block_size for m in self._groups.values())
            ):
                raise ValueError(
                    "AuxOutputConfig.object_key_prefix needs a plain full-attention "
                    "KV cache group of the target model whose block size is a "
                    "multiple of every other group's, prefix-cache hits on KV block "
                    "boundaries, at most one hash look-ahead token and no prefix "
                    "replay."
                )
            self._tier = SessionTierVersions.install(kv_connector, self)
        self._kv_connector = kv_connector
        self._key_prefix = key_prefix
        self._weight_version = "default"
        self._block_versions: dict[int, Label] = {}
        self._session_requests: dict[str, _SessionRequest] = {}

    def set_weight_version(self, weight_version: str) -> None:
        """Set the weight version of later steps; call while no step runs."""
        self._weight_version = weight_version

    def _session(self, request: Request) -> str | None:
        params = request.sampling_params
        if self._key_prefix is None or params is None:
            return None
        return params.aux_session

    def _object_version(
        self, state: _SessionRequest, index: int, group_id: int
    ) -> Label:
        label = state.hit_versions.get(group_id, {}).get(index)
        if label is not None:
            return label
        size = self._groups[group_id].block_size
        return _label(state.weight_versions, index * size, (index + 1) * size)

    def agrees(
        self, label: Label | None, start: int, end: int, full: Sequence[Label | None]
    ) -> bool:
        """Whether a block of positions `[start, end)`, which lies in one
        tracked block, has at every position the version of the tracked
        blocks `full` (from position 0)."""
        tracked = full[start // self._block_size]
        if label is None or tracked is None:
            return False
        if label == tracked:
            return True
        return _runs(label, start, end) == _runs(tracked, start, end)

    def _cached_blocks(
        self, blocks: Sequence[KVCacheBlock], group_id: int
    ) -> Iterator[tuple[int, int, KVCacheBlock]]:
        from vllm.v1.core.kv_cache_utils import null_head_len

        size = self._groups[group_id].block_size
        for index in range(null_head_len(blocks), len(blocks)):
            if not blocks[index].is_null:
                yield index * size, (index + 1) * size, blocks[index]

    def _first_mismatch(self, blocks: Sequence[Sequence[KVCacheBlock]]) -> int | None:
        size = self._block_size
        full = [
            self._block_versions.get(block.block_id) for block in blocks[self._group_id]
        ]
        mismatch = full.index(None) * size if None in full else None
        limit = len(full) * size if mismatch is None else mismatch
        for group_id in self._groups.keys() - {self._group_id}:
            for start, end, block in self._cached_blocks(blocks[group_id], group_id):
                if start >= limit:
                    break
                if not self.agrees(
                    self._block_versions.get(block.block_id), start, end, full
                ):
                    mismatch = limit = start
                    break
        return mismatch

    def request_scheduled(self, request: Request, num_new_tokens: int) -> None:
        """Label the blocks a session request writes in this step, right after
        their allocation, before a later request of the same `schedule()` can
        hit them."""
        session = self._session(request)
        if session is None:
            return
        request_id = request.request_id
        state = self._session_requests.setdefault(request_id, _SessionRequest(session))
        if request_id not in self._sent_hash_counts:
            state.weight_versions = []
        start = request.num_computed_tokens
        record_weight_version(state.weight_versions, start, self._weight_version)
        versions = state.weight_versions
        for manager in self._groups.values():
            size = manager.block_size
            blocks = manager.req_to_blocks[request_id]
            end = min((start + num_new_tokens) // size, len(blocks))
            for index in range(start // size, end):
                self._block_versions[blocks[index].block_id] = _label(
                    versions, index * size, (index + 1) * size
                )

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
            state = self._session_requests.get(request_id)
            if request_id not in self._sent_hash_counts:
                self._sent_hash_counts[request_id] = 0
                if state is not None:
                    sessions[request_id] = state.session
            packed = self._take_new_hashes(
                request, request.num_computed_tokens + num_tokens
            )
            if packed is not None:
                block_hashes_by_request[request_id] = packed
            self._loading.pop(request_id, None)
            if state is not None:
                scheduled_requests[request_id] = state.weight_versions[0][0]
                continue
            assert request.sampling_params is not None
            # A P/D prefiller may cut the prompt short of the requested start.
            scheduled_requests[request_id] = max(
                min(
                    request.sampling_params.routed_experts_prompt_start,
                    request.num_prompt_tokens,
                ),
                0 if request.num_output_tokens == 0 else request.num_tokens - 1,
            )
        for request_id in sessions:
            state = self._session_requests[request_id]
            start = state.weight_versions[0][0]
            state.hit_versions = {
                group_id: {
                    low // manager.block_size: self._block_versions[block.block_id]
                    for low, _, block in self._cached_blocks(
                        manager.req_to_blocks[request_id][
                            : start // manager.block_size
                        ],
                        group_id,
                    )
                }
                for group_id, manager in self._groups.items()
            }
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
            sessions,
            self._weight_version,
            self._block_size,
        )
        self._retained_hashes, self._released_hashes = [], []
        if self._tier is not None:
            self._tier.label_stores()
        for request_id in scheduler_output.finished_req_ids or ():
            self._session_requests.pop(request_id, None)
        return metadata

    def max_cache_hit(self, request: Request) -> int:
        if self._session(request) is not None:
            return self._session_hit(request)
        num_blocks = 0
        for block, keys in self._kv_blocks(request.block_hashes):
            if block in self._kv_cached:
                self._kv_cached.move_to_end(block)
            elif not self._gpu_cached(keys[-1]):
                break
            num_blocks += 1
        return min(num_blocks * self._block_size, request.num_tokens - 1)

    def _session_hit(self, request: Request) -> int:
        """Longest object boundary below the last token whose GPU hit has, in
        every KV group, the tracked group's version at every position. A hit
        lowered below the first mismatch is checked once more: a window group
        reads other blocks there. If it disagrees as well, nothing is hit."""
        size = self._block_size
        num_tokens = (request.num_tokens - 1) // size * size
        for _ in range(2):
            if len(self._groups) == 1 or not num_tokens:
                return num_tokens
            blocks, _, _ = self._coordinator.find_longest_cache_hit(
                request.block_hashes, request.get_max_cache_hit_length(num_tokens)
            )
            mismatch = self._first_mismatch(blocks)
            if mismatch is None:
                return num_tokens
            num_tokens = mismatch - mismatch % size
        return 0

    def cap_external_hit(
        self, request: Request, num_local: int, num_external: int, load_async: bool
    ) -> tuple[int, bool]:
        num_hit = num_local + num_external
        if self._session(request) is not None:
            return self._session_tier_hit(request, num_local, num_hit, load_async)
        if num_hit % self._block_size == 0 and num_hit <= self.max_cache_hit(request):
            return num_external, load_async
        return 0, False

    def _session_tier_hit(
        self, request: Request, num_local: int, num_hit: int, load_async: bool
    ) -> tuple[int, bool]:
        """Longest tier load that ends on an object boundary below the last
        token and whose entries have, in every KV group, the tracked group's
        version at every position. A load is lowered to the object boundary
        below the first mismatch until it agrees (a window group reads other
        entries at a lower boundary), then looked up again."""
        if self._tier is None:
            return 0, False
        size = self._block_size
        end = min(num_hit, request.num_tokens - 1) // size * size
        if end > num_local:
            end = self._tier.consistent_end(request.block_hashes, num_local, end)
        if end <= num_local:
            return 0, False
        if end < num_hit:
            scheduler = self._kv_connector.scheduler
            num_loaded, load_async = scheduler.get_num_new_matched_tokens(
                request, num_local, end - num_local
            )
            if num_loaded != end - num_local:
                return 0, False
        return end - num_local, load_async

    def request_loading(self, request: Request, num_tokens: int) -> None:
        if self._session(request) is not None:
            return
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
        step = self._step
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
        return request_output.rows[:local_end]

    def take_rows_and_keys(
        self,
        request: Request,
        output: dict[str, AuxRequestOutput] | None,
        stopped: bool,
    ) -> tuple[np.ndarray | None, list[str] | None]:
        """Rows and, for a stopped session request, object keys of one output."""
        rows = self.take_output(request, output)
        state = self._session_requests.get(request.request_id)
        if state is None or rows is None:
            return rows, None
        size = self._block_size
        token_end = request.num_tokens - 1
        num_hashes = self._sent_hash_counts.get(
            request.request_id, state.num_sent_hashes
        )
        boundary = min(token_end, num_hashes * self._hash_block_size) // size * size
        if token_end - len(rows) > boundary:
            assert state.tail is not None
            rows = np.concatenate((state.tail, rows))
        start = token_end - len(rows)
        assert start <= boundary, "session auxiliary output rows are not contiguous"
        state.tail = rows = rows[boundary - start :]
        keys = None
        if stopped:
            keys = [
                aux_object_key(
                    self._key_prefix,
                    state.session,
                    request.block_hashes[(index + 1) * self._step - 1].hex(),
                    last_version(self._object_version(state, index, self._group_id)),
                )
                for index in range(boundary // size)
            ]
        return rows if request.is_finished() else None, keys

    def request_finished(self, request: Request) -> None:
        """Queue a request's terminal event and final block hashes."""
        request_id = request.request_id
        self._loading.pop(request_id, None)
        num_blocks = request.num_computed_tokens // self._block_size
        if self._session(request) is not None:
            num_blocks = 0
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
        state = self._session_requests.get(request_id)
        if state is not None:
            state.num_sent_hashes = self._sent_hash_counts[request_id]
            state.stopped = request.is_finished()
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
        for request_id, block_hashes in self._finished_requests.items():
            state = self._session_requests.get(request_id)
            if state is not None and block_hashes is not None:
                state.num_sent_hashes -= (
                    len(block_hashes.data) // block_hashes.item_size
                )
        self._finished_requests.clear()
        self._generation += 1
        self._released_hashes.extend(chain.from_iterable(self._kv_cached.values()))
        self._kv_cached.clear()
        self._block_versions.clear()
