# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight versions of the session KV blocks held by a KV offloading tier."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.base import (
    OffloadingManager,
    get_offload_group_idx,
    make_offload_key,
)

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import (
        AuxOutputSchedulerConnector,
        Label,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
        GroupOffloadConfig,
    )
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_offload.base import (
        ConfigInfoMapping,
        LoadStoreSpec,
        LookupResult,
        OffloadingEvent,
        OffloadKey,
        PrepareStoreOutput,
        ReqContext,
        RequestOffloadingContext,
        StoreTrigger,
    )
    from vllm.v1.metrics.cache_hit_source import CacheHitSource


class SessionTierVersions(OffloadingManager):
    """Keep the weight version of every session KV block that the host tier
    of an `OffloadingConnector` holds; a load copies it to the GPU block it
    fills. A version is dropped with its entry."""

    def __init__(
        self,
        manager: OffloadingManager,
        connector: AuxOutputSchedulerConnector,
        groups: Sequence[GroupOffloadConfig],
    ) -> None:
        self._manager = manager
        self._connector = connector
        self._versions: dict[OffloadKey, Label] = {}
        self._stored: list[tuple[OffloadKey, str, int, int]] = []
        self._groups = groups

    @classmethod
    def install(
        cls, kv_connector: object, connector: AuxOutputSchedulerConnector
    ) -> SessionTierVersions | None:
        """Wrap the manager of `kv_connector` if its tier can be tracked: a
        host-only tier that stores one block per entry."""
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnector,
        )
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        if type(kv_connector) is not OffloadingConnector:
            return None
        scheduler = kv_connector.scheduler
        groups = [
            group
            for group in scheduler.config.kv_group_configs
            if group.group_idx in connector._groups
        ]
        if (
            type(scheduler.manager) is not CPUOffloadingManager
            or len(groups) != len(connector._groups)
            or any(group.tokens_per_chunk != group.tokens_per_block for group in groups)
        ):
            return None
        scheduler.manager = tier = cls(scheduler.manager, connector, groups)
        return tier

    def _object_index(
        self, key: OffloadKey, req_context: ReqContext
    ) -> tuple[int, int] | None:
        group_id = get_offload_group_idx(key)
        manager = self._connector._groups.get(group_id)
        position = req_context.get_offload_key_position(key)
        if manager is None or position is None or position % manager.block_size:
            return None
        return group_id, position // manager.block_size - 1

    def consistent_end(
        self, block_hashes: Sequence[BlockHash], start: int, end: int
    ) -> int:
        """Largest object boundary in `(start, end]` up to which every entry
        that a load reads has, in every KV group, the tracked group's version
        at every position, or `start`. Each lowered boundary scans only the
        entries it reads that no earlier boundary has checked."""
        connector = self._connector
        size = connector._block_size
        hash_size = connector._hash_block_size

        def version(group_id: int, position: int) -> Label | None:
            key = make_offload_key(block_hashes[position // hash_size - 1], group_id)
            return self._versions.get(key)

        full: list[Label | None] = [None] * (start // size) + [
            version(connector._group_id, p) for p in range(start + size, end + 1, size)
        ]
        if None in full[start // size :]:
            end = full.index(None, start // size) * size
        checked: dict[int, int] = {}
        while end > start:
            mismatch = end
            for group in self._groups:
                if group.group_idx == connector._group_id:
                    continue
                chunk = group.tokens_per_chunk
                ends = range(start + chunk, end + 1, chunk)
                window = group.load_window_size_in_chunks(end)
                if window is not None:
                    ends = ends[-window:]
                if not ends:
                    continue
                stop = min(checked.get(group.group_idx, end), mismatch)
                checked[group.group_idx] = min(ends[0] - chunk, stop)
                for chunk_end in ends:
                    low = chunk_end - chunk
                    if low >= stop:
                        break
                    label = version(group.group_idx, chunk_end)
                    if not connector.agrees(label, low, chunk_end, full):
                        mismatch = low
                        break
            if mismatch == end:
                return end
            end = mismatch - mismatch % size
        return start

    def label_stores(self) -> None:
        """Record the versions of the stores accepted in this `schedule()`."""
        for key, request_id, group_id, index in self._stored:
            state = self._connector._session_requests.get(request_id)
            if state is not None:
                self._versions[key] = self._connector._object_version(
                    state, index, group_id
                )
        self._stored.clear()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._manager.lookup(key, req_context)

    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return self._manager.get_load_source(key, req_context)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        for key in keys:
            index = self._object_index(key, req_context)
            if index is None:
                continue
            group_id, block_index = index
            manager = self._connector._groups[group_id]
            block_id = manager.req_to_blocks[req_context.req_id][block_index].block_id
            version = self._versions.get(key)
            if version is not None:
                self._connector._block_versions[block_id] = version
        return self._manager.prepare_load(keys, req_context)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        return self._manager.complete_load(keys, req_context)

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        trigger: StoreTrigger,
    ) -> PrepareStoreOutput | None:
        request_id = req_context.req_id
        connector = self._connector
        state = connector._session_requests.get(request_id)
        if state is not None and state.stopped:
            num_objects = state.num_sent_hashes // connector._step
            keys = [
                key
                for key in keys
                if (index := self._object_index(key, req_context)) is None
                or index[0] != connector._group_id
                or index[1] < num_objects
            ]
        output = self._manager.prepare_store(keys, req_context, trigger=trigger)
        if output is None:
            return None
        for key in output.evicted_keys:
            self._versions.pop(key, None)
        for key in output.keys_to_store:
            index = self._object_index(key, req_context)
            if index is not None:
                self._stored.append((key, request_id, *index))
        return output

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        trigger: StoreTrigger,
        success: bool = True,
    ):
        self._manager.complete_store(
            keys, req_context, trigger=trigger, success=success
        )
        if not success:
            for key in keys:
                self._versions.pop(key, None)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return self._manager.on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        return self._manager.on_request_finished(req_context)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._manager.take_events()

    def reset_cache(self) -> None:
        self._manager.reset_cache()
        self._versions.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        return self._manager.get_stats()

    def config_info(self) -> Sequence[ConfigInfoMapping]:
        return self._manager.config_info()
