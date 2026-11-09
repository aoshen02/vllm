# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side auxiliary-output data plane."""

from __future__ import annotations

import json as _json
import os
import queue
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from threading import Condition, Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as _re
import torch
from vllm.distributed.aux_output_connector.routed_experts import (
    RoutedExpertsBuffer,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)

from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector import oc_common as _oc
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxRequestOutput,
)
from vllm.distributed.aux_output_connector.store import (
    BackgroundBlockObjectStore,
    BlockObjectStore,
    BlockObjectStoreError,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch


def _clip_segs(segs, a, b):
    # Segments [start, version] restricted to [a, b), offsets relative to a.
    out = []
    for i, (s0, v) in enumerate(segs):
        s1 = segs[i + 1][0] if i + 1 < len(segs) else b
        if s1 > a and s0 < b:
            out.append([int(max(s0, a) - a), int(v)])
    return out or None


def _object_segs(segs, keys, bs, block_segs, total):
    # Every full block reports the versions of the object a reader gets (first put wins,
    # so a duplicate this request computed may differ); positions past the full blocks
    # and blocks without a record report this track's own segments.
    out = []
    for i, k in enumerate(keys):
        rec = block_segs.get(k)
        part = (
            [[i * bs + o, v] for o, v in rec]
            if rec
            else [
                [i * bs + o, v]
                for o, v in (_clip_segs(segs, i * bs, (i + 1) * bs) or [])
            ]
        )
        for p, v in part:
            if not out or out[-1][1] != v:
                out.append([int(p), int(v)])
    for p, v in _clip_segs(segs, len(keys) * bs, max(total, len(keys) * bs)) or []:
        if not out or out[-1][1] != v:
            out.append([int(len(keys) * bs + p), int(v)])
    return out


def _oc_keep_tail(w, kind: str, track, tail) -> None:
    # A few recent tails per trajectory, by start position, for forks that start from an
    # earlier point than the parent's latest tail.
    hist = getattr(w, "tail_hist", None)
    if hist is None:
        hist = w.tail_hist = {}
    h = hist.setdefault((kind, track), {})
    h[(int(tail[0]), int(tail[0]) + len(tail[1]))] = tail
    while len(h) > 32:
        h.pop(min(h, key=lambda k: k[1]))


def _oc_tag(track) -> str:
    import hashlib

    return hashlib.blake2b(str(track).encode(), digest_size=6).hexdigest()


def _set_rec(block_segs, key, rec):
    # First put wins, but a block recorded before any segment existed gets its real
    # record.
    if block_segs.get(key) is None:
        block_segs[key] = rec


def _with_hit_segs(segs, hashes, bs, block_segs):
    # Positions below this track's first segment came from prefix-cache hits: take their
    # versions from the blocks' records. Unknown blocks leave the gap visible.
    first = segs[0][0] if segs else len(hashes) * bs
    pre = []
    for i in range(min(len(hashes), (first + bs - 1) // bs)):
        rec = block_segs.get(hashes[i])
        if rec is None:
            return segs
        for off, v in rec:
            p = int(i * bs + off)
            if p >= first:
                break
            if not pre or pre[-1][1] != v:
                pre.append([p, int(v)])
    if not pre:
        return segs
    rest = [[int(x[0]), int(x[1])] for x in segs]
    if rest and rest[0][1] == pre[-1][1]:
        rest = rest[1:]
    return pre + rest


class _SerialStore:
    """Base store whose reads never overlap a mutation: the background store applies
    puts on its own thread, and reads also come from the oc-apply thread, outside _lock.
    """

    def __init__(self, store: BackgroundBlockObjectStore) -> None:
        self._store = store
        self._lock = Lock()

    def put(
        self,
        objects,
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        with self._lock:
            self._store.put(objects, retain_keys=retain_keys, release_keys=release_keys)

    def get_concatenated(self, keys: list[str]) -> bytes:
        with self._lock:
            return self._store.get_concatenated(keys)

    def close(self) -> None:
        with self._lock:
            self._store.close()


@dataclass
class _WorkerRequestState:
    aux_output_keys: list[str] = field(default_factory=list)
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    scheduled_cursor: int = 0
    emit_cursor: int = 0
    pending_outputs: int = 0
    # Terminal event received; teardown waits for in-flight step outputs.
    finished: bool = False
    session: str | None = None
    # Session plus trajectory: splice and version segments are per trajectory, so
    # branches of one session (GRPO samples, forks) never take each other's rows.
    oc_track: str | None = None
    # Parent trajectory of a fork, inherited at the first capture: at admission the
    # parent's last request may not be torn down yet, so its tail would be stale.
    oc_parent: str | None = None
    oc_cone_src: str | None = None
    preempted: bool = False
    oc_aborted: bool = False
    oc_parked_finish: bool = False
    oc_hash_skip: int = 0
    # Rows computed before a preemption that no block publish delivered yet, by start
    # position; positions below oc_freeze keep them (first computation wins).
    oc_stash: dict | None = None
    oc_freeze: int = 0
    oc_aux_start: int | None = None
    oc_splice_cap: int | None = None
    # Partial capture: this request's exact cells of the trimmed layers.
    oc_cone: object | None = None
    oc_cone_idx: object | None = None
    oc_cone_end: int | None = None
    oc_cone_prompt: int | None = None
    oc_cone_cut: int | None = None
    oc_cone_run: int | None = None
    oc_idx: dict | None = None
    # Per indexer block: which rows were actually written (captured, or kept from before
    # a preemption). Rows never written are not valid padding and must not be published.
    oc_idx_mask: dict | None = None
    # First position computed on invalid KV (scheduler rewind); rows from here are
    # reported invalid in the manifest.
    oc_invalid_from: int | None = None
    # Blocks held back while the trajectory's previous request is still being torn down.
    oc_deferred: list | None = None
    oc_closing_marked: bool = False
    oc_idx_holes: int = 0
    # Block index -> session-store name, for computed blocks whose bytes differ from the
    # block another trajectory published under the same hash.
    oc_alt: dict | None = None
    # Indexer blocks holding an overflowed row: never published (no other trajectory may
    # dedup onto them), reported missing instead.
    oc_idx_bad: set | None = None
    oc_hid: list | None = None
    oc_hid_start: int | None = None
    oc_hid_n: int = 0
    oc_hid_dim: int = 0
    # Raw rows' dtype name.
    oc_hid_dtype: str | None = None
    # A row arrived in another dtype or width than the request's first: the field is
    # withheld.
    oc_hid_bad: bool = False
    oc_hid_chunks: int = 0
    oc_hid_lens: list | None = None
    # Top-k logprobs: the hidden rules (sampled positions, chunks, preemption, no
    # inheritance).
    oc_tk: list | None = None
    oc_tk_start: int | None = None
    oc_tk_n: int = 0
    oc_tk_chunks: int = 0
    oc_tk_lens: list | None = None
    oc_tk_bad: bool = False


@dataclass
class PendingAuxOutput:
    """Own one step's R3 snapshot until its asynchronous copy is consumed."""

    connector: AuxOutputWorkerConnector
    request_ids: list[str]
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    # GPU snapshot taken on the main stream.
    routed_experts_gpu: torch.Tensor | None
    # Filled by enqueue_cpu_copy on the output copy stream.
    routed_experts: np.ndarray | None = None
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None

    def enqueue_cpu_copy(
        self, num_sampled: np.ndarray, num_rejected: np.ndarray
    ) -> None:
        """Enqueue asynchronous D2H copies; call on the output copy stream."""
        self.num_sampled = num_sampled
        self.num_rejected = num_rejected
        self.routed_experts = self.routed_experts_gpu.to(
            "cpu", non_blocking=True
        ).numpy()
        nb = self.routed_experts_gpu.numel() * self.routed_experts_gpu.element_size()
        for name in ("oc_idx", "oc_idx_ovf", "oc_idx_invalid", "oc_hq", "oc_tk"):
            t = getattr(self, name + "_gpu", None)
            setattr(
                self,
                name,
                None if t is None else t.to("cpu", non_blocking=True).numpy(),
            )
            if t is not None:
                nb += t.numel() * t.element_size()
        if _oc.PROF:
            e = _oc.ENGINE_PROF
            e["d2h_bytes"] = e.get("d2h_bytes", 0) + nb
            e["d2h_n"] = e.get("d2h_n", 0) + 1

    def drop_gpu(self) -> None:
        """The host copies are consumed; queued oc-apply events keep only those
        alive."""
        self.routed_experts_gpu = None
        for name in ("oc_idx", "oc_idx_ovf", "oc_idx_invalid", "oc_hq", "oc_tk"):
            if hasattr(self, name + "_gpu"):
                setattr(self, name + "_gpu", None)

    def process_output(self) -> dict[str, AuxRequestOutput]:
        if not _oc.PROF:
            return self.connector.process_output(self)
        t = time.perf_counter()
        out = self.connector.process_output(self)
        e = _oc.ENGINE_PROF
        e["proc_s"] = e.get("proc_s", 0.0) + time.perf_counter() - t
        e["proc_n"] = e.get("proc_n", 0) + 1
        w = _oc.writer()
        if w is not None:
            w.prof["engine"] = dict(e)
        return out


class AuxOutputWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=max_num_batched_tokens, vllm_config=vllm_config
        )
        if os.environ.get("OC_NOBIND") != "1":
            bind_routed_experts_capturer(model, capturer)
        self._capturer = capturer
        self._store: _SerialStore | None = None
        self._buffer: RoutedExpertsBuffer | None = None
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: AuxOutputConnectorMetadata | None = None
        # Steps whose asynchronous copy has not been consumed yet. The engine
        # consumes step outputs in order, so at most max_concurrent_batches
        # are outstanding; guarded by _lock against the output thread.
        self._pending_outputs: list[PendingAuxOutput] = []
        self._lock = Lock()
        self._max_concurrent_batches = vllm_config.max_concurrent_batches
        # Every TP rank participates in capture collectives, but only the
        # executor output rank owns the auxiliary output data plane.
        first_rank = get_tp_group().is_first_rank
        # Every oc_* field of a request, the stash contents and the writer's engine
        # tables are owned by the oc-apply thread. Events are posted under _lock, so
        # their order is the lock order. OC_APPLY_INLINE=1 applies each event when it is
        # posted.
        self._oc_inline = (
            os.environ.get("OC_APPLY_INLINE") == "1"
            or not first_rank
            or _oc.writer() is None
        )
        self._oc_q: queue.SimpleQueue = queue.SimpleQueue()
        self._oc_error: BaseException | None = None
        # Events posted and applied: process_output waits (outside _lock) while more
        # than OC_APPLY_MAX_PENDING are outstanding, so releases cannot fall arbitrarily
        # behind.
        self._oc_cv = Condition()
        self._oc_posted = self._oc_applied = 0
        self._oc_max_pending = int(os.environ.get("OC_APPLY_MAX_PENDING", "1024"))
        self._oc_thread: Thread | None = None
        if not self._oc_inline:
            self._oc_thread = Thread(target=self._oc_run, daemon=True, name="oc-apply")
            self._oc_thread.start()
        if not first_rank:
            return

        shape_per_token = self._capturer.shape_per_token
        dtype: np.dtype[Any] = np.dtype(self._capturer.output_dtype_name)
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // hash_block_size
        block_nbytes = hash_block_size * int(np.prod(shape_per_token)) * dtype.itemsize
        max_bytes = vllm_config.aux_output_config.max_bytes
        if max_bytes is None:
            max_bytes = kv_cache_config.num_blocks * hashes_per_kv_block * block_nbytes
        self._store = _SerialStore(
            BackgroundBlockObjectStore(
                BlockObjectStore(max_bytes=max_bytes, object_nbytes=block_nbytes),
                max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
            )
        )
        self._buffer = RoutedExpertsBuffer(
            dtype,
            shape_per_token,
            hash_block_size,
            vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens,
            vllm_config.max_concurrent_batches,
        )
        self._oc_step = 0
        # Stash records by request id: membership is kept under _lock, contents are
        # filled by the oc-apply thread.
        self._oc_preempted: dict[str, dict] = {}
        self._oc_track_closing: dict[str, int] = {}
        if _oc.EXTRA and _oc.writer() is not None:
            _oc.setup_extra(model, max_num_batched_tokens)

    def _oc_post(self, fn, *args) -> None:
        if self._oc_inline:
            fn(*args)
        else:
            self._oc_put((fn, args))

    def _oc_put(self, item: tuple | None) -> None:
        with self._oc_cv:
            self._oc_posted += 1
        self._oc_q.put(item)

    def _oc_backpressure(self) -> None:
        if self._oc_thread is None:
            return
        with self._oc_cv:
            if self._oc_posted - self._oc_applied <= self._oc_max_pending:
                return
            t = time.perf_counter()
            self._oc_cv.wait_for(
                lambda: self._oc_posted - self._oc_applied <= self._oc_max_pending
            )
        w = _oc.writer()
        if w is not None:
            w.stats["oc_apply_wait_s"] = (
                w.stats.get("oc_apply_wait_s", 0.0) + time.perf_counter() - t
            )
            w.stats["oc_apply_wait_n"] = w.stats.get("oc_apply_wait_n", 0) + 1

    def _oc_run(self) -> None:
        while True:
            item = self._oc_q.get()
            if item is None:
                return
            fn, args = item
            try:
                fn(*args)
            except BaseException as e:  # noqa: BLE001
                w = _oc.writer()
                if w is not None:
                    w.stats["oc_apply_errors"] = w.stats.get("oc_apply_errors", 0) + 1
                print(
                    f"oc-apply error in {getattr(fn, '__name__', fn)}: {e!r}",
                    flush=True,
                )
                if self._oc_error is None:
                    self._oc_error = e
            with self._oc_cv:
                self._oc_applied += 1
                self._oc_cv.notify_all()

    def _oc_drain(self) -> None:
        if self._oc_thread is not None:
            done = Event()
            self._oc_put((done.set, ()))
            done.wait()
        self._oc_raise()

    def _oc_stop(self) -> None:
        if self._oc_thread is not None:
            self._oc_put(None)
            self._oc_thread.join()
            self._oc_thread = None

    def _oc_raise(self) -> None:
        if self._oc_error is not None:
            raise self._oc_error

    def prepare_output(self, input_batch: InputBatch) -> PendingAuxOutput | None:
        """Snapshot one step's R3 tensor for asynchronous CPU transfer."""
        if self._buffer is None or self._step_metadata is None:
            return None

        request_ids = list(input_batch.req_ids)
        query_start_loc = input_batch.query_start_loc_np[: len(request_ids) + 1]
        num_rows = int(query_start_loc[-1])
        pending_output = PendingAuxOutput(
            connector=self,
            request_ids=request_ids,
            token_starts=input_batch.num_computed_tokens_np,
            query_start_loc=query_start_loc,
            routed_experts_gpu=self._capturer.snapshot_routing_data(num_rows),
        )
        self._oc_prepare(pending_output, num_rows)
        with self._lock:
            assert len(self._pending_outputs) < self._max_concurrent_batches, (
                "auxiliary output step outputs are not consumed in order"
            )
            self._pending_outputs.append(pending_output)
            for request_id in pending_output.request_ids:
                self._requests[request_id].pending_outputs += 1
        return pending_output

    def process_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        """Commit one consumed R3 snapshot and build request outputs.

        Request teardown deferred by begin_step is completed here once no
        in-flight step covers the request.
        """
        self._oc_backpressure()
        with self._lock:
            teardown = []
            release_keys: list[str] = []
            stash_release: list[str] = []
            try:
                outputs = self._commit_output(pending)
            finally:
                pending.drop_gpu()
                self._pending_outputs.remove(pending)
                for request_id in pending.request_ids:
                    state = self._requests[request_id]
                    state.pending_outputs -= 1
                    if state.pending_outputs == 0 and state.finished:
                        teardown.append(request_id)
                        (stash_release if state.preempted else release_keys).extend(
                            reversed(state.aux_output_keys)
                        )
            if teardown:
                self._teardown(teardown, release_keys, stash_release)
            self._oc_raise()
            return outputs

    def _commit_output(self, pending: PendingAuxOutput) -> dict[str, AuxRequestOutput]:
        buffer = self._buffer
        store = self._store
        assert buffer is not None and store is not None
        block_size = buffer.block_size

        routed_experts = pending.routed_experts
        num_sampled = pending.num_sampled
        num_rejected = pending.num_rejected
        assert (
            routed_experts is not None
            and num_sampled is not None
            and num_rejected is not None
        ), "auxiliary output CPU copy was not enqueued"

        # Publish the whole batch before materializing any consumer output.
        materialize_outputs: list[tuple[str, int, int]] = []
        block_batches = []
        outputs: dict[str, AuxRequestOutput] = {}

        # Use the ModelRunner's actual batch boundaries rather than rebuilding them.
        for request_id, token_start, start, end, sampled, rejected in zip(
            pending.request_ids,
            pending.token_starts,
            pending.query_start_loc[:-1],
            pending.query_start_loc[1:],
            num_sampled,
            num_rejected,
            strict=True,
        ):
            request_num_tokens = end - start
            assert request_num_tokens > 0, (
                "auxiliary output request token count must be positive"
            )
            state = self._requests[request_id]

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            assert 0 <= rejected <= request_num_tokens, (
                "auxiliary output rejected-token count is invalid"
            )
            rows = routed_experts[start : end - rejected]

            capture_start = token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                capture_cursor = capture_start

            skip = 0
            rewind_at = None
            if capture_start < capture_cursor:
                # The scheduler rewound this request (block-level KV-load failure
                # recovery, only possible for single-KV-cache-group models). Rows past
                # the rewind point were computed on invalid KV: the manifest marks them
                # invalid and the recompute is not spliced in. Keep capturing
                # contiguously from the old cursor.
                w_ = _oc.writer()
                if w_ is not None:
                    w_.stats["capture_rewinds"] = w_.stats.get("capture_rewinds", 0) + 1
                rewind_at = int(capture_start)
                skip = min(capture_cursor - capture_start, len(rows))
                rows = rows[skip:]
                capture_start = capture_cursor
            if capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                assert capture_cursor < state.scheduled_cursor, (
                    "auxiliary output capture has an unbacked token gap"
                )
                capture_start = capture_cursor

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            completed = buffer.capture(request_id, capture_start, rows)
            if _oc.DUMP_DIR and os.environ.get("OC_DUMP_COMMIT") == "1":
                # Reference for checkers: the rows this request actually kept this step
                # (the rejected speculative suffix already dropped), at their real
                # positions.
                commit_dump = getattr(self, "_oc_commit_dump", None)
                if commit_dump is None:
                    commit_dump = self._oc_commit_dump = []
                commit_dump.append(
                    (state.oc_track, int(capture_start), np.array(rows, copy=True))
                )
            self._oc_post(
                self._oc_capture,
                request_id,
                state,
                capture_start,
                pending,
                int(start) + skip,
                int(end) - rejected,
                int(sampled),
                int(token_start) + int(request_num_tokens) - 1 - rejected,
                int(token_start) + int(request_num_tokens),
                rewind_at,
                len(state.aux_output_keys),
            )
            token_end = capture_start + len(rows)
            state.capture_cursor = token_end
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            if sampled > 0 and emit_start <= token_end:
                if emit_start >= capture_start:
                    outputs[request_id] = AuxRequestOutput(
                        emit_start, rows[emit_start - capture_start :]
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        # A consumer may reuse a block produced earlier in the same batch.
        self._publish_blocks(block_batches)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            stored_end = (
                min(token_end // block_size, len(state.aux_output_keys)) * block_size
            )
            if emit_start < stored_end:
                first_block = emit_start // block_size
                keys = state.aux_output_keys[first_block : stored_end // block_size]
                try:
                    stored = materialize_routed_experts(
                        store,
                        keys,
                        shape_per_token=buffer.shape_per_token,
                        dtype=buffer.dtype,
                    )
                except BlockObjectStoreError:
                    w = _oc.writer()
                    if w is None or state.session is None:
                        raise
                    self._oc_drain()
                    parts = []
                    for i, k in enumerate(keys):
                        # A block computed before a preemption may exist only in the
                        # stash.
                        old = (state.oc_stash or {}).get((first_block + i) * block_size)
                        if old is not None and len(old) == block_size:
                            parts.append(
                                np.ascontiguousarray(old, dtype=buffer.dtype).tobytes()
                            )
                        else:
                            parts.append(
                                w.get(_oc.block_key(state.session, k.rsplit("/", 1)[1]))
                            )
                    if any(p is None for p in parts):
                        raise
                    self._oc_fallback_reads = getattr(self, "_oc_fallback_reads", 0) + 1
                    stored = np.frombuffer(b"".join(parts), dtype=buffer.dtype).reshape(
                        (-1, *buffer.shape_per_token)
                    )
                local_start = emit_start % block_size
                rows = stored[local_start : local_start + stored_end - emit_start]
                if stored_end < token_end:
                    rows = np.concatenate(
                        (rows, buffer.read(request_id, stored_end, token_end))
                    )
            else:
                rows = buffer.read(request_id, emit_start, token_end)
            outputs[request_id] = AuxRequestOutput(emit_start, rows)
            state.emit_cursor = token_end
        dump = getattr(self, "_oc_commit_dump", None)
        if dump:
            os.makedirs(_oc.DUMP_DIR, exist_ok=True)
            self._oc_commit_n = getattr(self, "_oc_commit_n", 0) + 1
            torch.save(
                dict(commit=dump),
                f"{_oc.DUMP_DIR}/c{os.getpid()}_{self._oc_commit_n:07d}.pt",
            )
            self._oc_commit_dump = []
        return outputs

    def _publish_blocks(
        self,
        batches: list[tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]],
        retain_keys: Sequence[str] = (),
        release_keys: Sequence[str] = (),
    ) -> None:
        store = self._store
        buffer = self._buffer
        assert store is not None and buffer is not None
        ready_batches = []
        released = []
        for state, completed in batches:
            blocks = state.pending_blocks + completed
            keyed_end = len(state.aux_output_keys) * buffer.block_size
            ready = [(start, rows) for start, rows in blocks if start < keyed_end]
            state.pending_blocks = [
                (start, buffer.retain_block(rows))
                for start, rows in blocks
                if start >= keyed_end
            ]
            if ready:
                released.extend(rows for _, rows in ready)
                clean = [(s, r) for s, r in ready if not _oc.r3_has_sentinel(r)]
                if clean:
                    ready_batches.append((state.aux_output_keys, clean))
                if _oc.writer() is not None and state.session is not None:
                    if not self._oc_inline:
                        ready = [(s, np.array(r, copy=True)) for s, r in ready]
                    self._oc_post(self._oc_publish, state, ready, self._generation)
        if ready_batches or retain_keys or release_keys:
            publish_routed_experts(
                store,
                batches=ready_batches,
                block_size=buffer.block_size,
                retain_keys=retain_keys,
                release_keys=release_keys,
            )
        for rows in released:
            buffer.release_block(rows)

    def _oc_publish(
        self,
        state: _WorkerRequestState,
        ready: list[tuple[int, np.ndarray]],
        generation: int,
    ) -> None:
        bs = self._buffer.block_size
        w = _oc.writer()
        closing = self._oc_track_closing
        busy = closing.get(state.oc_track, 0) > 0 and state.oc_track is not None
        todo = []
        if not busy and state.oc_deferred:
            todo, state.oc_deferred = state.oc_deferred, []
        for start, rows in ready:
            if busy and state.oc_aux_start is not None and start < state.oc_aux_start:
                # The trajectory's previous request finished but its teardown (which
                # records the tail spliced below aux_start) is still pending: publish
                # this block after it.
                if state.oc_deferred is None:
                    state.oc_deferred = []
                state.oc_deferred.append((start, np.array(rows, copy=True)))
                w.stats["deferred_blocks"] = w.stats.get("deferred_blocks", 0) + 1
            else:
                todo.append((start, rows))
        for start, rows in todo:
            key = state.aux_output_keys[start // bs]
            rows = self._oc_splice(state, start, rows)
            h = key.rsplit("/", 1)[1]
            # Versions of a block's rows travel with its object: a request that hits
            # it reports them, not its own. Objects are per session (session store,
            # first put wins) and per generation (base store), so are the records.
            bsegs = getattr(w, "block_segs", None)
            if bsegs is None:
                bsegs = w.block_segs = {}
            rec = _clip_segs(
                w.weight_versions.get(state.oc_track, []), start, start + bs
            )
            _set_rec(bsegs, ("", generation, h), rec)
            if (
                state.oc_aux_start is not None
                and h in w.published(state.session)
                and (
                    start + bs <= self._oc_splice_end(state)
                    or w.holds(state.session, _oc_tag(state.oc_track), h)
                )
            ):
                # Every row of this block precedes the request's first computation and
                # a computation of the same tokens is already published: that one came
                # first (the previous turn, or the parent of a fork). Share it. A block
                # that spans the aux start holds this trajectory's own rows past it, so
                # the version rule of put_block_checked decides.
                name, new = h, False
                w.note_has(state.session, _oc_tag(state.oc_track), {h: h})
                w.stats["shared_earlier_block"] = (
                    w.stats.get("shared_earlier_block", 0) + 1
                )
            elif _oc.r3_has_sentinel(rows):
                name = w.share_if_same_version(
                    state.session, h, _oc_tag(state.oc_track), rec
                )
                new = False
                if name is None and h in w.published(state.session):
                    # Another version holds this block: rows this trajectory did not
                    # compute are the ones its hit read, so they come from that object.
                    raw = w.get(_oc.block_key(state.session, h), timeout_s=5.0)
                    if raw and len(raw) == rows.nbytes:
                        src = np.frombuffer(raw, dtype=rows.dtype).reshape(rows.shape)
                        gap = (
                            rows[:, : _oc.R3_FIRST_TRIMMED] == np.iinfo(rows.dtype).max
                        ).any(axis=(1, 2))
                        rows = np.array(rows, copy=True)
                        rows[gap] = src[gap]
                        if not _oc.r3_has_sentinel(rows):
                            w.stats["r3_trim_filled"] = (
                                w.stats.get("r3_trim_filled", 0) + 1
                            )
                            name, new = w.put_block_checked(
                                state.session,
                                h,
                                rows.tobytes(order="C"),
                                _oc_tag(state.oc_track),
                                segs=rec,
                            )
                if name is None:
                    # A row the replay trim did not run: it was to come from a block
                    # that is not published. Reported as missing rather than delivered
                    # wrong.
                    w.stats["r3_trim_blocked"] = w.stats.get("r3_trim_blocked", 0) + 1
                    if w.stats["r3_trim_blocked"] <= 40:
                        sent = (rows == np.iinfo(rows.dtype).max).any(axis=(1, 2))
                        print(
                            f"OC_TRIM_BLOCK track={state.oc_track!r} start={start} "
                            f"aux={state.oc_aux_start} "
                            f"splice_end={self._oc_splice_end(state)} "
                            f"published={h in w.published(state.session)} "
                            "holds="
                            f"{w.holds(state.session, _oc_tag(state.oc_track), h)} "
                            f"sent_rows={int(sent.sum())} first={int(sent.argmax())} "
                            f"parent={state.oc_parent!r}",
                            flush=True,
                        )
                    if state.oc_idx is not None:
                        state.oc_idx.pop(start // bs, None)
                    continue
            else:
                name, new = w.put_block_checked(
                    state.session,
                    h,
                    rows.tobytes(order="C"),
                    _oc_tag(state.oc_track),
                    segs=rec,
                )
            if new:
                # The writer resolves rows from a prefix hit (-1) to the versions
                # of the object they came from.
                _set_rec(
                    bsegs,
                    (state.session, name),
                    w.object_segs(state.session, name) or rec,
                )
            if name != h:
                if state.oc_alt is None:
                    state.oc_alt = {}
                state.oc_alt[start // bs] = name
                # The trajectory's later requests hit this block: they must read its
                # own copy, not the session's first one.
                ta = getattr(w, "track_alt", None)
                if ta is None:
                    ta = w.track_alt = {}
                ta.setdefault(state.oc_track, {})[h] = name
            self._oc_put_idx(state, start, name)

    def _teardown(
        self,
        request_ids: Iterable[str],
        release_keys: Sequence[str] = (),
        stash_release: Sequence[str] = (),
    ) -> None:
        """Drop finished requests' temporary state, then release their keys.

        release_keys follow the finishes on the oc-apply thread: a finish may still read
        hit blocks from the base store, which the release would let it evict. A stash
        reads none, so stash_release is published here."""
        buffer = self._buffer
        assert buffer is not None
        items = []
        pinned: list[str] = []
        for request_id in request_ids:
            state = self._requests.pop(request_id)
            if state.oc_parked_finish and not self._oc_inline:
                # A parked finish holds no reference to the blocks it reads: pin them
                # until its finish has run, so later releases cannot let them be evicted
                # first.
                pinned.extend(reversed(state.aux_output_keys))
            rec = (
                None
                if not state.preempted
                else dict(session=state.session, track=state.oc_track)
                if state.oc_aborted
                else self._oc_stash_record(request_id, state)
            )
            items.append(
                (
                    request_id,
                    state,
                    rec,
                    state.oc_closing_marked,
                    *self._oc_rows_snapshot(request_id, state),
                )
            )
            for _, rows in state.pending_blocks:
                buffer.release_block(rows)
            buffer.discard(request_id)
        if pinned:
            self._publish_blocks([], retain_keys=pinned)
        if items or release_keys:
            self._oc_post(
                self._oc_teardown, items, [*release_keys, *pinned], self._generation
            )
        if stash_release:
            self._publish_blocks([], release_keys=stash_release)

    def _oc_rows_snapshot(
        self, request_id: str, state: _WorkerRequestState
    ) -> tuple[list[tuple[int, np.ndarray]], tuple[int, np.ndarray] | None]:
        # Unpublished rows of a request leaving the buffer: its slots are recycled at
        # once.
        pending = [(int(s), np.array(r, copy=True)) for s, r in state.pending_blocks]
        tail = self._buffer._requests.get(request_id)
        if tail is None or not tail.length:
            return pending, None
        return pending, (
            int(tail.block_start),
            np.array(self._buffer._rows[tail.slot, : tail.length], copy=True),
        )

    def _oc_stash_record(self, request_id: str, state: _WorkerRequestState) -> dict:
        rec = self._oc_preempted[request_id] = dict(
            session=state.session, track=state.oc_track
        )
        return rec

    def _oc_stash_now(self, request_id: str, state: _WorkerRequestState) -> None:
        self._oc_post(
            self._oc_stash_preempted,
            request_id,
            state,
            *self._oc_rows_snapshot(request_id, state),
            self._oc_stash_record(request_id, state),
        )

    def _oc_teardown(
        self, items: list, release_keys: list[str], generation: int
    ) -> None:
        error = None
        for request_id, state, rec, closing_marked, pending, tail in items:
            try:
                if rec is not None:
                    self._oc_stash_preempted(request_id, state, pending, tail, rec)
                else:
                    self._oc_finish(request_id, state, pending, tail, generation)
            except BaseException as e:  # noqa: BLE001
                error = error or e
            if closing_marked:
                self._oc_closing(state.oc_track, -1)
        if release_keys:
            self._publish_blocks([], release_keys=release_keys)
        if error is not None:
            raise error

    def _oc_closing(self, track: str, delta: int) -> None:
        closing = self._oc_track_closing
        closing[track] = closing.get(track, 0) + delta
        if closing[track] <= 0:
            del closing[track]

    def _oc_stash_preempted(
        self,
        request_id: str,
        state: _WorkerRequestState,
        pending: list[tuple[int, np.ndarray]],
        tail: tuple[int, np.ndarray] | None,
        rec: dict,
    ) -> None:
        # The request resumes later and recomputes from its prefix-cache hit. Keep what
        # only this computation has: unpublished rows (pending blocks and the tail),
        # indexer blocks not yet put, and the hidden cursor; the manifest is written at
        # the real finish.
        old = state.oc_stash or {}
        stash = dict(old)
        fresh = list(pending)
        if tail is not None:
            fresh.append(tail)
        for start, rows in fresh:
            rows = np.array(rows, copy=True)
            prev = old.get(start)
            if prev is not None:
                # Preempted again: positions frozen by the first preemption keep its
                # rows.
                k = min(len(prev), max(0, state.oc_freeze - start))
                if len(rows) < k:
                    rows = np.concatenate([rows, prev[len(rows) : k]])
                rows[:k] = prev[:k]
            stash[start] = rows
        if state.oc_hid:
            self._oc_put_hid(request_id, state)
        if state.oc_tk:
            self._oc_put_tk(request_id, state)
        rec.update(
            stash=stash,
            freeze=max(int(state.capture_cursor or 0), state.oc_freeze),
            aux_start=state.oc_aux_start,
            idx=state.oc_idx,
            idx_holes=state.oc_idx_holes,
            idx_bad=state.oc_idx_bad,
            idx_mask=state.oc_idx_mask,
            alt=state.oc_alt,
            invalid_from=state.oc_invalid_from,
            cone=(state.oc_cone, state.oc_cone_idx, state.oc_cone_prompt),
            hid=(
                state.oc_hid_start,
                state.oc_hid_n,
                state.oc_hid_dim,
                state.oc_hid_chunks,
            ),
            hid_lens=list(state.oc_hid_lens or []),
            hid_dtype=state.oc_hid_dtype,
            hid_bad=state.oc_hid_bad,
            tk=(
                state.oc_tk_start,
                state.oc_tk_n,
                state.oc_tk_chunks,
                list(state.oc_tk_lens or []),
                state.oc_tk_bad,
            ),
        )
        w = _oc.writer()
        if w is not None:
            w.stats["preempted"] = w.stats.get("preempted", 0) + 1

    def _oc_take_stash(
        self, request_id: str, state: _WorkerRequestState
    ) -> dict | None:
        saved = self._oc_preempted.pop(request_id, None)
        if saved is not None:
            state.session = state.session or saved["session"]
            state.oc_track = state.oc_track or saved["track"]
        return saved

    def _oc_resume(
        self, state: _WorkerRequestState, saved: dict | None, aux_start: int | None
    ) -> None:
        if saved is not None:
            self._oc_apply_stash(state, saved)
        if state.oc_aux_start is None and aux_start is not None:
            state.oc_aux_start = aux_start

    def _oc_apply_stash(self, state: _WorkerRequestState, saved: dict) -> None:
        state.oc_stash, state.oc_freeze = saved["stash"], saved["freeze"]
        state.oc_aux_start = saved["aux_start"]
        state.oc_idx, state.oc_idx_holes = saved["idx"], saved["idx_holes"]
        state.oc_idx_bad = saved.get("idx_bad")
        state.oc_idx_mask = saved.get("idx_mask")
        state.oc_alt = saved.get("alt")
        state.oc_cone, state.oc_cone_idx, state.oc_cone_prompt = saved["cone"]
        state.oc_cone_run = None
        if saved.get("invalid_from") is not None:
            state.oc_invalid_from = (
                saved["invalid_from"]
                if state.oc_invalid_from is None
                else min(state.oc_invalid_from, saved["invalid_from"])
            )
        state.oc_hid_start, state.oc_hid_n, state.oc_hid_dim, state.oc_hid_chunks = (
            saved["hid"]
        )
        state.oc_hid_lens = list(saved.get("hid_lens") or [])
        state.oc_hid_dtype = saved.get("hid_dtype")
        state.oc_hid_bad = saved.get("hid_bad", False)
        if state.oc_hid_start is not None:
            state.oc_hid = []
        if saved.get("tk") is not None:
            (
                state.oc_tk_start,
                state.oc_tk_n,
                state.oc_tk_chunks,
                state.oc_tk_lens,
                state.oc_tk_bad,
            ) = saved["tk"]
            if state.oc_tk_start is not None:
                state.oc_tk = []
        w = _oc.writer()
        if w is not None:
            w.stats["resumed"] = w.stats.get("resumed", 0) + 1

    def _oc_prepare(self, pending: PendingAuxOutput, num_rows: int) -> None:
        self._oc_step += 1
        pending.oc_wv = _oc.WEIGHT_VERSION[0]
        cap = _oc.IDX_CAP
        if cap is not None and cap.trimmed:
            pending.oc_idx_invalid_gpu = cap.invalid[:num_rows].clone()
            cap.trimmed = False
        if cap is not None and _oc.IDX_MODE in ("gpupack", "gpudelta"):
            # 20-bit pack on the GPU: (value + 1) in pairs, 5 bytes per pair, same byte
            # layout as idx_encode's tag 1; the host only runs zstd (GIL released).
            if _oc.IDX_MODE == "gpudelta":
                v = cap.buf[:, :num_rows].transpose(0, 1).reshape(num_rows, -1) + 1
                pending.oc_idx_ovf_gpu = ((v < 0) | (v >= (1 << 20))).any(dim=1)
                pending.oc_idx_gpu = _oc.dplane20_encode_torch(
                    v.clamp_(0, (1 << 20) - 1), *cap.shape
                )
            else:
                v = (
                    cap.buf[:, :num_rows]
                    .transpose(0, 1)
                    .reshape(num_rows, -1)
                    .to(torch.int64)
                    + 1
                )
                pending.oc_idx_ovf_gpu = (
                    ((v < 0) | (v >= (1 << 20))).flatten(1).any(dim=1)
                )
                pairs = v[:, 0::2] | (v[:, 1::2] << 20)
                pending.oc_idx_gpu = (
                    pairs.view(torch.uint8)
                    .reshape(num_rows, -1, 8)[:, :, :5]
                    .reshape(num_rows, -1)
                )
        elif cap is not None and _oc.IDX_MODE != "nod2h":
            pending.oc_idx_gpu = cap.buf[:, :num_rows].transpose(0, 1).contiguous()
        h = _oc.HIDDEN[0]
        _oc.HIDDEN[0] = None
        cu = None
        if isinstance(h, tuple):
            h, cu = h
        n_req = len(pending.request_ids)
        if (
            h is not None
            and cu is not None
            and len(cu) >= n_req + 1
            and int(cu[n_req]) == h.shape[0]
        ):
            pending.oc_hid_cu = np.asarray(cu[: n_req + 1], dtype=np.int64)
            self._oc_hid_snapshot(pending, h)
        elif h is not None and h.shape[0] == n_req:
            pending.oc_hid_cu = np.arange(n_req + 1, dtype=np.int64)
            self._oc_hid_snapshot(pending, h)
        t = _oc.TOPK[0]
        _oc.TOPK[0] = None
        if _oc.TOPK_ON:
            tk, cu = t if isinstance(t, tuple) else (None, None)
            if (
                tk is not None
                and cu is not None
                and len(cu) >= n_req + 1
                and int(cu[n_req]) == tk.shape[0]
            ):
                pending.oc_tk_gpu, pending.oc_tk_cu = (
                    tk,
                    np.asarray(cu[: n_req + 1], dtype=np.int64),
                )
            else:
                pending.oc_tk_missing = True
        if _oc.DUMP_DIR:
            self._oc_dump(pending, num_rows)

    @staticmethod
    def _oc_hid_snapshot(pending: PendingAuxOutput, h: torch.Tensor) -> None:
        pending.oc_hq_gpu, pending.oc_hid_dtype, pending.oc_hid_d = _oc.hid_snapshot(h)

    def _oc_dump(self, pending: PendingAuxOutput, num_rows: int) -> None:
        # Validation only: the step's batch layout with the hook's second copy of the
        # indexer rows (+ positions, per-layer forward counters) and the sampler's
        # lm_head input and logits, read synchronously.
        sess = [self._requests[r].session for r in pending.request_ids]
        rec = dict(
            req=[_oc.key_id(r) for r in pending.request_ids],
            sess=sess,
            tstart=np.array(pending.token_starts, dtype=np.int64),
            qsl=np.array(pending.query_start_loc, dtype=np.int64),
        )
        if os.environ.get("OC_DUMP_R3_ONLY") == "1":
            # Reference for the checker: the routing each request actually computed this
            # step.
            rec["sess"] = [self._requests[r].oc_track for r in pending.request_ids]
            rec["r3"] = pending.routed_experts_gpu[:num_rows].cpu().numpy()
            os.makedirs(_oc.DUMP_DIR, exist_ok=True)
            torch.save(rec, f"{_oc.DUMP_DIR}/s{self._oc_step:07d}.pt")
            return
        cap = _oc.IDX_CAP
        if cap is not None:
            rec["idx_layers"] = list(cap.layer_ids)
            rec["idx_count"] = cap.count.cpu().numpy()
            rec["idx_pos"] = cap.dpos[:, :num_rows].cpu().numpy()
            rec["idx"] = cap.dbuf[:, :num_rows].cpu().numpy()
        if _oc.DUMP_LOGITS:
            h, lg = _oc.DUMP_LOGITS[0]
            sel = [
                i for i, s in enumerate(sess) if s and _re.search(_oc.DUMP_LOGITS_RE, s)
            ]
            rec["hid"] = h.cpu()
            rec["logits_rows"] = sel
            rec["logits"] = lg[sel].cpu()
        torch.save(rec, f"{_oc.DUMP_DIR}/s{self._oc_step:07d}.pt")

    def _oc_inherit(
        self, state: _WorkerRequestState, keys_n: int | None = None
    ) -> None:
        # A fork starts from its parent's context at the fork point: the parent's tail
        # there (not its latest one: the parent may have finished more turns since), the
        # blocks the parent held, and its version segments.
        parent, state.oc_parent = state.oc_parent, None
        state.oc_cone_src = parent
        w = _oc.writer()
        if w is None:
            return
        bs = self._buffer.block_size
        aux = int(state.oc_aux_start or 0)
        if _oc.R3_PARTIAL:
            # The parent's cells below the fork point are this trajectory's history,
            # however far its later turns went. Rows from the fork point on are the
            # fork's own.
            cl = getattr(w, "cone_list", None)
            if cl is None:
                cl = w.cone_list = {}
            if state.oc_track not in cl:
                cl[state.oc_track] = [
                    dict(p, end=min(p["end"], aux + 1))
                    for p in cl.get(parent, [])
                    if p["lo"] <= aux
                ]
                ce = getattr(w, "cone_ends", None)
                if ce is None:
                    ce = w.cone_ends = {}
                ce[state.oc_track] = [
                    [p_, min(e_, aux + 1)]
                    for p_, e_ in ce.get(parent, [])
                    if p_ <= aux + 1
                ]
                hd = getattr(w, "cone_held", None)
                if hd is not None:
                    for pk, ck in (
                        (parent, state.oc_track),
                        ((parent, "idx"), (state.oc_track, "idx")),
                    ):
                        h_ = hd.get(pk)
                        if h_ is not None and ck not in hd and h_[0] <= aux:
                            hd[ck] = (
                                h_[0],
                                np.array(h_[1][: aux + 1 - h_[0]], copy=True),
                            )
        start = (aux // bs) * bs
        nblk = aux // bs
        keys_n = len(state.aux_output_keys) if keys_n is None else keys_n
        for kind, d in (("r3", w.last_tail), ("idx", w.last_tail_idx)):
            if state.oc_track in d:
                continue
            # The parent request the fork continues from ended at the fork point: its
            # tail is the one reaching aux with the smallest end (later turns of the
            # parent end further). Keys are (start, end); a request that ended on a
            # block boundary left an empty one.
            hist = dict(getattr(w, "tail_hist", {}).get((kind, parent), {}))
            if parent in d:
                s0, rows0 = d[parent]
                hist.setdefault((int(s0), int(s0) + len(rows0)), d[parent])
            cand = [(e0, s0) for (s0, e0) in hist if e0 >= aux and s0 <= aux + 1]
            sel = None
            if cand:
                e0, s0 = min(cand)
                sel = (s0, e0, hist[(s0, e0)])
            if sel is not None and sel[0] <= start and sel[1] > sel[0]:
                d[state.oc_track] = sel[2]
            elif (
                kind == "r3"
                and sel is not None
                and sel[0] == aux + 1 == start + bs
                and keys_n > nblk
            ):
                # With speculative decoding the parent's last step verified position aux
                # itself, completing the aux block before the fork point: the whole
                # block is the parent's computation and the fork holds it too.
                nblk += 1
            elif kind == "r3" and aux > start:
                # The fork-point tail is gone: the fork's own recompute of [start, aux)
                # is its own computation, labelled with its own version (no splice,
                # segment from start).
                state.oc_splice_cap = start
                w.stats["fork_tail_missing"] = w.stats.get("fork_tail_missing", 0) + 1
        prefix = [
            k.rsplit("/", 1)[1] for k in state.aux_output_keys[: min(nblk, keys_n)]
        ]
        ta = getattr(w, "track_alt", None)
        if ta and parent in ta and state.oc_track not in ta:
            ta[state.oc_track] = {h: ta[parent][h] for h in prefix if h in ta[parent]}
        w.inherit_has(state.session, _oc_tag(parent), _oc_tag(state.oc_track), prefix)
        if parent in w.weight_versions and state.oc_track not in w.weight_versions:
            # Segments up to where the fork's own rows begin (the parent's later turns
            # start their segments there or after).
            cut = self._oc_splice_end(state)
            w.weight_versions[state.oc_track] = [
                list(x) for x in w.weight_versions[parent] if x[0] < cut
            ] or None
            if w.weight_versions[state.oc_track] is None:
                del w.weight_versions[state.oc_track]

    def _oc_capture(
        self,
        request_id,
        state,
        capture_start,
        pending,
        lo,
        hi,
        sampled,
        pos,
        cone_end: int,
        rewind_at: int | None,
        keys_n: int,
    ) -> None:
        if rewind_at is not None:
            state.oc_invalid_from = (
                rewind_at
                if state.oc_invalid_from is None
                else min(state.oc_invalid_from, rewind_at)
            )
        state.oc_cone_end = cone_end
        if state.oc_parent:
            self._oc_inherit(state, keys_n)
        w = _oc.writer()
        if w is not None and state.session is not None and hi > lo:
            # Positions before aux_start keep the session's earlier rows (first
            # computation wins), so a version segment starts at the first position this
            # request actually delivers.
            if state.oc_parent:
                self._oc_inherit(state, keys_n)
            se = self._oc_splice_end(state)
            p = max(int(capture_start), se, state.oc_freeze)
            segs = [s for s in w.weight_versions.get(state.oc_track, []) if s[0] < p]
            if (
                state.oc_aux_start is not None
                and int(capture_start) > max(se, state.oc_freeze)
                and segs
                and not any(s[0] >= se for s in segs)
            ):
                # Positions from the splice end up to the first capture were served by a
                # prefix hit: their rows are those of the object hit, whose own record
                # gives the version (-1 = "as recorded by the object"), not the segment
                # inherited before them.
                segs.append([int(max(se, state.oc_freeze)), -1])
            v = getattr(pending, "oc_wv", _oc.WEIGHT_VERSION[0])
            if p < int(capture_start) + hi - lo and (not segs or segs[-1][1] != v):
                segs.append([p, v])
            w.weight_versions[state.oc_track] = segs
            if _oc.R3_PARTIAL:
                self._oc_cone_add(state, capture_start, pending, lo, hi, sampled, v)
        if w is None or state.session is None or not _oc.EXTRA:
            return
        bs = self._buffer.block_size
        if _oc.IDX_CAP is not None:
            if state.oc_idx is None:
                state.oc_idx = {}
            idx = getattr(pending, "oc_idx", None)
            ovf = getattr(pending, "oc_idx_ovf", None)
            bad = None if ovf is None or idx is None else np.flatnonzero(ovf[lo:hi])
            if bad is not None and len(bad):
                # Only the overflowed rows are lost; positions below the freeze keep
                # stashed rows.
                q = capture_start + bad
                q = q[q >= state.oc_freeze]
                if len(q):
                    w.stats["idx_overflow_rows"] = w.stats.get(
                        "idx_overflow_rows", 0
                    ) + len(q)
                    state.oc_idx_holes += len(q)
                    if state.oc_idx_bad is None:
                        state.oc_idx_bad = set()
                    state.oc_idx_bad.update(int(b) for b in np.unique(q // bs))
            if idx is None:
                state.oc_idx_holes += hi - lo
            else:
                rows = idx[lo:hi]
                shallow = _oc.idx_shallow_width(rows)
                inv = getattr(pending, "oc_idx_invalid", None)
                inv = None if inv is None else inv[lo:hi]
                off = 0
                while off < len(rows):
                    b, local = divmod(capture_start + off, bs)
                    c = min(bs - local, len(rows) - off)
                    blk = state.oc_idx.get(b)
                    if blk is None:
                        if (
                            rows.dtype == np.uint8
                        ):  # packed 0 == pad (-1); calloc, untouched pages stay free
                            blk = state.oc_idx[b] = np.zeros(
                                (bs, *rows.shape[1:]), rows.dtype
                            )
                        else:
                            blk = state.oc_idx[b] = np.full(
                                (bs, *rows.shape[1:]), -1, rows.dtype
                            )
                    if state.oc_idx_mask is None:
                        state.oc_idx_mask = {}
                    mask = state.oc_idx_mask.setdefault(b, np.zeros(bs, dtype=bool))
                    keep = max(0, min(c, state.oc_freeze - (capture_start + off)))
                    blk[local + keep : local + c, :shallow] = rows[
                        off + keep : off + c, :shallow
                    ]
                    mask[local + keep : local + c] = (
                        True if inv is None else ~inv[off + keep : off + c]
                    )
                    off += c
        hq = getattr(pending, "oc_hq", None)
        if hq is not None and sampled > 0:
            pos_ = getattr(pending, "oc_rid_pos", None)
            if pos_ is None:
                pos_ = pending.oc_rid_pos = {
                    r: k for k, r in enumerate(pending.request_ids)
                }
            i = pos_[request_id]
            cu = getattr(pending, "oc_hid_cu", None)
            r0 = int(cu[i]) if cu is not None else i
            n_rows = int(cu[i + 1]) - r0 if cu is not None else 1
            # The accepted rows are the request's first `sampled` logits rows. Positions
            # come from the captured range, not the scheduler's (async) token start:
            # logits rows sit on the last n_rows captured positions (prefill: only the
            # last one; decode and speculative verification: every one), so row j is at
            # capture_start + max(0, (hi - lo) - n_rows) + j.
            base = int(capture_start) + max(0, (hi - lo) - n_rows)
            take = min(int(sampled), n_rows)
            if take < int(sampled):
                w.stats["hid_short_rows"] = (
                    w.stats.get("hid_short_rows", 0) + int(sampled) - take
                )
            enc = (
                getattr(pending, "oc_hid_dtype", None),
                getattr(pending, "oc_hid_d", hq.shape[1]),
            )
            for j in range(take if not state.oc_hid_bad else 0):
                p = base + j
                if state.oc_hid_start is None:
                    state.oc_hid_start, state.oc_hid = p, []
                    state.oc_hid_dtype, state.oc_hid_dim = enc
                if enc != (state.oc_hid_dtype, state.oc_hid_dim):
                    # Rows of two encodings cannot share one field: report it missing
                    # instead.
                    state.oc_hid_bad = True
                    w.stats["hid_encoding_changed"] = (
                        w.stats.get("hid_encoding_changed", 0) + 1
                    )
                    break
                nxt = state.oc_hid_start + state.oc_hid_n
                if p == nxt:
                    state.oc_hid.append((hq, r0 + j))
                    state.oc_hid_n += 1
                    if len(state.oc_hid) >= _oc.HID_CHUNK:
                        self._oc_put_hid(request_id, state)
                elif p > nxt:
                    w.stats["hid_gaps"] = w.stats.get("hid_gaps", 0) + 1
                    if w.stats["hid_gaps"] <= 30:
                        print(
                            f"OC_HID_GAP rid={_oc.key_id(request_id)} i={i} r0={r0} "
                            f"n_rows={n_rows} "
                            f"sampled={sampled} take={take} j={j} p={p} pos={pos} "
                            f"nxt={nxt} "
                            f"hid_start={state.oc_hid_start} "
                            f"capture_start={capture_start} lo={lo} hi={hi}",
                            flush=True,
                        )

        if _oc.TOPK_ON and sampled > 0 and not state.oc_tk_bad:
            self._oc_topk_add(
                request_id, state, pending, capture_start, lo, hi, sampled, w
            )

    def _oc_topk_add(
        self, request_id, state, pending, capture_start, lo, hi, sampled, w
    ) -> None:
        """The step's top-k rows of this request, at the positions hidden rows take."""
        tk = getattr(pending, "oc_tk", None)
        if tk is None:
            # A sampled row without its top-k (sharded sampler, or no capture):
            # withheld.
            state.oc_tk_bad = True
            w.stats["topk_withheld"] = w.stats.get("topk_withheld", 0) + 1
            return
        pos_ = getattr(pending, "oc_rid_pos", None)
        if pos_ is None:
            pos_ = pending.oc_rid_pos = {
                r: k for k, r in enumerate(pending.request_ids)
            }
        i = pos_[request_id]
        r0 = int(pending.oc_tk_cu[i])
        n_rows = int(pending.oc_tk_cu[i + 1]) - r0
        base = int(capture_start) + max(0, (hi - lo) - n_rows)
        for j in range(min(int(sampled), n_rows)):
            p = base + j
            if state.oc_tk_start is None:
                state.oc_tk_start, state.oc_tk = p, []
            nxt = state.oc_tk_start + state.oc_tk_n
            if p == nxt:
                state.oc_tk.append((tk, r0 + j))
                state.oc_tk_n += 1
                if len(state.oc_tk) >= _oc.HID_CHUNK:
                    self._oc_put_tk(request_id, state)
            elif p > nxt:
                w.stats["topk_gaps"] = w.stats.get("topk_gaps", 0) + 1

    def _oc_put_tk(self, request_id: str, state) -> None:
        rows, state.oc_tk = state.oc_tk, []
        if not rows:
            return
        _oc.writer().put(
            _oc.topk_key(state.session, _oc.key_id(request_id), state.oc_tk_chunks),
            lambda rows=rows: np.stack([r[0][r[1]] for r in rows]).tobytes(),
        )
        state.oc_tk_chunks += 1
        state.oc_tk_lens = (state.oc_tk_lens or []) + [len(rows)]

    def _oc_cone_add(
        self, state, capture_start, pending, lo, hi, sampled, version
    ) -> None:
        rows = pending.routed_experts[lo:hi]
        deep = rows[:, _oc.R3_FIRST_TRIMMED :]
        if state.oc_cone is None:
            state.oc_cone = _oc.ConeParts(range(_oc.R3_FIRST_TRIMMED, rows.shape[1]))
        end = state.oc_cone_end
        if state.oc_cone_run is None:
            state.oc_cone_run = int(capture_start)
        state.oc_cone.add(int(capture_start), end, deep, version, state.oc_cone_run)
        idx = getattr(pending, "oc_idx", None)
        cap = _oc.IDX_CAP
        slots = (
            None
            if cap is None
            else [
                j for j, lyr in enumerate(cap.layer_ids) if lyr >= _oc.R3_FIRST_TRIMMED
            ]
        )
        ovf_ = getattr(pending, "oc_idx_ovf", None)
        if (
            idx is not None
            and slots
            and ovf_ is not None
            and bool(np.asarray(ovf_[lo:hi]).any())
        ):
            # A value the packed format cannot hold was clamped: these cells stay
            # uncovered.
            w_ = _oc.writer()
            if w_ is not None:
                w_.stats["cone_idx_ovf_steps"] = (
                    w_.stats.get("cone_idx_ovf_steps", 0) + 1
                )
            idx = None
        if os.environ.get("OC_CONE_IDX", "1") == "0":
            idx = None
        if idx is not None and slots:
            r = idx[lo:hi]
            r = r.reshape(len(r), len(cap.layer_ids), -1)[:, slots[0] :]
            if state.oc_cone_idx is None:
                state.oc_cone_idx = _oc.ConeParts(cap.layer_ids[slots[0] :])
            state.oc_cone_idx.add(
                int(capture_start), end, r, version, state.oc_cone_run
            )
        if state.oc_cone_prompt is None:
            if sampled > 0:
                state.oc_cone_prompt = end
            else:
                # Cells this far behind the current end cannot reach any sampled token.
                state.oc_cone.trim(end - _oc.CONE_MAX)
                if state.oc_cone_idx is not None:
                    state.oc_cone_idx.trim(end - _oc.CONE_MAX)

    def _oc_cone_finish(self, state, key_id, total: int, w):
        """Writes this request's patch; returns the manifest field: every patch of the
        trajectory, each with the end up to which its rows belong to this trajectory,
        and the count of cells this request's tokens depend on that no patch holds."""
        lists = getattr(w, "cone_list", None)
        if lists is None:
            lists = w.cone_list = {}
        held_map = getattr(w, "cone_held", None)
        if held_map is None:
            held_map = w.cone_held = {}
        aux = int(state.oc_aux_start or 0)
        # Rows from the splice end on are this request's, whatever earlier requests
        # computed there.
        cut = getattr(state, "oc_cone_cut", None)
        cut = aux if cut is None else int(cut)
        patches = [
            dict(p, end=min(p["end"], cut))
            for p in lists.get(state.oc_track, [])
            if p["lo"] < cut
        ]
        first = _oc.R3_FIRST_TRIMMED
        cone, cone_idx = state.oc_cone, state.oc_cone_idx
        prompt = state.oc_cone_prompt
        rows_end = int(total)
        base = max(
            0,
            (rows_end if prompt is None else min(prompt, rows_end))
            - _oc.CONE_MAX
            - _oc.R3_WINDOW,
        )
        keep_from = max(0, rows_end - _oc.CONE_MAX) - base

        def one(hkey, parts, layers, key):
            """Stores the cells of one field no earlier patch holds. Returns (lo, hi) or
            None, and the count of needed cells nobody holds."""
            held = held_map.get(hkey)
            # Only cells a sampled token depends on are stored; a request that sampled
            # nothing has none.
            cover = None if parts is None or prompt is None else parts.cover(total)
            if cover is not None:
                for j_, layer_ in enumerate(parts.layers):
                    cover[2][
                        : max(0, prompt - _oc.cone_size(layer_) - cover[0]), j_
                    ] = False
            union = np.zeros((rows_end - base, len(layers)), bool)
            if held is not None and held[0] < cut:
                h0, hc = held
                lo_, hi_ = max(h0, base), min(h0 + len(hc), cut, rows_end)
                if lo_ < hi_:
                    union[lo_ - base : hi_ - base] = hc[lo_ - h0 : hi_ - h0]
            stored = None
            if cover is not None:
                lo, hi, cov = cover
                lo_, hi_ = max(lo, base), hi
                keep = cov.copy()
                if lo_ < hi_:
                    keep[lo_ - lo : hi_ - lo] &= ~union[lo_ - base : hi_ - base]
                    union[lo_ - base : hi_ - base] |= cov[lo_ - lo : hi_ - lo]
                w.put(key, lambda c=parts, t=total, k=keep: c.encode(t, k))
                stored = (int(lo), int(hi))
            n = 0
            if prompt is not None:
                # Rows this request sampled from but did not deliver (aborted while
                # replaying after a preemption) are missing, not someone else's to fill.
                need_hi = max(prompt, int(getattr(state, "oc_freeze", 0) or 0))
                for j, layer in enumerate(layers):
                    need_lo = prompt - _oc.cone_need(layer)
                    a_ = max(base, need_lo)
                    if a_ < rows_end:
                        n += int((~union[a_ - base :, j]).sum())
                    if need_hi > rows_end:
                        n += need_hi - max(rows_end, need_lo, 0)
            held_map[hkey] = (base + keep_from, union[keep_from:])
            return stored, n

        key = _oc.cone_key(state.session, key_id)
        stored, uncovered = one(
            state.oc_track, cone, list(range(first, _oc.R3_LAST_LAYER + 1)), key
        )
        cap = _oc.IDX_CAP
        idx_layers = (
            cone_idx.layers
            if cone_idx is not None
            else None
            if cap is None
            else [lyr for lyr in cap.layer_ids if lyr >= first]
        )
        uncovered_idx = 0
        idx_stored = None
        if idx_layers:
            idx_key = _oc.cone_key(state.session, key_id, "idx")
            idx_stored, uncovered_idx = one(
                (state.oc_track, "idx"), cone_idx, idx_layers, idx_key
            )
        if stored is not None:
            entry = dict(key=key, lo=stored[0], end=stored[1], prompt_end=prompt)
            if idx_stored is not None:
                entry["idx_key"], entry["idx_codec"] = idx_key, _oc.IDX_CODEC
            patches.append(entry)
            w.stats["cone_patches"] = w.stats.get("cone_patches", 0) + 1
        elif idx_stored is not None:
            patches.append(
                dict(
                    lo=idx_stored[0],
                    end=idx_stored[1],
                    prompt_end=prompt,
                    idx_key=idx_key,
                    idx_codec=_oc.IDX_CODEC,
                )
            )
        if uncovered or uncovered_idx:
            w.stats["cone_uncovered"] = (
                w.stats.get("cone_uncovered", 0) + uncovered + uncovered_idx
            )
            if w.stats.get("cone_uncovered_logged", 0) < 20:
                w.stats["cone_uncovered_logged"] = (
                    w.stats.get("cone_uncovered_logged", 0) + 1
                )
                print(
                    f"OC_CONE_UNCOVERED track={state.oc_track!r} n={uncovered} "
                    f"idx={uncovered_idx} aux={aux} prompt={prompt} "
                    f"total={total} stored={stored}",
                    flush=True,
                )
        lists[state.oc_track] = patches
        ends_all = getattr(w, "cone_ends", None)
        if ends_all is None:
            ends_all = w.cone_ends = {}
        ends = [
            [p_, min(e_, cut)]
            for p_, e_ in ends_all.get(state.oc_track, [])
            if p_ <= cut
        ]
        if prompt is not None:
            ends.append([int(prompt), int(rows_end)])
        ends_all[state.oc_track] = ends
        return dict(
            partial=True,
            ends=ends,
            first_deep_layer=first,
            last_layer=_oc.R3_LAST_LAYER,
            window=_oc.R3_WINDOW,
            patches=patches,
            uncovered=uncovered,
            uncovered_idx=uncovered_idx,
            indexer_layers=idx_layers,
        )

    def _oc_idx_complete(self, state, start: int, n: int, w, want_cov=False):
        # Every row of [start, start + n) must be written by this request or spliced
        # from the trajectory's previous tail (positions before aux_start); anything
        # else is a gap.
        bs = self._buffer.block_size
        cov = np.zeros(n, dtype=bool)
        for b in range(start // bs, (start + n + bs - 1) // bs):
            m = (state.oc_idx_mask or {}).get(b)
            if m is None:
                continue
            lo, hi = max(start, b * bs), min(start + n, (b + 1) * bs)
            cov[lo - start : hi - start] = m[lo - b * bs : hi - b * bs]
        last = (
            w.last_tail_idx.get(state.oc_track) if state.oc_track is not None else None
        )
        if last is not None and state.oc_aux_start is not None:
            # The same rows _oc_splice takes: the previous tail at this block's offset,
            # up to the splice end (aux_start, or aux_start + 1 when the previous
            # request computed it).
            lo = max(start, int(last[0]))
            hi = min(
                start + n,
                int(last[0]) + len(last[1]),
                self._oc_splice_end(state, w.last_tail_idx),
            )
            if hi > lo:
                cov[lo - start : hi - start] = True
        if cov.all():
            return (True, cov) if want_cov else True
        if want_cov:
            return False, cov
        w.stats["idx_gap_rows"] = w.stats.get("idx_gap_rows", 0) + int((~cov).sum())
        w.stats["idx_gap_blocks"] = w.stats.get("idx_gap_blocks", 0) + 1
        if w.stats["idx_gap_blocks"] <= 20:
            miss = np.flatnonzero(~cov)
            print(
                f"OC_IDX_GAP track={state.oc_track!r} start={start} n={n} "
                f"gap={int(len(miss))} "
                f"first={int(start + miss[0])} last={int(start + miss[-1])} "
                f"capture_cursor={state.capture_cursor} aux={state.oc_aux_start} "
                f"freeze={state.oc_freeze} "
                f"last_tail={None if last is None else (last[0], len(last[1]))}",
                flush=True,
            )
        return False

    def _oc_put_idx(self, state, start: int, block_hash: str) -> None:
        w = _oc.writer()
        if w is None or state.session is None or state.oc_idx is None:
            return
        blk = state.oc_idx.pop(start // self._buffer.block_size, None)
        if blk is None:
            return
        if state.oc_idx_bad and start // self._buffer.block_size in state.oc_idx_bad:
            w.stats["idx_bad_blocks"] = w.stats.get("idx_bad_blocks", 0) + 1
            return
        ok, cov = self._oc_idx_complete(
            state, start, self._buffer.block_size, w, want_cov=True
        )
        rows = self._oc_splice(state, start, blk, w.last_tail_idx, "spliced_rows_idx")
        if (
            not ok
            and not cov.any()
            and block_hash.split("~", 1)[0] + _oc.IDX_SUFFIX
            in w.published(state.session)
        ):
            w.stats["idx_block_kept"] = w.stats.get("idx_block_kept", 0) + 1
            return
        if not ok:
            # Rows this request did not capture were served by a prefix hit that ended
            # mid-block: they are the rows of the session's object for this block (the
            # KV the hit used). Without that object the block stays missing (reported at
            # finish).
            raw = block_hash.split("~", 1)[0]
            src, why = None, "unpublished"
            if raw + _oc.IDX_SUFFIX in w.published(state.session):
                b = w.get(
                    _oc.block_key(state.session, raw) + _oc.IDX_SUFFIX, timeout_s=5.0
                )
                why = "get_failed"
                if b:
                    src = _oc.idx_rows_like(b, rows)
                    why = "short"
            if src is None or len(src) < len(rows):
                w.stats["idx_fill_fail_" + why] = (
                    w.stats.get("idx_fill_fail_" + why, 0) + 1
                )
                if w.stats["idx_fill_fail_" + why] <= 5:
                    n_pub_idx = sum(
                        1
                        for x in w.published(state.session)
                        if x.endswith(_oc.IDX_SUFFIX)
                    )
                    print(
                        f"OC_IDX_FILL_FAIL why={why} name={block_hash[:20]} "
                        f"start={start} "
                        f"track={state.oc_track!r} "
                        f"n_pub_idx={n_pub_idx}",
                        flush=True,
                    )
                self._oc_idx_complete(
                    state, start, self._buffer.block_size, w
                )  # counts and logs
                return
            rows = np.array(rows, copy=True)
            rows[~cov] = src[: len(rows)][~cov]
            w.stats["idx_gap_filled"] = w.stats.get("idx_gap_filled", 0) + 1
        w.put_block(
            state.session,
            block_hash,
            lambda r=rows: _oc.idx_encode(r),
            suffix=_oc.IDX_SUFFIX,
        )

    def _oc_put_hid(self, request_id: str, state) -> None:
        rows, state.oc_hid = state.oc_hid, []
        if not rows:
            return
        _oc.writer().put(
            _oc.hid_key(state.session, _oc.key_id(request_id), state.oc_hid_chunks),
            lambda rows=rows: np.stack([r[0][r[1]] for r in rows]).tobytes(),
        )
        state.oc_hid_chunks += 1
        # A preemption flushes a short chunk: the manifest lists every chunk's real row
        # count.
        state.oc_hid_lens = (state.oc_hid_lens or []) + [len(rows)]

    def _oc_finish_extra(self, request_id, state, key_id, blocks, tail_start, total):
        """Indexer: unpublished full blocks + tail; hidden: last chunk. Returns
        callables that build the manifest fields on the writer thread."""
        w = _oc.writer()
        bs = self._buffer.block_size
        fields = []
        if state.oc_idx is not None:
            published = w.published(state.session)
            missing = 0
            for i, h in enumerate(blocks):
                if state.oc_idx_bad and i in state.oc_idx_bad:
                    # This request's rows for the block overflowed: report it missing
                    # rather than list another trajectory's object.
                    state.oc_idx.pop(i, None)
                    missing += 1
                    continue
                if h + _oc.IDX_SUFFIX in published:
                    continue
                blk = state.oc_idx.pop(i, None)
                if blk is None or not self._oc_idx_complete(state, i * bs, bs, w):
                    missing += 1
                    continue
                rows = self._oc_splice(
                    state, i * bs, blk, w.last_tail_idx, "spliced_rows_idx"
                )
                w.put_block(
                    state.session,
                    h,
                    lambda r=rows: _oc.idx_encode(r),
                    suffix=_oc.IDX_SUFFIX,
                )
            tkey = _oc.idx_tail_key(state.session, key_id)
            n_tail = total - tail_start
            parts = [
                state.oc_idx.get(b)
                for b in range(tail_start // bs, (total + bs - 1) // bs)
            ]
            bad_tail = bool(state.oc_idx_bad) and any(
                b in state.oc_idx_bad
                for b in range(tail_start // bs, (total + bs - 1) // bs)
            )
            if (
                n_tail > 0
                and not bad_tail
                and all(p is not None for p in parts)
                and self._oc_idx_complete(state, tail_start, n_tail, w)
            ):
                rows = self._oc_splice(
                    state,
                    tail_start,
                    np.concatenate(parts)[:n_tail],
                    w.last_tail_idx,
                    "spliced_rows_idx",
                )
                w.last_tail_idx[state.oc_track] = (
                    tail_start,
                    np.array(rows, copy=True),
                )
                _oc_keep_tail(w, "idx", state.oc_track, w.last_tail_idx[state.oc_track])
                w.put(tkey, lambda r=rows: _oc.idx_encode(r))
            else:
                missing += int(n_tail > 0)
                w.put(tkey, b"\0")
            keys = [
                _oc.block_key(state.session, h) + _oc.IDX_SUFFIX for h in blocks
            ] + [tkey]
            cap, holes = _oc.IDX_CAP, state.oc_idx_holes
            fields.append(
                lambda: dict(
                    name="indexer_topk",
                    dtype="int32",
                    shape=list(cap.shape),
                    codec=_oc.IDX_CODEC,
                    layout="blocks+tail",
                    block_suffix=_oc.IDX_SUFFIX,
                    tail_key=tkey,
                    layers=list(cap.layer_ids),
                    rows=int(total),
                    tail_start=int(tail_start),
                    tail_rows=int(n_tail),
                    missing=missing,
                    holes=holes,
                    nbytes=sum(
                        w.sizes.get(k, 0) for k in keys if w.sizes.get(k, 0) > 1
                    ),
                )
            )
        if state.oc_hid is not None and not state.oc_hid_bad:
            self._oc_put_hid(request_id, state)
            hkeys = [
                _oc.hid_key(state.session, key_id, i)
                for i in range(state.oc_hid_chunks)
            ]
            n, d, start = state.oc_hid_n, state.oc_hid_dim, state.oc_hid_start
            fields.append(
                lambda: dict(
                    name="hidden",
                    dtype=state.oc_hid_dtype,
                    shape=[d],
                    codec="raw",
                    layout="chunks",
                    key_prefix=_oc.hid_key(state.session, key_id, ""),
                    start=int(start),
                    rows=int(n),
                    chunk_rows=_oc.HID_CHUNK,
                    chunks=len(hkeys),
                    chunk_lens=list(state.oc_hid_lens or []),
                    nbytes=sum(w.sizes.get(k, 0) for k in hkeys),
                )
            )
        if state.oc_tk is not None and not state.oc_tk_bad:
            self._oc_put_tk(request_id, state)
            tkeys = [
                _oc.topk_key(state.session, key_id, i)
                for i in range(state.oc_tk_chunks)
            ]
            fields.append(
                lambda: dict(
                    name="topk",
                    dtype="int32+float32",
                    shape=[_oc.TOPK_K],
                    codec="raw",
                    layout="chunks",
                    key_prefix=_oc.topk_key(state.session, key_id, ""),
                    start=int(state.oc_tk_start),
                    rows=int(state.oc_tk_n),
                    chunk_rows=_oc.HID_CHUNK,
                    chunks=len(tkeys),
                    chunk_lens=list(state.oc_tk_lens or []),
                    nbytes=sum(w.sizes.get(k, 0) for k in tkeys),
                )
            )
        return fields

    def _oc_splice(
        self,
        state: _WorkerRequestState,
        start: int,
        rows,
        tails=None,
        stat="spliced_rows",
    ):
        # First computation wins: positions before aux_start that this session already
        # delivered (the previous turn's tail) keep their rows; a later prefill that
        # recomputes the unfinished block may route a few of them differently.
        if state.oc_parent:
            self._oc_inherit(state)
        w = _oc.writer()
        if tails is None and w is not None:
            tails = w.last_tail
        if state.oc_stash and w is not None and tails is w.last_tail:
            old = state.oc_stash.get(start)
            k = min(len(rows), state.oc_freeze - start, 0 if old is None else len(old))
            if k > 0:
                rows = np.array(rows, copy=True)
                rows[:k] = old[:k]
                w.stats["stash_rows"] = w.stats.get("stash_rows", 0) + int(k)
        last = (
            None if w is None or state.oc_track is None else tails.get(state.oc_track)
        )
        if last is None or state.oc_aux_start is None:
            return rows
        # The previous tail may span several blocks (blocks that filled before their
        # hash arrived are delivered in it): take the rows at this block's offset into
        # it.
        off = start - int(last[0])
        if os.environ.get("OC_TRACE") == "1" and abs(off) <= 512:
            print(
                f"OC_SPL track={state.oc_track!r} start={start} "
                f"last=({last[0]},{len(last[1])}) "
                f"aux={state.oc_aux_start} n={len(rows)} stat={stat} "
                f"stash={bool(state.oc_stash)}",
                flush=True,
            )
        if off < 0 or off >= len(last[1]):
            return rows
        # Positions up to aux_start - 1 always came from the previous request. With
        # speculative decoding the previous request's last step can also have computed
        # aux_start itself (its input is the last sampled token): that row came first
        # too. Rows past it are truncated drafts and are never spliced.
        k = min(
            len(last[1]) - off, self._oc_splice_end(state, tails) - start, len(rows)
        )
        if k <= 0:
            return rows
        rows = np.array(rows, copy=True)
        rows[:k] = last[1][off : off + k]
        w.stats[stat] = w.stats.get(stat, 0) + int(k)
        return rows

    def _oc_splice_end(self, state, tails=None) -> int:
        aux = int(state.oc_aux_start or 0)
        w = _oc.writer()
        if tails is None and w is not None:
            tails = w.last_tail
        if state.oc_splice_cap is not None:
            return min(aux, int(state.oc_splice_cap))
        last = (
            None
            if tails is None or state.oc_track is None
            else tails.get(state.oc_track)
        )
        if last is not None and int(last[0]) <= aux < int(last[0]) + len(last[1]):
            return aux + 1
        return aux

    def _oc_finish(
        self,
        request_id: str,
        state: _WorkerRequestState,
        pending: list[tuple[int, np.ndarray]],
        tail: tuple[int, np.ndarray] | None,
        generation: int,
    ) -> None:
        w = _oc.writer()
        buffer = self._buffer
        if w is None or state.session is None or buffer is None:
            return
        bs = buffer.block_size
        if state.oc_parent:
            self._oc_inherit(state)
        aux_ = int(state.oc_aux_start or 0)
        raw_ = [k.rsplit("/", 1)[1].split("~", 1)[0] for k in state.aux_output_keys]
        blocks_ = getattr(w, "cone_blocks", None)
        if blocks_ is None:
            blocks_ = w.cone_blocks = {}
        prev_ = blocks_.get(state.oc_track)
        fork_ = prev_ is None and state.oc_cone_src is not None
        if fork_:
            prev_ = blocks_.get(state.oc_cone_src)
        cone_cut = _oc.cone_boundary_cut(
            aux_,
            self._oc_splice_end(state),
            state.oc_splice_cap is not None,
            bs,
            prev_,
            raw_,
            getattr(w, "cone_list", {}).get(state.oc_track, []),
            fork_,
        )
        if state.oc_track is not None:
            blocks_[state.oc_track] = raw_
        # Engine-internal ids are '<caller id>-<8 random chars>'; key by the caller's
        # id.
        key_id = (
            request_id[:-9]
            if len(request_id) > 9 and request_id[-9] == "-"
            else request_id
        )
        hashes = [k.rsplit("/", 1)[1] for k in state.aux_output_keys]
        own = getattr(w, "track_alt", {}).get(state.oc_track, {})
        if own:
            mapped = [i for i, h in enumerate(hashes) if h in own]
            if mapped and w.stats.get("alt_map_logged", 0) < 12:
                w.stats["alt_map_logged"] = w.stats.get("alt_map_logged", 0) + 1
                print(
                    f"OC_ALT_MAP rid={key_id} track={state.oc_track!r} "
                    f"n_mapped={len(mapped)} "
                    f"first_idx={mapped[:8]} n_blocks={len(hashes)} "
                    f"oc_alt={sorted((state.oc_alt or {}).keys())[:8]}",
                    flush=True,
                )
            hashes = [own.get(h, h) for h in hashes]
        for i, name in (state.oc_alt or {}).items():
            if i < len(hashes):
                hashes[i] = name
        total = int(state.capture_cursor or 0)
        if (state.oc_stash or state.oc_parked_finish) and state.oc_freeze > total:
            # Aborted while replaying after a preemption, or finished while parked: rows
            # up to the freeze point are the first computation, in published blocks or
            # the stash.
            total = int(state.oc_freeze)
        nfull = min(total // bs, len(hashes))
        published = w.published(state.session)
        missing = [i for i in range(nfull) if hashes[i] not in published]
        if state.oc_stash:
            # A block this request computed before a preemption is its first computation
            # even if a sibling published the same hash meanwhile: it goes through the
            # version rule (shared when the versions match, its own copy otherwise), not
            # straight to the sibling's object.
            tag_ = _oc_tag(state.oc_track)
            for i in range(nfull):
                rows_ = state.oc_stash.get(i * bs)
                if (
                    i not in missing
                    and rows_ is not None
                    and len(rows_) == bs
                    and not w.holds(state.session, tag_, hashes[i].split("~", 1)[0])
                ):
                    missing.append(i)
            missing.sort()
        if missing and self._store is not None:
            # Blocks this request hit but no request of this session published (c1).
            for i in missing:
                # A block completed before a preemption and hit by the resumed request
                # was never published under a hash; its rows are in the stash.
                rows = (state.oc_stash or {}).get(i * bs)
                bsegs = getattr(w, "block_segs", None)
                if bsegs is None:
                    bsegs = w.block_segs = {}
                if rows is None or len(rows) != bs:
                    try:
                        payload = self._store.get_concatenated(
                            [state.aux_output_keys[i]]
                        )
                    except Exception as e:
                        # The base object is gone (keys released at a preemption
                        # teardown, or a generation change while parked): report the
                        # block missing, never raise inside begin_step.
                        w.stats["missing_base_blocks"] = (
                            w.stats.get("missing_base_blocks", 0) + 1
                        )
                        if w.stats["missing_base_blocks"] <= 5:
                            print(
                                f"oc missing base block {state.aux_output_keys[i]}: "
                                f"{e!r}",
                                flush=True,
                            )
                        continue
                    rows = np.frombuffer(payload, dtype=buffer.dtype).reshape(
                        (-1, *buffer.shape_per_token)
                    )
                    rec = bsegs.get(("", generation, hashes[i]))
                else:
                    rec = _clip_segs(
                        w.weight_versions.get(state.oc_track, []), i * bs, (i + 1) * bs
                    )
                rows = self._oc_splice(state, i * bs, rows)
                if _oc.r3_has_sentinel(rows):
                    w.stats["r3_trim_blocked_finish"] = (
                        w.stats.get("r3_trim_blocked_finish", 0) + 1
                    )
                    continue
                _set_rec(bsegs, (state.session, hashes[i]), rec)
                name, new_ = w.put_block_checked(
                    state.session,
                    hashes[i],
                    rows.tobytes(order="C"),
                    _oc_tag(state.oc_track),
                    segs=rec,
                )
                if new_ and name != hashes[i]:
                    _set_rec(
                        bsegs,
                        (state.session, name),
                        w.object_segs(state.session, name) or rec,
                    )
                if name != hashes[i]:
                    if getattr(w, "track_alt", None) is None:
                        w.track_alt = {}
                    w.track_alt.setdefault(state.oc_track, {})[hashes[i]] = name
                    hashes[i] = name
                published = w.published(state.session)
        tail_start = nfull * bs
        if os.environ.get("OC_TRACE") == "1":
            print(
                f"OC_FIN rid={key_id} track={state.oc_track!r} total={total} "
                f"nfull={nfull} "
                f"nhash={len(hashes)} tail_start={tail_start} "
                f"buf={None if tail is None else (tail[0], len(tail[1]))} "
                f"pending={sorted(int(x[0]) for x in pending)} "
                f"aux={state.oc_aux_start} "
                f"missing={missing[:4]}",
                flush=True,
            )
        tail_bytes = b""
        raw_tail = None
        if tail is not None and tail[0] == tail_start:
            raw_tail = tail[1]
        elif total > tail_start and any(int(x[0]) >= tail_start for x in pending):
            # Full blocks completed after the last hash the request received (with
            # speculative decoding the final hashes can lag the accepted rows): they are
            # still pending without a key. Deliver them in the tail, in order, followed
            # by the buffer's incomplete block if there is one (none when the capture
            # ended on a block boundary).
            parts, at = [], tail_start
            for start, rows in sorted(pending, key=lambda x: x[0]):
                if start == at:
                    parts.append(np.asarray(rows))
                    at += len(rows)
            if tail is not None and tail[0] == at:
                parts.append(tail[1])
                at += len(tail[1])
            if at >= total:
                raw_tail = np.concatenate(parts)[: total - tail_start]
                w.stats["tail_from_pending"] = w.stats.get("tail_from_pending", 0) + 1
            else:
                w.stats["tail_gap"] = w.stats.get("tail_gap", 0) + 1
        elif state.oc_stash and tail_start in state.oc_stash and total > tail_start:
            # Finished while parked: the tail exists only in the stash, possibly as
            # several contiguous blocks (hash-less full blocks plus the partial one).
            parts, at = [], tail_start
            while at < total and at in state.oc_stash and len(state.oc_stash[at]):
                parts.append(np.asarray(state.oc_stash[at]))
                at += len(state.oc_stash[at])
            if at >= total:
                raw_tail = np.concatenate(parts)[: total - tail_start]
                w.stats["tail_from_stash"] = w.stats.get("tail_from_stash", 0) + 1
            else:
                w.stats["tail_gap"] = w.stats.get("tail_gap", 0) + 1
        if raw_tail is None and total == tail_start and total > 0:
            # Ended on a block boundary: record where it ended, so a fork from this
            # point knows the block before it is complete.
            _oc_keep_tail(w, "r3", state.oc_track, (tail_start, buffer._rows[0, :0]))
        if raw_tail is not None:
            tail_rows = self._oc_splice(state, tail_start, raw_tail)
            if _oc.r3_has_sentinel(tail_rows):
                w.stats["r3_trim_tail"] = w.stats.get("r3_trim_tail", 0) + 1
            else:
                w.last_tail[state.oc_track] = (
                    tail_start,
                    np.array(tail_rows, copy=True),
                )
                _oc_keep_tail(w, "r3", state.oc_track, w.last_tail[state.oc_track])
                tail_bytes = tail_rows.tobytes(order="C")
        w.put(_oc.tail_key(state.session, key_id), tail_bytes or b"\0")
        if hasattr(w, "note_has"):
            raw = [k.rsplit("/", 1)[1] for k in state.aux_output_keys[:nfull]]
            w.note_has(
                state.session, _oc_tag(state.oc_track), dict(zip(raw, hashes[:nfull]))
            )
        meta = dict(
            key_prefix=f"{_oc.PREFIX}/{state.session}",
            tail_key=_oc.tail_key(state.session, key_id),
            blocks=hashes[:nfull],
            tail_start=int(tail_start),
            rows=int(total),
            tail_nbytes=len(tail_bytes),
            dtype=str(buffer.dtype),
            shape=list(buffer.shape_per_token),
            block_size=bs,
            missing_from_session=len(missing),
            invalid_from=state.oc_invalid_from,
            weight_versions=_object_segs(
                [s for s in w.weight_versions.get(state.oc_track, []) if s[0] < total],
                [(state.session, h) for h in hashes[:nfull]],
                bs,
                getattr(w, "block_segs", {}),
                total,
            ),
        )
        if _oc.R3_PARTIAL:
            state.oc_cone_cut = cone_cut
            meta["r3_cone"] = self._oc_cone_finish(state, key_id, int(total), w)
        meta["fields"] = [
            dict(
                name="routed_experts",
                dtype=str(buffer.dtype),
                shape=list(buffer.shape_per_token),
                codec="raw",
                layout="blocks+tail",
                rows=int(total),
                nbytes=nfull
                * bs
                * int(np.prod(buffer.shape_per_token))
                * buffer.dtype.itemsize
                + len(tail_bytes),
            )
        ]
        extra = self._oc_finish_extra(
            request_id, state, key_id, hashes[:nfull], int(tail_start), int(total)
        )
        w.put(
            _oc.manifest_key(state.session, key_id),
            lambda: _json.dumps(
                dict(
                    meta,
                    fields=meta["fields"] + [f() for f in extra],
                    writer_stats=dict(w.stats),
                )
            ).encode(),
        )

    def begin_step(self, metadata: AuxOutputConnectorMetadata | None) -> None:
        w_ = _oc.writer()
        with self._lock:
            self._oc_raise()
            if w_ is not None:
                self._oc_post(w_.prune_engine_tables)
                if not self._oc_inline:
                    w_.stats["oc_apply_backlog"] = self._oc_posted - self._oc_applied
        """Apply one scheduler step's request and block-hash updates.

        While an unconsumed step covers a finished request, its teardown is
        deferred to that step's process_output.
        """
        self._step_metadata = metadata
        _oc.step_begin()
        if self._buffer is None or metadata is None:
            return
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "auxiliary output request cannot run and finish in one step"
        )
        assert metadata.generation >= self._generation, (
            "auxiliary output metadata generation moved backwards"
        )
        with self._lock:
            release_keys: list[str] = []
            stash_release: list[str] = []
            if metadata.generation > self._generation:
                # A generation change follows a prefix-cache reset, which the
                # scheduler only performs once all model output is consumed.
                assert not self._pending_outputs, (
                    "auxiliary output generation changed with output in flight"
                )
                # Requests that finished before the reset are finished in the old
                # generation: their final hashes name objects of that generation.
                done_before = [
                    r for r in metadata.finished_requests if r in self._requests
                ]
                old_batches, old_retained = [], []
                parked_now = set(getattr(metadata, "parked_finished", ()))
                for request_id in done_before:
                    state = self._requests[request_id]
                    state.finished = True
                    if request_id in parked_now:
                        # Finished while its preemption teardown was still deferred: the
                        # final hashes are resent from the start; skip the ones it
                        # already has.
                        state.preempted = False
                        state.oc_hash_skip = len(state.aux_output_keys)
                    if request_id in metadata.block_hashes:
                        keys = routed_experts_keys(
                            metadata.block_hashes[request_id], str(self._generation)
                        )
                        keys, state.oc_hash_skip = keys[state.oc_hash_skip :], 0
                        state.aux_output_keys.extend(keys)
                        old_retained.extend(keys)
                        old_batches.append((state, []))
                self._publish_blocks(old_batches, old_retained, [])
                for state in self._requests.values():
                    (release_keys if state.finished else stash_release).extend(
                        reversed(state.aux_output_keys)
                    )
                self._teardown(done_before)
                # reset_prefix_cache(reset_running_requests=True) (in-flight weight
                # updates) preempts every running request: stash them like a preemption
                # so the rows computed before the reset stay first. Every old request is
                # listed, so its old preemption event never touches a new state of the
                # same id.
                reset_stashed = set(done_before)
                for request_id, state in self._requests.items():
                    reset_stashed.add(request_id)
                    if state.session is not None and not state.finished:
                        self._oc_stash_now(request_id, state)
                self._buffer.reset()
                self._requests.clear()
                self._generation = metadata.generation
                self._oc_post(self._oc_drop_block_segs, self._generation)
            else:
                reset_stashed = set()
            for request_id, emit_start in metadata.requests.items():
                state = self._requests.get(request_id)
                new = state is None
                if new:
                    state = self._requests[request_id] = _WorkerRequestState(
                        emit_cursor=emit_start
                    )
                assert not state.preempted, (
                    "auxiliary output request resumed before its preemption teardown"
                )
                if metadata.sessions and request_id in metadata.sessions:
                    state.oc_track, _, parent = metadata.sessions[request_id].partition(
                        "\x1e"
                    )
                    state.session = state.oc_track.split("\x1f", 1)[0]
                    state.oc_parent = parent or None
                # Resumed with or without session metadata: its stash is consumed all
                # the same.
                saved = self._oc_take_stash(request_id, state)
                if new or saved is not None:
                    self._oc_post(self._oc_resume, state, saved, int(emit_start))
                if state.pending_outputs == 0:
                    # In-flight steps leave the worker cursor behind the
                    # scheduler's optimistic view; only check settled requests.
                    assert emit_start <= state.emit_cursor, (
                        "auxiliary output Scheduler emit cursor moved ahead"
                    )
            block_batches: list[
                tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
            ] = []
            retained_keys: list[str] = []
            for request_id in getattr(metadata, "parked_finished", ()):
                # Finished while parked (the output in flight at the preemption ended
                # it): finish from the stash. Its keys were never retained, so none are
                # released.
                if request_id in self._requests:
                    # Preemption teardown still deferred: finish it as a normal finish.
                    # The final hashes are resent from the start; skip the ones it
                    # already has.
                    state = self._requests[request_id]
                    state.preempted = False
                    state.oc_hash_skip = len(state.aux_output_keys)
                elif request_id in self._oc_preempted:
                    state = self._requests[request_id] = _WorkerRequestState(
                        oc_parked_finish=True
                    )
                    self._oc_post(
                        self._oc_resume,
                        state,
                        self._oc_take_stash(request_id, state),
                        None,
                    )
            for request_id, block_hashes in metadata.block_hashes.items():
                if request_id in reset_stashed and request_id not in metadata.requests:
                    continue
                state = self._requests[request_id]
                keys = routed_experts_keys(block_hashes, str(self._generation))
                keys, state.oc_hash_skip = keys[state.oc_hash_skip :], 0
                state.aux_output_keys.extend(keys)
                if not state.oc_parked_finish:
                    retained_keys.extend(keys)
                block_batches.append((state, []))
            finished_now: list[str] = []
            # A normal finish overrides a preemption named in the same metadata.
            parked = set(getattr(metadata, "parked_finished", ()))
            # A request that ended (finished or aborted) in the same step is not
            # resumable: its terminal event wins over the preemption, so nothing is
            # stashed for it.
            ended = set(metadata.finished_requests)
            preempted = [
                r
                for r in getattr(metadata, "preempted_requests", ())
                if r not in reset_stashed
                and r not in parked
                and r not in ended
                and r in self._requests
            ]
            for request_id in preempted:
                self._requests[request_id].preempted = True
            finished = [r for r in metadata.finished_requests if r not in reset_stashed]
            for request_id in dict.fromkeys((*finished, *preempted)):
                if request_id not in self._requests:
                    # Ended while preempted (abort): nothing to deliver, drop the stash.
                    self._oc_preempted.pop(request_id, None)
                    continue
                state = self._requests[request_id]
                state.finished = True
                if state.preempted and request_id in ended:
                    # Aborted before its preemption teardown ran: the stash is still
                    # written, but the request never resumes, so no record is kept for
                    # it.
                    state.oc_aborted = True
                if state.pending_outputs:
                    if (
                        not state.oc_closing_marked
                        and state.oc_track is not None
                        and not state.preempted
                    ):
                        self._oc_post(self._oc_closing, state.oc_track, 1)
                        state.oc_closing_marked = True
                    continue
                finished_now.append(request_id)
                if not state.oc_parked_finish:
                    (stash_release if state.preempted else release_keys).extend(
                        reversed(state.aux_output_keys)
                    )
            self._publish_blocks(block_batches, retained_keys, [])
            self._teardown(finished_now, release_keys, stash_release)

    def _oc_drop_block_segs(self, generation: int) -> None:
        # Base-store version records are per generation; older generations' objects are
        # gone.
        bsegs = getattr(_oc.writer(), "block_segs", None) if _oc.writer() else None
        if bsegs:
            for k in [k for k in bsegs if k and k[0] == "" and k[1] != generation]:
                del bsegs[k]

    def close(self) -> None:
        try:
            self._oc_drain()
        finally:
            self._oc_stop()
            if _oc.writer() is not None:
                _oc.writer().flush()
            with self._lock:
                self._pending_outputs.clear()
                self._requests.clear()
                getattr(self, "_oc_preempted", {}).clear()
                if self._store is not None:
                    try:
                        self._store.close()
                    finally:
                        self._store = None


def get_aux_output_connector(
    model: torch.nn.Module, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> AuxOutputWorkerConnector:
    return AuxOutputWorkerConnector(
        model=model,
        kv_cache_config=kv_cache_config,
        vllm_config=vllm_config,
    )
