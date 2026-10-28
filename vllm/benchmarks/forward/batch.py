# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a `SchedulerOutput` for an exact batch, without the scheduler."""

from dataclasses import dataclass

import numpy as np

from vllm.benchmarks.forward.shapes import BenchCase
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


@dataclass
class BenchBatch:
    scheduler_output: SchedulerOutput
    # req_id -> token IDs of the query tokens, in order.
    query_token_ids: dict[str, np.ndarray]
    # Per KV cache group: allocated (begin, end) block ranges.
    group_block_ranges: list[list[tuple[int, int]]]
    num_blocks: int
    # req_id -> the runner's expected (scheduled, computed, is_decode).
    expected: dict[str, tuple[int, int, bool]]

    @property
    def num_tokens(self) -> int:
        return self.scheduler_output.total_num_scheduled_tokens

    @property
    def req_ids(self) -> list[str]:
        return list(self.scheduler_output.num_scheduled_tokens)

    def finish(self) -> SchedulerOutput:
        """A step that only removes this batch's requests."""
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(self.req_ids)
        return output


def kv_block_size(spec: KVCacheSpec, dcp_size: int) -> int:
    """Tokens per block as the scheduler counts them: DCP spreads each
    block's tokens over the DCP ranks."""
    return spec.block_size * (dcp_size if spec.dcp_sharded else 1)


def _num_blocks(
    spec: KVCacheSpec, computed: int, kv_len: int, dcp_size: int
) -> tuple[int, int]:
    """Return (leading null blocks, live blocks) for one request."""
    block_size = kv_block_size(spec, dcp_size)
    if isinstance(spec, UniformTypeKVCacheSpecs):
        spec = spec.first_spec
    if isinstance(spec, (CircularBufferSpec, MambaSpec)):
        return 0, 1
    total = -(-kv_len // block_size)
    skipped = 0
    if isinstance(spec, SlidingWindowSpec):
        # Blocks wholly before the window are freed to the null block.
        skipped = max(0, computed - spec.sliding_window + 1) // block_size
    return skipped, total - skipped


def _group_capacities(
    num_blocks: int | tuple[int, ...], num_groups: int
) -> tuple[int, ...]:
    if isinstance(num_blocks, int):
        return (num_blocks,) * num_groups
    if len(num_blocks) != num_groups:
        raise ValueError("KV capacity vector must have one entry per cache group")
    if any(type(count) is not int or count < 1 for count in num_blocks):
        raise ValueError("KV capacity vector entries must be positive integers")
    return num_blocks


def build_batch(
    case: BenchCase,
    kv_cache_groups,
    num_kv_blocks: int | tuple[int, ...],
    vocab_size: int,
    rng: np.random.Generator,
    dcp_size: int = 1,
) -> BenchBatch:
    """Allocate disjoint blocks and build `NewRequestData` for every request.

    Scalar capacity uses a shared pool with globally unique block IDs. A
    capacity vector uses independent group pools. Block 0 is null in each pool.

    With prefix P = K - Q (KV length K, query length Q):

    | Kind               | prompt tokens | num_computed_tokens | spec tokens |
    | ------------------ | ------------- | ------------------- | ----------- |
    | Prefill            | K             | P                   | -           |
    | Decode             | P             | P                   | -           |
    | Verify (D drafts)  | P             | P                   | D           |

    Unless chunked, a prefill's prefix is a prefix hit, which replays its last
    R = min(window, P) tokens on models with SWA bounded replay (window 0
    otherwise), like the scheduler: it schedules R + Q tokens from
    replay_start = num_computed_tokens = P - R.
    """
    capacities = _group_capacities(num_kv_blocks, len(kv_cache_groups))
    independent = not isinstance(num_kv_blocks, int)
    replay_window = max(
        (group.kv_cache_spec.prefix_replay_tokens for group in kv_cache_groups),
        default=0,
    )
    expected: dict[str, tuple[int, int, bool]] = {}
    cursor = 1
    group_cursors = [1] * len(kv_cache_groups)
    new_reqs = []
    query_token_ids = {}
    group_block_ranges: list[list[tuple[int, int]]] = [[] for _ in kv_cache_groups]
    output = SchedulerOutput.make_empty()
    for i, req in enumerate(case.requests):
        req_id = f"bench-{i}"
        replay = min(replay_window, req.prefix_len) if req.prefix_hit else 0
        computed = req.prefix_len - replay
        block_ids = []
        for group_id, (group, ranges) in enumerate(
            zip(kv_cache_groups, group_block_ranges)
        ):
            skipped, live = _num_blocks(
                group.kv_cache_spec, computed, req.kv_len, dcp_size
            )
            begin = group_cursors[group_id] if independent else cursor
            block_ids.append([0] * skipped + list(range(begin, begin + live)))
            ranges.append((begin, begin + live))
            if independent:
                group_cursors[group_id] += live
            else:
                cursor += live
        tokens = rng.integers(0, vocab_size, req.kv_len, dtype=np.int64)
        prompt = tokens[: req.prefix_len] if req.is_decode else tokens
        query_token_ids[req_id] = tokens[computed:]
        expected[req_id] = (req.query_len + replay, computed, req.is_decode)
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt.tolist(),
                prefill_token_ids=prompt.tolist(),
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
                pooling_params=None,
                block_ids=tuple(block_ids),
                num_computed_tokens=computed,
                lora_request=None,
                replay_start=computed if replay else 0,
            )
        )
        output.num_scheduled_tokens[req_id] = req.query_len + replay
        if req.num_draft_tokens:
            output.scheduled_spec_decode_tokens[req_id] = tokens[
                req.prefix_len + 1 :
            ].tolist()
    if isinstance(num_kv_blocks, int) and cursor > num_kv_blocks:
        raise ValueError(
            f"Case {case.name!r} needs {cursor} KV blocks, but only "
            f"{num_kv_blocks} exist. Increase the KV cache size."
        )
    if independent:
        for group_id, (used, capacity) in enumerate(zip(group_cursors, capacities)):
            if used > capacity:
                raise ValueError(
                    f"Case {case.name!r} needs {used} KV blocks in group {group_id}, "
                    f"but only {capacity} exist. Increase the KV cache size."
                )
    output.scheduled_new_reqs = new_reqs
    output.total_num_scheduled_tokens = sum(output.num_scheduled_tokens.values())
    used_blocks = sum(group_cursors) if independent else cursor
    return BenchBatch(
        output, query_token_ids, group_block_ranges, used_blocks, expected
    )
