# SWA decode-end snapshot (VLLM_SWA_DECODE_SNAPSHOT=1)

Overlay for the DSV4.1 image (vLLM 0.32.0rc1.dev281+g49cb2bbf7). Scope study: `../../swa-two-snapshot-scope.md`.

## What changed

The baseline is `swa_bounded_replay=False` with `prefix_cache_retention_interval=0`. In that configuration each SWA group already hashes the 4-block window that ends at the prompt-end boundary. The patch adds a second window, the one at the end of decode. It is hashed when the request's blocks are freed: on finish, abort, or preemption.

| File | Change |
| --- | --- |
| `v1/core/single_type_kv_cache_manager.py` | `SlidingWindowManager` changes:<br>- `decode_snapshot` flag: env set, caching on, and the spec is prefix-cacheable.<br>- **Hold**: `get_num_skipped_tokens` frees nothing at or above `A-128`, where `A = floor(c/128)*128` and `c` is the processed (committed) token count. The 4 blocks a hit at `A` needs therefore stay allocated.<br>- Hit allocation (`get_num_hit_skipped_tokens`) still uses the plain window formula.<br>- New `cache_final_snapshot(request, committed)`: hashes blocks `[A/32-4, A/32)` through the stock `BlockPool.cache_full_blocks`, with a mask that skips blocks that already have a hash. It does nothing if any of those blocks is null. `committed` is capped at the group's `num_cached_block`, so blocks that are still loading for remote/connector KV are never hashed. |
| `v1/core/kv_cache_coordinator.py` | `cache_final_snapshot(request)`: `committed = min(num_tokens-1, num_computed_tokens - num_in_flight_tokens)`, the same commit point as the GPU prefix cache. Engine start refuses the mode with a ValueError when:<br>- EAGLE/MTP/any prefill lookahead is on;<br>- `retention_interval != 0`;<br>- there is no prefix-cacheable SWA group (bounded replay is on);<br>- `extra_retained_tokens` is set;<br>- the alignment is not block-representable. |
| `v1/core/kv_cache_manager.py` | `cache_final_snapshot` forwards to the coordinator when caching is on. |
| `v1/core/sched/scheduler.py` | `_free_request_blocks` calls the snapshot before `free` / `pop_blocks_for_free`. This covers finish (including delayed-free connector paths), abort and `_preempt_request`. |
| `v1/kv_cache_interface.py` | `SlidingWindowSpec.max_admission_blocks_per_request` adds `2*window` tokens (+8 blocks per SWA group) when the env is set, so the admission cap covers the hold. The hold is at most `align + 128 - window = 128` tokens; the validation checks the bound. |

Size: 121 added lines and 2 changed lines in 5 files. There is no model, kernel or CUDA-graph change. With the env unset, every new path returns early and `get_num_skipped_tokens` / admission are byte-for-byte unchanged in behavior.

Not included (follow-up): the OffloadingConnector store job. `_build_store_jobs` would need three changes:

- add the finish commit point to `reachable_boundaries`;
- rewind `next_stored_chunk_idx` over the tail chunks, which it already passed during decode (the existing `reconsider_final_swa_tail` rewind only runs when `retention_interval is None`);
- confirm the tail block ids are still live in its group state.

That is roughly 40 lines plus tests, and the Mooncake store coordinator needs the same change. Until it lands, with `OC_OFFLOAD_GB` the decode-end window lives only in the GPU prefix cache. Once it is evicted there, the CPU tier still serves only the prompt-end window.

## Enable

```bash
SP=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
python3 patch_swa_snapshot.py $SP   # once per container
export VLLM_SWA_DECODE_SNAPSHOT=1 OC_SWA_REPLAY=0      # x10_driver: swa_bounded_replay=False
unset OC_SPEC                                          # spec decode is refused
```

`prefix_cache_retention_interval` must stay at its default of 0. Re-running the patch on a patched tree fails its asserts (exit 1) by design.

## Tests

`test_swa_snapshot.py` runs with a bare python3 (the login node has 3.10, no torch, no pytest). It also runs under pytest with `VLLM_SRC=<vllm dir>`.

It cuts the real methods out of the source files with `ast` and runs them inside stand-in classes. Real code under test:

- the `SlidingWindowManager` methods (find_longest_cache_hit, reachable_block_mask, cache_blocks, remove_skipped_blocks, hold, snapshot);
- `BlockPool.cache_full_blocks`;
- `KVCacheCoordinator.cache_blocks` / `cache_final_snapshot` / validation and `replay_boundaries`;
- `KVCacheManager.cache_final_snapshot`;
- `Scheduler._free_request_blocks`;
- `SlidingWindowSpec.max_admission_blocks_per_request`.

The block pool's storage, the requests and the spec are fakes. The scenario tests drive one SWA group the way `allocate_slots` does: chunked prefill, then decode, then a scheduler free. Then turn 2's prompt (the full turn-1 sequence plus new tokens) is looked up.

```text
cd <dir with orig/ and patched/ copies of the vllm package>
python3 test_swa_snapshot.py patched/vllm   ->  40 passed, 0 failed
python3 test_swa_snapshot.py orig/vllm      ->  15 passed, 25 failed
```

On the pristine tree, the following fail: turn-2 hits at the decode end (4 of 5 cases; single-token output lands on the prompt-end window either way), preemption resume, committed-point hashing, snapshot-before-free, the hold, admission, and validation. The 15 tests that pass on both trees are the "unchanged when off" and safety cases: env-off hits, resend hit, hit allocation, no hashing for unloaded or incomplete tails.

Not covered: the hybrid coordinator fixed point with the compressed/full groups (unchanged code; those groups hash decode blocks densely), real LRU eviction, the worker side. Those need a real run.

## Verify on a real run

1. **Two-turn smoke.**
   - Turn 1: prompt P tokens, `max_tokens=300`, `ignore_eos`.
   - Turn 2: `TokensPrompt(P ids + turn-1 output ids + 64 new ids)`.
   - Expect turn 2's `RequestOutput.num_cached_tokens == floor((P+300-1)/128)*128`.
   - Without the env, expect `floor((P-1)/128)*128`.
2. **x10 run.** Use the same arm with `OC_SWA_REPLAY=0` and toggle only `VLLM_SWA_DECODE_SNAPSHOT`, on the same node.
   - The per-phase `engine_metrics` give the cumulative counters `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`. Take after minus before for each phase.
   - The hit tokens should rise by about the previous turns' generated tokens, rounded down to 128.
   - Per request, turn k's `num_cached_tokens` should be at least `floor((prev prompt + prev output - 1)/128)*128`. This only holds when the driver appends the previous output token-for-token. If it re-tokenizes or strips reasoning, the hit stays at the prompt-end window.
   - Prefill compute per turn should drop accordingly. Check the scheduler-stats `num_preemptions` and KV usage for the extra hold (≤ 4 blocks per running request per SWA group).
3. **Correctness.**
   - With BI on, turn-2 logprobs with the env on must equal a cold run of the same prompt (`reset_prefix_cache` between turns).
   - With BI off, compare to the bounded-replay arm within the usual noise.
   - Watch the log for the ValueError at start: it means a refused config (spec decode, bounded replay on, retention ≠ 0).

## Risks

- **Memory.** The hold keeps up to 4 extra SWA blocks per running request per SWA group: +128 rows ≈ 2.7 MB at 528 B/row over 40 layers, about 690 MB per GPU at 256 decodes. The admission cap adds 8 blocks per group (conservative). Snapshots are ordinary evictable LRU entries; nothing is pinned.
- **Offloading tier.** This is the follow-up above. Under GPU eviction pressure with `OC_OFFLOAD_GB`, the decode-end window is lost, and turn k falls back to the prompt-end window and full recompute of the gap.
- **Deferred frees.** Blocks are hashed before the in-flight step finishes. Only rows below `num_computed - num_in_flight` (already processed) are covered, which is stricter than the stock hash-at-schedule.
- **Aborts during remote/connector loads.** These are guarded by the `num_cached_block` cap, which the tests cover. A connector that marks blocks cached before their data lands would break this, but none in this image does.
- **Spec decode refused.** Supporting it needs `need=5`, an EAGLE-shifted commit point, and handling of the draft lookahead in the commit point.
- **Hits that land between snapshots** (shared system prompt, different turns) clamp to an earlier snapshot. The `shared_prefix_boundary` pin is kept, so later siblings recover it.
