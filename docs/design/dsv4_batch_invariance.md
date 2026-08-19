# RFC: Batch Invariance for DeepSeek-V4-Flash on GB200

Status: fork-internal roadmap (not for upstream submission in this form).
AI assistance: this work was developed with AI assistance (Claude); every
change is human-reviewed before merge.

## Motivation

RL post-training needs rollout and trainer to agree on logprobs. The
inference-side prerequisite is batch invariance (BI): a request's output must
be bitwise identical regardless of which other requests share its batch. Once
the rollout engine is a deterministic function of the request, the remaining
rollout–trainer gap is a property of the two stacks, not of scheduling noise
— and a zero-gap stack can skip the per-step scoring forward entirely
(IsoExec reports 348s -> 47s per step from exactly this).

vLLM's `VLLM_BATCH_INVARIANT=1` covered dense models (DeepSeek-V3.1 tested);
DeepSeek-V4-Flash adds operators it never covered: mHC (hyper-connections),
DSA sparse attention (indexer top-k + sparse MLA), a tiered router GEMM, a
KV compressor, and fp8/mxfp4 MoE expert stacks.

## Nondeterminism classes

1. Reduction-order drift: split/tile counts derived from `num_tokens` or SM
   count change the summation tree with the batch.
2. Dual implementations: batch-size-gated kernel forks (e.g. `<=16` fast
   paths) produce different rounding.
3. Discrete decision flips: exact-score ties resolved by atomics select
   different top-k sets — different KV / experts, i.e. divergence, not noise.
4. Batch-composition dependence: one request's call geometry (chunk packing,
   alignment ladders) depends on its neighbors.
5. Cache-state dependence: prefix-cache hits change the numeric path (BI is
   defined with prefix caching off; upstream #29125 does the same).

## Operator status (all with archived accuracy/perf/ncu evidence)

| Operator | Defect class | Fix | Cost after fix |
|---|---|---|---|
| mHC (4 entries) | 1+2 | pin K-split=32, disable small-fma fork | ~0 (n=1: +2us) |
| GateLinear router | 1+3 | fp32 direct `linear_batch_invariant` + small-N config | 28.7us vs 8-20us, ~+0.65ms/step |
| indexer top-k | 3 | deterministic tie-break; fused Triton kernel (binary-search select + scan emit) | 18.5us vs 8-14us baseline (was 90-133us) |
| sparse MLA decode | 1 | pin tile-scheduler plan; split each request by its own length | 0.99-1.04x |
| sparse MLA prefill | 4 | one request per chunk (call-geometry independence) | in e2e |
| compressor | — | none needed (row-local, split=8 constant); verified + sensitivity-calibrated | 0 |
| fp8 MoE (DeepGEMM) | 1+4 | zhuzilin fork `set_batch_invariant` + vLLM wiring, alignment ladder pinned | ~0 (grouped) |
| mxfp4 MegaMoE | 1+4 | DeepGEMM-side patch exists (fork branch), e2e not yet passed | +7.6..22% (op level) |

## e2e evidence (fp8 Flash-Base)

- Checker: fixed victim across batch sizes 1-64, three positions, mixed
  long/short compositions crossing the indexer shortcut boundary (2048).
  It runs with prefix caching off, because with caching on the victim's
  prefill is computed once and prefill invariance is never tested — this
  exact blind spot hid the sparse MLA prefill bug. Prefix caching is
  covered separately by a probe that varies the hit length instead (see
  "Configuration assumptions").
- 6L dummy 1-GPU and TP4/EP4: 0 diff x3 rounds; full 43-layer Flash-Base
  TP4/EP4: 0 diff.
- GSM8K (full 43-layer model, 1319 questions, 5-shot, greedy, 64-way
  concurrency): BI=1 accuracy 0.911 with **1319/1319 per-question agreement
  across two independent full reruns** (different batch compositions each
  time) — the live e2e demonstration of batch invariance. BI=0 same-backend
  control: 0.904 and 0.908 across two identical reruns (5 answers flip
  run-to-run), and 87/1319 answers (6.6%) differ from BI=1. No quality
  regression (within 1 sigma).
- Throughput (TP4/EP4 dummy, warm medians, both modes under cudagraphs —
  BI does NOT forfeit graphs in this build): BI total +55%, decomposed by
  counterfactual: the BI NCCL pins (single channel, NCCL_NTHREADS=1,
  tree/Simple, custom-AR/symm-mem off) alone cost +31%; all BI operators
  together +23%. Relaxing the comm pins (a deterministic multi-channel
  config) is the highest-value upstream perf item; every relaxation must
  re-verify invariance.
  That +23% operator term is now attributed per branch, by shadowing
  `VLLM_BATCH_INVARIANT` to `False` in one branch's files at a time (n=14 warm
  medians, 6L model, TP4/EP4, BI=1 baseline 0.955s): sparse-MLA scheduling
  −7.9%, DeepGEMM selection/alignment −5.2%, the fp32 router −3.7%,
  deterministic indexer top-k −1.6%, mHC −1.0%; the five sum to 19.4%. Two
  self-checks: disabling all five at once gives 17.3% (2.1pp of the 19.4% is
  non-additive, as expected when the branches share kernels), and a baseline
  re-measured at the end of the queue drifted 0.5% — a machine-drift floor
  below every one of the five effects, including the smallest. Compare BI=1
  against BI=1 only; the BI/non-BI ratio moved between rounds because the BI=0
  side got faster. All of the above is `data_parallel_size=1`; see the DP
  caveat below.

## Configuration assumptions (2026-08-17)

The e2e claim above was originally hedged with a long list of "…with these
options off". Each item was re-tested against the merged tree of every PR in
the map below. Judgement is >=38 checker rounds with 0 diff — the pre-fix
leak had a 38.7% per-round hit rate, so 38 clean rounds land near 1e-8 by
chance — or a purpose-built probe for the cases the checker cannot reach.

| Assumption | Verdict | Evidence |
|---|---|---|
| code = the tree the evidence was taken on | re-verified, then re-based | merged tree: 47/47 rounds and GSM8K 0.9113/0.9113 with 1319/1319 agreement on the 08-12 base; re-established on upstream `aa9903490` (the commit the runtime image is built from) after the rebase |
| prefix caching off | **not needed** | probe varying the victim's own cached prefix (0/256/512/1280/2304/3328 hit tokens, two independent readings): 18/18 bitwise equal; checker soak with caching on 64/64 |
| `cudagraph_mode` pinned to FULL | **removed** | the pin was stepped over and the production default `FULL_AND_PIECEWISE` ran 50/50 clean with both capture kinds non-zero, so the pin is gone; the merged tree now captures piecewise graphs and stays invariant |
| flashinfer autotune off | **not needed** | autotune on (autotuner confirmed running on all four workers): 42/42. Upstream never coupled the two; autotune resolves once during warmup, and a fixed choice cannot depend on the batch |
| async scheduling off | **was never true** | the serve config never set it and the field auto-enables; every result above and before was taken with `AsyncScheduler` |
| `use_fp4_indexer_cache=False` | **not needed** | the Blackwell recipe's `True`: 42/42 |
| `--block-size 256` | not a knob | `FLASHMLA_SPARSE_DSV4` supports exactly `[256]` |
| batch <=64, 8K context | holds at 2x | `max_num_seqs` 128, `max_model_len` 16384, checker batches through 128: 38/38 |
| the NCCL pins | required, and **not ours** | `init_batch_invariance()` sets them itself before the process group exists; the ablation above measures which ones matter |
| speculative decoding off, SM90 | still open | #27433 upstream; no Hopper part on hand |
| **`data_parallel_size == 1`** | **required, newly written down** | every result above is dp=1. With dp>1 the MoE goes through `AgRsAll2AllManager`, which concatenates every engine's tokens before the expert GEMMs, and measured on DP2xTP2+EP every round leaks. Narrowed to the path, not to the operator: the collective is *not* the cause — an offline probe pinning one rank's slice and varying only the neighbours' token counts finds `reduce_scatterv` bitwise invariant across five size combinations, with the NCCL pins both on and off, and with a repeat control and a 1-ulp sensitivity control passing. Graph-mode re-dispatch is also ruled out, since `--enforce-eager` leaks identically. The MoE block itself is ruled out too: a per-stage content-keyed dump (`a1 -> a1q_local -> a1q_gath -> expert -> combined`) over 1545 repeated input rows on rank 0, 935+ of which genuinely varied their step size (1 to 34 tokens), diverges at no stage. So is the sparse indexer's paged-MQA-logits schedule, this time by intervention rather than observation: with the victim row's q, weights, context length, block-table row and the 32 KV blocks it points at pinned byte for byte, varying only the co-batch lengths (7 shapes, 7 distinct schedule hashes) and the victim's own row index (rows 0-7) leaves `logits[0, :ctx]` bitwise identical in all 15 runs, with a repeat control and a 1-ulp sensitivity control passing. The attention side is not localised: the layer-wise dump that pointed there has an alignment key with no request identity, and it collapses -- 14023 records into 280 keys at layer 0 -- so its numbers were withdrawn rather than read. Characterised, not fixed |
| all2all backend | **the default is the broken one** | same DP2xTP2+EP shape, one variable apart: `allgather_reducescatter` leaks in every round with graphs on (103 rounds) *and* under `--enforce-eager` (33 rounds), while both DeepEP backends under `--enforce-eager` are clean — `deepep_high_throughput` for 31 rounds and `deepep_low_latency` for 35. Two independent all2all implementations hold; the default one does not. DeepEP with graphs on is still unanswered: HT dies in capture (`cudaErrorStreamCaptureUnjoined`, its all2all uses a side stream that never joins back) |

Upstream `tests/v1/determinism` as a regression gate on the merged tree,
since these branches live on a fork and its B200 job has never run them:
508 passed at the kernel level, and on the e2e file the invariance assertion
itself (`bs1_vs_bsN`) passes on all three backends. Three failures, none
attributable to these changes and each checked rather than assumed: one
negative control fails on the base tree too, one passes 4/4 when run alone,
and one is an engine that could not start (13.5 of 184 GiB free, the
previous test's model still resident).

## PR map (all against this fork; path-disjoint except where noted)

1. `bi/indexer-topk` — deterministic top-k + fused Triton kernel + tests
2. `bi/sparse-mla` — decode plan pin + per-request prefill chunks + tests.
   **Based on `bi/indexer-topk`, not on `bi/base`**: the indices this plan
   operates on come from the indexer's top-k, whose dispatch switches
   implementation on `num_rows <= 32`. Pinned alone, this branch would fix the
   schedule and leave the indices batch-dependent, so the dependency is a git
   base rather than a line in the description.
3. `bi/mhc` — tilelang split pins + tests
4. `bi/gate-linear` — fp32 router + small-N persistent matmul config + tests
5. `bi/deepgemm-fp8` — DeepGEMM BI wiring (fail-closed probe, alignment pin,
   `FallbackExperts` forwarding fix) + mocked tests
6. `bi/cudagraph-mode` — close the cudagraph dispatchers that pick a graph per
   step outside `cudagraph_mode`'s reach (microbatching is refused; the
   speculative-decoding proposer drops to `NONE`), warn when an explicit capture
   envelope is too small for the largest decode step, and warn that a piecewise
   mode leaves a per-step numeric choice. Policy, not numerics: on the
   configuration this stack is validated on, none of it changes a bit.

The root cause of the mixed-step leak — the `DeepseekV4Indexer` short-context
predicate evaluated once against the capture-time dummy batch and baked into the
graph — was a correctness bug independent of batch invariance, and **upstream
fixed it in #52492**. The branch that carried our version is deleted. Upstream
#51318 likewise removed the adaptive C128A metadata packing, which retired a
second pin of ours (`active_topk_width` no longer exists).

Serving recipe under BI: `--kv-cache-dtype fp8 --block-size 256
--no-enable-prefix-caching --no-enable-flashinfer-autotune`,
`max_num_batched_tokens <= 8192`, TP4/EP4 on this cluster adds `NCCL_MNNVL_ENABLE=0` (a node-local
workaround for a broken MNNVL/IMEX path, not a batch-invariance requirement);
cudagraphs stay ON and the mode is left alone — the default
`FULL_AND_PIECEWISE` is what the evidence is now taken under, and both capture
kinds are confirmed present in the server log. That is safe exactly while every
path the mode can select is bit-identical for this model, which is measured here
and unknown elsewhere; `cudagraph_mode=FULL` removes the choice for anyone who
has not measured it. Speculative decoding stays off (upstream #27433).
Keep `use_fp4_indexer_cache=False` (the upstream default) — see roadmap.

## Not covered yet (roadmap)

- ~~Full-scale mixed-step leak~~ **root-caused and fixed on 2026-08-16**:
  a 75-round soak on the full 43-layer model showed a composition-dependent
  logprob flip in 29/75 rounds. It was first attributed to **piecewise
  cudagraph replay**; the actual cause is a host-side predicate in
  `DeepseekV4Indexer` frozen at capture time, since fixed upstream in #52492.
  With the batch composition pinned by a hand-driven `LLMEngine` (no HTTP
  races), the same step sequence gave: eager 19/19 identical; `FULL` 17/17
  identical *and bit-equal to eager*; `PIECEWISE` equal to neither;
  `FULL_AND_PIECEWISE` flipping between the two values step by step. With the
  predicate fixed, `PIECEWISE` is bit-identical to both. What stands
  independently of that model bug: the per-step mode is chosen from batch
  properties (uniform decode? within `max_cudagraph_capture_size`?), so a
  request's path is picked by its neighbours and nothing checks that the
  selectable paths agree. Fix: pin
  `cudagraph_mode=FULL` under `VLLM_BATCH_INVARIANT` and keep the mixed-mode
  downgrade from handing piecewise back. Ruled out by single-variable
  experiments before landing there: cudagraph batch padding (dense capture
  sizes `[1..128]` still reproduce), the tile-scheduler plan family (graphs +
  forced skeleton still leaks at the same rate), `use_fp4_indexer_cache`,
  request history, indexer Q quant, and the MQA logits kernel. The
  piecewise/full divergence itself is upstream-reportable.
  Three lessons worth carrying: single-round 0-diff is underpowered against
  a ~0.4/round intermittent leak; a switch that changes *speed* (eager)
  also changes *scheduling*, so composition must be pinned before comparing;
  and a repro scenario has to make the target path actually execute — long
  fillers pushed the step past `max_cudagraph_capture_size`, so no graph ran
  and every configuration agreed for the wrong reason.
- NCCL pin ablation result (2026-08-15): the minimal sufficient set is
  **all five pins** — relaxing any one breaks 0-diff (dropping
  `NCCL_ALGO=allreduce:tree` fails 46/46; multi-channel, NTHREADS,
  symm-mem, custom-AR each leak intermittently, caught only on x2
  checker rounds). The +31% comm cost is not reducible by configuration;
  the upstream path is deterministic multi-channel collectives (new
  kernel work). Defaults unchanged.
- mxfp4/MegaMoE production path: DeepGEMM patch preserved on
  `codex/bugfix-deepgemm-fp4-bi-v261` (references/DeepGEMM); e2e first-pass
  pending — deferred by owner decision 2026-08-15.
- TP-axis invariance (TP1 vs TP4 bitwise): different problem; pik-style
  fixed reduction trees are the candidate approach.
- Trainer-side operator alignment (Phase 2 of the project plan).
- Upstream submission: several pieces are independently upstreamable (CUDA
  top-k tie nondeterminism report, `FallbackExperts` forwarding fix, fused
  deterministic top-k); duplicate-work checks against #27433 / #46639 /
  #36488 recorded in the project knowledge base.

## Known upstream context

Tracking issue #27433 (board 29). Related in-flight: #46639 (Marlin MoE BI,
mxfp4), #36488 (matmul_ogs bitwise-invariance flag), #51902 (declared-but-
violated BI — the cautionary tale this work's negative-control rule exists
for), #30321 (DP+EP out of scope upstream), #51290/#50136/#51187/#51287.
