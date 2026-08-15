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
  long/short compositions crossing the indexer shortcut boundary (2048),
  prefix caching OFF (with it on, the victim's prefill is computed once and
  prefill invariance is never tested — this exact blind spot hid the sparse
  MLA prefill bug).
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

## PR map (all against this fork, path-disjoint, independently mergeable)

1. `bi/indexer-topk` — deterministic top-k + fused Triton kernel + tests
2. `bi/sparse-mla` — decode plan pin + per-request prefill chunks + tests
3. `bi/mhc` — tilelang split pins + tests
4. `bi/gate-linear` — fp32 router + small-N persistent matmul config + tests
5. `bi/deepgemm-fp8` — DeepGEMM BI wiring (fail-closed probe, alignment pin,
   `FallbackExperts` forwarding fix) + mocked tests

Serving recipe under BI: `--kv-cache-dtype fp8 --block-size 256
--no-enable-prefix-caching --no-enable-flashinfer-autotune`,
`max_num_batched_tokens <= 8192`, TP4/EP4 adds `NCCL_MNNVL_ENABLE=0`;
cudagraphs stay ON (PIECEWISE+FULL capture verified under BI, including
the fused top-k); speculative decoding stays off (upstream #27433).
Keep `use_fp4_indexer_cache=False` (the upstream default) — see roadmap.

## Not covered yet (roadmap)

- **`use_fp4_indexer_cache=True` leak (found 2026-08-15)**: the official
  V4-Flash Blackwell recipe enables the fp4 indexer KV cache; under it the
  full 43-layer model shows one deterministic composition-dependent flip
  (long victim, batch 17, second decode token; bit-identical across server
  instances). Controls: fp8 default path clean on the same sequence, 6L
  dummy clean, BI=0 negative control fails loudly, no history dependence
  (solo requests bit-stable under any preceding traffic). Mechanism
  screening so far: indexer Q quant is per-(token, head); the DeepGEMM MQA
  logits kernel is elementwise over KV. Next: offline full-scale repro +
  per-op bisect. Until fixed, BI deployments keep the cache off.
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
