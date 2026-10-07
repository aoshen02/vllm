# Original RFC text (before the 2026-10 revision), verbatim

This is the body of vllm-project/vllm#57479 as it stood before the 2026-10 rewrite, copied byte for byte below the line.

---

# Motivation.

Bulk abort is expected to be the fast path for pausing online RL generation,
but its latency can scale with all output retained by the frontend. In a
DP8/EP8/TP1 deployment with eight API server processes, aborting 2,000
non-streaming completion requests at a reported approximately 3,500 generated
tokens per request produced 96-177 seconds of frontend tail latency. Actual
returned means were 4,850-5,445 tokens because generation continued while the
frontend lagged. The engines had already accepted the abort; API processes
were still draining queued outputs,
detokenizing sampled logprobs, constructing response models, materializing
Python dictionaries, and encoding JSON.

The largest amplifier was API-process load imbalance. The parent enabled
`SO_REUSEPORT`, but all workers inherited copies of one socket and therefore
shared one accept queue. In one run, the busiest API process owned 1,492 of
2,000 requests. Balanced DP-engine load did not imply balanced frontend load.

This matters beyond RL pause. The same work appears when long non-streaming
completion requests finish or are aborted together, especially when sampled
logprobs or token IDs are returned. A fast model can produce outputs faster
than one Python frontend process can consume and serialize them.

This RFC follows the completed pause/resume API RFC #32103, whose follow-up
list explicitly called out better abort handling. It does not change pause
mode semantics or DPEP coordination. At RFC creation, searches of issues and open PRs using
pause/abort, frontend serialization, logprobs, API server, and reuseport terms
found no overlapping proposal or implementation. Related work is
complementary:

- #56754 drains in-process requests before sleep.
- #52875 reduces per-frame cost for simple streaming completions.
- #47933 preserves `finish_reason="abort"` for scale-out token streams.

## Reproduction and evidence

The primary setup used two GB200 nodes, DP8/EP8/TP1, eight API servers, 2,000
requests, and sampled `logprobs=0`. A four-layer checkpoint made the frontend
bottleneck easy to reproduce; the final candidate was also validated with the
full 78-layer MXFP8 checkpoint.

On the reduced model, the unmodified frontend took 96.18 and 177.12 seconds
from pause to its last response serialization in two runs. The measured
seven-file candidate took 3.38 seconds at 7.28 million returned tokens, and
5.18 seconds at 10.08 million returned tokens versus 96.18 seconds at 10.89
million unmodified. The six-file patch proposed here differs only by removal
of a null-result lazy-context change and was itself measured as the valid
2.924-second `no_lazy_context` leave-one-out arm.

On the full model, one unmodified baseline was 6.97 seconds frontend and 7.44
seconds through all client results, with a balanced 187-270 requests per
frontend. Four additional unmodified runs using a two-process client that
concentrated connections on fewer workers reached 8.92-21.10-second frontend
tails, with 462-907 requests on the busiest process. Two used the 3,500-token
trigger and two used 3,800. They have no patched counterpart and are baseline
evidence, not an A/B comparison.

Two seven-file-candidate repeats completed frontend serialization in 3.63 and
3.90 seconds and all client parsing in 3.81 and 4.19 seconds; waiting for both
the pause HTTP response and all client results took 3.81 and 7.67 seconds. This
validates parity and absence of regression at full scale. It does not reproduce
the reduced model's 100+ second tail in a matched A/B or isolate the listener
change. GPU response parity was also captured on this measured seven-file
candidate; equivalence to the shipped six-file state rests on hash identity
with the measured `no_lazy_context` arm rather than a separate post-trim GPU
capture.

We then ran an eight-arm, same-allocation leave-one-out experiment. Every arm
aborted and rendered exactly 2,000 responses. The complete candidate was
bracketed at 2.953 and 3.054 seconds, giving a 0.102-second drift bound.

| Removed optimization | Frontend-tail regression |
| --- | ---: |
| Independent `SO_REUSEPORT` listener per API worker | +6.199 s |
| Bounded single-token decode cache | +1.075 s |
| Flat sampled-logprob storage | +1.039 s |
| Direct Pydantic JSON serialization | +1.176 s |
| Explicit event-loop yields | +0.125 s |
| Lazy byte-fallback context lookup | -0.080 s |

The first four effects are 10-61 times the measured run drift. The lazy
context change was removed. Event-loop yields are retained for a deterministic
fairness invariant: without them, a ready output queue can consume 100 batches
before a waiting control task runs. Leave-one-out effects interact and must
not be summed to reconstruct the original 100+ second result.

The listener result is one stochastic draw for a client opening approximately
2,000 connections. Its +6.199-second effect is not a general point estimate;
keep-alive-heavy clients can see a different distribution or no benefit.

## Implementation update — 2026-09-23

Phase 1 is proposed in **#58278: [Frontend] Give API workers independent reuseport listeners** (draft, not merged). The narrowed implementation changes only listener selection and per-worker socket ownership. General CLI startup/shutdown behavior stays identical to the base; lifecycle refactoring is deferred. The following GPU measurements belong to historical head `377818881b`, not the current revision.

A prior **listener-only** GPU/HTTP ABBA experiment completed on node18 using one node with four GB200s, four-layer HY4-preview BF16, DP4/EP4/TP1, four default API processes, 250 requests per replica (1,000 total), non-streaming completions and sampled `logprobs=0`.

| Metric | Shared listener (A, two runs) | Independent listeners (B, two runs) |
| --- | ---: | ---: |
| Last frontend serialization | 8.298 / 10.698 s | 4.551 / 4.746 s |
| Median frontend serialization tail | 9.498 s | 4.649 s (51.1% lower) |
| Last client response body received | 9.463 / 11.493 s | 6.687 / 6.693 s |
| Median client receive tail | 10.478 s | 6.690 s (36.1% lower) |
| Most requests on one frontend | 503 / 507 | 261 / 268 |
| Mean returned tokens per request | 3731.648 / 3681.236 | 3633.930 / 3662.330 |

All 4,000 requests returned HTTP 200 with `finish_reason="abort"`; every run had 250 requests running on each engine, zero waiting, and 1,000 conversion/render events. Four rank-pinned greedy smoke responses, including token IDs and logprobs, matched across all arms. The PR reports individual pause HTTP timings separately and makes no pause-HTTP speedup claim.

Timing boundaries matter: server serialization excludes ASGI send/network/client parsing. Client receive time is the last body read, before parsing that final body. The harness also records time until **both** `/pause` returns and all responses are parsed; that combined metric is capped by pause return in A1 and must not be relabeled as client receive latency.

These are two repeats per arm with a 1.6% median token-count difference, not equal-token measurements or a confidence-interval estimate. GPU testing ports only the reviewed listener block onto pinned serving base `0fd2e8d503`; the PR is based on `94f4170df3` and also has an exact-base CPU/socket A/B. This single-node result must not be compared directly with the original DP8 100+ second baseline or treated as validation of the entire roadmap. Both valid GPU arms used identical lazy safetensors loading and `NCCL_MNNVL_ENABLE=0`; failed node-preflight starts were excluded.

Claude Opus 5.5 and Kimi K3 adversarial code and final evidence audits passed, including correction of the client timing label. See #58278 for test commands, complete per-run results, and compatibility limits. Phase 0's broader benchmark matrix and worker-failure integration coverage remain follow-up requirements, not completed by this experiment.

## Phase 2 implementation update — 2026-09-23

**Draft PR #58314** implements native JSON rendering for ordinary non-streaming completions, with legacy compatibility fallback. It changes one runtime file (+45/-2), reuses Pydantic and Starlette, and preserves load-counter cleanup. The earlier proposal to convert non-finite values into `null` is withdrawn from this performance change. Prompt logprobs, transfer metadata, metrics and arbitrary extras retain the legacy path initially; JSON numeric byte formatting may differ while parsed values match.

After the listener fix, an add-one comparison selected this direction over decode caching and merely enabling flat logprobs for the pause-drain critical path. The cache remains valuable for generation CPU and memory (related #54050, whose eviction policy differs from the experiment), and flat storage retains its memory benefit. Those are separate follow-ups; their gains must not be added to these results.

The final guarded implementation, rather than the earlier unguarded prototype, now has its own single-node four-layer HY4 BF16 / DP4/EP4/TP1 online ABBA BAAB experiment (four repeats/arm, 1,000 requests at about 3,500 tokens, sampled `logprobs=0`). Both arms already include the identical listener fix.

Server serialization-tail median **4.974 → 3.740 s (24.8% lower)**; client last-body-read median **6.312 → 5.692 s (9.8% lower)**.

| Run | Mean returned tokens | Requests per frontend (sorted) | Last serialization | Last client body-read | Pause HTTP |
|---|---:|---|---:|---:|---:|
| 1 A | 3518.950 | 227 / 248 / 261 / 264 | 4.953 s | 6.432 s | 6.435 s |
| 2 B | 3531.042 | 227 / 245 / 252 / 276 | 4.050 s | 5.747 s | 4.935 s |
| 3 B | 3540.671 | 225 / 241 / 258 / 276 | 3.721 s | 5.724 s | 5.727 s |
| 4 A | 3517.024 | 231 / 255 / 256 / 258 | 4.548 s | 6.191 s | 6.194 s |
| 5 B | 3522.804 | 223 / 244 / 260 / 273 | 3.657 s | 5.578 s | 5.581 s |
| 6 A | 3536.653 | 233 / 235 / 255 / 277 | 4.995 s | 5.550 s | 5.335 s |
| 7 A | 3511.551 | 203 / 234 / 280 / 283 | 5.161 s | 6.645 s | 6.648 s |
| 8 B | 3517.053 | 231 / 238 / 259 / 272 | 3.759 s | 5.659 s | 4.198 s |

Both balanced blocks are token-matched within 5%; mean B/A token ratios 1.0051 / 0.9988. A's observed tail range is 0.613 s; median tail saving is 1.234 s. Predeclared strong-live gate: PASS. All 8,000 responses were HTTP 200 / abort, all engine counts were 250 active / zero waiting, and all 13 greedy API parity cases matched across starts. Each patched run had 1,000 successful native encodes and zero legacy renders. The PR records commands, compatibility tests (59 passed), source-pin boundaries and adversarial reviews. Human line review/test rerun remain pending while the PR is draft.

This is a serialization-tail result, not a `/pause` return improvement. Client last-body-read remains an application observation influenced by client event-loop scheduling. No statistical-significance, full-model, DP8, chat-completions, streaming-throughput or original-100s-baseline claim is made. The patch is ported onto serving base `0fd2e8d503` with only two import relocations from PR base `955bd6abef`. An initial one-A-run attempt was preserved but excluded after a type-annotation fix; all eight final runs were restarted. Shared-host supplementary high-top-k/fallback checks supply parity evidence, not a throughput guarantee. Phase 0's broader workload matrix remains open.

# Proposed Change.

Land a sequence of small changes under one shared benchmark and compatibility
contract. No new public abstraction is needed for the first stages.

## Compatibility contract

- Preserve response fields, token IDs, text, finite logprob values, finish
  reasons, and the existing meaning of pause/abort.
- Keep prompt-logprob storage and schema unchanged.
- Keep single-API-server and Unix-domain socket behavior unchanged.
- Preserve per-request Unicode repair; cache only context-free token decoding.
- Preserve the existing non-streaming behavior for non-finite floats and arbitrary extension values. The native JSON fast path falls back to the existing `JSONResponse(model_dump())` renderer for these cases. The earlier proposal to change NaN/Inf from render errors into HTTP 200 with `null` is withdrawn from this performance work; changing that API behavior requires a separate decision.
- Continue returning the accumulated partial response on abort. Any compact
  abort response mode would require a separate API proposal.

## Roadmap

### Phase 0: benchmark and observability

Add a reproducible frontend-drain benchmark that records:

- time from abort dispatch to engine acknowledgment, last frontend render,
  last response send, and last client parse;
- actual generated and returned token counts;
- accepted request count and render count per API process;
- output-queue backlog and conversion/render duration histograms.

The benchmark should test at least 1 and 8 API servers, streaming and
non-streaming completions, no logprobs and sampled logprobs, and uniform and
keep-alive-heavy connection patterns. Performance gates should use tail
latency and exact response-count invariants rather than only aggregate
throughput.

### Phase 1: correct multi-frontend socket distribution

**Status: #58278 is draft pending human review and test rerun. Scope narrowed to independent listeners; historical GPU ABBA belongs to `377818881b`.**

HTTP launch supplies an optional socket factory for multi-worker TCP, reusing existing socket creation without a new OS allowlist. The manager scopes fresh parent handles and borrows the default socket. No new partial-start rollback or finalizer reordering: these follow baseline behavior. Production diff:24additions/11deletions, serve.py8addedlines.

18 related tests pass (one unrelated Rust device-inference test deselected); lint/mypy3.10 pass. Prior Linux CPU/socket and GPU evidence above remains historical; not rerun after removing added rollback.

For Linux TCP with multiple API servers and `SO_REUSEPORT`, bind one listener
per worker instead of passing copies of one parent descriptor. Keep the parent
socket for address reservation and close temporary parent descriptors after
spawn. Add a real spawn regression that compares listener inodes and verifies
single-worker and Unix-domain behavior. The initial PR should gate this path to
Linux; non-Linux behavior remains unchanged until its accept semantics are
separately validated.

This addresses the largest measured amplifier. It balances connections, not
requests multiplexed over an existing keep-alive connection, so the benchmark
must report both connection and request distributions.

The parent reservation socket must remain bound but must never enter
`listen()`: otherwise reuseport hashing can send connections to an accept queue
that no worker serves. The spawn regression should assert this invariant.
Independent listeners also change worker-failure behavior: connections already
queued on a crashed worker's accept backlog are reset instead of being
accepted by another worker. This trade-off should be documented and covered by
a worker-failure integration test.

### Phase 2: remove duplicate terminal-response work

Code-size audit: direct serialization alone needs +7/-2 lines (five nonblank added lines); preserving existing special-value/error behavior accounts for the additional 38 added lines, including imports and spacing. The +7/-2 variant is illustrative, not the submitted implementation. A follow-up removes one repeated attribute lookup; the GPU A/B above measures d04c21b46b before that cleanup and was not rerun for it.

**Status: draft PR #58314; standalone guarded-renderer A/B above. Not merged.**

For non-streaming completion responses, serialize the validated Pydantic model
directly with `model_dump_json()` instead of building a nested Python
`model_dump()` tree and asking Starlette to encode it again. Keep error
responses on their existing path. Add route-level parsed-value and response-header tests,
including unchanged `NaN` and `-inf` error behavior, arbitrary extension fields, schema changes, and load-counter cleanup. Reuse the existing `JSONResponse` type so response headers and background load accounting keep their lifecycle; do not add a general serialization framework. Prompt logprobs, per-request metrics, and transfer metadata retain the legacy path initially.

### Phase 3: reduce per-token Python work and retained objects

1. Add a tokenizer-local, bounded cache for context-free `decode([token_id])`
   results. Keep request-specific Unicode correction outside the cache and
   cover tokenizers with leading-space markers.
2. Store sampled completion logprobs in the existing flat representation
   during accumulation, then materialize the OpenAI response schema once.
   Prompt logprobs remain list-based because they are exposed directly.

These should be separate PRs because the cache changes tokenizer state while
flat storage changes output-processor representation. Each PR needs parsed
response parity for text, token IDs, top-k logprobs, echo, prompt logprobs, and
streaming exclusions.

### Phase 4: guarantee control-plane fairness

Yield after enqueuing ready IPC outputs and between processed output batches,
even when receive calls complete immediately. Validate with deterministic
tests that a control task runs before a preloaded queue is exhausted. Treat
this as a starvation/cancellation fix; the measured end-to-end effect was
0.125 seconds at an experiment resolution limit of 0.102 seconds and is not
claimed as a performance benefit.

### Phase 5: larger follow-up directions

These need separate design review after Phases 0-4 establish the baseline:

1. **Abort-aware output compaction.** Stop retaining intermediate Python
   objects that can be reconstructed from token IDs, while preserving the
   current full partial-response contract.
2. **Incremental materialization with bounded backpressure.** Bound the amount
   of engine output one frontend can accumulate and expose backlog metrics.
   This must avoid throttling unrelated requests behind a slow client.
3. **Connection-aware dispatch.** Investigate frontend-aware routing or client
   connection guidance for workloads dominated by a few persistent
   connections; independent listeners alone cannot rebalance them.
4. **Native/Rust frontend parity.** Carry the benchmark and abort contract to
   the Rust frontend and compare object-free token/logprob serialization before
   moving more Python paths.
5. **Pause completion semantics.** Decide whether operators need separate
   acknowledgments for engine pause and frontend/client drain. Changing when
   `/pause` returns is an API decision and is intentionally outside the
   initial fixes.

## Alternatives considered

1. **Only add event-loop yields.** This fixes starvation but leaves the
   dominant socket imbalance and O(tokens) Python work intact.
2. **Move all handling to the scheduler/core.** This could reduce frontend
   backlog, but it couples OpenAI response semantics to the engine and is too
   invasive for the measured bottlenecks.
3. **Wait for a native frontend replacement.** That may be the long-term
   boundary, but the Python frontend remains deployed and the first four fixes
   are small, independently measurable, and useful to both implementations as
   contract tests.

The proposed sequence chooses local fixes first, with one benchmark and
compatibility matrix tying them together. Each optimization can be reviewed,
measured, and reverted independently.

## Success criteria

- Exact response-count and parsed-response parity across the compatibility
  matrix.
- No event-loop starvation under a preloaded output queue.
- Near-even connection ownership with many independent client connections.
- No last-frontend-serialization tail above 10 seconds for the documented
  2,000-request, approximately 3,500-token workload on the reference
  deployment. This gate excludes ASGI send, network transfer, client parsing,
  and the independent pause HTTP response. Those boundaries must still be
  reported separately; the reduced-model baseline-sized patched run reached
  all client results in 13.65 seconds.
- No material throughput regression for ordinary streaming or requests that
  do not request logprobs.

## Requested feedback

1. Should independent per-worker listeners be the default whenever multiple
   TCP API servers use `SO_REUSEPORT`?
2. Separately from these performance changes, should non-streaming non-finite-logprob behavior ever be aligned with streaming? Phase 2 preserves the existing render-error behavior and does not decide that API question.
3. Should sampled completion logprobs adopt flat storage globally, or should
   it be limited to the OpenAI completion path first?
4. Which boundary should own backlog metrics: `AsyncLLM`, the output
   processor, or the serving endpoint?
5. Should pause expose separate engine-acknowledged and response-drained states
   in a later RFC?

# Feedback Period.

Two weeks.

# CC List.

No explicit CCs for the initial post; relevant pause/resume and frontend
owners are welcome to opt in.

# Any Other Things.

The earlier combined implementation candidate remains local historical evidence. The isolated implementation PRs are #58278 (listeners) and draft #58314 (native completion JSON with compatibility fallback). The later roadmap changes are not included in either PR; #58314 remains draft pending human per-line review and test rerun. This RFC and implementation were prepared with AI assistance. The human submitter requested publication and remains accountable for the contribution. The roadmap continues to define the scope and validation required for subsequent PRs.

### Before submitting a new issue...

- [x] I searched existing issues and open PRs. The closest completed RFC is
  #32103; the open items linked above cover different serving paths or
  lifecycle semantics.




2026-10-02: removed the newly introduced Linux-only runtime gate at the user’s request; neighboring socket setup already uses SO_REUSEPORT without an OS allowlist. Independent listeners now apply to multi-worker TCP wherever the existing helper supports binding them. Load-distribution performance is validated only on Linux; no claim for other operating systems.
2026-10-02: removed added try/except and restored finalizer registration after the worker loop, as explicitly requested. Removed rollback-only fault tests; baseline partial-start failure behavior remains outside scope.

2026-10-02, phase2/#58314: combined sampled and top-logprob finite checks into one lazy iterator/check; no compatibility guard or fallback removed. Runtime diff+44/-2 (one fewer line),59completiontests pass, lint and exactClaudeOpus5.5/KimiK3 reviewpass. EarlierGPUAB remains historical, not rerun for cleanup. Directbytes output deferred.
