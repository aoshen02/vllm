# Generate-endpoint abort optimization — contract (Claude, 2026-10-04)

Owner: Claude Code (Opus 5.5) orchestrator. Supersedes Codex "quantify then stop"
goal. User instructions (2026-10-04): optimize as far as possible, rounds ordered
by cost-effectiveness, sub-agents optimize Python and Rust, reuse existing
optimizations, never stop unless the user says so, no time limit.
`/inference/v1/generate` may be made as lean as possible **but opt-in**.
CPU-only protocol mock (no weights / GPU / DP coordinator). Nodes node01-node04.
Audit with several agent types including Claude.

## Workload (unchanged from frozen campaign)
256 concurrent requests total; input 16384; 245760 output tokens consumed by the
frontend before one `POST /pause?mode=abort`; top-128 output logprobs; no prompt
logprobs; stream=false; ignore_eos; no stop; max_tokens 245761. Mock pool 1024.
Endpoint is now **`POST /inference/v1/generate`** (token in / token out).

## Repos
Base SHA `0fd2e8d5038f958e1fabe3cd448030d7066d68b7` (same as Codex campaign).
- Python: `worktrees/claude-generate-python`, branch `claude/generate-compact-python`
- Rust:   `worktrees/claude-generate-rust`,   branch `claude/generate-compact-rust`
Git is always `git -C <worktree>`. Project dir is not a repo.
Existing optimizations to reuse: `frozen-r1/python-listeners.patch` (per-process
SO_REUSEPORT listeners), the idea of `python-json.patch` (direct JSON render).
Rejected (do not reuse): sampled-logprob reuse, Rust decoder move.

## Option: `logprobs_format` (request top-level field, both frontends)
- absent / `"openai"`: today's response, byte-for-byte semantics unchanged.
- `"compact"`: `choices[0].logprobs` is `null`; `choices[0].compact_logprobs` is
  ```json
  {"num_positions": N, "num_slots": S, "dtype_token_ids": "int32",
   "dtype_logprobs": "float32", "byteorder": "little",
   "token_ids": "<base64 of int32[N*S], row-major>",
   "logprobs":  "<base64 of float32[N*S], row-major>",
   "ranks":     "<base64 of int32[N]>"}
  ```
  Layout is exactly the engine `LogprobsLists` row: slot 0 = sampled token,
  slots 1..S-1 = top-k in engine order (S = k+1; sampled token may also appear
  in the top-k). `ranks[i]` = engine rank of the sampled token. Values are the
  raw engine float32 (no -9999 clamp; non-finite allowed and preserved bitwise).
  `token_ids` of the choice remains the sampled ids list as today.
  Must be identical bytes-for-bytes in meaning between Python and Rust (base64
  strings must decode to the same arrays for the same engine input).
- Unknown value -> 400. Streaming + compact: per-chunk compact block (same
  schema for that chunk's positions) or reject with 400 — implementer decides,
  document it, test it.
- Opt-in only. Default path must stay compatible; default-path optimizations are
  allowed only if output semantics are unchanged (verify).

## Measurement rules (inherit Codex lessons)
Fresh services per cell, ABBA for each optimization, same nodes/topology/client.
Never change concurrency, lengths, top-k, client process count or data to make
speed. Record: pause HTTP return, last server serialization, all bodies received,
client parse done, bytes, peak memory, CPU. Overlapping times are never added.
Small scale for debugging only; acceptance only at full 256.
Failures (timeouts, memory floor) are kept as results, never replaced.
Every results/ artifact needs a `.meta.json` sidecar (cmd/git_commit/env_pins/ts/agent).

## Amendments (orchestrator decisions, 2026-10-04 round 2)
- When logprobs are not requested, the `compact_logprobs` key is OMITTED (not `null`), stream and non-stream.
- Data the compact layout cannot represent (rows narrower than k+1 / changing width, token id or
  rank outside int32) → that request fails with HTTP 500 (generation error); other requests unaffected.
- `num_slots` is k+1 for known k>=0 regardless of engine data (wider rows truncated); k=-1 → engine width
  (0 when no positions). Zero-token abort → 200 with an empty block of that width.
- (round 2b) The compact block is emitted iff the request asks the engine for logprobs: `sampling_params.logprobs`
  set OR `logprob_token_ids` non-empty; S = k+1 with k = logprobs if set else len(logprob_token_ids).
  k=-1: every row must have the first row width, else 500. Streaming: the terminal zero-token chunk omits the
  key; non-stream zero-token abort returns an empty block. Default (openai) path semantics remain exactly base.
- (round 2c) `logprobs_format: "compact"` with `logprobs=-1` → 400 (full-vocab engine payload not representable);
  this supersedes the k=-1 rules above. Accumulated positions must equal generated tokens (except zero-token
  abort) else 500. Compact stream chunks carry explicit `"logprobs": null`.

## Amendment v3 (2026-10-05, user input: RL needs token_ids, R3 routed_experts, top-k ids + logprobs; top-128 f32;
## not the sampled slot nor ranks; form = JSON + field switches, opt-in)
- Optional request fields valid only with logprobs_format="compact" (400 if set to a non-default value otherwise):
  `compact_include_sampled` (bool, default true), `compact_include_ranks` (bool, default true).
  include_sampled=false: arrays hold only the k top-k slots (engine slots 1..k), num_slots=k, block carries
  `"sampled_slot": false` (omitted when true, keeping existing bytes). include_ranks=false: `ranks` omitted.
- routed_experts (R3): existing wire format kept exactly (choices[0].routed_experts = base64 of numpy .npy of
  shape (P+N-1, num_layers, topk), uint8 if experts<=256 else uint16; null if no forward happened). Must be
  produced by both frontends, byte-identical, for default and compact formats, cheaply at abort scale.
- Workload addition: R3 cells with the production MoE layer count (pending from user) and 4 layers (assets).

## 2026-10-05 user: R3 out of scope for now. R3 work limited to safety/no-regression; L=61 placeholder runs void; final matrix excludes R3.

## Audit scope rule (2026-10-05, orchestrator)
Robustness bar: new code must not be worse than base, and must contain errors originating from data or logic
of the new paths (request-local failure, batch continues). Failures that base would also propagate in the same
situation (e.g. a generic MemoryError injected into allocations that base also performs) are out of scope.

## Amendment 2026-10-06 (round 16, Rust stack slimming; orchestrator decision, recorded by the Rust PR-stack engineer)
- Rust: `stream: true` with `logprobs_format: "compact"` -> 400 (user: no streaming for now; the SPEC already
  allowed reject-with-400). The default streaming path is exactly base.
- Rust: the compact block is sized and emitted only from `sampling_params.logprobs` (k >= 0; S = k+1). The round 2b
  rule that `logprob_token_ids` alone also emits a block (S = len(logprob_token_ids)+1) is dropped for Rust: such a
  request gets `"logprobs": null` and no `compact_logprobs` key, exactly as the default format omits logprobs then.
  The RL contract (token_ids + top-k ids + float32 logprobs with `logprobs: k`) is unchanged; cross-language byte
  parity is required for requests with `logprobs` set.
- Rust: an unknown `logprobs_format` value is a 400 from request parsing (no `param` field); an explicit `null`
  means the default format.

## Amendment 2026-10-07 (round 17, final Rust compact schema; supersedes amendment v3 and the round-2 `ranks`/S=k+1 wording for Rust)
- The Rust compact block is always exactly these 7 keys in this order, no whitespace:
  `{"num_positions":N,"num_slots":k,"dtype_token_ids":"int32","dtype_logprobs":"float32","byteorder":"little","token_ids":"<b64>","logprobs":"<b64>"}`
  with engine slots 1..=k (top-k, engine order; the sampled slot 0 is not repeated, it is `choices[0].token_ids`), raw f32 bits, no ranks.
- `compact_include_sampled` / `compact_include_ranks` are no longer implemented; sent values are ignored as unknown fields.
- Unchanged: emitted token ids outside int32 -> 500 for that request (round-2 rule; restored in round 17).
- Cross-language parity target (requests with `logprobs` set): RL-lean genbench body sha256
  `1a5360f0d3c4f20017f409079951a00bb716f62fb45eaec0037b1ddd7f3fe359`, 337,043,223 bytes.
