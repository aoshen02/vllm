# Handoff: RL pause(abort) frontend optimization (RFC #57479)

Owner until 2026-10-07: aoshen02. This file is the entry point for whoever takes over the work.

## Goal
Make vLLM's frontend cheap after an RL `pause(mode="abort")`.

The workload is:
- 245,760 output tokens per request;
- top-128 logprobs;
- a 16,384-token input;
- non-streaming;
- production topology DP8 EP8.

RL needs only `token_ids`, the top-k ids and float32 logprobs. R3 and streaming are out of scope for now.

The theory, results and methodology are in the RFC (vllm-project/vllm#57479) and in [`rfc-57479/`](./) on this branch.

## Where things are

| What | Where |
|---|---|
| RFC (rewritten 2026-10-07) | https://github.com/vllm-project/vllm/issues/57479 |
| Rust PRs (final, three-way audited) | aoshen02/vllm#107 (A: streamed default render) → #108 (B: opt-in compact) |
| Python PRs (round-24 tips, validated at DP8) | aoshen02/vllm#109 (container hook) → #110 (start_generate) → #111 (default fast render) → #112 (RL compact plugin) |
| Docs, theory, harness | branch `rfc57479/docs`, directory `rfc-57479/` (harness in [`harness/`](harness/)) |
| All branches | aoshen02/vllm, branches `rfc57479/*` (list below) |
| Upstream PRs by aoshen02 | vllm-project/vllm#58278 (listeners), #58314 (native JSON render) |

### Branches in aoshen02/vllm
- `rfc57479/base`: base commit 0fd2e8d503 that every PR stacks on.
- `rfc57479/rs-prA-streamed-render`, `rfc57479/rs-prB-compact`: the PR #107 and #108 heads.
- `rfc57479/rs-backup-wire-decode`, `rfc57479/rs-backup-parallel-decode`: not proposed. Their gain appears only on the mock engine; real engine rates leave about 12x ingest headroom.
- `rfc57479/py-pr1`, `py-pr2`, `py-pr3`, `py-plugin`: the PR #109–#112 heads.
- `rfc57479/r25-unvalidated/*`: Python slimming round 25 (drop replace_route, stdlib thread pool, compact streaming returns 400). It was stopped before validation, so do not use it without re-running the acceptance checks.
- `rfc57479/intree-python`, `rfc57479/intree-rust`: the original monolithic optimization branches, kept for reference.
- `rfc57479/upstream-58278-rewrite`: the proposed replacement for #58278, a standalone commit on upstream main. Not yet applied to #58278; see "Open decisions".

## Results in one paragraph
DP8 mock engines and the real DPCoordinator were used throughout.

- **Rust:** PR A brings n32-def from 32.9 s / 202 GiB to 20.7 s / 14.5 GiB. PR B (compact) is about 1.8 s / 11.9 GiB for n32-rl.
- **Python stack:** n32-def 21.3 s and cohort-def-wide 54.7 s (base fails at the memory floor). RL compact n32-rl is about 1.8 s.
- **Byte identity:** default-format bytes are identical to base everywhere, and the compact bytes equal the Rust reference.
- **Request distribution:** the shared socket beats SO_REUSEPORT for one-connection-per-request clients. Round-robin is best. The #58278 rewrite (one accept per wakeup plus a 1 ms pause) removes the burst greed: busiest worker 341–1005 → 253–258 of 2000, and DP8 59.5 → 55.9 s.

The full tables are in the RFC.

## Open decisions / next steps
1. **#58278 rewrite:** replace the upstream PR with `rfc57479/upstream-58278-rewrite`.
   - It is +125/−1 production lines and +234 test lines, audited by Codex, Kimi and Claude.
   - The draft PR body is in the session workspace (`reports/generate-opt-20261004/rfc/pr-bodies/upstream-58278-rewrite.md`).
   - Option: slim it first.
2. **Upstream order:** when moving the fork PRs upstream, go Rust A first (pure perf, no API change), then Python PR3, then the compact/plugin API PRs after RFC discussion. Per AGENTS.md, a human must review every line and run the tests before submitting.
3. **Python slimming:** decide whether to resume round 25 (see minimality-py review). It needs measurements M1–M6.
4. **Upstream conflicts:**
   - #58839 overlaps Python PR2's `replace_route`.
   - #60251 adds `stop_reason`, which Rust PR A's writer must then emit.
5. **Real-GPU validation** of the final stacks has not been done; the mock understates DP pause cost (real is about 0.3–1.6 s).

## Workspace and data
The original workspace is `nvidia-gb200-login:/home/inf-aoshen/vllm/projects/vllm-rl-frontend`:
- `agent_run/reports/generate-opt-20261004/`: PROGRESS.md is the master log; SPEC.md; round reports; audits.
- `agent_run/results/generate-opt-20261004/`: raw results, about 17 TB shared, not copied.

A copy is on `gcp-gb200-head:~aoshen/vllm/projects/vllm-rl-frontend`. It holds the reports, scripts, small result files (1.4 GB), worktrees and `README-copy.md`.

## Claude Code session
The whole work was driven from one Claude Code session, which you can resume.
- Package: `claude-session-handoff/` next to the workspace copy. It contains the transcript, the subagent transcripts, memory and `install-session.sh`.
- Install and resume:
  ```bash
  bash claude-session-handoff/install-session.sh ~/vllm/projects/vllm-rl-frontend
  cd ~/vllm/projects/vllm-rl-frontend && claude --resume dbfff990-d2ca-454d-90f0-ad1335a4bfb6
  ```
- The first message after resuming should say: "Read HANDOFF.md. You now work for <you>; use my accounts, hosts and Slurm allocation."
- The transcript refers to aoshen's accounts (`inf-aoshen`, nodes nova-hazel-cn01..04, Slurm hold job 18145), which will not work for you.

## Gotchas
- **Shared node lock:** measurements take `NODES-IN-USE.txt` with 30 s polling. Never use `pkill -f`; kill by exact PID only.
- **Long remote jobs:** launch with `setsid nohup ... & disown`. Don't use single-quoted ssh heredocs, because apostrophes break them.
- **Sidecars:** every artifact has a `.meta.json` sidecar. Keep that convention.
- **Pre-merge validation:** judge by A/B in the ABBA order, and check byte identity by body sha. Default sha is `572252e6`; compact is `1a5360f0`.
- **Codex login:** Codex on the NVIDIA cluster had a revoked login (HTTP 401), so audits ran Codex locally.
