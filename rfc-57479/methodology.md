# Methodology and reproduction

## Workload

| Parameter | Value |
|---|---|
| Endpoint | `POST /inference/v1/generate` (token in, token out), non-streaming |
| Input | 16,384 tokens |
| Output | 245,760 tokens consumed by the frontend before `POST /pause?mode=abort` (`ignore_eos`, no stop, `max_tokens` 245,761) |
| Logprobs | top-128 output logprobs; no prompt logprobs |
| Formats | default (OpenAI-style per-entry JSON, bytes identical to base) and compact (opt-in, `logprobs_format: "compact"`, top-k ids and float32 logprobs as base64) |
| Concurrency | n = 32 (main cells); bs = 256 (cohort cells) |
| Out of scope | R3 (routed experts), streaming |

## Engine mock and topology

**Engines (`dp-engine.py`).** There is one mock engine process per DP rank.
- **Wire format:** EngineCore ZMQ/msgpack, using vLLM's own `MsgpackEncoder` with its 256 B
  zero-copy aux-frame threshold.
- **Output messages:** one `EngineCoreOutputs` per client per step, sent from one PUSH per client
  output address. Outputs are routed by `request.client_index`.
- **Abort on pause:** empty-token ABORT outputs plus `finished_requests`, sent before the pause
  reply.
- **DP coordinator protocol, engine side:**
  - stats;
  - wave complete;
  - `START_DP_WAVE`;
  - dropping stale waves while paused;
  - lockstep dummy steps;
  - two-phase pause.

**Coordinator.** The real `DPCoordinatorProc` from the vLLM tree.

**Frontends.**
- **Python:** 32 API servers per node. n = 32 cells run on cn02 only; bs = 256 cells on cn02 and
  cn03. No independent listeners in the stack arms.
- **Rust:** one process with 64 request threads (32 in some earlier cells). This is the deployable
  topology: every Rust process reports `client_index` 0, so a second process would receive no
  outputs, and the launcher runs one.

**Client.** client v3 (native Rust; 8 cores on cn04), which does sha256 plus strict semantic
validation of every response. "Wide" cells give the same binary 72 cores and 64 parse threads, as
a measurement aid only.

**Nodes.** GB200 nodes cn01–cn04, Grace CPUs, 200 Gb/s NICs.

## Cells (`final-validation.sh`)

| Cell | Description | Baseline arm |
|---|---|---|
| `n32-def` | n = 32, default format | yes |
| `n32-rl` | n = 32, compact | none (base has no compact format) |
| `cohort-rl-v3` | bs = 256, compact, 8-core client | none |
| `cohort-rl-wide` | bs = 256, compact, wide client | none |
| `cohort-def-wide` | bs = 256, default, wide client | none: base Rust aborted on a 4 GiB allocation failure (DP4, dp-round4), and the base Python storage path fails at the memory floor (dp-round9) |
| `wf-paced` | bs = 256 with lognormal output lengths. The engine is paced at 32 tokens per 20 ms; pause at 10 s, sleep 5 s, resume, then resubmit the aborted partials | none |

**Recorded per run:**
- pause HTTP return;
- last server serialisation or format end;
- all bodies received;
- all parsed;
- bytes per request;
- frontend peak RSS (VmHWM) per process;
- frontend CPU;
- the busiest Python server's request count;
- wave checks;
- body sha256 lists (Rust) or byte-length lists (Python, whose bodies contain `created`).

## Fairness rules

- **Services:** fresh services for every cell.
- **Ordering:** each A/B runs ABBA (A B | B A | A B for 3 repeats).
- **Same environment:** both arms use the same nodes, topology and client binary.
- **No tuning for speed:** concurrency, lengths, top-k, client process count and data are never
  changed to gain speed.
- **Failures are results:** timeouts and memory-floor stops are kept as results, never replaced.
- **Node lock:** `NODES-IN-USE.txt` on the shared allocation is taken per cell (30 s polling), so
  no two benchmark jobs overlap.
- **Snapshots:** every Python tree runs from a node-local `git archive` snapshot, verified with
  `tar -d` (0 differences) and with `.pyc` warmed.
- **Rust binaries:** built `--locked --offline --release`; their sha256 is recorded.
- **Sidecars:** every artifact has a `.meta.json` with the command, commit, environment pins,
  timestamp and agent.

## Byte identity

**Rust.** A genbench micro-bench renders one 245,760 × 129 response. Its payload is either built
directly or encoded as Python-shaped msgpack engine messages with aux frames and decoded by
`engine-core-client`.

| Body | sha256 |
|---|---|
| Default (base and every tip) | `572252e6…`, 3,408,730,749 B |
| Compact | `1a5360f0d3c4f20017f409079951a00bb716f62fb45eaec0037b1ddd7f3fe359`, 337,043,223 B |

**Python.**
- **Golden scenarios:** 27 `/inference/v1/generate` scenarios through the real router and
  `OutputProcessor`. The status, messages, Content-Length and sha all match.
- **Base-path bench:** one 245,760 × 129 response; body sha `e663245f53` at base, P1, P2 and P3.
- **Per-request byte-length lists:** 1 distinct list per cell across arms.

## Mock fidelity (dp-round7)

17 behaviours were compared with the real `EngineCoreProc`, `DPEngineCoreProc`, `Scheduler`,
`Sampler`, `LogprobsLists`, `MsgpackEncoder` and `DPCoordinatorProc`. Two differences are
material:

1. **Granularity and dtype.**
   - **Real:** 1 token per request per step, int32 ids.
   - **Mock default:** 1,024 per step, int64 ids.
   - **Effect:** the A/B at real granularity (`--int32-ids`, 1 token per step) changed no
     post-pause metric. It only lowers pre-pause ingest, which keeps ≈12× (Rust) and ≥5× (Python)
     headroom over real decode rates.
2. **DP pause latency.**
   - **Real:** consensus only at `step_counter % 32 == 0`, with real dummy forwards, so 0.3–1.6 s.
   - **Mock:** 20–70 ms.
   - **Effect:** engine-side and identical for both frontends.

**Why the conclusions hold without GPUs.** Every optimisation removes frontend work per
(position, slot) entry after the engine has accepted the abort. That work is a function of the
response contents, which the mock reproduces exactly (shapes, dtypes, framing). It does not
depend on how the tokens were produced.

## Reproduction

The harness is published next to this file in [`harness/`](harness/README.md). It contains the DP mock engine and
launcher, `final-validation.sh`, the client v3 source and the Python base-path bench. Site-specific values are
environment variables (`GENOPT_ROOT`, `GENOPT_PYTHON`, `GENOPT_IP_NODE01..04`, `SLURM_HOLD_JOB`); model assets, binaries
and results data are not included (see the harness README). The workload contract is [`SPEC.md`](SPEC.md).

```bash
# DP8 final-validation chain (all cells; with a baseline arm where one can run)
harness/genopt-dp/final-validation.sh --impl rust \
  --candidate <vllm-rs binary> --baseline <base binary> \
  --out <dir> --repeats 3 --rust-threads 64
harness/genopt-dp/final-validation.sh --impl python \
  --candidate <node-local tree on node02 and node03> --baseline <base tree> \
  --out <dir> --repeats 3 [--cand-env VLLM_PLUGINS=rl_compact]
harness/genopt-dp/final-summary.py <dir>          # final-summary.{json,md}
```

**Rust checks per tip** (rustc 1.95.0):
```bash
cd rust
cargo test --offline --workspace
cargo clippy --offline -p vllm-server -p vllm-engine-core-client -p vllm-llm -p vllm-cmd -p vllm-managed-engine --all-targets -- -D warnings
cargo fmt --all --check
cargo build --locked --offline --release -p vllm-cmd --bin vllm-rs --features native-tls-vendored
```

**Python checks per tip** (CPU test node; the environmental failures are identical on base and
are not validation):
```bash
PYTHONPATH=$TREE:$TREE/tests/plugins/vllm_add_dummy_endpoint_plugin VLLM_TARGET_DEVICE=cpu \
  python -m pytest -q -rfE --continue-on-collection-errors \
  tests/entrypoints/scale_out/token_in_token_out/ tests/v1/engine/test_output_processor.py \
  tests/test_logprobs.py tests/plugins_tests/ tests/entrypoints/serve/middleware/
PYTHONPATH=$TREE VLLM_TARGET_DEVICE=cpu python -m pytest -q -rfE tests/entrypoints/serve
python -m pytest tests/v1/engine/test_sample_logprobs_container.py \
  tests/entrypoints/scale_out/token_in_token_out/test_array_logprobs.py
```

**Base-path bench (Python):** `harness/python-bench/claude-genopt-py-bench.py`.
- It runs one default-format request of 245,760 × 129, pinned to 16 cores, from fresh
  same-length snapshots.
- Rounds are interleaved as B0 P P B0.
- It reports consumption µs/position, build and render seconds, VmHWM, GC time and the body sha.
