# Measurement harness for RFC #57479

This is the harness that produced the numbers in RFC vllm-project/vllm#57479 and in `../results.md`:
- a CPU mock of the EngineCore output protocol;
- a DP launcher around the real `DPCoordinatorProc`;
- the final-validation chain;
- the native measurement client (client v3);
- the Python base-path bench.

It is published as measurement tooling, not product code. The scripts are as they were run, with one exception: site-specific values (absolute paths, host names, IP addresses, Slurm job ids) are replaced by the environment variables below.

## Contents

| Path | What it is |
|---|---|
| `genopt-dp/dp-engine.py` | Mock DP engine, one process per rank. It speaks EngineCore ZMQ/msgpack through vLLM's `MsgpackEncoder` and implements the engine side of the DP coordinator protocol (stats, wave complete, `START_DP_WAVE`, stale waves, lockstep dummy steps, two-phase pause). It routes outputs by `client_index`. Options: `--int32-ids`, `--tokens-per-step`, `--idle-step-ms` |
| `genopt-dp/dp-coordinator.py` | Starts the real `vllm.v1.engine.coordinator.DPCoordinatorProc` from the vLLM tree under test |
| `genopt-dp/dp-python-server.py`, `dp-rust-server.py` | Frontend launchers: 32 Python API servers per node, or one Rust process |
| `genopt-dp/dp-client.py` | Python RL-workflow driver for `wf-paced`: submit, pause(abort), sleep, resume, resubmit the partials, validate. The cohort and n32 cells use client v3 |
| `genopt-dp/dp-run.py` | Runs one cell end to end (engines, coordinator, frontends, monitor, client) and writes `run.json` / `result.json`. Request assignment: shared socket (default), per-worker `SO_REUSEPORT` listeners (`--listeners`; `final-validation.sh` turns them on for the Python candidate unless `--no-listeners`), or round-robin over per-server ports (`--round-robin-ports`, used with `dp-python-server.py --per-server-ports`; dp-round12) |
| `genopt-dp/final-validation.sh`, `final-summary.py` | The DP8 cell chain used for RFC §5, and its summary (`final-summary.{json,md}`) |
| `genopt-dp/dp-analyze.py`, `r3-pytimeline.py`, `r8-rates.py`, `r11-real-output-routing.py`, `r13-*.py` | Analysis helpers used in the dp rounds |
| `genopt/genopt-run.py`, `genopt-engine.py`, `genopt-*-server.py`, `genopt-monitor.py`, `genopt-summarize.py`, `frontend-mock-payload.py` | The earlier single-engine harness (n = 32 two-node cells). The DP harness reuses its monitor and payload code |
| `genopt/genopt-client.py`, `genopt-client2.py` | The earlier Python clients (v1, v2), kept as validation oracles for client v3 |
| `genopt/client3/` | **Client v3** source (Rust; v3.1 = `b0c7182e` plus the change that accepts the compact block with or without `"sampled_slot": false`), its HTTP and mutation test suites, and the replay bench |
| `python-bench/claude-genopt-py-bench.py` (+ `-gc.py` wrapper) | Python base-path bench: one 245,760 × 129 default-format request through the real `OutputProcessor` and generate builder. Reports consumption µs per position, build and render seconds, VmHWM, GC and body sha |
| `../SPEC.md` | The workload and compact-format contract with its dated amendments |

**Not included:**
- binaries;
- results data;
- the frozen base tree;
- the model assets: config and tokenizer files, plus a 1,024-entry token pool `[{"id": int, "decoded": str}, ...]` drawn from that tokenizer.

Supply your own model assets as described below. The engine needs no weights.

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `GENOPT_ROOT` | Project root. The scripts expect `$GENOPT_ROOT/agent_run/scripts/{genopt,genopt-dp}` (symlink or copy these directories) and write under `$GENOPT_ROOT/agent_run/results/` | `.` (Python), required (shell) |
| `GENOPT_PYTHON` | Python interpreter with the vLLM tree's dependencies | current interpreter / `python3` |
| `GENOPT_IP_NODE01` … `GENOPT_IP_NODE04` | Data-plane IP of node01–node04: node01 builds and benches, node02/node03 run frontends and engines, node04 runs the client | `127.0.0.1` |
| `SLURM_HOLD_JOB` | Slurm job id whose allocation `srun --overlap` steps join (`final-validation.sh --job` overrides it) | none |

**Expected assets** under `$GENOPT_ROOT/agent_run/results/frontend-mock-256k-20261004/frozen-r1/`:
- `model-assets/`: `config.json`, `generation_config.json`, the tokenizer and chat template, and `token-pool-v2.json`;
- `python-baseline/`: a checkout of base `0fd2e8d503`.

**Client binaries** are looked up under `$GENOPT_ROOT/agent_run/results/generate-opt-20261004/src/client3/genopt-client3-<id>`. Build one with:

```bash
cd genopt/client3 && cargo +1.95.0 build --release --offline --locked
```

`build-cn01.sh` does the same inside a Slurm step.

## Running

```bash
export GENOPT_ROOT=/path/to/project GENOPT_PYTHON=/path/to/venv/bin/python SLURM_HOLD_JOB=<id>
export GENOPT_IP_NODE01=... GENOPT_IP_NODE02=... GENOPT_IP_NODE03=... GENOPT_IP_NODE04=...
# Rust candidate vs base, all DP8 cells, 3 repeats (ABBA where a baseline can run)
genopt-dp/final-validation.sh --impl rust --candidate <vllm-rs binary> --baseline <base binary> \
  --out <dir> --repeats 3 --rust-threads 64
# Python stack (node-local tree on node02 and node03; plugin enabled through the candidate env)
genopt-dp/final-validation.sh --impl python --candidate <tree> --baseline <base tree> --out <dir> \
  --cand-env VLLM_PLUGINS=rl_compact
genopt-dp/final-summary.py <dir>
```

**The lock file.** `final-validation.sh` serialises cells through `$GENOPT_ROOT/agent_run/results/generate-opt-20261004/NODES-IN-USE.txt` (noclobber, polled every 30 s), so that several users of one allocation never overlap.

**Topology assumptions** are those of RFC §3:
- GB200 nodes, 200 Gb/s NICs;
- engines on node02;
- Python frontends on node02 and node03 (n = 32 cells: node02 only);
- one Rust process on node02;
- the client on node04 with 8 cores, or 72 cores for the `-wide` cells.

Core pinning (`taskset` ranges) is in the scripts; adjust it to your machines.
