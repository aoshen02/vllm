# Aux outputs for RL: implementation snapshot

The code behind `docs/design/rl_aux_output/`, as run in the experiments there. It is an overlay on a vLLM build with DeepSeek-V4.1 support (commit `da8922bf0f`) and Mooncake `0.3.13.post1`, not yet a vLLM change set. Splitting it into reviewable pull requests is the next step.

## Layout

| Path | What it is | Installed as |
| --- | --- | --- |
| `vllm_overlay/connector.py` | Aux output connector: scheduler side, sessions and trajectories | `vllm/distributed/aux_output_connector/connector.py` |
| `vllm_overlay/worker.py` | Worker side: capture (on its own thread), first-computation-wins bookkeeping, preemption stash, manifests | `vllm/distributed/aux_output_connector/worker.py` |
| `vllm_overlay/oc_common.py` | Store client, writer (put retry, release fence, failed-write policy), codecs (indexer `gpudelta`, hidden raw, top-k) | `vllm/distributed/aux_output_connector/oc_common.py` |
| `vllm_overlay/oc_reader.py` | Trainer-side reader: `read_sample`, existence check of every listed object, replay masks | used by the trainer |
| `vllm_overlay/model_runner.py` | Model runner with the hidden-state and top-k capture hooks | `vllm/v1/worker/gpu/model_runner.py` |
| `vllm_overlay/patch_r3_trim.py` | Patch: R3 with the decoder replay trim (bounded-replay mode) | applied at container start |
| `swa_snapshot/` | Engine patch: keep the sliding-window state at decode end as a prefix-cache snapshot (`VLLM_SWA_DECODE_SNAPSHOT=1`), with tests | applied at container start |
| `drivers/x10_driver.py` | Single-replica workload and checker (sessions, turns, forks, compaction, fault injection, reference dumps) | experiment driver |
| `drivers/x9_driver.py` | Multi-replica top-scenario driver and checker | experiment driver |
| `tests/` | Unit and regression tests with in-memory store fakes (`tests/fixtures/`) | `python tests/<t>.py <overlay dir>` |

## State

- Delivery: block-wise, session-scoped objects with manifests; first computation wins per trajectory; rows carry their weight version; a failed store write fails the whole request (reader checks every listed object exists).
- Store: Mooncake with local SSD offload on eviction, admission by free disk, force removal at release.
- Outputs: R3, indexer selections, exact hidden states (raw), top-k logprobs (store path, `OC_EXTRA_TOPK=1`).
- Design decision pending in code: full-depth capture with `swa_bounded_replay=False` replaces partial capture (cone patches). The partial-capture path is still in this snapshot and will be removed once the full-depth path is verified at scale.
- Not included: cluster launch scripts.
