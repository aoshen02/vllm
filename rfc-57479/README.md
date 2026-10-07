# RFC #57479 supporting documents

These files and the harness support the revised RFC
[vllm-project/vllm#57479](https://github.com/vllm-project/vllm/issues/57479). They live on the
fork branch `rfc57479/docs` of `aoshen02/vllm`, so they do not enter vLLM's tree.

| File | Contents |
|---|---|
| [theory.md](theory.md) | Every model term of RFC §2, derived step by step with the constants and their sources |
| [results.md](results.md) | The full measurement tables behind RFC §0, §4 and §5 |
| [methodology.md](methodology.md) | Harness, topology, cells, fairness rules, mock fidelity, and reproduction commands |
| [history-original-rfc.md](history-original-rfc.md) | The original RFC text (before the 2026-10 revision), verbatim |
| [SPEC.md](SPEC.md) | Workload and compact-format contract, with dated amendments |
| [harness/](harness/README.md) | The measurement harness (mock engines, DP launcher, final-validation chain, client v3 source, Python bench) |

**Source keys.** `[Sn]` refers to the source table at the end of the RFC body. Paths are
relative to `agent_run/results/generate-opt-20261004/` in the submitter's project tree unless
marked `reports/`, which means `agent_run/reports/generate-opt-20261004/`.

**Status.** Draft of 2026-10-07. Values marked `{{TBD:...}}` are filled in when the last runs
finish:
- Python: the final stack after round 25 (slimming).
- Rust: round 17.
- dp-round12: shared accept queue vs reuseport vs round-robin at about 1, 4 and 16 requests per server.
