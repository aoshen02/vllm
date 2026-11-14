# Routed Experts Return (MoE Routing Replay / R3)

`--enable-return-routed-experts` makes vLLM return the routed experts (the expert IDs each MoE layer's router selected) for every token of a request. RL trainers use it for MoE routing replay (R3): the training forward pass reuses the rollout's routing.

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
# offline: LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
```

The flag sets `AuxOutputConfig.enable_return_routed_experts` (full form: `--aux-output-config '{"enable_return_routed_experts": true, "max_bytes": N}'`). There is no per-request on/off switch.

## Requirements

Startup fails with a `ValueError` unless:

- the runner is `generate` and the model is **MoE**;
- **prefix caching is enabled** (the default). `--no-enable-prefix-caching` fails; the old v0.21 advice to disable it is wrong.

Rejected at startup: adaptive speculative verification, PP > 1, DCP/PCP > 1, and the `NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MoRIIOConnector`, `MooncakeConnector` KV connectors. Regular speculative decoding works.

### How it works: aux output connector

Routing is stored by the aux output connector (`vllm/distributed/aux_output_connector/`, configured via `--aux-output-config`), keyed by KV block hash and shared with prefix caching, which is why prefix caching is required. `max_bytes` is its host-memory LRU capacity (default: derived from KV cache capacity). If a prefix-cache hit needs evicted rows, the worker raises `BlockObjectStoreError`; increase `max_bytes`.

### Backend row restoration

A backend's `MoERunner` or `RoutedExpertsCaptureSource` can optionally implement
`restore_routed_experts_rows(topk_ids, ctx) -> Tensor`. During binding, the capturer
selects the first such module in `model.modules()` order and uses its callback
for every captured layer. Without a provider, the existing DP/TP recovery is used.
Existing capture sources need no changes.

The callback must restore this DP rank's token-ordered logical expert IDs for all
captured layers, using the supplied `ForwardContext` rather than the provider's
layer-specific state. It must preserve the input, device, dtype and top-k width,
and keep collective participation consistent across ranks, including empty
ranks. Operations must support graph replay. Models requiring different per-layer
restorers cannot use this hook. Repeated layer IDs do not select a different
restorer; the existing buffer-write semantics are unchanged.

## Output format

| API | Field |
| --- | --- |
| Offline (`LLM`, `AsyncLLM`) | `CompletionOutput.routed_experts`: `np.ndarray` or `None` |
| `/v1/completions`, `/v1/chat/completions` | `choices[i].routed_experts`: base64 `.npy`, **non-streaming only** |
| `/inference/v1/generate` | `routed_experts` per choice (base64 `.npy`); with `stream: true`, on the final chunk |

- **Shape:** `(num_tokens - 1 - routed_experts_prompt_start, num_moe_layers, num_experts_per_tok)`, `num_tokens` = prompt + output. The last sampled token has no row.
- **Rows:** row `i` is position `routed_experts_prompt_start + i` of `prompt_token_ids + output_token_ids`.
- **dtype:** `uint8` for at most 256 experts, else `uint16`; logical expert IDs.
- Filled only when the request finishes. `None` if the flag is off or the request aborted before any forward pass.

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

## `routed_experts_prompt_start`

Per-request, default `0`: drops the first N prompt-token rows (e.g. routing already received in earlier turns). Must be `0 <= value <= prompt_len`, else HTTP 400. Pass it top-level on `/v1/*completions`, inside `sampling_params` on `/inference/v1/generate`, or as `SamplingParams(routed_experts_prompt_start=...)`.

Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).

## Delivering rows through a shared object store

Returning every row inline costs one copy per turn through the engine output path. With `--aux-output-config '{"enable_return_routed_experts": true, "object_store": {"backend": "mooncake", "key_prefix": "<job>"}}'` (the Mooncake client is configured by `MOONCAKE_CONFIG_PATH`, as for the Mooncake KV connector), a request that sets `SamplingParams(aux_session=...)` gets its rows through the store instead:

- The engine writes one immutable object per KV cache block, the prefix-cache hit granularity, named `<key_prefix>/<aux_session>/<hash of the object's last hash block>@<weight version>`, where the weight version is the one set with `update_weight_version` when the block's last row was computed. A session writes each key once; a block recomputed under a new weight version gets a new key.
- The finished response lists in `aux_output_keys` the keys of the objects covering rows `[0, B)`, where `B` is the last block boundary at or below `num_tokens - 1`, and returns rows `[B, num_tokens - 1)` in `routed_experts`. Every turn lists the full key list, so the caller replaces what it holds from the first key that differs. `routed_experts_prompt_start` must be `0`.
- Read the rows with `vllm.distributed.aux_output_connector.objects.read_aux_rows(store, keys, "routed_experts", tail)`, which waits for objects that are still being written.
- A session request's prefix-cache hit covers only objects this session has published, so every listed key exists. An object counts as published once its rows have been committed by the worker; rows of an aborted step that never completed are not, so a later request recomputes those positions. If another session or a request without a session has since recomputed the block under a different weight version, the cached KV may come from that computation, so the hit stops there and the session recomputes the block. Requests without `aux_session` keep the inline behavior.

Objects stay in the store until the caller removes them:

- After reading a session, remove the union of the keys its responses listed with `store.remove(keys)` (a forced batch remove on Mooncake). A key that is still being written may fail to remove; the job-end sweep catches it.
- Never send another request with a released `aux_session`: the engine may still allow it to hit objects that were removed. Continue under a new session name instead, which recomputes the prefix.
- At the end of the job, `store.remove_prefix("<key_prefix>/")` removes everything left, including objects of aborted or discarded samples.
- The engine forgets a session that has had no running request for `session_idle_timeout_s` (default one hour); its next request then hits only objects it publishes again.
