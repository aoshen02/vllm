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

## Sessions: delivering rows through a shared object store

`routed_experts_prompt_start` keeps earlier turns out of a response, but the client must then hold every row it has received, and each sibling request or trainer worker that needs the same prefix must be handed them. A session stores the rows of full KV cache blocks once in a shared object store instead: any request of the session, and any process that can reach the store, reads them by key, and each key names the weight version of the KV it belongs to.

Start the server with `--aux-output-config '{"enable_return_routed_experts": true, "object_key_prefix": "<run>"}'` and send `aux_session` inside `sampling_params` on `/inference/v1/generate`. Requests without a session keep the inline behavior.

- **Store.** The Mooncake client is configured by `MOONCAKE_CONFIG_PATH`, as for the Mooncake KV connector, and must use `"mode": "standalone-store"`: objects placed in the memory of an embedded client disappear when that client exits. Objects are hard-pinned and never evicted, so size the pool for everything written between two removals. A failed write (for example a full pool) is logged and stops the engine at its next step.
- **Key prefix.** `object_key_prefix` enables sessions. It is one path segment without `/`. Objects are immutable, so never reuse a prefix: choose a new one for every start of a deployment, including a restart of the same job, and a different one for every deployment that shares the store. Engines of one deployment that share a configuration (data-parallel ranks, replicas of one model) use the same prefix and need no other setting: a key is written by whichever engine computes its block, and engines that compute the same block of a session at the same weight version write the same key (see Weight versions).
- **Requests.** `aux_session` is a non-empty string without `/`. It is rejected without `object_key_prefix`, together with `stop` strings (the frontend ends such a request before the engine can return its keys; use `stop_token_ids`), and on streaming-input requests. `routed_experts_prompt_start` is ignored for a session request. Only the Python API server accepts `aux_session`.
- **KV cache layout.** An object covers one block of the full-attention KV cache group, and a prefix-cache hit of a session request must end on a block of that group. The engine refuses to start with `object_key_prefix` without prefix caching, if the target model has no plain full-attention KV cache group whose block size is a multiple of every other group's, if prefix-cache hits may end inside a KV block, if block hashes look ahead more than one token, or with prefix replay (disable `CacheConfig.swa_bounded_replay` for models that use it).
- **Objects.** The engine writes one immutable object per block of the full-attention KV cache group, named `<object_key_prefix>/<aux_session>/<hash of the block's last hash block>@<weight version>`. The weight version is the one set with `update_weight_version` (`default` before the first call) when the step that computes the block's last row is scheduled. Change it only while no step is in flight, that is while the engine is paused: `update_weight_version` does not wait for running steps, and with speculative decoding a change during a step can list a key whose object is never written.
- **Response.** The final output of a request (the last chunk when streaming) lists in `aux_output_keys` the keys of the objects covering rows `[0, B)` and returns rows `[B, num_tokens - 1)` in `routed_experts`. `B` is a block boundary at or below `num_tokens - 1`: normally the last one, an earlier one when the engine could not name the last blocks before the request stopped (their hash needs a token that was sampled last, or the request was preempted). Every turn lists all its keys, about 100 bytes per KV cache block of the request; the caller replaces what it holds from the first key that differs. Read the rows with `object_store.read_aux_rows(object_store.MooncakeAuxObjectStore.create(), keys, "routed_experts", tail)` from `vllm.distributed.aux_output_connector`, with the same `MOONCAKE_CONFIG_PATH`; it waits up to 60 seconds for objects that are still being written. Create the store client in a process that does not start an engine afterwards: a Mooncake client created before the engine breaks CUDA initialization in the engine's workers.
- **Prefix cache.** The session is part of the cache salt: only requests of the same session share KV blocks, so the first request of a session computes its whole prompt. A later request of the session, also one admitted in the same step, hits the blocks of the session that are in the GPU prefix cache; their objects are written even if the request that computed them is aborted or preempted afterwards. For models with sliding-window KV cache groups (DeepSeek-V4 class), set `prefix_cache_retention_interval` to `null` or the block size so that a later turn hits the outputs of earlier turns: with the default, sliding-window state is kept only at the end of each prompt, so the hit stops at the previous turn's prompt. With the native KV offloading connector (`OffloadingConnector`), a session request also loads the session's blocks from its host-memory tier, for example after a sleep that keeps the tier (`reset_connector=False`); a loaded block is listed under the weight version of the block that was stored. This works with a single CPU tier (the `CPUOffloadingSpec` manager, no secondary tiers) that holds one KV block per entry (neither `blocks_per_chunk` nor `block_size` in `kv_connector_extra_config`). Earlier turns' outputs are reusable from the tier only with `offload_prompt_only` set to `false`; by default the tier keeps prompt tokens only. A tier hit is lowered to the last block of the full-attention group below the last prompt token, and further as described under weight versions; a lowered hit is looked up in the tier again. With any other KV connector or tier layout a session request loads no KV from the connector: what is not in the GPU prefix cache is computed again. A sleep or pause with `mode="keep"` and cache reset fails while a request waits for a load from the tier (also without sessions); with the offload tier, use `mode="wait"` or `"abort"`, or sleep after the rollout's requests have finished.
- **Weight versions.** A key names the weight version of the KV the request used for that block. After a weight update that keeps the prefix cache, hit blocks keep their earlier version and only blocks computed afterwards carry the new one; nothing is masked, the trainer decides how to treat older blocks. With several KV cache groups (full attention with sliding-window or compressed state, DeepSeek-V4 class), the engine keeps the weight version of every position of every group's blocks, as written by the target model (the draft KV that multi-module MTP computes again does not change the routed experts and does not lower a hit): a hit, from the GPU prefix cache or from the tier, is lowered to the last block boundary at or below the first position where a group it reads holds KV of another version than the full-attention group; because a sliding-window or state group reads other blocks at the lowered boundary, these are checked again: a tier hit is lowered until they agree, and a GPU hit is checked once more and drops to 0 if they disagree as well. A tier hit lowered to the GPU hit loads nothing from the tier; the GPU hit stays. The part of the prompt that is not used is computed again at the current version. A block filled across a weight update is named by the version of its last row and holds rows of both versions. The store keeps one of the objects written under a key: a second computation of the same block at the same version, by a sibling or after an eviction, lists the same key, although it may have attended to a prefix of another version.

The engine only writes objects; the consumer deletes them with a Mooncake client (`mooncake.store.MooncakeDistributedStore`) connected to the same store: `batch_remove(keys, True)` for the keys the responses listed, or `remove_by_regex("^" + re.escape(object_key_prefix + "/"), True)` for everything under the prefix, which also catches objects of aborted requests. Delete by key after reading the objects, and by prefix only after the engines that write under it have shut down or stopped writing to it: Mooncake skips objects that are still being written, even with force, and they stay pinned. The `True` forces the removal: without it, Mooncake keeps an object until the lease of its last read (`default_kv_lease_ttl` of the master) expires. Do not send another request with a released `aux_session`: the engine still lets it hit the blocks of the removed objects. Continue under a new session name, which recomputes the prefix.
