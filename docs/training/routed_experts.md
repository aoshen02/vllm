# Routed Experts and Other Auxiliary Outputs

RL trainers need more from a rollout than its tokens. vLLM can return, for every position of a request:

| Output | Config field | Use |
| --- | --- | --- |
| Routed experts (expert IDs each MoE layer selected) | `enable_return_routed_experts` | MoE routing replay (R3) in the training forward pass |
| Sparse-attention indexer top-k positions | `enable_return_indexer_topk` | Indexer replay on models such as DeepSeek-V4 |
| Final hidden states (the input of the LM head) | `enable_return_hidden_states` | Value heads, probes, distillation |
| Top-k next-token ids and log-probabilities | `return_topk_logprobs: k` | Teacher scoring and distillation |

All four are fields of `AuxOutputConfig`, set with `--aux-output-config '{...}'` or `LLM(..., aux_output_config={...})`. `--enable-return-routed-experts` is shorthand for the first. They are engine-wide; there is no per-request switch.

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash --aux-output-config \
  '{"enable_return_routed_experts": true, "enable_return_indexer_topk": true}'
```

## Requirements

Startup fails with a `ValueError` unless:

- the runner is `generate`;
- **prefix caching is enabled** (the default), because rows are tracked per KV cache block;
- the model is MoE, if routed experts are enabled. The other outputs also work on dense models; the indexer output needs a sparse-attention model.

Also rejected at startup: adaptive speculative verification, PP > 1, DCP/PCP > 1, and the `NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MoRIIOConnector` and `MooncakeConnector` KV connectors. Regular speculative decoding works.

## Which positions a response covers

A request of `num_tokens` tokens (prompt + output) has positions `0 .. num_tokens - 2` with rows. The last sampled token never went through the model, so it has none. Every output is an array whose first axis is positions; row `i` is position `start + i`, so `start = num_tokens - 1 - len(rows)`.

Prefix-cache hits decide `start`:

- **Routed experts** are cached with the KV blocks, so a hit returns the rows of the computation that produced the cached KV. Rows start at 0, or at [`routed_experts_prompt_start`](#routed_experts_prompt_start).
- **The other outputs** are not cached with the KV. Returned inline, they cover only the positions this request computed. Returned through a [session](#sessions-delivering-rows-through-an-object-store), every position is covered, because the session's earlier objects hold the rows of the hit.

Rows always describe the KV that later tokens read. When weights change between turns, a hit keeps KV computed with the earlier weights, and its rows are those earlier rows; only positions from the first recomputed one onward reflect the new weights. Nothing is masked: the session object keys carry the weight version each block was computed with, and the trainer decides how to treat older positions.

## Inline delivery

Without `object_store`, every request returns its rows when it finishes:

| API | Routed experts | Other outputs |
| --- | --- | --- |
| Offline (`LLM`, `AsyncLLM`) | `CompletionOutput.routed_experts`: `np.ndarray` or `None` | `CompletionOutput.aux_fields`: `dict[str, np.ndarray]` |
| `/v1/completions`, `/v1/chat/completions` | `choices[i].routed_experts`: base64 `.npy`, **non-streaming only** | not exposed |
| `/inference/v1/generate` | `routed_experts` per choice (base64 `.npy`); with `stream: true`, on the final chunk | not exposed |

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

Outputs are `None` if the request aborted before any forward pass.

### `routed_experts_prompt_start`

Per request, default `0`: drops up to the first N prompt-token rows of routed experts, for example the routing already received in earlier turns. Must be `0 <= value <= prompt_len`, else HTTP 400. Pass it top-level on `/v1/*completions`, inside `sampling_params` on `/inference/v1/generate`, or as `SamplingParams(routed_experts_prompt_start=...)`.

Rows start at `min(routed_experts_prompt_start, first recomputed position)`. The prefix-cache hit can end earlier than the caller expects: part of the previous turn was evicted, the hit stops at the cache-hit granularity, or the cache salt changed. The positions from the hit onward are computed again and their new rows returned, because the rows received earlier for them may describe another computation. Derive `start` from the row count and replace the held rows from `start` on. Example: [examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py).

## Sessions: delivering rows through an object store

Returning every row inline copies it through the engine output path on every turn, and loses the non-cached outputs on prefix-cache hits. With `"object_store": {"backend": "mooncake", "key_prefix": "<job>"}` (the Mooncake client is configured by `MOONCAKE_CONFIG_PATH`, as for the Mooncake KV connector; `"backend": "local"` works within one process), a request that sets `SamplingParams(aux_session=...)` gets its rows through the store instead. `aux_session` without `object_store` is rejected. Requests without a session keep inline routed experts and return no other outputs. Sessions, `aux_output_keys` and `aux_fields` are offline-API features (`LLM`, `AsyncLLM`); the HTTP endpoints do not expose them.

- **Objects.** The engine writes one immutable object per KV cache block, the prefix-cache hit granularity, holding the rows of every enabled output for that block. It is named `<key_prefix>/<aux_session>/<hash of the block's last hash block>@<weight version>`, where the weight version is the one set with `update_weight_version` (`default` before the first call) when the block's last row was computed. A session writes each key once; a block recomputed under a new weight version gets a new key, and a block served from the prefix cache keeps its key.
- **Response.** The finished response lists in `aux_output_keys` the keys of the objects covering rows `[0, B)`, where `B` is the last block boundary at or below `num_tokens - 1`. Rows `[B, num_tokens - 1)` come inline: routed experts in `routed_experts`, the others in `aux_fields`. Every turn lists the full key list, so the caller replaces what it holds from the first key that differs.
- **Reading.** `vllm.distributed.aux_output_connector.objects.read_aux_rows(store, keys, field, tail)` returns all rows of one field (call it once per field), waiting for objects that are still being written; for example `read_aux_rows(store, keys, "routed_experts", output.routed_experts)` or `read_aux_rows(store, keys, "topk_logprobs", output.aux_fields["topk_logprobs"])`.
- **Hits.** A session request's prefix-cache hit covers only objects this session has published, so every listed key exists. An object is published once the worker has committed its rows; rows of an aborted step are not, so a later request recomputes them. If any other request, with another session or without one, has since recomputed a block under a different weight version, the cached KV may come from that computation, so the hit stops there and the session recomputes the block. A request that skips prefix-cache reads (for example with `prompt_logprobs`) counts too, even though the cached KV may still be the older one: the hit is lost, but rows never disagree with the KV.
- **Dense models.** Routed experts are optional in this mode. Without them, objects carry only the other enabled fields. This serves, for example, a dense teacher with `return_topk_logprobs`.

[examples/rl/aux_output_teacher_scoring.py](../../examples/rl/aux_output_teacher_scoring.py) scores a conversation turn by turn: each turn is one request with `max_tokens=1` in the same session, and only its new tokens are computed.

Objects stay in the store until the caller removes them:

- After reading a session, remove the union of the keys its responses listed with `store.remove(keys)` (a forced batch remove on Mooncake). A key that is still being written may fail to remove; the job-end sweep catches it.
- Never send another request with a released `aux_session`: the engine may still let it hit removed objects. Continue under a new session name, which recomputes the prefix.
- At the end of the job, `store.remove_prefix("<key_prefix>/")` removes everything left, including objects of aborted or discarded samples.
- The engine forgets a session that has had no running request for `session_idle_timeout_s` (default one hour); its next request then hits only objects it publishes again.

## Field formats

| Field | Shape per position | dtype |
| --- | --- | --- |
| `routed_experts` | `[num_moe_layers, num_experts_per_tok]`, logical expert IDs | `uint8` for at most 256 experts, else `uint16` |
| `indexer_topk` | `[indexer layers, index_topk]`, `-1` pads unused slots | `int16` when `max_model_len / compression ratio <= 32768`, else `int32` |
| `hidden_states` | `[hidden_size]` | the model dtype; bfloat16 rows are int16 bit patterns: `torch.from_numpy(rows).view(torch.bfloat16)` |
| `topk_token_ids`, `topk_logprobs` | `[k]`, sorted by log-probability | `int32`, `float32` (log-softmax of the raw logits) |

## Cost

- Top-k logprobs add one LM-head pass over the prefill tokens of every request that returns them: every request inline, session requests only with `object_store`. Unlike `prompt_logprobs`, they do not disable prefix-cache hits.
- Routed experts keep a host-memory row cache that serves prefix-cache and KV offload hits. `max_bytes` sets its capacity (default: derived from the KV cache capacity) and requires `enable_return_routed_experts` (startup fails otherwise). A hit on a block whose rows were released is recomputed. With a KV offload tier, size it for 1.5-2x the tier's block count: one block takes `block_size × num_hidden_layers × num_experts_per_tok × (1 if num_experts <= 256 else 2)` bytes.

## Replaying recorded routing

With `"enable_routed_experts_replay": true` (requires `enable_return_routed_experts`), a request can pass `SamplingParams(routed_experts_replay=rows)`: the routed experts recorded for prompt positions `[0, len(rows))`, shaped like the returned rows. When prefill computes those positions, every MoE layer takes the given experts instead of selecting them, and their weights are recomputed from the current router logits with the router's scoring function. The returned rows for those positions are the replayed ones.

Replay applies only to positions the request computes. Positions served from the prefix cache keep the routing that computed their KV, and return those cached rows. The KV and rows a replaying request computes are cached like any other, so a later request with the same prefix and cache salt reuses them; give requests whose cached routing must not be shared different cache salts.

Supported for non-monolithic MoE kernels with softmax, sigmoid or sqrtsoftplus routers, without fused shared experts and without data parallelism.

## Backend row restoration

A backend's `MoERunner` or `RoutedExpertsCaptureSource` can optionally implement `restore_routed_experts_rows(topk_ids, ctx) -> Tensor`. During binding, the capturer selects the first such module in `model.modules()` order and uses its callback for every captured layer. Without a provider, the existing DP/TP recovery is used. Existing capture sources need no changes.

The callback must restore this DP rank's token-ordered logical expert IDs for all captured layers, using the supplied `ForwardContext` rather than the provider's layer-specific state. It must preserve the input, device, dtype and top-k width, and keep collective participation consistent across ranks, including empty ranks. Operations must support graph replay. Models requiring different per-layer restorers cannot use this hook. Repeated layer IDs do not select a different restorer; the existing buffer-write semantics are unchanged.
