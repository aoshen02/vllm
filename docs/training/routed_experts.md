# Routed Experts Return (MoE Routing Replay / R3)

`--enable-return-routed-experts` makes vLLM return, for every token of a request, the expert IDs that each MoE layer's router selected. RL trainers use it for routing replay (R3, "rollout routing replay"): the training forward pass reuses the rollout's routing so that the policy being trained routes tokens to the same experts as the policy that sampled them.

## Enable it

```bash
vllm serve Qwen/Qwen3-30B-A3B --enable-return-routed-experts
```

```python
from vllm import LLM

llm = LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
```

The flag sets `AuxOutputConfig.enable_return_routed_experts` (the full config can also be passed as `--aux-output-config '{"enable_return_routed_experts": true, "max_bytes": ...}'`).

## Requirements

Startup fails with a `ValueError` unless all of the following hold:

- **Model Runner V2.** MRV2 is the default when it supports your configuration. If vLLM falls back to the V1 runner (it logs `Model Runner V2 does not yet support ...`), you get `AuxOutput Connector requires Model Runner V2; set VLLM_USE_V2_MODEL_RUNNER=1.` TPU and out-of-tree platforms do their own validation.
- **A generate runner** (`--runner generate`, the default for generative models).
- **An MoE model.**
- **Prefix caching enabled** (the default). Do not pass `--no-enable-prefix-caching`; that fails with `AuxOutput Connector requires prefix caching.` Routing rows are stored per KV block hash, so prompt tokens served from the prefix cache still get their routing back.

!!! warning
    The removed v0.21 page `training/routed_experts_replay` told you to disable prefix caching. On current vLLM that prevents the server from starting.

## Incompatibilities

`--enable-return-routed-experts` is rejected at startup with:

- adaptive speculative verification (`speculative_config.enable_adaptive_verification`);
- pipeline parallelism (`--pipeline-parallel-size` > 1);
- decode or prefill context parallelism (DCP or PCP > 1);
- disaggregated prefill/decode KV connectors: `NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MoRIIOConnector`, `MooncakeConnector`.

Regular speculative decoding works. Rows for rejected draft tokens are dropped.

## Output format

| API | Field | Type |
| --- | --- | --- |
| Offline (`LLM`, `AsyncLLM`) | `CompletionOutput.routed_experts` | `np.ndarray` or `None` |
| HTTP | `routed_experts` on each choice | base64-encoded `.npy` string or `None` |

- **Shape:** `(num_tokens - 1 - routed_experts_prompt_start, num_moe_layers, num_experts_per_tok)`, where `num_tokens` = prompt tokens + output tokens. The last sampled token has no row because it was never fed through the model.
- **Row alignment:** row `i` is the routing for the token at position `routed_experts_prompt_start + i` of `prompt_token_ids + output_token_ids`.
- **dtype:** `uint8` when the model has at most 256 experts, otherwise `uint16`. The values are logical expert IDs.
- **When it is returned:** only once the request has finished. Intermediate streaming outputs carry `None`.
- **`None`** when the flag is off, or when the request was aborted before any forward pass.

To decode the HTTP field, use:

```python
import base64, io
import numpy as np

experts = np.load(io.BytesIO(base64.b64decode(choice["routed_experts"])))
```

## Skip already-returned prompt tokens: `routed_experts_prompt_start`

`routed_experts_prompt_start` is a per-request parameter, default `0`. It drops the first N prompt tokens from the returned rows. In multi-turn agent rollouts, set it to the number of tokens whose routing you already received in earlier turns. The value must satisfy `0 <= routed_experts_prompt_start <= prompt_len`; otherwise the request is rejected with a validation error (HTTP 400).

## Request it

The server flag does all the enabling. There is no per-request on/off switch.

### Offline

```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-30B-A3B", enable_return_routed_experts=True)
out = llm.generate(
    "The capital of France is",
    SamplingParams(max_tokens=32, routed_experts_prompt_start=0),
)[0]
experts = out.outputs[0].routed_experts  # (num_tokens - 1, layers, topk)
```

### OpenAI-compatible (`/v1/completions`, `/v1/chat/completions`)

These endpoints fill the field in non-streaming responses only; streaming responses do not include it.

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-30B-A3B",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 32,
  "routed_experts_prompt_start": 0
}'
# -> choices[0].routed_experts: "<base64 .npy>"
```

### Token-in/token-out (`/inference/v1/generate`)

Pass the parameter inside `sampling_params`. Each choice carries `routed_experts`. With `stream: true`, it arrives on the final chunk of each choice.

```bash
curl http://localhost:8000/inference/v1/generate -H 'Content-Type: application/json' -d '{
  "token_ids": [9707, 11, 847, 829, 374],
  "sampling_params": {"max_tokens": 32, "routed_experts_prompt_start": 0}
}'
```

## Limitations

- **Rust HTTP frontend:** the Rust frontend does not return `routed_experts` on any endpoint, and it always sends `routed_experts_prompt_start = 0` to the engine. Use the Python API server for routing replay.
- **Capacity:** routing for cached prefix blocks is kept in a host-memory LRU store. By default it is sized to the KV cache capacity; you can override that with `--aux-output-config '{"max_bytes": N}'`. If a prefix-cache hit needs rows that were already evicted, the worker raises `BlockObjectStoreError` with a message saying to increase `aux_output_config.max_bytes`. Do not set `max_bytes` below the default unless you have to.

## Example

[examples/rl/routed_experts_e2e.py](../../examples/rl/routed_experts_e2e.py) captures routing with `AsyncLLM`, checks that the expert IDs are in range, and checks that they are deterministic across runs.
