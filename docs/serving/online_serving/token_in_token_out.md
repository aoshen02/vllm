# Tokens In <> Tokens Out API

`POST /inference/v1/generate` is the token-in/token-out (TITO) HTTP endpoint: it takes prompt token IDs and returns generated token IDs, logprobs keyed by integer token ID, and `finish_reason`/`stop_reason`, with no detokenization. Use it when the caller does its own tokenization and chat templating, such as an RL rollout worker. It is also the generation step of the `render → generate → derender` pipeline used in disaggregated serving (see the [Renderer APIs](renderer.md) and [Derenderer APIs](derenderer.md)).

## Enabling the endpoint (`--enable-scale-out`, `--tokens-only`)

| Launch | `/inference/v1/generate` | Tokenizer |
| --- | --- | --- |
| `vllm serve <model>` | not registered: returns 403 `Scale-out endpoints are disabled. Restart vllm serve with --enable-scale-out to enable them.` | loaded |
| `vllm serve <model> --enable-scale-out` | registered, with `/render`, `/derender` and `/inference/v1/abort_requests` | loaded |
| `vllm serve <model> --tokens-only` | always registered, only the Tokens In <> Tokens Out endpoints | not loaded, detokenization is forced off |

- `--enable-scale-out` replaces the `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` environment variable, which no longer exists.
- When `--api-key` is set, `/inference/v1/generate` requires it.
- `--tokens-only` forces `detokenize=False` on every request, so `sampling_params.stop` (stop strings) is rejected with a 400. Use `stop_token_ids`, or check stop strings on the caller side.

```bash
vllm serve Qwen/Qwen3-0.6B --enable-scale-out

curl -X POST http://localhost:8000/inference/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"token_ids": [151644, 872, 198], "sampling_params": {"max_tokens": 32, "temperature": 1.0, "logprobs": 1}}'
```

A full client that tokenizes with the model's chat template is in [examples/scale_out/token_generation_client.py](../../../examples/scale_out/token_generation_client.py).

## Request fields

| Field | Type | Notes |
| --- | --- | --- |
| `token_ids` | `list[int]` | Required. The already-templated prompt. At least one ID, no negative IDs. |
| `sampling_params` | `SamplingParams` | Required. Any [SamplingParams](../../api/README.md#inference-parameters) field, e.g. `max_tokens`, `temperature`, `logprobs`, `prompt_logprobs`, `stop_token_ids`, `n`. If `max_tokens` is omitted, the server applies the same default as the OpenAI endpoints (the remaining context length or the model's generation config), not the dataclass default of 16. `n` must not exceed `--max-num-seqs`. |
| `output_mode` | `"tokens"` \| `"text"` | Default `"tokens"`. See [Output modes](#output-modes). |
| `stream` | `bool` | Default `false`. `true` returns SSE chunks (`data: {...}`), ending with `data: [DONE]`. |
| `stream_options` | `{"include_usage", "continuous_usage_stats"}` | Usage reporting in streams. |
| `request_id` | `str` | Defaults to a random UUID. The `X-Request-Id` header takes precedence. The server prefixes the ID with `generate-tokens-`. |
| `return_token_ids` | `bool` | Echo the final prompt IDs (after multimodal placeholder expansion) as `prompt_token_ids`, plus `mm_placeholders`. Generated IDs are always returned. In a stream, only the first chunk carries them. |
| `features` / `content_parts` | object / `list[dict]` | Multimodal input: `features` from `/render`, or raw `image_url`/`audio_url`/`video_url` parts that the server resolves. Mutually exclusive. |
| `cache_salt` | `str` | Prefix cache salt, 1 to 1024 characters. |
| `priority` | `int` | Default 0. Non-zero values need priority scheduling. |
| `model` | `str` | Served model or LoRA adapter name. |
| `kv_transfer_params`, `ec_transfer_params` | `dict` | KV and encoder cache transfer parameters for disaggregated serving. |

## Response fields

Top level: `request_id`, `model`, `created`, `output_mode`, `choices`, `usage`, `prompt_token_ids` (with `return_token_ids`), `prompt_logprobs` (with `sampling_params.prompt_logprobs`), `mm_placeholders`, `kv_transfer_params`, `ec_transfer_params`.

Each entry of `choices` (`output_mode: "tokens"`):

| Field | Notes |
| --- | --- |
| `index` | Choice index, for `n > 1`. |
| `token_ids` | Generated token IDs. |
| `finish_reason` | For example `stop`, `length` or `abort`. |
| `stop_reason` | The stop token ID or stop string that ended generation. `null` on EOS, length and abort. |
| `logprobs` | `{"content": [{"token_id", "logprob", "rank", "top_logprobs": [{"token_id", "logprob", "rank"}]}]}`, one entry per generated token. The sampled token comes first in `top_logprobs`. `null` unless `sampling_params.logprobs` is set. |
| `weight_versions` | Weight version spans over the output tokens, see [Weight versions](#weight-versions). |
| `routed_experts` | Base64-encoded `.npy` of per-token MoE expert IDs, with `--enable-return-routed-experts`. |
| `sampling_mask` | Per-token sampling support, with `--return-sampling-mask`, see [Sampling Mask](../../training/sampling_mask.md). |

Logprob entries carry an integer `token_id`, not the `"token_id:<id>"` strings that older releases returned:

```json
"logprobs": {"content": [
  {"token_id": 9707, "logprob": -0.12, "rank": 1,
   "top_logprobs": [{"token_id": 9707, "logprob": -0.12, "rank": 1},
                    {"token_id": 13, "logprob": -2.31, "rank": 2}]}
]}
```

With `stream: true`, each chunk carries the same per-choice fields for the new tokens. `finish_reason` and `stop_reason` are set on the final chunk of each choice.

## Targeting a data-parallel rank (`X-data-parallel-rank`)

Send the `X-data-parallel-rank: <int>` header to pin a request to one data-parallel (DP) engine rank, for example to keep a multi-turn rollout on the rank that holds its prefix cache. There is no body field for it.

```bash
curl -X POST http://localhost:8000/inference/v1/generate \
  -H "Content-Type: application/json" -H "X-data-parallel-rank: 1" \
  -d '{"token_ids": [151644, 872, 198], "sampling_params": {"max_tokens": 32}}'
```

- Without the header, the server's DP load balancer picks the rank.
- A header that is not an integer is ignored, and the request is load balanced.
- The rank must be in `[0, data_parallel_size)`, or `[0, data_parallel_size_local)` when the API server only manages its local engines (`local_engines_only`). Out of range ranks are rejected with `data_parallel_rank N is out of range [0, M).` (a 400 for non-streaming requests).

## Output modes

The request field `output_mode` selects how much of the postprocessing the server does for the caller.

| `output_mode` | Response adds | Server needs |
| --- | --- | --- |
| `tokens` (default) | nothing, token IDs only | nothing |
| `text` | `text` on every choice and decoded tokens with `bytes` in `logprobs` | a tokenizer |

`text` returns text and token IDs in a single call, without a separate [derender](derenderer.md) hop per chunk. This suits latency sensitive streaming, where the derender hop lands on every output token.

Every response and every stream chunk echoes `output_mode`. Servers that predate the field ignore it and return token IDs only with a 200, so check that the `output_mode` in the response matches the one you sent. Older servers don't return it.

```python
import httpx

response = httpx.post(
    "http://localhost:8000/inference/v1/generate",
    json={
        "token_ids": [151644, 872, 198],
        "sampling_params": {"max_tokens": 32},
        "output_mode": "text",
    },
).json()

assert response["output_mode"] == "text"
print(response["choices"][0]["text"])
print(response["choices"][0]["token_ids"])
```

### Text

- `text` comes from the request's own detokenizer, so `skip_special_tokens`, `spaces_between_special_tokens`, `stop` and `include_stop_str_in_output` behave as they do on `/v1/completions`.
- A matched stop string is cut from `text` unless `include_stop_str_in_output` is set. `token_ids` keeps every generated token, so decoding `token_ids` yourself brings the stop string back.
- `/derender` only accepts `output_mode: "tokens"` responses and returns a 400 for text responses which are already detokenized.
- With `stream: true`, each choice's `text` is the delta since the previous chunk. A chunk is sent whenever the engine output carries new text or a `finish_reason`, even with no new token IDs. That covers text held back for stop string matching and the final output after an abort.

### Logprobs

With `output_mode: "tokens"`, logprobs are `GenerateLogProbs` with an integer `token_id` and `rank` per entry and no `token` or `bytes` (see [Generate Output Logprobs](renderer.md#generate-output-logprobs)). With `output_mode: "text"`, they are `ChatCompletionLogProbs` carrying the decoded token strings and their UTF-8 `bytes` for the sampled token and every entry in `top_logprobs`; a server started with `--return-tokens-as-token-ids` returns `token_id:N` placeholders there instead, as `/v1/completions` does.

### Errors

The server returns a 400 for:

- `output_mode: "text"` on a server without a tokenizer (`--tokens-only` or `--skip-tokenizer-init`).
- `output_mode: "text"` with `sampling_params.detokenize: false`.
- An unsupported `output_mode` value.

For using `output_mode` with separate prefill and decode pools, see [Disaggregated Prefilling](../../features/disagg_prefill.md#generate-api-output-modes).

## Weight versions

The final output of every choice carries `weight_versions`, the label of the weights that sampled each range of its output tokens. Ranges are half-open over output-token indexes (prompt excluded), contiguous and in generation order, so a request that crossed one weight update gets two:

```json
"weight_versions": [
  {"version": "v7", "start": 0, "end": 57},
  {"version": "v8", "start": 57, "end": 128}
]
```

- With `stream: true`, only the last chunk of each choice carries the list, including the last chunk after an abort. Its offsets cover the whole output. Earlier chunks carry `null` or omit the field.
- An output with no tokens, such as a request aborted before its first token, gets `[]`.
- The labels are the ones you set with `/update_weight_version` or `/finish_weight_update`, see [Async Reinforcement Learning](../../training/async_rl.md#weight-versions-in-outputs). vLLM doesn't order or check them, and the label is `"default"` until you set one.

## Aborting requests

`POST /inference/v1/abort_requests` aborts in-flight requests. It is registered wherever `/inference/v1/generate` is and requires the API key when `--api-key` is set, like `/inference/v1/generate`.

```bash
curl -X POST http://localhost:8000/inference/v1/abort_requests \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"request_ids": ["generate-tokens-42"]}'
```

The request ID is the `request_id` the server returned: the `X-Request-Id` header or body `request_id` you sent, prefixed with `generate-tokens-`. A missing `request_ids` returns a 400. The response is empty and the abort finishes in the background.

With `--tokens-only`, the same handler is also served at `POST /abort_requests` for existing deployments. That path doesn't require the API key, even when `--api-key` is set. See [Security](../../usage/security.md#unprotected-endpoints-no-api-key-required).
