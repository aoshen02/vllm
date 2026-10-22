# Trace Replay

Trace replay (forced decoding, teacher forcing of decode tokens) replays a previously generated trajectory through vLLM: set `SamplingParams.trace_decode_token_ids` and the engine emits exactly those token IDs while computing real logprobs from the model's unmodified logits. The sampled token is overwritten before logprobs are computed, so `logprobs` and ranks are scored at the forced token. The primary use case is comparing logprob distributions across different configurations:

- **Inference config diff**: compare logprobs between different quantization schemes, tensor parallelism layouts, or attention backends for the same model.
- **Train vs inference diff**: replay a training-time token sequence through the inference engine to detect logprob divergence caused by numerical differences between training and serving frameworks.

## Requirements

- Start the engine with `--enable-trace-replay` (or `LLM(..., enable_trace_replay=True)`). It is off by default because it reserves a per-request trace buffer. Without it, a request with `trace_decode_token_ids` fails with `trace_decode_token_ids is set but trace replay is not enabled. Start the engine with --enable-trace-replay to use it.`
- Model Runner V2 is required. It is the default when Triton is available and the config has no feature MRV2 does not support yet; otherwise vLLM falls back to the V1 runner. With `--enable-trace-replay` on the V1 runner, engine startup fails with `trace replay requires Model Runner V2`. Set `VLLM_USE_V2_MODEL_RUNNER=1` to force V2, and do not set it to `0`.

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enable_trace_replay=True)

# Decode token IDs captured from a previous run or training log
trace_tokens = [15, 284, 1026, 374]

params = SamplingParams(
    trace_decode_token_ids=trace_tokens,
    max_tokens=len(trace_tokens),  # the default of 16 truncates longer traces
    logprobs=5,
)
outputs = llm.generate(["Once upon a time"], sampling_params=params)

for token, logprob in zip(
    outputs[0].outputs[0].token_ids,
    outputs[0].outputs[0].logprobs,
):
    print(f"token={token}  logprob={logprob[token].logprob:.4f}")
```

The output tokens will always be `[15, 284, 1026, 374]`. The prompt is passed as usual; the trace holds only the decoded (response) tokens. The logprobs reflect the model's probability for each forced token under the current inference configuration, following `logprobs_mode` (default `raw_logprobs`, before temperature and penalties; use `processed_logprobs` for post-processing values). To score only the forced token instead of top-k, use `logprob_token_ids`.

To compare configurations, run the same trace against two engine setups and diff the per-token logprobs.

## Over HTTP

Use the Tokens In <> Tokens Out endpoint [`/inference/v1/generate`](token_in_token_out.md): its `sampling_params` object accepts every `SamplingParams` field, including `trace_decode_token_ids`. Start the server with `--enable-trace-replay` plus `--enable-scale-out` (or `--tokens-only`), which registers the endpoint. `/v1/completions` and `/v1/chat/completions` do not expose `trace_decode_token_ids`.

```bash
vllm serve Qwen/Qwen3-0.6B --enable-trace-replay --enable-scale-out
```

```python
import httpx

prompt_ids = [151644, 872, 198]
trace_tokens = [15, 284, 1026, 374]

response = httpx.post(
    "http://localhost:8000/inference/v1/generate",
    json={
        "token_ids": prompt_ids,
        "sampling_params": {
            "trace_decode_token_ids": trace_tokens,
            "logprobs": 1,
        },
    },
).json()

choice = response["choices"][0]
assert choice["token_ids"] == trace_tokens
print(choice["logprobs"])
```

When `sampling_params.max_tokens` is omitted, `/inference/v1/generate` applies the server default (up to `max_model_len - prompt_len`) instead of 16, so the full trace is replayed. A request that breaks a rule under [Limitations](#limitations) gets an error response and is not run.

## Behavior

When `trace_decode_token_ids` is set:

- The trace is truncated to `max(max_model_len - prompt_len, 1)` tokens.
- `max_tokens` becomes `min(len(trace), max_tokens)`. It is not raised to the trace length: with the `SamplingParams` default `max_tokens=16`, a longer trace is silently cut to 16 tokens. Set `max_tokens=len(trace)` (or `max_tokens=None`, which resolves to `max_model_len - prompt_len`).
- All stop conditions are disabled: EOS (`ignore_eos=True`), `stop`, `stop_token_ids` and `min_tokens`. An EOS inside the trace does not end generation.
- Generation produces exactly the trace tokens (up to `max_tokens`), then stops.
- Watermarking is skipped for trace replay requests, with a one-time warning.

## Limitations

The trace must be a non-empty list of non-negative integers below the vocabulary size. `trace_decode_token_ids` is incompatible with the following features (the request raises `ValueError`):

| Feature | Reason |
| --- | --- |
| `n > 1` | Trace replay produces a single deterministic sequence |
| `prompt_logprobs` | Expanded logit layout conflicts with trace kernel indexing |
| `prompt_logprob_token_ids` | Same as `prompt_logprobs` |
| Speculative decoding (any `--speculative-config`) | Multi-token speculation conflicts with single-token trace stepping |
| Structured outputs | Grammar constraints conflict with forced token injection |
| `repetition_detection` | Would terminate the request on repeated patterns in the trace |
| `thinking_token_budget` | Logit masking corrupts logprobs for trace tokens |
| `bad_words` (non-empty) | Logit masking corrupts logprobs for trace tokens |

To score a sequence on the prefill path instead of the decode path, put it in the prompt and use [`prompt_logprob_token_ids`](../../training/prompt_token_id_logprobs.md); that does not reproduce decode numerics.

See also the offline example [examples/generate/trace_replay_offline.py](../../../examples/generate/trace_replay_offline.py).
