# Async Reinforcement Learning

## Overview

In a standard RL training loop, generation and training happen sequentially: the policy generates rollouts, then training runs on those rollouts, and the cycle repeats. During generation the training accelerators sit idle, and vice versa.

The **one-off pipelining** approach separates the generation and training phases into two parallel coroutines, allowing the model to generate new samples while simultaneously training on previously generated data. This can lead to better GPU utilization and greater training throughput.

However, this overlap introduces a complication: weights must be updated in the inference engine mid-flight, while requests may still be in progress.

## The Pause and Resume API

To safely update weights while the inference engine is running, vLLM provides `pause_generation` and `resume_generation` methods (HTTP: `POST /pause`, `POST /resume`). These let the trainer coordinate a clean window for weight synchronization without losing in-flight work.

### pause_generation

```python
await engine.pause_generation(mode="keep", clear_cache=True)
```

The `mode` parameter controls how in-flight requests are handled, and whether
new requests are admitted while paused:

| Mode | In-flight requests | New requests |
| ---- | ------------------ | ------------ |
| `"abort"` | Aborted immediately, returning partial results (default) | Rejected with HTTP 503 until resume |
| `"wait"` | Running requests finish before the pause completes; queued ones run after resume | Rejected with HTTP 503 until resume |
| `"keep"` | Frozen in the queue; they resume when `resume_generation` is called | Accepted and queued |

`"abort"` and `"wait"` treat the pause as a generation boundary: from the moment
the pause starts (for `"wait"`, including while in-flight requests drain) until
`resume_generation` returns, the engine rejects new requests from every API server
rather than carrying them across the boundary. Clients should retry on 503. A
request racing the pause call itself may instead be treated as in-flight, and
input to an already-open streaming-input session is still accepted. Other pauses,
such as `sleep`, keep queueing new requests.

`clear_cache` (default `True`; marked deprecated but still honored in every mode, including `"keep"`) resets the prefix, KV connector, multimodal and encoder caches when the pause completes. Requests added while paused are queued until resume.

### resume_generation

```python
await engine.resume_generation()
```

Resumes the scheduler after a pause. Any requests frozen with `mode="keep"` will continue generating; `await engine.is_paused()` returns the state. Call it only after `pause_generation` returns: with DP on an MoE model, resuming while a pause is in flight raises `RuntimeError` (HTTP 500).

### HTTP Endpoints

With `VLLM_SERVER_DEV_MODE=1`, the vLLM HTTP server exposes the same functionality via:

- `POST /pause?mode=abort|wait|keep&clear_cache=true|false` - Pause generation (default `mode=abort`); returns `{"status": "paused"}`
- `POST /resume` - Resume generation
- `GET /is_paused` - Returns `{"is_paused": true|false}`
- `POST /abort_requests` - Abort in-flight requests without pausing the scheduler (send `{}` to abort all, or `{"request_ids": [...]}`); returns `{"status": "aborted", "aborted": <count>}`
    - IDs are the `X-Request-Id` (or body `request_id`) you sent plus the endpoint prefix: `chatcmpl-<id>` (chat), `cmpl-<id>-<i>` (prompt `i` of completions).
    - `--enable-scale-out`/`--tokens-only` also expose `POST /inference/v1/abort_requests` (no dev mode, `--api-key`, `request_ids` required); see [Aborting requests](../serving/online_serving/token_in_token_out.md#aborting-requests).
- `POST /update_weight_version` - Set the `weight_version` label without changing weights (send `{"new_version": "v8"}`)
- `GET /weight_info` - Return the latest committed `weight_version`

!!! note "Data Parallelism"
    With the **internal load balancer** (one `vllm serve --data-parallel-size N`, `mp` or `ray` backend), a single pause/resume call reaches every DP rank (MoE ranks also agree on the pause via all-reduce). With an **external** (`--data-parallel-external-lb`, `--data-parallel-rank`, or instances behind a proxy) or **hybrid** (`--data-parallel-hybrid-lb`) load balancer, each server controls only its own ranks: send pause, resume and the weight version to **every** server.

## Typical Async RL Flow

A typical async RL loop with weight syncing looks like this:

1. Start generating rollouts from the current policy
2. Once trainer has new weights to update to, pause generation with `mode="keep"`
3. Sync the updated weights from the trainer to the inference engine (see [Weight Transfer](weight_transfer/README.md)) and set their weight version
4. Resume generation -- in-flight requests continue with the new weights
5. Repeat

The key insight is that requests paused with `mode="keep"` will produce tokens from the **old** weights before the pause and tokens from the **new** weights after resume. With `clear_cache=True` (default), frozen requests are preempted and re-prefilled with the new weights on resume (a recompute); with `clear_cache=False`, their pre-pause KV cache is kept and stays stale. Either way, tokens sampled before the pause are old-policy; see [weight versions](#weight-versions-in-outputs).

## Weight Versions in Outputs

To tell which tokens came from which weights, set a weight-version label while generation is paused, after the new weights are loaded: pass `weight_version` to `finish_weight_update`, or call `update_weight_version` (`POST /update_weight_version` over HTTP). Labels are strings you choose, such as the trainer step; vLLM doesn't order or check them.

The final output of every sequence then carries `weight_versions`, one span per label over its output-token indexes. A request paused with `mode="keep"` across one update gets two spans:

```python
completion = final_output.outputs[0]
for span in completion.weight_versions:
    # The tokens in [span.start, span.end) were sampled by weights `span.version`.
    print(span.version, completion.token_ids[span.start : span.end])
```

The offsets cover the whole output, so with streamed delta outputs, collect the tokens before slicing. `/inference/v1/generate` returns the same list on each choice; see [Tokens In <> Tokens Out API](../serving/online_serving/token_in_token_out.md#weight-versions).

`/v1/completions` and `/v1/chat/completions` return it on each choice when the request sets `return_weight_versions: true`. The offsets count the choice's output tokens, which `return_token_ids` returns as `token_ids`, and a streamed choice gets the list on its last chunk.

An aborted request returns spans over its partial output. If you resubmit that output as part of a new prompt, the new request's spans start again at 0, so shift them by the number of tokens already generated before joining the two lists.

## Example

The [async RLHF example](../../examples/rl/rlhf_async_new_apis.py) demonstrates this pattern with `vllm.AsyncLLMEngine`, NCCL weight transfer, and mid-flight pause/resume with validation.
