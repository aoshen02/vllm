# Async Reinforcement Learning

## Overview

In a standard RL training loop, generation and training happen sequentially: the policy generates rollouts, then training runs on those rollouts, and the cycle repeats. During generation the training accelerators sit idle, and vice versa.

The **one-off pipelining** approach separates the generation and training phases into two parallel coroutines, allowing the model to generate new samples while simultaneously training on previously generated data. This can lead to better GPU utilization and greater training throughput.

However, this overlap introduces a complication: weights must be updated in the inference engine mid-flight, while requests may still be in progress.

## The Pause and Resume API

To safely update weights while the inference engine is running, vLLM provides `pause_generation` and `resume_generation` methods (HTTP: `POST /pause`, `POST /resume`), including across data-parallel ranks. These let the trainer coordinate a clean window for weight synchronization without losing in-flight work.

### pause_generation

```python
await engine.pause_generation(mode="keep")
```

The `mode` parameter controls how in-flight requests are handled:

| Mode | Behavior |
| ---- | -------- |
| `"abort"` | Abort all in-flight requests immediately and return partial results with `finish_reason="abort"` (default) |
| `"wait"` | Stop scheduling new requests and wait for all in-flight requests to finish before pausing |
| `"keep"` | Freeze requests in the queue; they resume when `resume_generation` is called |

In every mode, requests added while paused are queued and start after resume. `pause_generation` returns once the pause has completed (for `"wait"`, after the in-flight requests have drained).

`clear_cache` (default `True`) is marked **deprecated** in the API docstrings, but is still honored in every mode, including `"keep"`: when the pause completes, it resets the prefix cache, the KV connector cache, and the multimodal and encoder caches. See [Typical Async RL Flow](#typical-async-rl-flow) for its effect on frozen requests. Pass `clear_cache=False` to keep the caches.

### resume_generation

```python
await engine.resume_generation()
```

Resumes the scheduler after a pause. Any requests frozen with `mode="keep"` will continue generating, and requests queued during the pause start. `await engine.is_paused()` returns the current state.

Call `resume_generation` only after `pause_generation` has returned. With data parallelism on an MoE model, calling it while a pause is still in flight raises `RuntimeError` ("resume_scheduler called while pause is still in flight"); over HTTP, `/resume` then returns 500.

### HTTP Endpoints

With `VLLM_SERVER_DEV_MODE=1`, the vLLM HTTP server exposes the same functionality via:

- `POST /pause?mode=abort|wait|keep&clear_cache=true|false` - Pause generation (default `mode=abort`); returns `{"status": "paused"}`
- `POST /resume` - Resume generation; returns `{"status": "resumed"}`
- `GET /is_paused` - Returns `{"is_paused": true|false}`
- `POST /abort_requests` - Abort in-flight requests without pausing the scheduler (send `{}` to abort all, or `{"request_ids": [...]}` with the external request IDs). Aborted requests return partial results with `finish_reason="abort"`; the response is `{"status": "aborted", "aborted": <count>}`
    - The IDs are the `X-Request-Id` header (or body `request_id`) you sent, with the endpoint prefix: `chatcmpl-<id>` for `/v1/chat/completions`, `cmpl-<id>-<i>` for prompt `i` of `/v1/completions`.
    - With `--enable-scale-out` or `--tokens-only`, `POST /inference/v1/abort_requests` also exists without dev mode and is guarded by `--api-key`. It requires `request_ids` (no abort-all), returns an empty 200 and aborts in the background; see [Aborting requests](../serving/online_serving/token_in_token_out.md#aborting-requests).
- `POST /update_weight_version` - Set the `weight_version` label without changing weights (send `{"new_version": "v8"}`)
- `GET /weight_info` - Return the latest committed `weight_version`

!!! note "Data Parallelism"
    With vLLM's **internal load balancer** (one `vllm serve --data-parallel-size N` server, with either the `mp` or `ray` DP backend), the server sends each pause and resume call to every DP rank -- a single call is sufficient. For MoE models, the DP ranks also agree on the pause through an all-reduce before it completes. With an **external load balancer** (`--data-parallel-external-lb`, `--data-parallel-rank`, or multiple independent vLLM instances behind a proxy) or **hybrid load balancing** (`--data-parallel-hybrid-lb`, one API server per node), each server controls only its own ranks: send pause and resume to **every** server individually before and after the weight update, and set the weight version on every server too.

## Typical Async RL Flow

A typical async RL loop with weight syncing looks like this:

1. Start generating rollouts from the current policy
2. Once trainer has new weights to update to, pause generation with `mode="keep"`
3. Sync the updated weights from the trainer to the inference engine (see [Weight Transfer](weight_transfer/README.md)) and set their weight version
4. Resume generation -- in-flight requests continue with the new weights
5. Repeat

The key insight is that requests paused with `mode="keep"` will produce tokens from the **old** weights before the pause and tokens from the **new** weights after resume. The deprecated `clear_cache` parameter decides what happens to their KV cache:

- `clear_cache=True` (the default): the frozen requests are preempted and the prefix cache is reset. On resume, their prompt and already-generated tokens are prefilled again with the new weights, so all KV used after resume comes from the new weights, at the cost of a recompute.
- `clear_cache=False`: the existing KV cache is kept, so resume is faster, but the context computed before the pause still reflects the old weights (stale KV cache).

Either way, tokens already sampled before the pause stay old-policy tokens; use [weight versions](#weight-versions-in-outputs) to tell them apart.

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
