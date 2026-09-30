# Sampling Mask (Distribution Replay)

When using top-k/top-p sampling for RL rollouts (e.g. GRPO), there is a
systematic mismatch between the truncated distribution the sampler actually
drew from and the full-vocabulary softmax used to compute log-probabilities
during training. The **sampling mask** feature closes this gap by returning
the exact set of token IDs that survived top-k/top-p/min-p filtering at each
generation step, so the training side can normalize over the same support.

## Background

This feature implements the **Keep Sampling Mask** strategy described in the
[DeepSeek-V3.2 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)
(Section 3.3). The key insight: top-k/top-p truncation during rollout sampling
introduces a mismatch between the action spaces of `π_old` and `π_θ`, which
violates the principles of importance sampling and destabilizes training. By
preserving the truncation masks from `π_old` and applying them to `π_θ` during
training, both policies share identical action subspaces. DeepSeek reports that
combining top-p sampling with the Keep Sampling Mask strategy effectively
preserves language consistency during RL training.

## Quick start

```bash
vllm serve <model> \
    --return-sampling-mask \
    --logprobs-mode processed_logprobs
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model, return_sampling_mask=True,
           logprobs_mode="processed_logprobs")
output = llm.generate(
    "The capital of France is",
    SamplingParams(temperature=1.0, top_k=50, top_p=0.95, logprobs=1),
)
mask = output[0].outputs[0].sampling_mask
# mask.token_ids: [[187, 326, 512], [42, 88], ...]
#   mask.token_ids[i] = token IDs in the sampling support for generated token i
```

The mask is also available via the `/inference/v1/generate` HTTP endpoint:

```json
{
  "choices": [{
    "token_ids": [187, 42, 303],
    "sampling_mask": [[187, 326, 512], [42, 88], [303, 11, 22]],
    "finish_reason": "stop"
  }]
}
```

For score centering, add `--return-sampling-mask-logprobs` to return
`sampling_mask_logprobs` alongside each row of `sampling_mask`. The scores are
normalized over the post-filtering support and have the same order as the token
IDs, so the training side gets `π_old` over the whole nucleus without requesting
full-vocabulary `logprobs=-1`. In streaming mode, each token chunk carries its
own mask and scores.

The paired transfer uses a fixed `[num_requests, max top_k]` device buffer
(`max top_k` is the batch maximum, capped at 2048) that rides the same
asynchronous GPU → CPU copy as the sampled tokens, so it never synchronizes
with the device. Requests must set `0 < top_k <= 2048`. A top-k boundary tie
that keeps more than `top_k` tokens cannot be represented in that buffer; the
engine then fails the request (`finish_reason: "error"`) instead of returning
a truncated support. Lower `top_k` (or avoid exact logit ties) if this occurs.

## Requirements

| Requirement | Reason |
| --- | --- |
| `--return-sampling-mask` | Engine-level opt-in (disables FlashInfer sampler) |
| `--return-sampling-mask-logprobs` | Optional paired logprobs; requires `--return-sampling-mask` |
| `--logprobs-mode processed_logprobs` | Returned logprobs are normalized over the nucleus, not full vocab |
| `temperature > 0` | Greedy has no truncated distribution |
| `top_k > 0` | Bounds mask size; pure top-p can produce vocab-sized masks |
| `top_k <= 2048` | With `--return-sampling-mask-logprobs`: sizes the fixed paired buffer |
| Model Runner V2 | Required by the async D2H copy pipeline |

The engine rejects unsupported combinations at startup or request time:

- Speculative decoding
- Diffusion models
- Custom logits processors (engine-level `--logits-processors`)

## How it works

1. The sampler applies all logit processors (penalties, logit bias, bad words,
   temperature, min-p) and then top-k/top-p filtering, which sets excluded
   logits to `-inf`.
2. After sampling, `torch.isfinite(processed_logits)` identifies the surviving
   token IDs — this is the sampling mask.
3. The mask is transferred GPU → CPU asynchronously alongside sampled tokens.
   With paired logprobs, the GPU gathers the post-filtering log-softmax at the
   compacted support IDs into a fixed `[num_requests, max top_k]` buffer that
   is transferred the same way.
4. Per-step masks are converted to `list[list[int]]` for the response.

## RL training usage

The training side needs two things for the importance ratio `π_θ/π_old`:

**`π_old(a|s)` — old policy's nucleus-normalized logprob:**
Already returned by vLLM when `--logprobs-mode processed_logprobs` is set.
The `log_softmax` is computed over processed logits (where filtered tokens
are `-inf`), so the denominator only includes the nucleus.

**`π_θ(a|s)` — current policy's nucleus-normalized logprob:**
Computed by the training framework using the mask:

```python
# mask_ids: list[int], the sampling support for this token
# logits: the training model's raw logits for this position
keep = torch.zeros(vocab_size, dtype=torch.bool)
keep[mask_ids] = True
masked_logits = logits.masked_fill(~keep, float("-inf"))
log_prob = log_softmax(masked_logits)[sampled_token_id]
```

Both sides normalize over the same token set, so the importance ratio is
consistent.

## Limitations

- **Engine-level flag:** `--return-sampling-mask` globally disables the
  FlashInfer fused sampler. All requests pay the cost of the PyTorch sampling
  path, even if they don't need the mask.
- **Streaming:** Token chunks contain aligned masks and, when enabled, paired
  logprobs.
- **Paired-logprob capacity:** The paired buffer is sized by the batch maximum
  `top_k` (at most 2048), with no per-step device synchronization. Ties at the
  top-k boundary that exceed `top_k` fail the request rather than degrading
  to the bitmask fallback used by the mask-only path.
