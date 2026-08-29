import torch

from vllm.v1.worker.gpu.sample.prompt_logprob import (
    compute_prompt_token_logprobs_with_chunking,
)


def test_prompt_token_logprobs_fixed_ids_preserves_order_and_shape():
    hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
    candidate_ids = torch.tensor([[4, 1], [0, 3], [2, 4]], dtype=torch.int64)

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        # Make each row's vocab logits deterministic and independent of hidden.
        return torch.arange(5, dtype=torch.float32).repeat(x.shape[0], 1) + x[:, :1]

    result = compute_prompt_token_logprobs_with_chunking(
        candidate_ids, hidden, logits_fn, logprobs_mode="raw_logits"
    )

    assert torch.equal(result.token_ids, candidate_ids)
    expected = torch.tensor([[4.0, 1.0], [2.0, 5.0], [6.0, 8.0]])
    assert torch.equal(result.logprobs, expected)


def test_prompt_token_logprobs_chunks_without_adding_target_column():
    hidden = torch.zeros((1025, 1))
    candidate_ids = torch.zeros((1025, 1), dtype=torch.int64)

    calls = []

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        calls.append(x.shape[0])
        return torch.zeros((x.shape[0], 2))

    result = compute_prompt_token_logprobs_with_chunking(
        candidate_ids, hidden, logits_fn, logprobs_mode="raw_logits"
    )

    assert calls == [1024, 1]
    assert result.token_ids.shape == (1025, 1)
    assert result.logprobs.shape == (1025, 1)
