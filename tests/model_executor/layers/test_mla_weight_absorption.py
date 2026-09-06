# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import split_kv_b_proj


def _kv_b_proj(kv_lora_rank: int, num_heads: int, nope: int, v: int):
    """An unquantized ``kv_b_proj`` stand-in: ``[out, in]`` weight, no quant."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(num_heads * (nope + v), kv_lora_rank), requires_grad=False
    )
    layer.quant_method = None
    return layer


def test_split_kv_b_proj_matches_per_head_slices():
    """``W_UK[:, n]`` / ``W_UV[:, n]`` are head ``n``'s ``[k_nope; v]`` rows of
    ``kv_b_proj``, transposed to project from the latent."""
    kv_lora_rank, num_heads, nope, v = 8, 3, 4, 2
    layer = _kv_b_proj(kv_lora_rank, num_heads, nope, v)

    W_UK, W_UV = split_kv_b_proj(layer, torch.float32, kv_lora_rank, num_heads, nope, v)

    assert W_UK.shape == (kv_lora_rank, num_heads, nope)
    assert W_UV.shape == (kv_lora_rank, num_heads, v)
    per_head = layer.weight.detach().view(num_heads, nope + v, kv_lora_rank)
    for n in range(num_heads):
        assert torch.equal(W_UK[:, n, :], per_head[n, :nope, :].T)
        assert torch.equal(W_UV[:, n, :], per_head[n, nope:, :].T)


def test_split_kv_b_proj_casts_to_act_dtype():
    layer = _kv_b_proj(8, 2, 4, 2)
    W_UK, W_UV = split_kv_b_proj(layer, torch.bfloat16, 8, 2, 4, 2)
    assert W_UK.dtype == W_UV.dtype == torch.bfloat16


def test_split_kv_b_proj_rejects_mismatched_geometry():
    layer = _kv_b_proj(8, 2, 4, 2)
    with pytest.raises(AssertionError, match="kv_b_proj weight"):
        split_kv_b_proj(layer, torch.float32, 8, 3, 4, 2)
