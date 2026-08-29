# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.deepseek_v4.nvidia.flashmla as flashmla_module
import vllm.model_executor.layers.sparse_attn_indexer as indexer_module
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_gather_lens,
    zero_invalid_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention


class _WorkspaceManager:
    def get_simultaneous(self, *specs):
        return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specs)


def _attention(compress_ratio: int) -> DeepseekV4FlashMLAAttention:
    attention = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
    nn.Module.__init__(attention)
    attention.compress_ratio = compress_ratio
    attention.window_size = 4
    attention.scale = 0.125
    attention.attn_sink = torch.zeros(2, dtype=torch.float32)
    attention.topk_indices_buffer = torch.tensor(
        [[0, 1, 2, 3, -1, -1, -1, -1]] * 8,
        dtype=torch.int32,
    )
    attention.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.empty((2, 256, 584), dtype=torch.uint8)
    )
    return attention


def _swa_metadata():
    return SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_valid_token=torch.tensor([True, True]),
        decode_swa_indices=torch.zeros((2, 1, 4), dtype=torch.int32),
        decode_swa_lens=torch.tensor([3, 4], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        block_size=256,
    )


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_decode_sparse_reuses_prefill_kernel(
    monkeypatch: pytest.MonkeyPatch,
    compress_ratio: int,
):
    attention = _attention(compress_ratio)
    swa_metadata = _swa_metadata()
    q = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    compressed_cache = (
        None
        if compress_ratio == 1
        else torch.empty((2, 256 // compress_ratio, 584), dtype=torch.uint8)
    )
    attn_metadata = (
        None
        if compress_ratio == 1
        else SimpleNamespace(
            block_table=torch.zeros((2, 2), dtype=torch.int32),
            block_size=256,
            c128a_global_decode_topk_indices=(
                torch.zeros((2, 1, 128), dtype=torch.int32)
                if compress_ratio == 128
                else None
            ),
            c128a_decode_topk_lens=(
                torch.tensor([2, 2], dtype=torch.int32)
                if compress_ratio == 128
                else None
            ),
        )
    )
    gathered = []
    captured = {}

    def fake_gather(out, cache, **kwargs):
        out.zero_()
        gathered.append((cache, kwargs))

    def fake_combine(local_topk, *args, out, **kwargs):
        captured["local_topk"] = local_topk.clone()
        indices, lengths = out
        indices.fill_(-1)
        lengths.fill_(1)
        return indices, lengths

    def fake_sparse_fwd(**kwargs):
        captured["sparse"] = kwargs
        kwargs["out"].zero_()

    monkeypatch.setattr(
        flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "sparse", raising=False
    )
    monkeypatch.setattr(
        flashmla_module, "current_workspace_manager", lambda: _WorkspaceManager()
    )
    monkeypatch.setattr(
        flashmla_module, "dequantize_and_gather_k_cache", fake_gather
    )
    monkeypatch.setattr(
        flashmla_module,
        "fill_c128_topk",
        lambda out, lengths: out.copy_(
            torch.where(
                torch.arange(out.shape[1])[None, :] < lengths[:, None],
                torch.arange(out.shape[1])[None, :],
                -1,
            ).to(torch.int32)
        ),
    )
    monkeypatch.setattr(flashmla_module, "combine_topk_swa_indices", fake_combine)
    monkeypatch.setattr(flashmla_module, "flash_mla_sparse_fwd", fake_sparse_fwd)
    monkeypatch.setattr(
        flashmla_module,
        "flash_mla_with_kvcache",
        lambda **kwargs: pytest.fail("paged decode kernel must not run"),
    )

    attention._forward_decode(
        q=q,
        kv_cache=compressed_cache,
        swa_metadata=swa_metadata,
        attn_metadata=attn_metadata,
        swa_only=compress_ratio == 1,
        output=output,
    )

    assert len(gathered) == (1 if compress_ratio == 1 else 2)
    assert captured["sparse"]["q"] is q
    assert captured["sparse"]["out"] is output
    if compress_ratio == 128:
        expected = torch.full((128,), -1, dtype=torch.int32)
        expected[:2] = torch.arange(2, dtype=torch.int32)
        torch.testing.assert_close(captured["local_topk"][0], expected, rtol=0, atol=0)
    elif compress_ratio == 4:
        torch.testing.assert_close(
            captured["local_topk"],
            attention.topk_indices_buffer[:2],
            rtol=0,
            atol=0,
        )


def test_decode_paged_remains_default(monkeypatch: pytest.MonkeyPatch):
    attention = _attention(1)
    swa_metadata = _swa_metadata()
    swa_metadata.tile_sched_swaonly = object()
    q = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    captured = {}

    def fake_paged(**kwargs):
        captured.update(kwargs)
        return kwargs["out"], None

    monkeypatch.setattr(
        flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "paged", raising=False
    )
    monkeypatch.setattr(flashmla_module, "flash_mla_with_kvcache", fake_paged)
    monkeypatch.setattr(
        flashmla_module,
        "flash_mla_sparse_fwd",
        lambda **kwargs: pytest.fail("sparse decode kernel must not run"),
    )

    attention._forward_decode(
        q=q,
        kv_cache=None,
        swa_metadata=swa_metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert captured["tile_scheduler_metadata"] is swa_metadata.tile_sched_swaonly
    assert captured["out"].shape == (2, 1, 2, 8)


def test_decode_sparse_fails_closed_without_scheduler_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    attention = _attention(1)
    swa_metadata = _swa_metadata()
    swa_metadata.seq_lens_cpu = None
    monkeypatch.setattr(
        flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "sparse", raising=False
    )

    with pytest.raises(RuntimeError, match="finalized scheduler metadata"):
        attention._forward_decode(
            q=torch.zeros((2, 2, 8), dtype=torch.bfloat16),
            kv_cache=None,
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
            output=torch.empty((2, 2, 8), dtype=torch.bfloat16),
        )


def test_compute_gather_lens_clamps_prefix_and_handles_empty_query() -> None:
    seq_lens = torch.tensor([0, 2, 10], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 0, 1, 3], dtype=torch.int32)
    output = torch.empty(3, dtype=torch.int32)

    compute_gather_lens(seq_lens, query_start_loc, output, window_size=4)

    # q_lens=[0, 1, 2], and the prefix contribution is capped at 3.
    torch.testing.assert_close(
        output, torch.tensor([0, 2, 5], dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compute_gather_lens_handles_noncontiguous_cuda_views() -> None:
    seq_base = torch.tensor([99, 0, 99, 2, 99, 10, 99], dtype=torch.int32,
                            device="cuda")
    seq_lens = seq_base[1::2]
    start_base = torch.tensor([0, 99, 0, 99, 1, 99, 3], dtype=torch.int32,
                              device="cuda")
    query_start_loc = start_base[::2]
    output_base = torch.empty(6, dtype=torch.int32, device="cuda")
    output = output_base[::2]

    compute_gather_lens(seq_lens, query_start_loc, output, window_size=4)

    torch.testing.assert_close(
        output, torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda"),
        rtol=0, atol=0
    )


def test_zero_invalid_lens_preserves_valid_rows() -> None:
    lens = torch.tensor([7, 0, 3], dtype=torch.int32)
    is_valid = torch.tensor([True, False, True])

    zero_invalid_lens(lens, is_valid)

    torch.testing.assert_close(
        lens, torch.tensor([7, 0, 3], dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_invalid_lens_handles_noncontiguous_cuda_views() -> None:
    lens_base = torch.tensor([99, 7, 99, 3, 99], dtype=torch.int32, device="cuda")
    valid_base = torch.tensor([False, True, False, False, False], device="cuda")
    lens = lens_base[1::2]
    is_valid = valid_base[1::2]

    zero_invalid_lens(lens, is_valid)

    torch.testing.assert_close(
        lens, torch.tensor([7, 0], dtype=torch.int32, device="cuda"),
        rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fill_c128_topk_handles_zero_and_full_lengths() -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import fill_c128_topk

    output = torch.empty((2, 128), dtype=torch.int32, device="cuda")
    lengths = torch.tensor([0, 128], dtype=torch.int32, device="cuda")

    fill_c128_topk(output, lengths)

    expected = torch.full_like(output, -1)
    expected[1] = torch.arange(128, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_topk_wrapper_uses_native_op_for_wide_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {}

    def fake_wide_fallback(*args):
        called["args"] = args

    monkeypatch.setattr(indexer_module.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        indexer_module, "_top_k_per_row_prefill_wide_fallback", fake_wide_fallback
    )
    logits = torch.empty((1, 4096), dtype=torch.float32)
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.tensor([4096], dtype=torch.int32)
    indices = torch.empty((1, 8), dtype=torch.int32)

    indexer_module._top_k_per_row_prefill(
        logits, starts, ends, indices, 1, 4096, 1, 8
    )

    args = called["args"]
    assert args[0] is logits
    assert args[1] is starts
    assert args[2] is ends
    assert args[3] is indices
    assert args[4:] == (1, 8)
