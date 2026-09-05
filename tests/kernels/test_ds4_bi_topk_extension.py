# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import (
    _top_k_per_row_prefill,
)
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def enable_batch_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("length", [128, 1024, 4012, 4096, 6000, 8175])
@pytest.mark.parametrize("width", [8192, 65536])
@pytest.mark.parametrize("top_k", [512, 2048])
@torch.inference_mode()
def test_packed_short_row_ties_preserve_request_local_order(
    length: int, width: int, top_k: int
) -> None:
    """A wider sibling must not change the chosen subset of equal scores."""
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)
    logits = torch.ones((4, width), device="cuda")
    starts = torch.full((4,), 17, device="cuda", dtype=torch.int32)
    ends = starts + length
    output = torch.empty((4, top_k), device="cuda", dtype=torch.int32)
    expected = torch.full_like(output, -1)
    count = min(length, top_k)
    expected[:, :count] = torch.arange(
        length - 1, length - count - 1, -1, device="cuda", dtype=torch.int32
    )
    for _ in range(3):
        torch.ops.ds4_bi.top_k_per_row_prefill(
            logits, starts, ends, output, 4, *logits.stride(), top_k
        )
        assert torch.equal(output, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("width", [8192, 16384, 32768, 65536])
@torch.inference_mode()
def test_coarse_histogram_overflow_does_not_drop_larger_scores(width: int) -> None:
    """More than a temporary buffer of scores can share one coarse FP16 bin."""
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)
    logits = torch.linspace(1.0, 1.0001, width, device="cuda")[None].repeat(4, 1)
    starts = torch.zeros(4, device="cuda", dtype=torch.int32)
    ends = starts + width
    output = torch.empty((4, 512), device="cuda", dtype=torch.int32)
    expected = torch.arange(
        width - 1, width - 513, -1, device="cuda", dtype=torch.int32
    ).expand_as(output)
    torch.ops.ds4_bi.top_k_per_row_prefill(
        logits, starts, ends, output, 4, *logits.stride(), 512
    )
    assert torch.equal(output, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_single_pivot_tie_in_two_ctas_cannot_overwrite_next_row() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)
    logits = torch.zeros((4, 65536), device="cuda")
    logits[:, :511] = 10
    logits[:, 1024] = 1
    logits[:, -1] = 1
    starts = torch.zeros(4, device="cuda", dtype=torch.int32)
    ends = starts + 65536
    output = torch.full((5, 512), -777, device="cuda", dtype=torch.int32)
    expected = torch.cat(
        (
            torch.arange(510, -1, -1, device="cuda", dtype=torch.int32),
            torch.tensor([65535], device="cuda", dtype=torch.int32),
        )
    ).expand(4, -1)
    torch.ops.ds4_bi.top_k_per_row_prefill(
        logits, starts, ends, output, 4, *logits.stride(), 512
    )
    assert torch.equal(output[:4], expected)
    assert torch.all(output[4] == -777)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_bi_dispatch_uses_standalone_topk_with_request_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.ops.load_library(library)

    top_k = 64
    row_starts = torch.tensor([0, 17, 65], dtype=torch.int32, device="cuda")
    row_ends = row_starts + 512
    logits = torch.ones((3, 1024), dtype=torch.float32, device="cuda")
    expected = (
        (row_ends - row_starts)[:, None]
        - 1
        - torch.arange(top_k, dtype=torch.int32, device="cuda")
    )

    for _ in range(3):
        indices = torch.empty((3, top_k), dtype=torch.int32, device="cuda")
        _top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_standalone_topk_matches_vllm_score_and_tie_order() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)

    logits = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    logits[0, 5] = 3
    logits[0, 2] = 4
    logits[0, 7] = 4
    logits[0, 1] = 2
    row_starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([16], dtype=torch.int32, device="cuda")
    expected = torch.tensor([[7, 2, 5, 1]], dtype=torch.int32, device="cuda")

    for _ in range(3):
        indices = torch.empty((1, 4), dtype=torch.int32, device="cuda")
        torch.ops.ds4_bi.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            1,
            logits.stride(0),
            logits.stride(1),
            4,
        )
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_wide_batch_exact_pivot_tie_is_deterministic() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)

    rows, cols, top_k = 128, 65536, 512
    logits = torch.zeros((rows, cols), dtype=torch.float32, device="cuda")
    logits[:, :1000] = 1.0
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    expected = torch.arange(999, 999 - top_k, -1, dtype=torch.int32, device="cuda")
    for _ in range(3):
        indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
        _top_k_per_row_prefill(
            logits,
            starts,
            ends,
            indices,
            rows,
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        torch.testing.assert_close(indices, expected.expand(rows, -1), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_short_histogram_uses_high_index_for_partial_tie() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)

    rows, cols, top_k = 128, 32768, 512
    logits = torch.full((rows, cols), -1.0, dtype=torch.float32, device="cuda")
    logits[:, :511] = 10.0
    logits[:, 1000:1200] = 1.0
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    expected = torch.cat(
        (
            torch.arange(510, -1, -1, dtype=torch.int32, device="cuda"),
            torch.tensor([1199], dtype=torch.int32, device="cuda"),
        )
    )
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    _top_k_per_row_prefill(
        logits,
        starts,
        ends,
        indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    torch.testing.assert_close(indices, expected.expand(rows, -1), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_wide_rows_support_generic_topk_values() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)

    rows, cols, top_k = 2, 8192, 64
    logits = torch.ones((rows, cols), dtype=torch.float32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    _top_k_per_row_prefill(
        logits,
        starts,
        ends,
        indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    torch.accelerator.synchronize()
    expected = torch.arange(
        cols - 1, cols - top_k - 1, -1, dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(indices, expected.expand(rows, -1), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_wide_rows_mix_short_and_large_sequences() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)

    rows, cols, top_k = 64, 65536, 512
    logits = torch.full((rows, cols), -1.0, dtype=torch.float32, device="cuda")
    logits[:, :top_k] = 10.0
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    ends[::2] = 1024
    indices = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    _top_k_per_row_prefill(
        logits,
        starts,
        ends,
        indices,
        rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    torch.accelerator.synchronize()
    expected = torch.arange(top_k - 1, -1, -1, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(indices, expected.expand(rows, -1), rtol=0, atol=0)
