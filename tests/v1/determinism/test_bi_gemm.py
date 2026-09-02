# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA batch-invariant GEMM behind matmul_persistent
(csrc/batch_invariant/bi_gemm.cu): correctness for every operand layout and
tail the kernel handles, and batch invariance across M, row order and repeats.
References are computed in fp64 so TF32 matmul defaults cannot mask an error.
DSv4 shapes: indexer weights_proj M x 4096 @ 4096 x 64 (split-K) and lm_head
M x 4096 @ 4096 x 129280 (single pass)."""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers import batch_invariant as bi
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="bi_gemm is the CUDA path"
)


def _operands(M, K, N, dtype, b_layout="kmajor", seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", generator=g).to(dtype)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(dtype)
    b = w.t() if b_layout == "kmajor" else w.t().contiguous()
    return x, b


def _tol(dtype):
    return (
        {"rtol": 2e-2, "atol": 2e-2}
        if dtype != torch.float32
        else {"rtol": 1e-4, "atol": 1e-4}  # fp32 sequential sum vs fp64
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("b_layout", ["kmajor", "nmajor"])
@pytest.mark.parametrize(
    ("M", "K", "N"),
    [
        (1, 4096, 64),  # weights_proj decode, split-K
        (130, 4096, 64),
        (5, 4096, 129280),  # lm_head
        (17, 4104, 63),  # K, N tails inside a 16 B chunk
        (33, 100, 65),  # K % 8 != 0 -> zero-padded K
        (64, 2048, 127),
        (3, 1000, 129),  # K < 1024 -> no split
        (3, 1000, 64),  # K tail without a split, tcgen05 path
        (33, 4104, 64),  # K tail in the last chunk, tcgen05 path
        (16, 1, 1),
        (0, 4096, 64),
        (7, 0, 9),
    ],
)
def test_matmul_persistent_matches_fp32_reference(M, K, N, dtype, b_layout):
    x, b = _operands(M, K, N, dtype, b_layout)
    g = torch.Generator(device="cuda").manual_seed(1)
    bias = torch.randn(N, device="cuda", generator=g).to(dtype)
    ref = (x.double() @ b.double()).float()
    torch.testing.assert_close(bi.matmul_persistent(x, b).float(), ref, **_tol(dtype))
    torch.testing.assert_close(
        bi.matmul_persistent(x, b, bias).float(), ref + bias.float(), **_tol(dtype)
    )


@skip_unsupported
def test_matmul_persistent_noncontiguous_a():
    x, b = _operands(64, 4096, 64, torch.bfloat16)
    ref = (x.double() @ b.double()).float()
    rows = x[::2]  # strided rows
    torch.testing.assert_close(
        bi.matmul_persistent(rows, b).float(), ref[::2], **_tol(torch.bfloat16)
    )
    cols = torch.cat([x, x], dim=1)[:, 4096:]  # 16 B aligned slice, larger pitch
    torch.testing.assert_close(
        bi.matmul_persistent(cols, b).float(), ref, **_tol(torch.bfloat16)
    )
    cols = torch.cat([x, x], dim=1)[:, 4:4100]  # unaligned -> copied
    torch.testing.assert_close(
        bi.matmul_persistent(cols, b).float(),
        (cols.double() @ b.double()).float(),
        **_tol(torch.bfloat16),
    )


def test_split_k_depends_only_on_k_and_n():
    plan = torch.ops.vllm_batch_invariant.bi_gemm_split_k
    bi._bi_gemm()
    if torch.cuda.get_device_capability()[0] == 10:
        # tcgen05 path: the K chunking is fixed by K alone (N <= 2048); wide N
        # runs one pass; N % 8 != 0 falls back to mma.sync split-K.
        assert plan(4096, 64) == 8
        assert plan(4096, 2048) == 8
        assert plan(65536, 64) == 16
        assert plan(1024, 64) == 2
        assert plan(1023, 64) == 1
        assert plan(4096, 129280) == 1
        assert plan(4096, 63) == 16
        return
    assert plan(4096, 64) == 16
    assert plan(4096, 1024) == 16
    assert plan(4096, 1025) == 1
    assert plan(4096, 129280) == 1
    assert plan(1023, 64) == 1
    assert plan(1024, 64) == 4
    assert plan(65536, 64) == 16


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("K", "N"),
    [(4096, 64), (4096, 129280), (4104, 63), (4104, 64), (2048, 4096)],
)
@pytest.mark.parametrize("M", [1, 2, 3, 7, 16, 17, 32, 33, 63, 64, 65, 129, 1000])
def test_matmul_persistent_batch_invariant(M, K, N, dtype):
    """Row m must not depend on the batch it is computed in: each row alone,
    the whole batch, and a permuted batch give bitwise-identical rows; and a
    repeated call is bitwise identical."""
    x, b = _operands(M, K, N, dtype)
    bias = torch.randn(N, device="cuda").to(dtype)
    y = bi.matmul_persistent(x, b, bias)
    per_row = torch.cat([bi.matmul_persistent(x[i : i + 1], b, bias) for i in range(M)])
    assert torch.equal(y, per_row), f"batch dependence at M={M}"
    perm = torch.randperm(M, device="cuda")
    assert torch.equal(bi.matmul_persistent(x[perm], b, bias), y[perm]), (
        f"position dependence at M={M}"
    )
    assert torch.equal(y, bi.matmul_persistent(x, b, bias)), (
        f"nondeterministic at M={M}"
    )


@skip_unsupported
def test_linear_batch_invariant_dispatches_to_bi_gemm():
    x, b = _operands(32, 4096, 64, torch.bfloat16)
    w = b.t()
    y = bi.linear_batch_invariant(x, w, None)
    assert torch.equal(y, torch.ops.vllm_batch_invariant.bi_gemm(x, w.t(), None))
