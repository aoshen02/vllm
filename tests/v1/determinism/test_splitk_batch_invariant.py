# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K batch-invariant matmul used by matmul_persistent for skinny-N /
long-K linears (DSv4 indexer weights_proj: M x 4096 @ 4096 x 64)."""

import pytest
import torch
from utils import skip_unsupported

from vllm.model_executor.layers import batch_invariant as bi
from vllm.model_executor.layers.bi_splitk_matmul import matmul_splitk, splitk_plan

K, N = 4096, 64


def _inputs(M: int):
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    return x, w.t()  # strided [K, N] view, as linear_batch_invariant passes it


def test_splitk_plan_depends_only_on_k_and_n():
    assert splitk_plan(4096, 64) is not None
    assert splitk_plan(4096, 128) is not None
    assert splitk_plan(4096, 256) is None
    assert splitk_plan(1024, 64) is None
    assert splitk_plan(4100, 64) is None


@skip_unsupported
@pytest.mark.parametrize("M", [1, 2, 3, 7, 16, 32, 33, 63, 64, 100, 129, 1000, 4096])
def test_splitk_batch_invariant_and_deterministic(M: int):
    # Goes through matmul_persistent so the production dispatch config
    # (split_k=8, block_m=64) is what gets checked, not matmul_splitk defaults.
    x, b = _inputs(M)
    ref = torch.cat([bi.matmul_persistent(x[i : i + 1], b) for i in range(M)])
    y = bi.matmul_persistent(x, b)
    assert torch.equal(y, ref), f"batch dependence at M={M}"
    perm = torch.randperm(M, device="cuda")
    assert torch.equal(bi.matmul_persistent(x[perm], b), ref[perm]), (
        f"position dependence at M={M}"
    )
    assert torch.equal(y, bi.matmul_persistent(x, b)), f"nondeterministic at M={M}"


@skip_unsupported
def test_splitk_matches_fp32_reference():
    x, b = _inputs(256)
    y = bi.matmul_persistent(x, b).float()
    ref = x.float() @ b.float()
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)


@skip_unsupported
def test_matmul_persistent_dispatches_to_splitk():
    x, b = _inputs(32)
    y = bi.matmul_persistent(x, b)
    assert torch.equal(y, matmul_splitk(x, b, split_k=8, block_m=64))
