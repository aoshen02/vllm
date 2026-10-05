# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM paths selected under batch invariance give a row the same bits whether
it runs alone or inside a larger batch."""

import pytest
import torch
from utils import skip_if_not_cuda

from vllm.platforms import current_platform

ROWS = [1, 2, 7, 16, 33, 64, 256, 1024]
MAX_ROWS = 4096

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="requires SM10x",
)


def _assert_rows_invariant(fn, x):
    full = fn(x)
    for m in ROWS:
        assert torch.equal(fn(x[:m].contiguous()), full[:m]), m


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_gate_skips_small_m_cute_gemm_under_bi(monkeypatch, batch_invariant):
    """The small-M CuTe router GEMM reduces differently from the large-M path."""
    from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
    from vllm.model_executor.layers.fused_moe.router import gate_linear

    monkeypatch.setattr(gate_linear.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(ll_bf16, "is_available", lambda: True)
    gate = gate_linear.GateLinear.__new__(gate_linear.GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.empty(128, 2688, dtype=torch.bfloat16))
    gate.allow_specialized_router_gemm = True
    gate.allow_cublas_router_gemm = False
    gate._router_gemm_cublas_capable = True
    gate.out_dtype = None
    gate.set_out_dtype(torch.float32)
    assert gate.allow_ll_bf16_gemm is not batch_invariant
    # cuBLAS bf16 -> fp32 stays; see test_router_cublas_gemm_is_row_invariant.
    assert gate.allow_cublas_router_gemm


_ROUTER_CHECK = """
import torch
from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
init_batch_invariance()
torch.manual_seed(0)
x = torch.randn(16384, 2688, device="cuda", dtype=torch.bfloat16)
w = torch.randn(128, 2688, device="cuda", dtype=torch.bfloat16) / 50
full = torch.mm(x, w.T, out_dtype=torch.float32)
for m in (1, 2, 7, 16, 33, 64, 256, 1024, 4096, 8192):
    part = torch.mm(x[:m].contiguous(), w.T, out_dtype=torch.float32)
    assert torch.equal(part, full[:m]), m
"""


@requires_sm100
def test_router_cublas_gemm_is_row_invariant():
    """GateLinear's cuBLAS bf16 x bf16 -> fp32 GEMM is row invariant once
    init_batch_invariance() has configured cuBLAS, which must happen before the
    first cuBLAS call, so run in a fresh process."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, VLLM_BATCH_INVARIANT="1")
    subprocess.run([sys.executable, "-c", _ROUTER_CHECK], env=env, check=True)


@requires_sm100
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(10304, 2688), (2688, 4096)])
def test_flashinfer_fp8_gemm_is_row_invariant(n, k):
    from vllm.utils.flashinfer import (
        _bmm_fp8_backend,
        flashinfer_scaled_fp8_mm,
        has_flashinfer,
    )

    if not has_flashinfer():
        pytest.skip("FlashInfer is not installed")
    assert _bmm_fp8_backend() == "cutlass"
    torch.manual_seed(0)
    weight = (torch.randn(n, k, device="cuda") / 30).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(0.02, device="cuda")
    scale_b = torch.tensor(0.01, device="cuda")
    x = (torch.randn(MAX_ROWS, k, device="cuda") * 4).to(torch.float8_e4m3fn)
    _assert_rows_invariant(
        lambda t: flashinfer_scaled_fp8_mm(
            t, weight.t(), scale_a, scale_b, torch.bfloat16
        ),
        x,
    )


@skip_if_not_cuda
def test_batch_invariant_w4a16_linear_uses_humming(monkeypatch, caplog_vllm):
    from vllm.model_executor import kernels
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.kernels.linear.nvfp4.humming import (
        HummingNvFp4LinearKernel,
    )

    monkeypatch.setattr(
        linear, "_get_linear_backend", lambda quantization: "flashinfer-cutedsl"
    )
    kernel = kernels.linear.init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, HummingNvFp4LinearKernel)
    assert "overrides --linear-backend=flashinfer-cutedsl" in caplog_vllm.text
