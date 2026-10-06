# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-H batch-invariant pieces outside the BI kernel library.

Each kernel must give a token the same bits whether it runs alone or inside a
larger batch: the first m rows of an M-row call equal an m-row call. Shapes are
the Nemotron-H Lightning ones (hidden 2688, 128 experts, shared expert 3712).
"""

import pytest
import torch
from utils import skip_if_not_cuda

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

HIDDEN = 2688
ROWS = [1, 2, 7, 16, 33, 64, 256, 1024]
MAX_ROWS = 4096

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="Nemotron-H batch-invariant kernels target SM10x",
)


def _assert_rows_invariant(fn, x):
    full = fn(x)
    for m in ROWS:
        part = fn(x[:m].contiguous())
        if isinstance(full, tuple):
            for p, f in zip(part, full):
                assert torch.equal(p, f[:m]), m
        else:
            assert torch.equal(part, full[:m]), m


@skip_if_not_cuda
@torch.inference_mode()
def test_nemotron_h_norms_are_row_invariant():
    from vllm.model_executor.layers.layernorm import cuda_rms_norm as rms_forward
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        grouped_gated_rms_norm as gated_forward,
    )

    torch.manual_seed(0)
    x = torch.randn(MAX_ROWS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    _assert_rows_invariant(lambda t: rms_forward(t, weight, 1e-5), x)
    _assert_rows_invariant(
        lambda t: rms_forward(t, weight, 1e-5, residual[: t.shape[0]]), x
    )
    # Out of place: the caller's tensors are left untouched.
    x_copy, residual_copy = x.clone(), residual.clone()
    rms_forward(x, weight, 1e-5, residual)
    assert torch.equal(x, x_copy) and torch.equal(residual, residual_copy)

    # Mamba gated norm: intermediate 4096 in 8 groups.
    z = torch.randn(MAX_ROWS, 4096, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(z)
    gate_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    _assert_rows_invariant(
        lambda t: gated_forward(t, z[: t.shape[0]], gate_weight, 512, 1e-5), y
    )


@requires_sm100
@torch.inference_mode()
@pytest.mark.parametrize("n,k", [(3712, HIDDEN), (HIDDEN, 3712)])
def test_nemotron_h_shared_expert_w4a16_is_row_invariant(n, k):
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        NemotronSharedNvFp4LinearKernel,
    )

    supported, reason = NemotronSharedNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(0)
    weight = (torch.randn(n, k, device="cuda") / 30).to(torch.bfloat16)
    global_scale = (448.0 * 6.0 / weight.abs().max()).float()
    packed, block_scale = ops.scaled_fp4_quant(
        weight, global_scale, is_sf_swizzled_layout=False
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(block_scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        (1.0 / global_scale).reshape(()), requires_grad=False
    )
    layer.output_size_per_partition = n
    expected_scale = 1.0 / (1.0 / layer.weight_global_scale.detach().clone())
    kernel = NemotronSharedNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    assert torch.equal(
        layer.weight_global_scale.reshape(()), expected_scale.reshape(())
    )

    x = torch.randn(MAX_ROWS, k, device="cuda", dtype=torch.bfloat16) / 4
    _assert_rows_invariant(lambda t: kernel.apply_weights(layer, t), x)
