# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the MoE router gate GEMM (GateLinear).

The gate's output is the expert scores: if a row's scores move with the batch,
expert *selection* flips, which is divergence rather than rounding noise. The
router top-k is already BI-guarded (sorted=...), but that only fixes tie
ordering. Under VLLM_BATCH_INVARIANT the forward skips the shape-dispatched
tiers 1-5 and uses F.linear, which UnquantizedLinearMethod.apply routes to
linear_batch_invariant. No global init is needed: that route keys on the env
flag alone (linear.py), so these tests stay process-clean.
"""

import pytest
import torch

import vllm.envs as envs
from tests.v1.determinism.utils import skip_if_not_cuda
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.utils.torch_utils import set_random_seed

# DeepSeek V4: hidden_size=4096, n_routed_experts=256. With these dims tiers
# 2/3 are shape-ineligible and tier 4 is opt-in, so dispatch is a clean
# binary: x.shape[0] <= 16 -> cuteDSL ll_bf16_gemm, else cuBLAS.
HIDDEN_SIZE = 4096
NUM_EXPERTS = 256
BOUNDARIES = [1, 8, 15, 16, 17, 32, 33, 64]


def _make_gate(device) -> GateLinear:
    set_random_seed(0)
    gate = GateLinear(
        input_size=HIDDEN_SIZE,
        output_size=NUM_EXPERTS,
        bias=False,
        out_dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        gate.weight.copy_(
            torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        )
    return gate


def _row0(gate, victim, n, seed):
    set_random_seed(seed)
    filler = torch.randn(
        (n - 1, HIDDEN_SIZE), dtype=torch.bfloat16, device=victim.device
    )
    batch = torch.cat([victim, filler]) if n > 1 else victim
    out = gate(batch)
    scores = out[0] if isinstance(out, tuple) else out
    return scores[0].clone()


@skip_if_not_cuda
def test_gate_linear_batch_invariance(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    device = "cuda"
    gate = _make_gate(device)
    victim = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    base = _row0(gate, victim, BOUNDARIES[0], 1)
    assert base.dtype == torch.float32
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        out = _row0(gate, victim, n, i)
        assert torch.equal(base, out), (
            f"gate scores for row 0 changed at batch size {n}: "
            f"max diff {(base - out).abs().max().item():.3e}"
        )


@skip_if_not_cuda
def test_gate_linear_correctness(monkeypatch):
    """The invariant path must match fp32 F.linear within bf16 input noise."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    device = "cuda"
    gate = _make_gate(device)
    for n in (1, 16, 64):
        set_random_seed(n)
        x = torch.randn((n, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        out = gate(x)
        scores = out[0] if isinstance(out, tuple) else out
        ref = torch.nn.functional.linear(x.float(), gate.weight.float())
        # Output is fp32 but inputs are bf16, so the bound is set by the bf16
        # products, not fp32 ULPs: |err| <~ K * eps_bf16 * |x||w| per element.
        torch.testing.assert_close(scores, ref, atol=5e-2, rtol=1e-2)


@skip_if_not_cuda
def test_gate_linear_negative_control(monkeypatch):
    """The two default-path implementations that BI mode bypasses must be
    shown to actually disagree; otherwise the BI test proves nothing."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    device = "cuda"
    gate = _make_gate(device)
    if not gate.allow_ll_bf16_gemm:
        pytest.skip(
            "cuteDSL ll_bf16_gemm unavailable; on this build the default "
            "path never leaves cuBLAS for (4096, 256)"
        )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    set_random_seed(1)
    x = torch.randn((16, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    tier1 = ll_bf16_gemm(x, gate.weight)
    tier5 = torch.mm(x, gate.weight.T, out_dtype=torch.float32)
    assert not torch.equal(tier1, tier5), (
        "cuteDSL and cuBLAS agree bitwise on (16, 4096) x (4096, 256); "
        "the 16->17 dispatch flip would then be harmless and this suite "
        "should be re-examined"
    )
