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

import tempfile

import pytest
import torch

import vllm.envs as envs
from tests.v1.determinism.utils import batch_with_victim, skip_if_not_cuda
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.utils.torch_utils import set_random_seed


@pytest.fixture(autouse=True)
def _tp1(default_vllm_config):
    if not model_parallel_is_initialized():
        with tempfile.NamedTemporaryFile() as f:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{f.name}",
            )
            initialize_model_parallel(1, 1)
    yield


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
    (batch,) = batch_with_victim((victim,), n, seed)
    out = gate(batch)
    scores = out[0] if isinstance(out, tuple) else out
    return scores[0].clone()


@skip_if_not_cuda
def test_gate_linear_batch_invariance():
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
def test_gate_linear_correctness():
    """The invariant path computes in fp32 end to end.

    Guards against the persistent matmul's input-dtype output rounding the
    fp32 logits through bf16 (~4e-3 relative) before the cast: the bound here
    is fp32 reassociation only, which a bf16 round trip exceeds by ~1000x.
    """
    device = "cuda"
    gate = _make_gate(device)
    for n in (1, 16, 64):
        set_random_seed(n)
        x = torch.randn((n, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        out = gate(x)
        scores = out[0] if isinstance(out, tuple) else out
        assert scores.dtype == torch.float32
        ref = torch.nn.functional.linear(x.float(), gate.weight.float())
        # Measured: <=1 element per 16k at rel ~8e-5 (tf32/reassociation).
        # A bf16 round trip is ~4e-3 relative on nearly every element, so
        # this bound still fails loudly on the input-dtype-output bug.
        torch.testing.assert_close(scores, ref, atol=2e-3, rtol=1e-4)


@skip_if_not_cuda
def test_gate_linear_negative_control(monkeypatch):
    """The default path must be shown to actually vary; otherwise the BI
    test proves nothing. Sweep the public forward across batch sizes; if the
    stack happens to be invariant (e.g. cuBLAS never re-picks its algorithm
    for these shapes), also try the tier-1-vs-tier-5 comparison, and only
    then record a skip."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    device = "cuda"
    gate = _make_gate(device)
    victim = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    base = _row0(gate, victim, BOUNDARIES[0], 1)
    diffs = [
        n
        for i, n in enumerate(BOUNDARIES[1:], start=2)
        if not torch.equal(base, _row0(gate, victim, n, i))
    ]
    if diffs:
        return

    if gate.allow_ll_bf16_gemm:
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

        set_random_seed(1)
        x = torch.randn((16, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        tier1 = ll_bf16_gemm(x, gate.weight)
        tier5 = torch.mm(x, gate.weight.T, out_dtype=torch.float32)
        assert not torch.equal(tier1, tier5), (
            "cuteDSL and cuBLAS agree bitwise and the batch sweep was "
            "invariant; the BI test cannot distinguish fixed from broken"
        )
        return

    pytest.skip(
        "default-path variance not reproduced on this stack: cuteDSL is "
        "unavailable and cuBLAS stayed bitwise stable across the sweep "
        "(it does use split-k here, so the risk remains cross-version)"
    )


def test_gate_linear_bias_survives_the_fp32_path(monkeypatch, default_vllm_config):
    """The fp32 batch-invariant path must still add ReplicatedLinear's bias.

    That path bypasses super().forward() to keep the matmul in fp32, so the bias
    has to be folded in by hand. Dropping it silently shifts every router score,
    and router scores decide which experts run.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(0)
    layer = GateLinear(
        input_size=128, output_size=8, bias=True, out_dtype=torch.float32
    ).cuda()
    with torch.no_grad():
        layer.bias.copy_(torch.arange(8, device="cuda", dtype=layer.bias.dtype) + 1)

    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    out, returned_bias = layer(x)

    unbiased = torch.nn.functional.linear(x.float(), layer.weight.float())
    assert returned_bias is None
    torch.testing.assert_close(out, unbiased + layer.bias.float(), rtol=0, atol=0)
