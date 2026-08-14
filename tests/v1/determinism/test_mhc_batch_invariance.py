# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the mHC tilelang ops.

A row's outputs must be bitwise identical no matter how many other rows share
the batch. The variant mechanisms are (a) ``compute_num_split`` deriving the
K-split count from the token-tile count and (b) the small-token FMA kernel in
``mhc_fused_post_pre`` switching implementations at ``num_tokens == 16``; both
are disabled under ``VLLM_BATCH_INVARIANT``.
"""

import torch

import vllm.envs as envs
import vllm.model_executor.kernels.mhc  # noqa: F401
from tests.kernels.test_mhc_kernels import mhc_pre_ref
from tests.v1.determinism.utils import skip_if_not_cuda
from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
from vllm.utils.torch_utils import set_random_seed

HC_MULT = 4
HIDDEN_SIZE = 4096
HC_MULT3 = 2 * HC_MULT + HC_MULT * HC_MULT
RMS_EPS = HC_PRE_EPS = HC_SINKHORN_EPS = 1e-6
SINKHORN_REPEAT = 20
HC_POST_ALPHA = 1.0

# Dispatch flips only at specific token counts: the small-FMA branch at 8 and
# 16, and compute_num_split whenever n_sms // cdiv(num_tokens, 64) drops.
# Uniform sweeps miss these; enumerate the flip points instead.
BOUNDARIES = [1, 7, 8, 15, 16, 17, 63, 64, 65, 128, 129, 192, 193, 256]


def _set_bi(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", value)
    compute_num_split.cache_clear()


def _mhc_weights(device):
    """Weights at checkpoint-realistic magnitudes.

    The correctness suite's ``fn * 1e-4`` keeps mixes so small that sigmoid
    compresses a ~10-ULP GEMM reassociation diff below one fp32 ULP of the
    output — the negative control then can't fail. The real checkpoint has
    ``hc_attn_scale ~= [2.08, 0.019, 0.245]`` and O(1) mixes, where the same
    diff survives into the fp32 outputs.
    """
    set_random_seed(0)
    fn = (
        torch.randn(
            (HC_MULT3, HC_MULT, HIDDEN_SIZE), dtype=torch.float32, device=device
        )
        * (HC_MULT * HIDDEN_SIZE) ** -0.5
    ).flatten(1, 2)
    hc_scale = torch.tensor([2.0, 0.02, 0.25], dtype=torch.float32, device=device)
    hc_base = torch.randn((HC_MULT3,), dtype=torch.float32, device=device) * 0.1
    return fn, hc_scale, hc_base


def _run_pre(residual, fn, hc_scale, hc_base):
    return torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_ALPHA,
        SINKHORN_REPEAT,
    )


def _pre_row0(n, victim, filler_seed, fn, hc_scale, hc_base):
    """Row 0 of every mhc_pre output for a batch of n rows led by victim."""
    set_random_seed(filler_seed)
    filler = torch.randn(
        (n - 1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=victim.device
    )
    batch = torch.cat([victim, filler]) if n > 1 else victim
    outs = _run_pre(batch, fn, hc_scale, hc_base)
    return [o[0].clone() for o in outs]


@skip_if_not_cuda
def test_mhc_pre_batch_invariance(monkeypatch):
    _set_bi(monkeypatch, True)
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    base = _pre_row0(BOUNDARIES[0], victim, 1, fn, hc_scale, hc_base)
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = _pre_row0(n, victim, i, fn, hc_scale, hc_base)
        for name, a, b in zip(("post_mix", "comb_mix", "layer_input"), base, outs):
            assert torch.equal(a, b), (
                f"mhc_pre {name} row 0 changed at batch size {n}: "
                f"max diff {(a.float() - b.float()).abs().max().item():.3e}"
            )


@skip_if_not_cuda
def test_mhc_pre_negative_control(monkeypatch):
    """With BI off, the same sweep must show a bitwise difference.

    This proves the harness can detect the defect; if the default path were
    already invariant, the BI test above would be vacuous.
    """
    _set_bi(monkeypatch, False)
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    base = _pre_row0(BOUNDARIES[0], victim, 1, fn, hc_scale, hc_base)
    diffs = []
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = _pre_row0(n, victim, i, fn, hc_scale, hc_base)
        if any(not torch.equal(a, b) for a, b in zip(base, outs)):
            diffs.append(n)
    assert diffs, (
        "default mhc_pre was bitwise invariant across all boundaries; "
        "the BI test cannot distinguish fixed from broken"
    )


@skip_if_not_cuda
def test_mhc_pre_correctness(monkeypatch):
    """The pinned-split path must still match the reference implementation."""
    _set_bi(monkeypatch, True)
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    for n in (1, 16, 129):
        set_random_seed(n)
        residual = torch.randn(
            (n, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        ref = mhc_pre_ref(
            residual,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_ALPHA,
            SINKHORN_REPEAT,
        )
        out = _run_pre(residual, fn, hc_scale, hc_base)
        # Same tolerance as tests/kernels/test_mhc_kernels.py: the tf32 GEMM
        # plus fused sinkhorn diverge from the fp32 reference well below this.
        for actual, expected in zip(out, ref, strict=True):
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


def _fused_row0(n, victims, filler_seed, fn, hc_scale, hc_base):
    device = victims[0].device
    set_random_seed(filler_seed)
    pads = (
        torch.randn((n - 1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((n - 1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((n - 1, HC_MULT, 1), dtype=torch.float32, device=device),
        torch.randn((n - 1, HC_MULT, HC_MULT), dtype=torch.float32, device=device),
    )
    args = [
        torch.cat([v, p]) if n > 1 else v for v, p in zip(victims, pads, strict=True)
    ]
    outs = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        *args,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_ALPHA,
        SINKHORN_REPEAT,
    )
    return [o[0].clone() for o in outs]


@skip_if_not_cuda
def test_mhc_fused_post_pre_batch_invariance(monkeypatch):
    _set_bi(monkeypatch, True)
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    victims = (
        torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((1, HC_MULT, 1), dtype=torch.float32, device=device),
        torch.randn((1, HC_MULT, HC_MULT), dtype=torch.float32, device=device),
    )
    names = ("residual", "post_mix", "comb_mix", "layer_input")

    base = _fused_row0(BOUNDARIES[0], victims, 1, fn, hc_scale, hc_base)
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = _fused_row0(n, victims, i, fn, hc_scale, hc_base)
        for name, a, b in zip(names, base, outs):
            assert torch.equal(a, b), (
                f"mhc_fused_post_pre {name} row 0 changed at batch size {n}: "
                f"max diff {(a.float() - b.float()).abs().max().item():.3e}"
            )


@skip_if_not_cuda
def test_mhc_fused_post_pre_negative_control(monkeypatch):
    _set_bi(monkeypatch, False)
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    victims = (
        torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((1, HC_MULT, 1), dtype=torch.float32, device=device),
        torch.randn((1, HC_MULT, HC_MULT), dtype=torch.float32, device=device),
    )
    base = _fused_row0(BOUNDARIES[0], victims, 1, fn, hc_scale, hc_base)
    diffs = []
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = _fused_row0(n, victims, i, fn, hc_scale, hc_base)
        if any(not torch.equal(a, b) for a, b in zip(base, outs)):
            diffs.append(n)
    assert diffs, (
        "default mhc_fused_post_pre was bitwise invariant across all "
        "boundaries; the BI test cannot distinguish fixed from broken"
    )


@skip_if_not_cuda
def test_mhc_pre_with_norm_batch_invariance(monkeypatch):
    """Production always fuses RMSNorm (norm_weight path); cover that
    big_fuse variant too."""
    _set_bi(monkeypatch, True)
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    set_random_seed(3)
    norm_w = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    def row0(n, seed):
        set_random_seed(seed)
        filler = torch.randn(
            (n - 1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        batch = torch.cat([victim, filler]) if n > 1 else victim
        outs = torch.ops.vllm.mhc_pre_tilelang(
            batch,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_ALPHA,
            SINKHORN_REPEAT,
            norm_weight=norm_w,
        )
        return [o[0].clone() for o in outs]

    base = row0(BOUNDARIES[0], 1)
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = row0(n, i)
        for name, a, b in zip(("post_mix", "comb_mix", "layer_input"), base, outs):
            assert torch.equal(a, b), (
                f"mhc_pre(norm) {name} row 0 changed at batch size {n}"
            )


# The broadcast entry computes its splits from k=hidden_size (cap 16), so the
# count first drops at cdiv(n, 64) = 10, i.e. n = 577.
BROADCAST_BOUNDARIES = [1, 7, 8, 15, 16, 17, 64, 65, 512, 576, 577, 640]


def _broadcast_row0(n, victim, seed, fn, fn_b, hc_scale, hc_base, norm_w):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_broadcast_tilelang

    set_random_seed(seed)
    filler = torch.randn(
        (n - 1, HIDDEN_SIZE), dtype=torch.bfloat16, device=victim.device
    )
    batch = torch.cat([victim, filler]) if n > 1 else victim
    outs = mhc_pre_broadcast_tilelang(
        batch,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_ALPHA,
        SINKHORN_REPEAT,
        norm_weight=norm_w,
        fn_broadcast=fn_b,
    )
    return [o[0].clone() for o in outs]


def _broadcast_setup(device):
    fn, hc_scale, hc_base = _mhc_weights(device)
    set_random_seed(4)
    fn_b = (
        torch.randn((HC_MULT3, HIDDEN_SIZE), dtype=torch.float32, device=device)
        * HIDDEN_SIZE**-0.5
    )
    norm_w = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    victim = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    return fn, fn_b, hc_scale, hc_base, norm_w, victim


@skip_if_not_cuda
def test_mhc_pre_broadcast_batch_invariance(monkeypatch):
    """Layer 0's production entry (residual broadcast from (T, H))."""
    _set_bi(monkeypatch, True)
    fn, fn_b, hc_scale, hc_base, norm_w, victim = _broadcast_setup("cuda")
    names = ("residual", "post_mix", "comb_mix", "layer_input")
    base = _broadcast_row0(1, victim, 1, fn, fn_b, hc_scale, hc_base, norm_w)
    for i, n in enumerate(BROADCAST_BOUNDARIES[1:], start=2):
        outs = _broadcast_row0(n, victim, i, fn, fn_b, hc_scale, hc_base, norm_w)
        for name, a, b in zip(names, base, outs):
            assert torch.equal(a, b), (
                f"mhc_pre_broadcast {name} row 0 changed at batch size {n}"
            )


@skip_if_not_cuda
def test_mhc_pre_broadcast_negative_control(monkeypatch):
    _set_bi(monkeypatch, False)
    fn, fn_b, hc_scale, hc_base, norm_w, victim = _broadcast_setup("cuda")
    base = _broadcast_row0(1, victim, 1, fn, fn_b, hc_scale, hc_base, norm_w)
    diffs = []
    for i, n in enumerate(BROADCAST_BOUNDARIES[1:], start=2):
        outs = _broadcast_row0(n, victim, i, fn, fn_b, hc_scale, hc_base, norm_w)
        if any(not torch.equal(a, b) for a, b in zip(base, outs)):
            diffs.append(n)
    assert diffs, (
        "default mhc_pre_broadcast was bitwise invariant across all "
        "boundaries; the BI test cannot distinguish fixed from broken"
    )


@skip_if_not_cuda
def test_hc_head_batch_invariance(monkeypatch):
    """hc_head has no K-split and each token maps to its own block, so it
    should be invariant even without the flag; assert that holds under BI."""
    _set_bi(monkeypatch, True)
    device = "cuda"
    set_random_seed(0)
    fn = (
        torch.randn(
            (HC_MULT, HC_MULT * HIDDEN_SIZE), dtype=torch.float32, device=device
        )
        * 1e-4
    )
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((HC_MULT,), dtype=torch.float32, device=device) * 0.1
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    def row0(n, seed):
        set_random_seed(seed)
        filler = torch.randn(
            (n - 1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        batch = torch.cat([victim, filler]) if n > 1 else victim
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            batch, fn, hc_scale, hc_base, RMS_EPS, HC_PRE_EPS
        )
        return out[0].clone()

    base = row0(BOUNDARIES[0], 1)
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        out = row0(n, i)
        assert torch.equal(base, out), f"hc_head row 0 changed at batch size {n}"
