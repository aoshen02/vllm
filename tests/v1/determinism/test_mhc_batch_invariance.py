# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the mHC tilelang ops.

A row's outputs must be bitwise identical no matter how many other rows share
the batch. The variant mechanisms are (a) ``compute_num_split`` deriving the
K-split count from the token-tile count and (b) the small-token FMA kernel in
``mhc_fused_post_pre`` switching implementations at ``num_tokens == 16``; both
are disabled under ``VLLM_BATCH_INVARIANT``.
"""

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.kernels.mhc  # noqa: F401
from tests.kernels.test_mhc_kernels import mhc_post_ref, mhc_pre_ref
from tests.v1.determinism.utils import batch_with_victim, skip_if_not_cuda
from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
from vllm.utils.torch_utils import set_random_seed

HC_MULT = 4
HIDDEN_SIZE = 4096
HC_MULT3 = 2 * HC_MULT + HC_MULT * HC_MULT
RMS_EPS = HC_PRE_EPS = HC_SINKHORN_EPS = NORM_EPS = 1e-6
SINKHORN_REPEAT = 20
HC_POST_ALPHA = 1.0

# Dispatch flips only at specific token counts: the small-FMA branch at 8 and
# 16, and compute_num_split whenever n_sms // cdiv(num_tokens, 64) drops.
# Uniform sweeps miss these; enumerate the flip points instead.
BOUNDARIES = [1, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 129, 192, 193, 256]

# The non-DeepGEMM prenorm GEMM has two more dispatch flips of its own, on
# x.shape[0] rather than on the split count: a block-M kernel at >= 1024 and a
# wider-tile config below 128. B200 takes the DeepGEMM path, so nothing above
# reaches them; the helper is called directly instead.
PRENORM_BOUNDARIES = [1, 64, 127, 128, 129, 1023, 1024, 1025]


@pytest.fixture(autouse=True)
def _fresh_split_cache():
    """compute_num_split caches across the BI toggle; every test (the BI=True
    conftest default and the BI=False negative controls alike) must start
    with a cleared cache or the toggle silently does nothing."""
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
    (batch,) = batch_with_victim((victim,), n, filler_seed)
    outs = _run_pre(batch, fn, hc_scale, hc_base)
    return [o[0].clone() for o in outs]


@skip_if_not_cuda
def test_mhc_pre_batch_invariance():
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
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
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
def test_mhc_pre_correctness():
    """The pinned-split path must still match the reference implementation."""
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
    args = batch_with_victim(victims, n, filler_seed)
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
def test_mhc_fused_post_pre_batch_invariance():
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
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
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
def test_mhc_pre_with_norm_batch_invariance():
    """Production always fuses RMSNorm (norm_weight path); cover that
    big_fuse variant too."""
    device = "cuda"
    fn, hc_scale, hc_base = _mhc_weights(device)
    set_random_seed(3)
    norm_w = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    def row0(n, seed):
        (batch,) = batch_with_victim((victim,), n, seed)
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

    (batch,) = batch_with_victim((victim,), n, seed)
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
def test_mhc_pre_broadcast_batch_invariance():
    """Layer 0's production entry (residual broadcast from (T, H))."""
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
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
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
def test_hc_head_batch_invariance():
    """hc_head has no K-split and each token maps to its own block, so it
    should be invariant even without the flag; assert that holds under BI."""
    device = "cuda"
    set_random_seed(0)
    # Checkpoint-realistic magnitudes (see _mhc_weights): the old ``* 1e-4``
    # weights compress reassociation diffs below one output ULP, making the
    # assertion vacuously easy.
    fn = (
        torch.randn(
            (HC_MULT, HC_MULT * HIDDEN_SIZE), dtype=torch.float32, device=device
        )
        * (HC_MULT * HIDDEN_SIZE) ** -0.5
    )
    hc_scale = torch.tensor([2.0], dtype=torch.float32, device=device)
    hc_base = torch.randn((HC_MULT,), dtype=torch.float32, device=device) * 0.1
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    def row0(n, seed):
        (batch,) = batch_with_victim((victim,), n, seed)
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            batch, fn, hc_scale, hc_base, RMS_EPS, HC_PRE_EPS
        )
        return out[0].clone()

    base = row0(BOUNDARIES[0], 1)
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        out = row0(n, i)
        assert torch.equal(base, out), f"hc_head row 0 changed at batch size {n}"


@skip_if_not_cuda
def test_mhc_post_batch_invariance():
    """mhc_post (the end-of-loop post-mix catch-up) never goes through
    compute_num_split — per-token CTA, reductions only over hc_mult with a
    vectorized fixed order — so it must be invariant even without the flag;
    pin that fact under BI so a future K-split refactor trips this test."""
    device = "cuda"
    set_random_seed(0)
    victim_x = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    victim_res = torch.randn(
        (1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    victim_post = torch.randn((1, HC_MULT, 1), dtype=torch.float32, device=device)
    victim_comb = torch.randn((1, HC_MULT, HC_MULT), dtype=torch.float32, device=device)

    def row0(n, seed):
        x, res, post, comb = batch_with_victim(
            (victim_x, victim_res, victim_post, victim_comb), n, seed
        )
        out = torch.ops.vllm.mhc_post_tilelang(x, res, post, comb)
        return out[0].clone()

    base = row0(BOUNDARIES[0], 1)
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        out = row0(n, i)
        assert torch.equal(base, out), f"mhc_post row 0 changed at batch size {n}"


def _fused_norm_outs(n, victims, filler_seed, fn, hc_scale, hc_base, norm_w):
    args = batch_with_victim(victims, n, filler_seed)
    return torch.ops.vllm.mhc_fused_post_pre_tilelang(
        *args,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_ALPHA,
        SINKHORN_REPEAT,
        norm_weight=norm_w,
        norm_eps=NORM_EPS,
    )


def _fused_norm_setup(device):
    fn, hc_scale, hc_base = _mhc_weights(device)
    set_random_seed(5)
    norm_w = torch.randn((HIDDEN_SIZE,), dtype=torch.bfloat16, device=device)
    victims = (
        torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        torch.randn((1, HC_MULT, 1), dtype=torch.float32, device=device),
        torch.randn((1, HC_MULT, HC_MULT), dtype=torch.float32, device=device),
    )
    return fn, hc_scale, hc_base, norm_w, victims


@skip_if_not_cuda
def test_mhc_fused_post_pre_with_norm_batch_invariance():
    """The shape production actually calls between attention and FFN.

    The unfused entry above shares the dispatch but not the epilogue: with
    norm_weight the kernel takes a second pass that reduces over hidden_size
    to form the RMS, so a split-count change reaches the output through a
    path the norm-free test does not cover.
    """
    fn, hc_scale, hc_base, norm_w, victims = _fused_norm_setup("cuda")
    names = ("residual", "post_mix", "comb_mix", "layer_input")

    first = _fused_norm_outs(BOUNDARIES[0], victims, 1, fn, hc_scale, hc_base, norm_w)
    base = [o[0].clone() for o in first]
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = _fused_norm_outs(n, victims, i, fn, hc_scale, hc_base, norm_w)
        for name, a, b in zip(names, base, outs):
            b = b[0]
            assert torch.equal(a, b), (
                f"mhc_fused_post_pre(norm) {name} row 0 changed at batch size {n}: "
                f"max diff {(a.float() - b.float()).abs().max().item():.3e}"
            )


@skip_if_not_cuda
def test_mhc_fused_post_pre_with_norm_negative_control(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    fn, hc_scale, hc_base, norm_w, victims = _fused_norm_setup("cuda")
    first = _fused_norm_outs(BOUNDARIES[0], victims, 1, fn, hc_scale, hc_base, norm_w)
    base = [o[0].clone() for o in first]
    diffs = []
    for i, n in enumerate(BOUNDARIES[1:], start=2):
        outs = _fused_norm_outs(n, victims, i, fn, hc_scale, hc_base, norm_w)
        if any(not torch.equal(a, b[0]) for a, b in zip(base, outs)):
            diffs.append(n)
    assert diffs, (
        "default mhc_fused_post_pre(norm) was bitwise invariant across all "
        "boundaries; the BI test cannot distinguish fixed from broken"
    )


def _rmsnorm_ref(x, weight, eps):
    xf = x.float()
    return (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).bfloat16()


@skip_if_not_cuda
def test_mhc_fused_post_pre_with_norm_correctness():
    """Pinning the dispatch must not change what the fused-norm path computes.

    The reference is composed from the kernel suite's own post and pre
    references plus an fp32 RMSNorm, so it shares no code with the fused
    implementation.
    """
    device = "cuda"
    fn, hc_scale, hc_base, norm_w, _ = _fused_norm_setup(device)
    for n in (1, 16, 129):
        set_random_seed(n)
        x = torch.randn((n, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            (n, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        post_layer_mix = torch.randn(
            (n, HC_MULT, 1), dtype=torch.float32, device=device
        )
        comb_res_mix = torch.randn(
            (n, HC_MULT, HC_MULT), dtype=torch.float32, device=device
        )

        residual_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
        post_ref, comb_ref, layer_input_ref = mhc_pre_ref(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_ALPHA,
            SINKHORN_REPEAT,
        )
        normed_ref = _rmsnorm_ref(layer_input_ref, norm_w, NORM_EPS)

        out = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_ALPHA,
            SINKHORN_REPEAT,
            norm_weight=norm_w,
            norm_eps=NORM_EPS,
        )
        for actual, expected in zip(
            out, (residual_ref, post_ref, comb_ref, normed_ref), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


def _prenorm_row0(n, victim, seed, fn):
    """Row 0 of the non-DeepGEMM prenorm GEMM for a batch of n rows."""
    from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm

    (batch,) = batch_with_victim((victim,), n, seed)
    x = batch.view(n, HC_MULT * HIDDEN_SIZE)
    out = torch.empty(1, n, HC_MULT3, dtype=torch.float32, device=x.device)
    sqrsum = torch.empty(1, n, dtype=torch.float32, device=x.device)
    _tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, HIDDEN_SIZE, HC_MULT)
    return out[0, 0].clone(), sqrsum[0, 0].clone()


@skip_if_not_cuda
def test_prenorm_gemm_no_deep_gemm_batch_invariance():
    """The fallback GEMM's own dispatch, which DeepGEMM hardware never reaches.

    ``mhc_pre_tilelang`` only calls this helper when DeepGEMM is unavailable,
    so on B200 every test above goes through tf32_hc_prenorm_gemm instead and
    the >= 1024 and < 128 branches stay dark. Call the helper directly.
    """
    device = "cuda"
    fn, _, _ = _mhc_weights(device)
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    base = _prenorm_row0(PRENORM_BOUNDARIES[0], victim, 1, fn)
    for i, n in enumerate(PRENORM_BOUNDARIES[1:], start=2):
        outs = _prenorm_row0(n, victim, i, fn)
        for name, a, b in zip(("mul", "sqrsum"), base, outs):
            assert torch.equal(a, b), (
                f"hc_prenorm_gemm {name} row 0 changed at batch size {n}: "
                f"max diff {(a.float() - b.float()).abs().max().item():.3e}"
            )


@skip_if_not_cuda
def test_prenorm_gemm_no_deep_gemm_negative_control(monkeypatch):
    """Show which fallback branch actually moves a row's bits.

    Only the < 128 one does. The >= 1024 block-M variant keeps the same
    n_thr and tile_n, so it reassociates K in the same order and comes out
    bitwise equal to the generic kernel at DSv4's shape -- it is pinned under
    the flag because that equality is a property of the current tile config,
    not a guarantee, and asserting it here would pin the wrong thing.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    device = "cuda"
    fn, _, _ = _mhc_weights(device)
    victim = torch.randn((1, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    base = _prenorm_row0(128, victim, 1, fn)
    diffs = set()
    for i, n in enumerate(PRENORM_BOUNDARIES, start=2):
        outs = _prenorm_row0(n, victim, i, fn)
        if any(not torch.equal(a, b) for a, b in zip(base, outs)):
            diffs.add(n)
    assert diffs & {1, 64, 127}, (
        f"the < 128 wide-tile branch left row 0 bitwise unchanged (diffs at "
        f"{sorted(diffs)}); the BI test above cannot prove it is pinned"
    )


def test_mhc_refuses_a_deep_gemm_that_cannot_be_pinned(monkeypatch):
    """Fail closed, do not fall back.

    ``tf32_hc_prenorm_gemm`` is DeepGEMM's and the mHC path has no other
    implementation, so a ``deep_gemm`` without ``set_batch_invariant`` would
    keep running with its config chosen from the batch. Disabling DeepGEMM MoE,
    the loader's answer to such a build, does nothing here.
    """
    import vllm.model_executor.kernels.mhc.tilelang as tl
    import vllm.utils.deep_gemm as dg

    guard = tl._require_batch_invariant_deep_gemm
    monkeypatch.setattr(tl.envs, "VLLM_BATCH_INVARIANT", True)

    monkeypatch.setattr(dg, "deep_gemm_batch_invariant_enabled", lambda: True)
    guard.cache_clear()
    guard()  # pinned: no complaint

    monkeypatch.setattr(dg, "deep_gemm_batch_invariant_enabled", lambda: False)
    guard.cache_clear()
    with pytest.raises(RuntimeError, match="set_batch_invariant"):
        guard()

    # With the flag off the same unpinned build is fine -- nothing was promised.
    monkeypatch.setattr(tl.envs, "VLLM_BATCH_INVARIANT", False)
    guard.cache_clear()
    guard()
    guard.cache_clear()
