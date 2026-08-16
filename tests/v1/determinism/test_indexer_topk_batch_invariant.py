# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the sparse-indexer top-k under VLLM_BATCH_INVARIANT.

All three CUDA top-k kernels (cooperative/persistent/per-row) resolve exact
score ties through shared-memory atomics, so on tie plateaus — routine for
``relu``-based DSA scores — both the selected set and its order change run to
run. ``_topk_indices_batch_invariant`` replaces them under the flag with a
stable sort whose result depends only on the row's own contents.
"""

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import (
    _topk_indices_batch_invariant,
    _topk_indices_batch_invariant_ref,
)

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

TOPK = 512
NUM_COLS = 2048
VICTIM_SEQ = 1500
N_POS = 300  # fewer positives than TOPK forces the cutoff into the 0.0 plateau
PLATEAU = 700


def _victim_row() -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(7)
    row = torch.full((NUM_COLS,), -100.0, device="cuda", dtype=torch.float32)
    perm = torch.randperm(VICTIM_SEQ, generator=gen, device="cuda")
    row[perm[:N_POS]] = (
        torch.arange(1, N_POS + 1, device="cuda", dtype=torch.float32) / 64.0
    )
    row[perm[N_POS : N_POS + PLATEAU]] = 0.0
    return row


def _batch_with_victim(
    victim: torch.Tensor, num_rows: int, victim_at: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        (num_rows, NUM_COLS), generator=gen, device="cuda", dtype=torch.float32
    )
    seq_lens = torch.randint(
        200, NUM_COLS, (num_rows,), generator=gen, device="cuda", dtype=torch.int32
    )
    logits[victim_at] = victim
    seq_lens[victim_at] = VICTIM_SEQ
    return logits, seq_lens


def _run_bi_topk(logits: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    out = torch.empty((logits.shape[0], TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant(logits, None, seq_lens, out, TOPK)
    return out


def _assert_matches_ref(logits, start, end, msg: str, got=None) -> None:
    """Run the dispatcher and the torch reference on the same input and
    require bitwise-equal output. The buffers get distinct sentinels: both
    paths must write every slot, so a slot either path skips mismatches
    even if the other path skips it too."""
    ref = torch.full((logits.shape[0], TOPK), -8, device="cuda", dtype=torch.int32)
    if got is None:
        got = torch.full_like(ref, -7)
    else:
        got.fill_(-7)
    _topk_indices_batch_invariant(logits, start, end, got, TOPK)
    _topk_indices_batch_invariant_ref(logits, start, end, ref, TOPK)
    assert torch.equal(got, ref), msg


def test_topk_batch_invariant_deterministic_and_invariant():
    """Same victim row -> bitwise-identical indices, for repeated runs and for
    every batch composition around the CUDA kernels' dispatch boundaries."""
    victim = _victim_row()
    reference = None
    for num_rows in (1, 8, 32, 33, 64):
        victim_at = num_rows // 2
        logits, seq_lens = _batch_with_victim(victim, num_rows, victim_at, num_rows)
        outs = [_run_bi_topk(logits, seq_lens)[victim_at] for _ in range(5)]
        for o in outs[1:]:
            assert torch.equal(o, outs[0]), f"nondeterministic at rows={num_rows}"
        if reference is None:
            reference = outs[0]
        assert torch.equal(outs[0], reference), f"batch-variant at rows={num_rows}"


def test_topk_batch_invariant_matches_reference():
    """Selected set must be score-descending / index-ascending-on-ties,
    emitted ascending — checked against a pure-Python reference."""
    victim = _victim_row()
    logits, seq_lens = _batch_with_victim(victim, 4, 1, 42)
    got = _run_bi_topk(logits, seq_lens)[1].tolist()

    row = victim.tolist()
    ranked = sorted(range(VICTIM_SEQ), key=lambda i: (-row[i], i))[:TOPK]
    expect = sorted(ranked)
    assert got == expect


def test_topk_batch_invariant_short_row_padding():
    """Rows with fewer candidates than top-k pad with trailing -1, matching
    the _fill_short_context_topk_indices convention."""
    logits = torch.randn((2, NUM_COLS), device="cuda", dtype=torch.float32)
    seq_lens = torch.tensor([100, 900], device="cuda", dtype=torch.int32)
    out = _run_bi_topk(logits, seq_lens)
    assert (out[0, :100] >= 0).all() and (out[0, :100] < 100).all()
    assert (out[0, 100:] == -1).all()
    assert torch.equal(out[0, :100], out[0, :100].sort().values), (
        "valid indices must be ascending"
    )


def test_topk_batch_invariant_prefill_row_start():
    """Prefill contract: columns below cu_seqlen_ks are excluded and emitted
    indices are row-local (relative to ks), matching the CUDA prefill kernel
    (sampler.cu writes ``rowIt``, not ``rowIt + rowStart``)."""
    logits = torch.zeros((1, NUM_COLS), device="cuda", dtype=torch.float32)
    logits[0, :50] = 10.0  # would win top-k if the start bound leaked
    ks = torch.tensor([50], device="cuda", dtype=torch.int32)
    ke = torch.tensor([600], device="cuda", dtype=torch.int32)
    out = torch.empty((1, TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant(logits, ks, ke, out, TOPK)
    valid = out[out >= 0]
    assert valid.min() >= 0 and valid.max() < 550
    assert valid.numel() == 512


def test_topk_batch_invariant_narrow_logits():
    """Prefill chunks can carry fewer candidate columns than top-k; the CUDA
    kernels fill the remainder with -1 (TP4 hit this live with 30 columns)."""
    cols = 30
    logits = torch.randn((2, cols), device="cuda", dtype=torch.float32)
    ke = torch.tensor([cols, 20], device="cuda", dtype=torch.int32)
    out = torch.empty((2, TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant(logits, None, ke, out, TOPK)
    assert (out[0, :cols] >= 0).all() and (out[0, cols:] == -1).all()
    assert (out[1, :20] >= 0).all() and (out[1, 20:] == -1).all()


def test_topk_batch_invariant_row_chunk_boundary():
    """The torch reference processes rows in chunks of 1024; identical rows on
    both sides of that boundary must produce identical outputs on the fused
    path and on the reference itself."""
    victim = _victim_row()
    n = 1100
    logits = torch.randn((n, NUM_COLS), device="cuda", dtype=torch.float32)
    seq_lens = torch.full((n,), VICTIM_SEQ, device="cuda", dtype=torch.int32)
    logits[1023] = victim
    logits[1024] = victim
    out = _run_bi_topk(logits, seq_lens)
    assert torch.equal(out[1023], out[1024])
    ref = torch.empty((n, TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant_ref(logits, None, seq_lens, ref, TOPK)
    assert torch.equal(ref[1023], ref[1024])


def test_topk_batch_invariant_narrow_with_row_start():
    """Narrow candidate width combined with a nonzero cu_seqlen_ks: row-local
    coords plus the -1 tail must both survive."""
    cols = 40
    logits = torch.zeros((1, cols), device="cuda", dtype=torch.float32)
    logits[0, :10] = 5.0  # below ks; must be excluded
    ks = torch.tensor([10], device="cuda", dtype=torch.int32)
    ke = torch.tensor([cols], device="cuda", dtype=torch.int32)
    out = torch.empty((1, TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant(logits, ks, ke, out, TOPK)
    valid = out[0][out[0] >= 0]
    assert valid.numel() == 30 and valid.min() >= 0 and valid.max() < 30
    assert (out[0, 30:] == -1).all()


def test_topk_batch_invariant_prefill_shifted_ks():
    """The same row content packed at different cu_seqlen_ks offsets (i.e.
    different preceding requests in the workspace) must give identical
    row-local output — absolute-column emit would break exactly here."""
    seg = 550
    content = _victim_row()[:seg]
    outs = []
    for ks_val in (0, 50, 300):
        logits = torch.full((1, NUM_COLS), -100.0, device="cuda", dtype=torch.float32)
        logits[0, ks_val : ks_val + seg] = content
        ks = torch.tensor([ks_val], device="cuda", dtype=torch.int32)
        ke = torch.tensor([ks_val + seg], device="cuda", dtype=torch.int32)
        out = torch.empty((1, TOPK), device="cuda", dtype=torch.int32)
        _topk_indices_batch_invariant(logits, ks, ke, out, TOPK)
        outs.append(out[0])
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def _adversarial_content(
    kind: str, rows: int, cols: int, gen: torch.Generator
) -> torch.Tensor:
    x = torch.randn((rows, cols), generator=gen, device="cuda", dtype=torch.float32)
    if kind == "randn":
        return x
    if kind == "plateau":  # relu-style exact-0.0 plateau straddling the cutoff
        x = torch.relu(x)
        x[x < 1.2] = 0.0
        return x
    if kind == "quantized":  # dense exact ties at many value levels
        return (x * 4).round() / 4
    if kind == "signed_zero":  # -0.0 vs +0.0 must compare equal, like torch.sort
        # Guarantee a mixed +-0.0 plateau inside every row that straddles
        # the top-k cutoff: few positives, the rest alternating zero signs.
        x = torch.relu(x)
        x[x < 2.0] = 0.0
        x[:, 1::2] = torch.where(
            x[:, 1::2] == 0, torch.tensor(-0.0, device="cuda"), x[:, 1::2]
        )
        return x
    if kind == "denormal":  # keying must not flush denormals
        return x * 1e-40
    raise ValueError(kind)


def test_topk_triton_matches_torch_reference_bitwise():
    """The Triton fast path must be bitwise-equal to the torch stable-sort
    reference on adversarial contents: tie plateaus at the cutoff, dense
    quantized ties, mixed-sign zeros, denormals, empty / full / start==end
    rows, and narrow chunks, with and without ``cu_seqlen_ks`` offsets."""
    seed = 0
    for rows in (1, 3, 64, 257, 1100):
        for cols in (30, 512, 2048, 8192):
            for kind in ("randn", "plateau", "quantized", "signed_zero", "denormal"):
                for has_start in (False, True):
                    seed += 1
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    logits = _adversarial_content(kind, rows, cols, gen)
                    end = torch.randint(
                        0, cols + 1, (rows,), generator=gen, device="cuda"
                    ).to(torch.int32)
                    end[0] = cols
                    if rows > 1:
                        end[1] = 0
                    start = None
                    if has_start:
                        start = (
                            torch.rand((rows,), generator=gen, device="cuda")
                            * end.clamp(min=1)
                        ).to(torch.int32)
                        if rows > 2:
                            start[2] = end[2]
                    _assert_matches_ref(
                        logits,
                        start,
                        end,
                        f"rows={rows} cols={cols} kind={kind} start={has_start}",
                    )


def test_topk_triton_matches_reference_noncontiguous():
    """Sliced-column (padded row stride), strided-element (stride 1 != 1),
    and padded-output layouts must all go through the strided load/store
    paths and stay bitwise-equal to the reference."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    big = torch.randn((8, 4096), generator=gen, device="cuda", dtype=torch.float32)
    out_buf = torch.empty((8, TOPK + 88), device="cuda", dtype=torch.int32)
    for logits in (big[:, :1500], big[:, ::2]):
        cols = logits.shape[1]
        end = torch.randint(1, cols + 1, (8,), generator=gen, device="cuda").to(
            torch.int32
        )
        _assert_matches_ref(
            logits,
            None,
            end,
            f"cols={cols} stride={logits.stride()}",
            got=out_buf[:, :TOPK],
        )


def test_topk_triton_bucket_boundaries_and_degenerate_rows():
    """NCOLS power-of-two bucket edges, all-equal rows, all--inf valid
    windows, and single/zero-column shapes must match the reference."""
    for cols in (1, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4097):
        gen = torch.Generator(device="cuda").manual_seed(cols)
        logits = torch.randn(
            (4, cols), generator=gen, device="cuda", dtype=torch.float32
        )
        logits[1] = 0.25  # fully tied row
        logits[2] = float("-inf")  # valid window entirely -inf
        end = torch.randint(0, cols + 1, (4,), generator=gen, device="cuda").to(
            torch.int32
        )
        end[0] = cols
        _assert_matches_ref(logits, None, end, f"cols={cols}")


def test_topk_dispatcher_fallback_paths_match_reference():
    """Out-of-envelope inputs (columns beyond the fused kernel's 8192 limit;
    int64 bounds at and beyond int32 range) must route through the fallback
    or the clamped narrowing and still match the reference exactly."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn((3, 8193), generator=gen, device="cuda", dtype=torch.float32)
    end = torch.randint(1, 8194, (3,), generator=gen, device="cuda").to(torch.int32)
    _assert_matches_ref(logits, None, end, "8193-column fallback diverged")

    logits = torch.randn((2, 640), generator=gen, device="cuda", dtype=torch.float32)
    end64 = torch.tensor([2**31, 2**40], device="cuda", dtype=torch.int64)
    _assert_matches_ref(logits, None, end64, "int64 row_end must not wrap in int32")


# NOTE: the negative control for the CUDA kernels (persistent_topk /
# top_k_per_row_{decode,prefill} returning different tie SETS run to run) is
# deliberately NOT an asserted test: nondeterminism is permitted, not
# guaranteed, so asserting it would be flaky by construction. The manual
# probe and its archived evidence live at
# agent_run/scripts/probe_indexer_topk_determinism.py and
# agent_run/results/batch_invariance/shared/indexer-topk/accuracy/.


@pytest.mark.parametrize("all_neg_inf", [True, False])
def test_masked_lanes_do_not_spend_the_tie_budget(all_neg_inf):
    """The same logical row must select the same local indices wherever the
    packer puts its window.

    Out-of-window lanes are excluded from the output, but they still have a
    key. Keying them on ``-inf`` ties them with in-window lanes whose score
    really is ``-inf``; because the tie budget is spent in column order, the
    lanes sitting before ``row_start`` would eat it and the number of emitted
    indices would drop by exactly ``row_start``. That is a direct batch
    dependence: ``row_start`` is where this request landed in the packed
    prefill, i.e. a function of its neighbours.

    Comparing against the torch reference cannot catch this on its own -- the
    reference has to make the same choice, so both would be wrong together.
    Assert the invariant itself instead.
    """
    width = 600
    outs = []
    for start in (0, 1, 100, 511, 512, 977):
        cols = start + width
        row = torch.full((1, cols), float("-inf"), device="cuda", dtype=torch.float32)
        if not all_neg_inf:
            # Mixed: a handful of finite scores, the rest of the window -inf,
            # so the cutoff still lands on the -inf plateau.
            row[0, start : start + 5] = torch.arange(
                5, device="cuda", dtype=torch.float32
            )
        row_start = torch.tensor([start], device="cuda", dtype=torch.int32)
        row_end = torch.tensor([cols], device="cuda", dtype=torch.int32)
        out = torch.empty((1, TOPK), device="cuda", dtype=torch.int32)
        _topk_indices_batch_invariant(row, row_start, row_end, out, TOPK)
        outs.append((start, out.clone()))

        ref = torch.empty((1, TOPK), device="cuda", dtype=torch.int32)
        _topk_indices_batch_invariant_ref(row, row_start, row_end, ref, TOPK)
        assert torch.equal(out, ref), f"kernel != reference at start={start}"

    base_start, base = outs[0]
    n_emitted = int((base >= 0).sum())
    assert n_emitted == min(TOPK, width), (
        f"expected {min(TOPK, width)} emitted indices, got {n_emitted}"
    )
    for start, got in outs[1:]:
        assert torch.equal(got, base), (
            f"row_start={start} selected different indices than "
            f"row_start={base_start}: masked lanes are consuming the tie budget"
        )


def test_topk_sentinel_collision_and_narrow_windows_match_reference():
    """Bit patterns and shapes where the out-of-window sentinel could leak.

    A negative NaN maps to exactly the INT32_MIN sentinel, so an in-window lane
    can collide with the out-of-window ones. The kernel keeps them apart by
    restricting its tie scan to in-window lanes; the reference by ranking on
    validity before key. Also covers a window narrower than top-k inside a
    wider row, where the cutoff lands on the sentinel.
    """
    # -1 and -2 as int32 are 0xffffffff and 0xfffffffe: two distinct negative
    # NaNs whose keys are exactly INT32_MIN and INT32_MIN + 1 -- the only two
    # patterns adjacent to the sentinel, and the reason no value can be reserved
    # as one (lifting keys off it would merge these two).
    neg_nan = torch.tensor([-1], dtype=torch.int32).view(torch.float32).item()
    neg_nan2 = torch.tensor([-2], dtype=torch.int32).view(torch.float32).item()

    cols = 601
    row = torch.full((1, cols), neg_nan, device="cuda", dtype=torch.float32)
    start = torch.tensor([1], device="cuda", dtype=torch.int32)
    end = torch.tensor([cols], device="cuda", dtype=torch.int32)
    _assert_matches_ref(row, start, end, "negative-NaN plateau vs sentinel")

    # Both sentinel-adjacent patterns in one window: their keys differ by one,
    # so any scheme that clamps in-window keys off the sentinel would tie them.
    row = torch.full((1, cols), neg_nan, device="cuda", dtype=torch.float32)
    row[0, 1::2] = neg_nan2
    _assert_matches_ref(row, start, end, "two distinct negative NaNs in-window")

    # valid_count (30) < k_eff (512) while num_cols (630) > TOPK.
    gen = torch.Generator(device="cuda").manual_seed(31)
    row = torch.randn((1, 630), generator=gen, device="cuda", dtype=torch.float32)
    start = torch.tensor([600], device="cuda", dtype=torch.int32)
    end = torch.tensor([630], device="cuda", dtype=torch.int32)
    _assert_matches_ref(row, start, end, "narrow window inside a wide row")


# Written as raw int32 so the patterns survive: float32 -> Python float quiets
# signaling NaNs, so `.item()` round-trips cannot construct them.
_BIT_PATTERNS = [
    ("pos_zero_neg_zero", [0x00000000, 0x80000000]),
    ("infinities", [0x7F800000, 0xFF800000]),
    ("quiet_nans", [0x7FC00000, 0xFFC00000]),
    ("signaling_nans", [0x7F800001, 0xFF800001]),
    ("min_denormals", [0x00000001, 0x80000001]),
    ("max_denormals", [0x007FFFFF, 0x807FFFFF]),
    ("sentinel_adjacent", [0xFFFFFFFF, 0xFFFFFFFE]),
    (
        "everything",
        [
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0x7F800001,
            0xFF800001,
            0x00000001,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0xFFFFFFFF,
            0xFFFFFFFE,
        ],
    ),
]


def _row_from_bits(bits: list[int], cols: int) -> torch.Tensor:
    as_signed = [b - 2**32 if b >= 2**31 else b for b in bits]
    raw = torch.tensor(as_signed, dtype=torch.int32, device="cuda")
    row = raw.repeat((cols + len(bits) - 1) // len(bits))[:cols]
    return row.view(torch.float32).unsqueeze(0)


@pytest.mark.parametrize("name,bits", _BIT_PATTERNS, ids=[n for n, _ in _BIT_PATTERNS])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_topk_every_float_class_matches_reference(name, bits, topk):
    """Kernel and reference must agree on every fp32 class, not just on scores.

    Ranking is done on a monotone integer key, so the classes that matter are
    the bit patterns near its edges: the sentinel-adjacent NaNs, the signed
    zeros the key canonicalizes, the infinities that bound it, and denormals.
    Signaling NaNs are included because nothing in the pipeline quiets them.
    """
    # The last shape keeps the in-window count above every topk under test, so
    # the selection actually truncates instead of emitting the whole window.
    for start, width in ((0, 600), (1, 600), (600, 30), (1, 2600)):
        cols = start + width
        row = _row_from_bits(bits, cols)
        row_start = torch.tensor([start], device="cuda", dtype=torch.int32)
        row_end = torch.tensor([cols], device="cuda", dtype=torch.int32)
        out = torch.empty((1, topk), device="cuda", dtype=torch.int32)
        ref = torch.empty((1, topk), device="cuda", dtype=torch.int32)
        _topk_indices_batch_invariant(row, row_start, row_end, out, topk)
        _topk_indices_batch_invariant_ref(row, row_start, row_end, ref, topk)
        assert torch.equal(out, ref), (
            f"{name}: kernel != reference at start={start} width={width} topk={topk}"
        )
