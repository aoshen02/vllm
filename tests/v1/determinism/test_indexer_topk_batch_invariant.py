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
    assert torch.equal(
        out[0, :100], out[0, :100].sort().values
    ), "valid indices must be ascending"


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
    """The implementation processes rows in chunks of 1024; identical rows on
    both sides of that boundary must produce identical outputs."""
    victim = _victim_row()
    n = 1100
    logits = torch.randn((n, NUM_COLS), device="cuda", dtype=torch.float32)
    seq_lens = torch.full((n,), VICTIM_SEQ, device="cuda", dtype=torch.int32)
    logits[1023] = victim
    logits[1024] = victim
    out = _run_bi_topk(logits, seq_lens)
    assert torch.equal(out[1023], out[1024])


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
        logits = torch.full(
            (1, NUM_COLS), -100.0, device="cuda", dtype=torch.float32
        )
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
                    got = torch.full((rows, TOPK), -7, device="cuda", dtype=torch.int32)
                    ref = torch.full((rows, TOPK), -7, device="cuda", dtype=torch.int32)
                    _topk_indices_batch_invariant(logits, start, end, got, TOPK)
                    _topk_indices_batch_invariant_ref(logits, start, end, ref, TOPK)
                    assert torch.equal(got, ref), (
                        f"rows={rows} cols={cols} kind={kind} start={has_start}"
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
        got = out_buf[:, :TOPK]
        ref = torch.empty((8, TOPK), device="cuda", dtype=torch.int32)
        _topk_indices_batch_invariant(logits, None, end, got, TOPK)
        _topk_indices_batch_invariant_ref(logits, None, end, ref, TOPK)
        assert torch.equal(got, ref), f"cols={cols} stride={logits.stride()}"


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
        got = torch.empty((4, TOPK), device="cuda", dtype=torch.int32)
        ref = torch.empty((4, TOPK), device="cuda", dtype=torch.int32)
        _topk_indices_batch_invariant(logits, None, end, got, TOPK)
        _topk_indices_batch_invariant_ref(logits, None, end, ref, TOPK)
        assert torch.equal(got, ref), f"cols={cols}"


def test_topk_dispatcher_fallback_paths_match_reference():
    """Out-of-envelope inputs (columns beyond the fused kernel's 8192 limit;
    int64 bounds at and beyond int32 range) must route through the fallback
    or the clamped narrowing and still match the reference exactly."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn((3, 8193), generator=gen, device="cuda", dtype=torch.float32)
    end = torch.randint(1, 8194, (3,), generator=gen, device="cuda").to(torch.int32)
    got = torch.empty((3, TOPK), device="cuda", dtype=torch.int32)
    ref = torch.empty((3, TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant(logits, None, end, got, TOPK)
    _topk_indices_batch_invariant_ref(logits, None, end, ref, TOPK)
    assert torch.equal(got, ref)

    logits = torch.randn((2, 640), generator=gen, device="cuda", dtype=torch.float32)
    end64 = torch.tensor([2**31, 2**40], device="cuda", dtype=torch.int64)
    _topk_indices_batch_invariant(logits, None, end64, got[:2], TOPK)
    _topk_indices_batch_invariant_ref(logits, None, end64, ref[:2], TOPK)
    assert torch.equal(got[:2], ref[:2]), "int64 row_end must not wrap in int32"


# NOTE: the negative control for the CUDA kernels (persistent_topk /
# top_k_per_row_{decode,prefill} returning different tie SETS run to run) is
# deliberately NOT an asserted test: nondeterminism is permitted, not
# guaranteed, so asserting it would be flaky by construction. The manual
# probe and its archived evidence live at
# agent_run/scripts/probe_indexer_topk_determinism.py and
# agent_run/results/batch_invariance/shared/indexer-topk/accuracy/.
