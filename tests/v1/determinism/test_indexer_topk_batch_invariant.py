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
    RADIX_TOPK_WORKSPACE_SIZE,
    _topk_indices_batch_invariant,
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
    """Prefill semantics: columns below cu_seqlen_ks are excluded."""
    logits = torch.zeros((1, NUM_COLS), device="cuda", dtype=torch.float32)
    logits[0, :50] = 10.0  # would win top-k if the start bound leaked
    ks = torch.tensor([50], device="cuda", dtype=torch.int32)
    ke = torch.tensor([600], device="cuda", dtype=torch.int32)
    out = torch.empty((1, TOPK), device="cuda", dtype=torch.int32)
    _topk_indices_batch_invariant(logits, ks, ke, out, TOPK)
    valid = out[out >= 0]
    assert valid.min() >= 50 and valid.max() < 600
    assert valid.numel() == 512


def test_cuda_persistent_topk_negative_control():
    """The disease this module guards against: persistent_topk on a tie
    plateau returns a different selected set run to run. If this ever starts
    passing deterministically, the BI branch may be removable — re-evaluate."""
    victim = _victim_row()
    logits, seq_lens = _batch_with_victim(victim, 33, 16, 33)
    sets = set()
    for _ in range(20):
        out = torch.full((33, TOPK), -1, device="cuda", dtype=torch.int32)
        workspace = torch.zeros(
            (RADIX_TOPK_WORKSPACE_SIZE,), device="cuda", dtype=torch.uint8
        )
        torch.ops._C.persistent_topk(
            logits, seq_lens, out, workspace, TOPK, NUM_COLS
        )
        torch.cuda.synchronize()
        kept = out[16][out[16] >= 0]
        sets.add(tuple(kept.sort().values.tolist()))
    assert len(sets) > 1, (
        "persistent_topk resolved ties deterministically 20/20 times; "
        "negative control lost its teeth"
    )
