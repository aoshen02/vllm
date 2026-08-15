# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 sparse MLA decode must not depend on the rest of the batch.

FlashMLA's decode planner sizes each partition's payload as
``ceil_div(total_num_blocks, num_sm_parts)``, so a request's KV is cut at
boundaries that move with the other requests' lengths. On GB200 a pinned row's
output starts changing at batch size 17. ``build_pinned_sched_meta`` replaces
that plan with one that never splits a request.

Kernel-level on purpose: this catches the defect without paying a model load.
"""

import pytest
import torch

from vllm.platforms import current_platform

flashmla = pytest.importorskip(
    "vllm.third_party.flashmla.flash_mla_interface",
    reason="sparse MLA decode needs the FlashMLA extension",
)

from vllm.config import CompilationConfig, SchedulerConfig, VllmConfig  # noqa: E402
from vllm.config.compilation import CUDAGraphMode  # noqa: E402
from vllm.config.vllm import set_current_vllm_config  # noqa: E402
from vllm.v1.attention.backends.mla.sparse_swa import (  # noqa: E402
    build_pinned_sched_meta,
    record_num_sm_parts,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(100),
    reason="DeepSeek-V4 sparse MLA decode is SM100-only",
)

D_QK, D_V = 576, 512
H_Q, H_K = 64, 1
PAGE = 256
TOPK = 512
# sparse_decode.h: 512 fp8 nope + 4 fp32 block scales + 64 bf16 rope.
BYTES_PER_TOKEN = 512 + 64 * 2 + (512 // 128) * 4
# The kernel's own plan changes shape at b=17 on a 152-SM GB200; straddle it.
BATCH_SIZES = [1, 2, 8, 16, 17, 33, 65]
# The split plan is budgeted against max_num_seqs, which the fixture sets to 16.
SPLIT_BATCH_SIZES = [1, 2, 8, 15, 16]


def _kv_cache(num_blocks: int, device: torch.device) -> torch.Tensor:
    shape = (num_blocks, PAGE, H_K)
    nope = torch.randn(*shape, 512, device=device).to(torch.float8_e4m3fn)
    scales = torch.full((*shape, 4), 0.05, dtype=torch.float32, device=device)
    rope = torch.randn(*shape, 64, dtype=torch.bfloat16, device=device)
    kv = torch.empty(*shape, BYTES_PER_TOKEN, dtype=torch.uint8, device=device)
    kv[..., :512] = nope.view(torch.uint8)
    kv[..., 512:528] = scales.view(torch.uint8)
    kv[..., 528:] = rope.view(torch.uint8)
    return kv


def _row(gen: torch.Generator, device: torch.device, kv_tokens: int, valid: int):
    q = torch.randn(1, 1, H_Q, D_QK, dtype=torch.bfloat16, device=device, generator=gen)
    idx = torch.full((1, 1, TOPK), -1, dtype=torch.int32, device=device)
    idx[0, 0, :valid] = torch.randperm(kv_tokens, device=device, generator=gen)[
        :valid
    ].to(torch.int32)
    return q, idx, valid


def _decode(q, kv, idx, topk_len, sched, extra=None):
    """``extra`` mirrors production: V4 always passes a second (cache, indices,
    lengths) triple for the compressed KV alongside the SWA one."""
    extra_kv, extra_idx, extra_len = extra if extra else (None, None, None)
    out, _ = flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=kv,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=D_V,
        tile_scheduler_metadata=sched,
        indices=idx,
        topk_length=topk_len,
        extra_k_cache=extra_kv,
        extra_indices_in_kvcache=extra_idx,
        extra_topk_length=extra_len,
        causal=False,
        is_fp8_kvcache=True,
    )
    return out


def _batch(
    device,
    kv_tokens,
    b: int,
    victim,
    victim_at: int,
    with_extra=False,
    victim_extra=None,
):
    """Batch with ``victim`` at ``victim_at``; fillers get differing lengths so
    the shipped planner sees a different total workload for every batch size."""
    gen = torch.Generator(device=device).manual_seed(9000 + b)
    qs, idxs, lens, extra_idxs, extra_lens = [], [], [], [], []
    for j in range(b):
        if j == victim_at:
            q, idx, n = victim
        else:
            q, idx, n = _row(gen, device, kv_tokens, 32 + (j * 97) % (TOPK - 32))
        qs.append(q)
        idxs.append(idx)
        lens.append(n)
        if with_extra:
            if j == victim_at:
                # The victim's extra top-k must be pinned too, or its output is
                # entitled to change and the test proves nothing.
                _, e_idx, e_n = victim_extra
            else:
                # Filler extra lengths do not track the main ones: the planner
                # rounds the main length up to a whole block before adding the
                # extra one, so independent values exercise that rule.
                extra_valid = 16 + (j * 61) % (TOPK - 16)
                _, e_idx, e_n = _row(gen, device, kv_tokens, extra_valid)
            extra_idxs.append(e_idx)
            extra_lens.append(e_n)
    packed = (
        torch.cat(qs),
        torch.cat(idxs),
        torch.tensor(lens, dtype=torch.int32, device=device),
    )
    if not with_extra:
        return packed, None
    return packed, (
        torch.cat(extra_idxs),
        torch.tensor(extra_lens, dtype=torch.int32, device=device),
    )


def _pinned(topk_len, device, extra_len=None, split=False):
    """``split`` passes the layer's top-k caps, which turns on the plan that cuts
    a request at multiples of its own block count instead of handing it a whole
    partition."""
    sched = build_pinned_sched_meta(
        h_q=H_Q,
        s_q=1,
        topk_length=topk_len,
        extra_topk_length=extra_len,
        has_extra=extra_len is not None,
        device=device,
        topk_cap=TOPK if split else None,
        extra_topk_cap=TOPK if (split and extra_len is not None) else None,
    )
    assert sched is not None, (
        "build_pinned_sched_meta returned None; record_num_sm_parts must observe "
        "one real plan first (the engine's warmup decode does this)"
    )
    if split:
        # Guard against a vacuous pass: if the split budget silently collapsed
        # to one partition per request, this parametrisation would be testing
        # the same thing as split=False.
        meta = sched.tile_scheduler_metadata
        b = topk_len.shape[0]
        busy = int((meta[:, 0] <= meta[:, 1]).sum())
        assert busy > b, (
            f"split plan used {busy} partitions for {b} requests; expected more "
            "than one per request, so the split budget did not take effect"
        )
    return sched


@pytest.fixture(scope="module")
def small_batch_config():
    """Splitting needs two things from config: a max_num_seqs small enough to
    leave partitions to spare (16 gives 152 // 16 = 9), and cudagraphs, without
    which building the plan costs more host time than the split saves on GPU."""
    with set_current_vllm_config(
        VllmConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=16, max_model_len=4096, is_encoder_decoder=False
            ),
            compilation_config=CompilationConfig(
                cudagraph_mode=CUDAGraphMode.PIECEWISE
            ),
        )
    ):
        yield


@pytest.fixture
def bootstrapped():
    """Let the kernel plan once so the partition count is known, as at warmup."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    kv = _kv_cache(64, device)
    extra_kv = _kv_cache(64, device)
    kv_tokens = 64 * PAGE
    gen = torch.Generator(device=device).manual_seed(1)
    q, idx, n = _row(gen, device, kv_tokens, TOPK // 2)
    sched = flashmla.get_mla_metadata()[0]
    _decode(q, kv, idx, torch.tensor([n], dtype=torch.int32, device=device), sched)
    record_num_sm_parts(H_Q, 1, sched)
    return device, kv, extra_kv, kv_tokens


@pytest.mark.parametrize("split", [False, True], ids=["whole_request", "split"])
@pytest.mark.parametrize("with_extra", [False, True], ids=["swa_only", "with_extra_kv"])
def test_pinned_plan_makes_decode_batch_invariant(
    bootstrapped, small_batch_config, with_extra, split
):
    """A row's output must be bitwise identical however the batch is composed.

    ``with_extra_kv`` is the production shape: V4 passes an SWA top-k and a
    compressed-KV top-k together, and the planner rounds the first up to a whole
    block before adding the second. ``split`` covers the plan that cuts a request
    at multiples of its own block count, where the combine kernel really does
    reduce several partials per row.
    """
    device, kv, extra_kv, kv_tokens = bootstrapped
    gen = torch.Generator(device=device).manual_seed(1234)
    victim = _row(gen, device, kv_tokens, TOPK // 2)
    victim_extra = _row(gen, device, kv_tokens, TOPK // 4)

    reference = None
    for b in SPLIT_BATCH_SIZES if split else BATCH_SIZES:
        for victim_at in {0, (b - 1) // 2, b - 1}:
            packed, extra = _batch(
                device, kv_tokens, b, victim, victim_at, with_extra, victim_extra
            )
            q, idx, topk_len = packed
            extra_len = extra[1] if extra else None
            sched = _pinned(topk_len, device, extra_len, split)
            triple = (extra_kv, extra[0], extra[1]) if extra else None
            out = _decode(q, kv, idx, topk_len, sched, triple)
            row = out[victim_at]
            if reference is None:
                reference = row.clone()
            else:
                assert torch.equal(reference, row), (
                    f"pinned row moved at batch={b} position={victim_at}"
                )


@pytest.mark.parametrize("split", [False, True], ids=["whole_request", "split"])
@pytest.mark.parametrize("with_extra", [False, True], ids=["swa_only", "with_extra_kv"])
def test_pinned_plan_does_not_drop_work(
    bootstrapped, small_batch_config, with_extra, split
):
    """Invariance must not come from computing less.

    Compared against the shipped planner on identical input, so the reduction
    order differs by design and the results are close, not bitwise equal.

    The bar is 8 bf16 ULP at the output's own magnitude. Reordering was measured
    at 1-2 ULP; dropping one KV block of four moved the row by ~160 ULP and
    leaving a request unowned by ~64000. The gap is three orders of magnitude,
    so 8 separates the two cleanly with room for a different random draw.
    """
    device, kv, extra_kv, kv_tokens = bootstrapped
    tolerance_ulp = 8 * 2.0**-8
    for b in SPLIT_BATCH_SIZES if split else BATCH_SIZES:
        gen = torch.Generator(device=device).manual_seed(4321 + b)
        victim = _row(gen, device, kv_tokens, TOPK // 2)
        victim_extra = _row(gen, device, kv_tokens, TOPK // 4)
        packed, extra = _batch(
            device, kv_tokens, b, victim, 0, with_extra, victim_extra
        )
        q, idx, topk_len = packed
        extra_len = extra[1] if extra else None
        triple = (extra_kv, extra[0], extra[1]) if extra else None

        shipped = _decode(q, kv, idx, topk_len, flashmla.get_mla_metadata()[0], triple)
        sched = _pinned(topk_len, device, extra_len, split)
        pinned = _decode(q, kv, idx, topk_len, sched, triple)

        expected = shipped.float()
        deviation = (expected - pinned.float()).abs().max().item()
        tolerance = tolerance_ulp * expected.abs().max().item()
        assert deviation <= tolerance, (
            f"batch={b}: pinned plan deviates by {deviation:.3e}, beyond 8 bf16 "
            f"ULP ({tolerance:.3e}) — it is likely skipping KV blocks"
        )


def test_prefill_chunk_plan_batch_invariant():
    """Under BI the prefill chunk plan must emit one request per chunk with
    the same per-request (chunk_N, chunk_M) the greedy planner would compute:
    the sparse-prefill kernel plans its splits from the call geometry, so a
    request's chunk must not depend on which neighbors share the batch."""
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

    def make_meta(seq_lens, query_lens):
        meta = object.__new__(DeepseekSparseSWAMetadata)
        meta.num_prefills = len(seq_lens)
        meta.prefill_seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
        meta.prefill_query_lens_cpu = torch.tensor(query_lens, dtype=torch.int32)
        meta.prefill_window_size = 128
        meta.prefill_max_model_len = 8192
        meta.prefill_max_num_batched_tokens = 8192
        return meta

    seq_lens = [40, 2400, 6, 512]
    query_lens = [40, 2400, 6, 512]
    plan = make_meta(seq_lens, query_lens).get_prefill_chunk_plan(
        compress_ratio=4, prefill_chunk_size=64
    )
    assert [(c[0], c[1]) for c in plan] == [(i, i + 1) for i in range(4)]
    for i, (start, end, chunk_n, chunk_m) in enumerate(plan):
        solo = make_meta(seq_lens[i : i + 1], query_lens[i : i + 1])
        solo_plan = solo.get_prefill_chunk_plan(compress_ratio=4, prefill_chunk_size=64)
        assert solo_plan == [(0, 1, chunk_n, chunk_m)], (
            f"request {i} chunk geometry depends on neighbors"
        )
