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

# The backend accepts compute capability 9 and 10
# (DeepseekV4FlashMLABackend.supports_compute_capability), but the kernel tests
# below were only ever run on SM100 and the numbers quoted in the docstring are
# GB200's. Gate the ones that launch FlashMLA rather than the whole module, so
# the structural and control-flow coverage still runs everywhere -- including on
# the SM90 half of the supported range, where the kernel side is untested.
requires_flashmla_gpu = pytest.mark.skipif(
    not current_platform.is_device_capability(100),
    reason="sparse MLA decode kernel coverage has only been validated on SM100",
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
@requires_flashmla_gpu
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


@requires_flashmla_gpu
def test_cold_shape_cannot_be_pinned_until_a_plan_is_observed():
    """The precondition the cold-start rerun is built on.

    ``_forward_decode`` runs the shipped planner once for an unseen
    ``(h_q, s_q)`` precisely because the partition count is not derivable
    without it, then reruns with the pinned plan so the batch-dependent result
    is never the one returned. That control flow is not exercised here -- it
    needs the model -- but the fact it hinges on is: cold gives None, and only
    observing a real plan makes pinning possible.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import _num_sm_parts_cache

    device = torch.device("cuda:0")
    cold = (H_Q + 7, 3)
    assert cold not in _num_sm_parts_cache
    topk_len = torch.tensor([64], dtype=torch.int32, device=device)
    assert (
        build_pinned_sched_meta(
            h_q=cold[0],
            s_q=cold[1],
            topk_length=topk_len,
            extra_topk_length=None,
            has_extra=False,
            device=device,
        )
        is None
    ), "an unobserved shape must not produce a plan out of thin air"

    sched = flashmla.get_mla_metadata()[0]
    gen = torch.Generator(device=device).manual_seed(7)
    kv = _kv_cache(64, device)
    q, idx, n = _row(gen, device, 64 * PAGE, TOPK // 2)
    _decode(q, kv, idx, torch.tensor([n], dtype=torch.int32, device=device), sched)
    record_num_sm_parts(*cold, sched)

    assert (
        build_pinned_sched_meta(
            h_q=cold[0],
            s_q=cold[1],
            topk_length=topk_len,
            extra_topk_length=None,
            has_extra=False,
            device=device,
        )
        is not None
    ), "after one observed plan the same shape must be pinnable"


@requires_flashmla_gpu
def test_pinned_plan_batch_invariant_above_the_partition_count(bootstrapped):
    """More requests than partitions, on the real kernel.

    ``BATCH_SIZES`` tops out at 65 while a GB200 plans 152 partitions, so every
    test above stays in the regime where each request gets a partition to
    itself. Above the count a partition holds several whole requests, which is
    the branch that replaced the fallback to the shipped planner -- and the only
    coverage it had was a CPU assertion on the skeleton's shape.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import _num_sm_parts_cache

    device, kv, _extra_kv, kv_tokens = bootstrapped
    num_sm_parts = _num_sm_parts_cache[(H_Q, 1)]
    gen = torch.Generator(device=device).manual_seed(99)
    victim = _row(gen, device, kv_tokens, TOPK // 2)

    reference = None
    for b in (num_sm_parts - 1, num_sm_parts, num_sm_parts + 1, 2 * num_sm_parts + 1):
        for victim_at in {0, b - 1}:
            packed, _ = _batch(device, kv_tokens, b, victim, victim_at, False, None)
            q, idx, topk_len = packed
            out = _decode(q, kv, idx, topk_len, _pinned(topk_len, device))
            row = out[victim_at]
            if reference is None:
                reference = row.clone()
            else:
                assert torch.equal(reference, row), (
                    f"pinned row moved at batch={b} (partitions={num_sm_parts}) "
                    f"position={victim_at}"
                )


@requires_flashmla_gpu
def test_grouped_plan_agrees_with_the_shipped_planner(bootstrapped):
    """Above the partition count the grouped plan must still be *right*.

    ``test_pinned_plan_does_not_drop_work`` makes this comparison, but only
    over batch sizes below the partition count, where each request still owns a
    partition to itself. The grouped layout is a shape FlashMLA's own planner
    never emits and whose metadata columns we write by hand, so agreement there
    is what says both that no KV is being skipped and that each column means
    what we think the kernel reads it as -- getting a column wrong would move
    the output.

    Same bar as the sibling test: 8 bf16 ULP separates reduction-order noise
    (1-2) from a dropped KV block (~160).
    """
    from vllm.v1.attention.backends.mla.sparse_swa import _num_sm_parts_cache

    device, kv, _extra_kv, kv_tokens = bootstrapped
    num_sm_parts = _num_sm_parts_cache[(H_Q, 1)]
    tolerance_ulp = 8 * 2.0**-8

    for b in (num_sm_parts + 1, 2 * num_sm_parts + 1):
        gen = torch.Generator(device=device).manual_seed(555 + b)
        victim = _row(gen, device, kv_tokens, TOPK // 2)
        packed, _ = _batch(device, kv_tokens, b, victim, 0, False, None)
        q, idx, topk_len = packed

        shipped = _decode(q, kv, idx, topk_len, flashmla.get_mla_metadata()[0])
        pinned = _decode(q, kv, idx, topk_len, _pinned(topk_len, device))

        expected = shipped.float()
        deviation = (expected - pinned.float()).abs().max().item()
        tolerance = tolerance_ulp * expected.abs().max().item()
        assert deviation <= tolerance, (
            f"grouped plan at batch={b} (partitions={num_sm_parts}) deviates by "
            f"{deviation:.3e}, beyond 8 bf16 ULP ({tolerance:.3e}) -- it is "
            "either skipping KV or writing a metadata column the kernel reads "
            "differently"
        )


@requires_flashmla_gpu
def test_pinned_plan_above_the_partition_count_covers_every_request(bootstrapped):
    """Grouping whole requests must not drop any of them.

    Invariance alone would be satisfied by a plan that quietly stopped at the
    partition count and left the tail unattended, so check the spans as the
    kernel receives them.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import _num_sm_parts_cache

    device, _kv, _extra_kv, kv_tokens = bootstrapped
    num_sm_parts = _num_sm_parts_cache[(H_Q, 1)]
    gen = torch.Generator(device=device).manual_seed(100)
    victim = _row(gen, device, kv_tokens, TOPK // 2)

    b = 2 * num_sm_parts + 1
    packed, _ = _batch(device, kv_tokens, b, victim, 0, False, None)
    meta = _pinned(packed[2], device).tile_scheduler_metadata.cpu()
    busy = meta[meta[:, 0] <= meta[:, 1]]
    covered = torch.cat(
        [torch.arange(int(lo), int(hi) + 1) for lo, hi in busy[:, :2]]
    )
    assert torch.equal(covered, torch.arange(b)), (
        f"partitions covered {covered.tolist()[:8]}... for {b} requests; spans "
        "must be contiguous, disjoint and complete"
    )


@pytest.mark.parametrize("split", [False, True], ids=["whole_request", "split"])
@pytest.mark.parametrize("with_extra", [False, True], ids=["swa_only", "with_extra_kv"])
@requires_flashmla_gpu
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


def _stub_decode_layer(monkeypatch, record_works=True):
    """A DeepseekV4FlashMLAAttention with only what the swa-only decode reads.

    The cold-start control flow lives in _forward_decode: plan missing -> run
    the shipped planner once -> record the partition count from what it left
    behind -> rebuild pinned -> rerun. Nothing about it needs weights or a real
    kernel, so model each participant closely enough that dropping any one of
    them shows up:

    - the fake kernel leaves a shipped plan on the struct, as the real planner
      does on its first call;
    - the fake record only counts as an observation if it was handed that plan;
    - the fake build returns a pinned plan only once an observation happened.

    So removing ``record_num_sm_parts``, or calling it before the kernel, leaves
    the build returning None and the decode failing closed -- which is what the
    real cold start would do.
    """
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.models.deepseek_v4.nvidia import flashmla as fm

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    calls = []
    observed = []
    shipped = torch.zeros(1, dtype=torch.int32)
    tile = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)

    def fake_kernel(**kw):
        meta = kw["tile_scheduler_metadata"]
        calls.append(meta.tile_scheduler_metadata)
        if meta.tile_scheduler_metadata is None:
            # The in-kernel planner allocates the plan on its first call.
            meta.tile_scheduler_metadata = shipped
        kw["out"].fill_(len(calls))
        return kw["out"], None

    def fake_record(h_q, s_q, sched):
        if record_works and sched.tile_scheduler_metadata is not None:
            observed.append((h_q, s_q))

    pinned = SimpleNamespace(
        tile_scheduler_metadata=torch.ones(1, dtype=torch.int32),
        num_splits=torch.ones(2, dtype=torch.int32),
    )

    def fake_build(**kw):
        return pinned if observed else None

    monkeypatch.setattr(fm, "flash_mla_with_kvcache", fake_kernel)
    import vllm.v1.attention.backends.mla.sparse_swa as swa

    monkeypatch.setattr(swa, "build_pinned_sched_meta", fake_build)
    monkeypatch.setattr(swa, "record_num_sm_parts", fake_record)

    layer = object.__new__(fm.DeepseekV4FlashMLAAttention)
    layer.compress_ratio = 1
    layer.scale = 1.0
    layer.attn_sink = None
    layer.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros(1, PAGE, 1), split_budget_cfg=None
    )
    swa_meta = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        decode_swa_indices=torch.zeros(1, 1, TOPK, dtype=torch.int32),
        decode_swa_lens=torch.ones(1, dtype=torch.int32),
        tile_sched_swaonly=tile,
        is_valid_token=None,
        token_to_req_indices=None,
    )
    return layer, swa_meta, calls, tile, shipped, pinned


def test_cold_start_reruns_with_the_pinned_plan(monkeypatch):
    """The shipped planner's output must never be the one that comes back.

    A cold (h_q, s_q) has to run the real planner once because the partition
    count is not observable any other way. The contract is that this first
    result is discarded: the kernel runs a second time with the pinned plan
    installed, and that is what the caller sees.
    """
    layer, swa_meta, calls, tile, shipped, pinned = _stub_decode_layer(monkeypatch)
    out = torch.zeros(1, 64, D_V)

    layer._forward_decode(
        q=torch.zeros(1, 64, D_QK),
        kv_cache=None,
        swa_metadata=swa_meta,
        attn_metadata=None,
        swa_only=True,
        output=out,
    )

    assert len(calls) == 2, (
        f"cold shape ran the kernel {len(calls)} time(s); it must run once to "
        "observe the plan and once more with the pinned one"
    )
    assert calls[0] is None, "the first call must be the shipped planner's"
    assert calls[1] is pinned.tile_scheduler_metadata, (
        "the second call must carry the pinned plan, not the shipped one"
    )
    # Both fields, by identity: installing only the metadata leaves the split
    # counts describing the shipped layout.
    assert tile.tile_scheduler_metadata is pinned.tile_scheduler_metadata
    assert tile.num_splits is pinned.num_splits, (
        "num_splits must be installed too, or the rerun reads the pinned plan "
        "with the shipped plan's split counts"
    )
    assert torch.equal(out, torch.full_like(out, 2.0)), (
        "the returned output is the first kernel call's, i.e. the "
        "batch-dependent one this flag exists to avoid"
    )


def test_cold_start_fails_closed_when_the_plan_stays_unbuildable(monkeypatch):
    """If pinning still fails after the observation, raise rather than return.

    Returning here would hand back the shipped planner's batch-dependent
    output under a flag that promises the opposite, and it would do it
    silently. Driven by making the observation itself fail, which is also what
    dropping record_num_sm_parts from the production path would look like.
    """
    layer, swa_meta, *_ = _stub_decode_layer(monkeypatch, record_works=False)

    with pytest.raises(RuntimeError, match="could not pin the tile-scheduler"):
        layer._forward_decode(
            q=torch.zeros(1, 64, D_QK),
            kv_cache=None,
            swa_metadata=swa_meta,
            attn_metadata=None,
            swa_only=True,
            output=torch.zeros(1, 64, D_V),
        )


@pytest.mark.parametrize("b", [1, 3, 4, 5, 11])
def test_plan_skeleton_groups_whole_requests_for_any_batch(b):
    """Every request is owned by exactly one partition, whatever the batch size.

    With more requests than partitions a partition holds several whole
    requests. What must never happen is a request spanning partitions -- that
    is what would make a row's reduction order depend on the batch. The old
    code refused to plan at all above the partition count and fell back to the
    shipped planner, which is batch-dependent by construction.
    """
    from vllm.v1.attention.backends.mla.sparse_swa import _plan_skeleton

    P = 4
    device = torch.device("cpu")
    meta, (busy, last_owned), num_splits = _plan_skeleton(b, P, device)

    assert meta.shape[0] == P
    # One split per request, cumulative.
    assert torch.equal(num_splits, torch.arange(b + 1, dtype=torch.int32))

    covered = []
    for row, last in zip(busy.tolist(), last_owned.tolist()):
        begin, end = int(meta[row, 0]), int(meta[row, 1])
        assert begin <= end, "a busy partition must own at least one request"
        assert last == end, "the caller writes the block count of the last request"
        covered.extend(range(begin, end + 1))
    assert covered == list(range(b)), (
        f"requests must be covered once, contiguously and in order: {covered}"
    )

    idle = sorted(set(range(P)) - set(busy.tolist()))
    for row in idle:
        assert int(meta[row, 0]) > int(meta[row, 1]), "idle rows stay begin > end"

    # Nothing is ever marked as split.
    assert int(meta[:, 2].abs().sum()) == 0
    assert int(meta[:, 4:7].abs().sum()) == 0


def test_c128a_topk_width_is_not_sized_from_the_batch():
    """The C128A top-k width must not follow this step's longest sequence.

    It reaches the kernel as the split granularity, so deriving it from
    cm.max_seq_len lets a long neighbour push a request into a different
    power-of-two bucket and change how its own work is cut.

    Structural: building real C128A metadata needs a populated metadata builder
    and a KV cache spec. What is asserted is that the batch-invariant branch
    binds the width to the configured maximum and never reads the batch-wide
    length -- which is what would regress if someone folded the branches back
    together.
    """
    import inspect

    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadataBuilder

    src = inspect.getsource(DeepseekV4FlashMLAMetadataBuilder._build_c128a_metadata)
    start = src.index("VLLM_BATCH_INVARIANT")
    bi_branch = src[start : src.index("else:", start)]
    # Comments in that branch explain what it is avoiding, so look at code only.
    code = "\n".join(
        line for line in bi_branch.splitlines() if not line.strip().startswith("#")
    )
    assert "active_topk_width = self.c128a_max_compressed" in code
    assert "max_seq_len" not in code, (
        "the batch-invariant width must not read this step's max_seq_len"
    )
