# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 attention with a fixed split-KV schedule (batch invariance)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

BLOCK = 1024
NUM_HEADS, NUM_KV_HEADS, HEAD = 32, 2, 128


@pytest.mark.parametrize(
    "head_size,kv_cache_dtype,sliding_window,max_model_len,reason",
    [
        (128, "fp8_e4m3", None, 16384, None),
        (64, "fp8_e4m3", None, 16384, "head_size"),
        (128, "auto", None, 16384, "FP8 E4M3"),
        (128, "fp8_e4m3", 4096, 16384, "sliding-window"),
        (128, "fp8_e4m3", None, 16385, "max_model_len"),
    ],
)
def test_fixed_schedule_rejects_what_it_cannot_serve(
    monkeypatch, head_size, kv_cache_dtype, sliding_window, max_model_len, reason
):
    """Unsupported layers get a reason, which FlashAttnFixedSplitImpl raises at
    construction instead of exceeding the schedule at runtime."""
    from vllm.v1.attention.backends import flash_attn_fixed_split

    monkeypatch.setattr(
        flash_attn_fixed_split.current_platform,
        "is_device_capability_family",
        lambda family, device_id=0: family == 100,
    )
    actual = flash_attn_fixed_split.fixed_split_unsupported_reason(
        head_size, kv_cache_dtype, sliding_window, max_model_len
    )
    if reason is None:
        assert actual is None
    else:
        assert reason in actual


def _make_impl(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_fixed_split

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=flash_attn_fixed_split.MAX_SEQ_LEN),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    monkeypatch.setattr(
        flash_attn_fixed_split, "get_current_vllm_config", lambda: config
    )
    return flash_attn_fixed_split.FlashAttnFixedSplitImpl(
        NUM_HEADS, HEAD, HEAD**-0.5, NUM_KV_HEADS, None, None, "fp8_e4m3"
    )


def _run(impl, layer, queries, cache, pages, seq_lens):
    """Attend the given query rows of each sequence over its first seq_lens keys."""
    q = torch.cat(queries)
    cu = [0]
    for rows in queries:
        cu.append(cu[-1] + rows.shape[0])
    table = torch.zeros(len(pages), max(map(len, pages)), dtype=torch.int32)
    for i, row in enumerate(pages):
        table[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
    metadata = SimpleNamespace(
        use_cascade=False,
        causal=True,
        max_query_len=max(rows.shape[0] for rows in queries),
        max_seq_len=max(seq_lens),
        num_actual_tokens=q.shape[0],
        query_start_loc=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        block_table=table.cuda(),
    )
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    # The impl reads the cache as (blocks, kv_heads, block, K|V).
    return impl.forward(
        layer, q, None, None, cache.transpose(1, 2), metadata, out
    ).clone()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Fixed-schedule FA4 attention requires SM10x",
)
@torch.inference_mode()
def test_fixed_schedule_rows_ignore_batch_and_phase(monkeypatch):
    """A sequence's output rows are bitwise the same alone, next to other
    sequences, and as a one-token decode after its prefill."""
    impl = _make_impl(monkeypatch)
    gen = torch.Generator(device="cuda").manual_seed(0)
    lengths = [2000, 700, 3100]
    first_page = [0, 2, 3]
    cache = torch.zeros(
        7, BLOCK, NUM_KV_HEADS, 2 * HEAD, dtype=torch.float8_e4m3fn, device="cuda"
    )
    queries, pages = [], []
    for length, start in zip(lengths, first_page):
        kv = torch.randn(length, NUM_KV_HEADS, 2 * HEAD, generator=gen, device="cuda")
        row_pages = list(range(start, start + -(-length // BLOCK)))
        for j, page in enumerate(row_pages):
            chunk = kv[j * BLOCK : (j + 1) * BLOCK]
            cache[page, : chunk.shape[0]] = chunk.to(torch.float8_e4m3fn)
        q = torch.randn(length, NUM_HEADS, HEAD, generator=gen, device="cuda")
        queries.append(q.to(torch.float8_e4m3fn))
        pages.append(row_pages)
    layer = SimpleNamespace(
        **{
            name: torch.tensor([[value]], device="cuda")
            for name, value in (
                ("_q_scale", 0.0625),
                ("_k_scale", 0.03125),
                ("_v_scale", 0.046875),
            )
        }
    )

    alone = _run(impl, layer, queries[:1], cache, pages[:1], lengths[:1])
    batched = _run(impl, layer, queries, cache, pages, lengths)
    assert torch.equal(alone, batched[: lengths[0]])

    last_rows = [q[-1:] for q in queries]
    decode = _run(impl, layer, last_rows, cache, pages, lengths)
    ends = torch.tensor(lengths).cumsum(0) - 1
    assert torch.equal(decode, batched[ends.cuda()])
