# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    _layer_mapping,
    _opaque_fallback_mapping,
    _RankContext,
    _verify_tiling,
    derive_canonical_mappings,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVQuantMode,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

NUM_BLOCKS = 3


def _ctx(rank, tp=1, dcp=1, pcp=1, interleave=1, total=None, heads=2):
    return _RankContext(
        tp_size=tp,
        dcp_size=dcp,
        pcp_size=pcp,
        interleave=interleave,
        total_kv_heads=heads * tp if total is None else total,
        rank=rank,
    )


def _full_spec(num_kv_heads: int = 2, **kwargs) -> FullAttentionSpec:
    # block_size=4, head_dim=64, int8
    return TritonAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=num_kv_heads,
            head_size=64,
            dtype=torch.int8,
            **kwargs,
        )
    )


def _mla_spec(**kwargs) -> MLAAttentionSpec:
    # page = 4 * 64 = 256B, one 64B latent row per token
    return MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=64, dtype=torch.int8, **kwargs
    )


def _split_nhd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    )


def _split_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.num_kv_heads,
        spec.block_size,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 1, 3, 2, 4)


def _packed_nhd_cache(spec) -> torch.Tensor:
    """Logical (num_blocks, heads, block_size, 2 * head_size) over an NHD
    physical layout — the FlashAttention/FlashInfer/Triton/Flex form."""
    return torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        spec.num_kv_heads,
        2 * spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3)


def _packed_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        spec.num_kv_heads,
        spec.block_size,
        2 * spec.head_size,
        dtype=torch.int8,
    )


CACHE_BUILDERS = {
    "split_nhd": _split_nhd_cache,
    "split_hnd": _split_hnd_cache,
    "packed_nhd": _packed_nhd_cache,
    "packed_hnd": _packed_hnd_cache,
}


def _try_mapping(spec, kv_cache, ctx) -> CanonicalPageMapping | None:
    return _layer_mapping(spec, kv_cache, NUM_BLOCKS, ctx)


def _mapping(spec, kv_cache, ctx) -> CanonicalPageMapping:
    mapping = _try_mapping(spec, kv_cache, ctx)
    assert mapping is not None
    return mapping


def _triples(runs: tuple[CopyRun, ...]) -> list[tuple[int, int, int]]:
    """Expand runs to explicit (local_offset, canonical_offset, size) copies."""
    out = []
    for run in runs:
        for i in range(run.num_fragments):
            out.append(
                (
                    run.local_offset + i * run.local_stride,
                    run.canonical_offset + i * run.canonical_stride,
                    run.fragment_size,
                )
            )
    return out


# ---------------------------------------------------------------------------
# TP-only placement (byte-compatible with the uniform interleave layout)
# ---------------------------------------------------------------------------


def test_split_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _split_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert mapping.canonical_page_size_bytes == 4 * 1024
    assert mapping.parallelism_agnostic
    k_dst = [256, 768, 1280, 1792]
    assert _triples(mapping.runs) == [
        (local, canonical, 128)
        for local, canonical in zip(
            [0, 128, 256, 384, 512, 640, 768, 896],
            k_dst + [2048 + o for o in k_dst],
        )
    ]
    # Heads are sharded, not replicated: this rank writes every block
    assert mapping.num_writers == 1


def test_packed_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert _triples(mapping.runs) == [
        (0, 512, 256),
        (256, 1536, 256),
        (512, 2560, 256),
        (768, 3584, 256),
    ]


def test_packed_hnd_placement_rank1_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_hnd_cache(spec), _ctx(rank=1, tp=4))
    # Two heads, each a contiguous canonical head region of 4 tokens x 128B
    assert _triples(mapping.runs) == [(0, 1024, 1024)]


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_single_rank_coalesces_to_one_run(form):
    spec = _full_spec()
    mapping = _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank=0))
    assert mapping.canonical_page_size_bytes == 1024
    assert _triples(mapping.runs) == [(0, 0, 1024)]


# ---------------------------------------------------------------------------
# Replication and writer election
# ---------------------------------------------------------------------------


def test_gqa_replicated_heads_rotate_writer():
    # total 2 KV heads on tp=4: replication factor 2, head shard = rank // 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, total=2)  # noqa: E731
    rank2 = _mapping(spec, cache, ctx(2))
    rank3 = _mapping(spec, cache, ctx(3))
    assert rank2.canonical_page_size_bytes == 2 * 512
    # K region: head shard 1 at 64B offsets within 128B token rows
    assert _triples(rank2.runs)[:4] == [
        (0, 64, 64),
        (64, 192, 64),
        (128, 320, 64),
        (192, 448, 64),
    ]
    # Same head shard, so identical bytes: the two take alternate blocks
    assert rank3.runs == rank2.runs
    assert (rank2.is_writer(0), rank3.is_writer(0)) == (True, False)
    assert (rank2.is_writer(1), rank3.is_writer(1)) == (False, True)
    _verify_tiling("gqa", [_mapping(spec, cache, ctx(r)) for r in range(4)])


def test_mla_replicas_rotate_writer():
    spec = _mla_spec()
    rank0 = _mapping(spec, None, _ctx(rank=0, tp=2))
    rank1 = _mapping(spec, None, _ctx(rank=1, tp=2))
    # Latent pages are stored once per block, not once per rank
    assert rank0.canonical_page_size_bytes == 256
    assert _triples(rank0.runs) == [(0, 0, 256)]
    assert rank1.runs == rank0.runs
    assert (rank0.is_writer(0), rank1.is_writer(0)) == (True, False)
    assert (rank0.is_writer(1), rank1.is_writer(1)) == (False, True)


# ---------------------------------------------------------------------------
# DCP / PCP token sharding
# ---------------------------------------------------------------------------


def test_dcp_interleaves_tokens_within_replicas():
    # tp=4, dcp=2, total 2 KV heads: head shard = rank // 2, cp rank = rank % 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, dcp=2, total=2)  # noqa: E731
    per_rank = [_mapping(spec, cache, ctx(rank)) for rank in range(4)]
    assert all(m is not None for m in per_rank)
    # 8 canonical tokens x 2 heads x 64B per region
    assert per_rank[0].canonical_page_size_bytes == 2048
    assert not per_rank[0].parallelism_agnostic
    # rank 2 = head shard 1, cp rank 0: K tokens 0,2,4,6 at head offset 64
    assert _triples(per_rank[2].runs)[:4] == [
        (0, 64, 64),
        (64, 320, 64),
        (128, 576, 64),
        (192, 832, 64),
    ]
    # every rank contributes (dcp == replication: no residual replicas)
    assert all(m.num_writers == 1 for m in per_rank)
    _verify_tiling("dcp", per_rank)


def test_mla_dcp_shards_latent_tokens():
    spec = _mla_spec()
    ctx = lambda rank: _ctx(rank, tp=2, dcp=2)  # noqa: E731
    rank0 = _mapping(spec, None, ctx(0))
    rank1 = _mapping(spec, None, ctx(1))
    assert rank0.canonical_page_size_bytes == 512
    assert _triples(rank0.runs) == [(o, 2 * o, 64) for o in (0, 64, 128, 192)]
    assert _triples(rank1.runs) == [(o, 2 * o + 64, 64) for o in (0, 64, 128, 192)]
    _verify_tiling("mla-dcp", [rank0, rank1])


def test_pcp_tokens_and_tp_heads_compose():
    # tp=2 x pcp=2: rank = pcp_rank * 2 + tp_rank; 4 workers tile the page
    spec = _full_spec(num_kv_heads=1)
    cache = _packed_nhd_cache(spec)
    per_rank = [
        _mapping(spec, cache, _ctx(rank, tp=2, pcp=2, total=2)) for rank in range(4)
    ]
    assert all(m is not None and m.runs for m in per_rank)
    _verify_tiling("pcp", per_rank)


def test_interleave_chunks_stay_contiguous():
    # interleave=2: chunks of 2 tokens alternate between the 2 cp ranks and
    # coalesce into one contiguous fragment per chunk
    spec = _mla_spec()
    mapping = _mapping(spec, None, _ctx(rank=0, tp=2, dcp=2, interleave=2))
    assert _triples(mapping.runs) == [(0, 0, 128), (128, 256, 128)]
    _verify_tiling(
        "interleave",
        [
            _mapping(spec, None, _ctx(rank, tp=2, dcp=2, interleave=2))
            for rank in range(2)
        ],
    )


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_all_ranks_tile_canonical_page(form):
    spec = _full_spec()
    per_rank = [
        _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank, tp=4))
        for rank in range(4)
    ]
    _verify_tiling("layer", per_rank)


# ---------------------------------------------------------------------------
# Byte-level round trips
# ---------------------------------------------------------------------------


def _store_all(mappings, pages, size: int, block_id: int = 0) -> bytes:
    buf = bytearray(size)
    for mapping, page in zip(mappings, pages):
        if not mapping.is_writer(block_id):
            continue
        for local, canonical, n in _triples(mapping.runs):
            buf[canonical : canonical + n] = page[local : local + n]
    return bytes(buf)


def _load_one(mapping, canonical_bytes: bytes) -> bytes:
    page = bytearray(mapping.local_page_size_bytes)
    for local, canonical, n in _triples(mapping.runs):
        page[local : local + n] = canonical_bytes[canonical : canonical + n]
    return bytes(page)


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_cross_tp_store_load(form):
    """Bytes stored under one TP size are the bytes another TP size loads."""
    total_heads, canonical_size = 8, 4096
    reference = bytes((7 + 31 * i) % 256 for i in range(canonical_size))

    def mappings_at(tp: int):
        spec = _full_spec(num_kv_heads=total_heads // tp)
        cache = CACHE_BUILDERS[form](spec)
        return [
            _mapping(spec, cache, _ctx(rank, tp=tp, total=total_heads))
            for rank in range(tp)
        ]

    for tp in (4, 2, 1):
        mappings = mappings_at(tp)
        assert all(m is not None for m in mappings)
        pages = [_load_one(m, reference) for m in mappings]
        assert _store_all(mappings, pages, canonical_size) == reference


def test_cp_round_trip():
    # tp=4 / dcp=2 / 2 KV heads: 4 workers jointly hold one canonical page
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    mappings = [
        _mapping(spec, cache, _ctx(rank, tp=4, dcp=2, total=2)) for rank in range(4)
    ]
    reference = bytes((3 + 17 * i) % 256 for i in range(2048))
    pages = [_load_one(m, reference) for m in mappings]
    assert _store_all(mappings, pages, 2048) == reference


def test_replica_rotation_round_trip():
    """Whichever replica a block elects reproduces the same canonical page."""
    spec = _mla_spec()
    mappings = [_mapping(spec, None, _ctx(rank, tp=2)) for rank in range(2)]
    reference = bytes((5 + 11 * i) % 256 for i in range(256))
    pages = [_load_one(m, reference) for m in mappings]
    for block_id in (0, 1):
        assert _store_all(mappings, pages, 256, block_id) == reference


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_fail_closed_cases():
    spec = _full_spec()
    nhd = _split_nhd_cache(spec)
    # Spec heads inconsistent with total heads / tp
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, total=2)) is None
    # tp not divisible by total KV heads
    one_head = _full_spec(num_kv_heads=1)
    assert (
        _try_mapping(one_head, _split_nhd_cache(one_head), _ctx(0, tp=3, total=2))
        is None
    )
    # DCP wider than the KV replication factor (tokens would shard across
    # ranks holding different heads)
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, dcp=2)) is None
    # Interleave must divide the block size
    assert _try_mapping(spec, nhd, _ctx(0, tp=2, dcp=2, interleave=3, total=2)) is None
    # Per-token-head scales are packed with the data
    quant_spec = _full_spec(kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert _try_mapping(quant_spec, _split_nhd_cache(quant_spec), _ctx(0, tp=4)) is None
    # Compressed MLA slots are not 1:1 with tokens
    assert (
        _try_mapping(_mla_spec(tokens_per_state=2), None, _ctx(0, tp=2, dcp=2)) is None
    )
    # Unrecognized physical layouts
    swapped = torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        2,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3, 4)
    assert _try_mapping(spec, swapped, _ctx(0, tp=4)) is None
    # Non-attention specs
    assert _try_mapping(KVCacheSpec(block_size=4), None, _ctx(0, tp=4, total=8)) is None


def test_opaque_fallback_places_page_whole():
    mapping = _opaque_fallback_mapping(1024, 4, 2)
    assert mapping.canonical_page_size_bytes == 4096
    assert not mapping.parallelism_agnostic
    assert _triples(mapping.runs) == [(0, 2048, 1024)]
    assert mapping.num_writers == 1
    _verify_tiling("opaque", [_opaque_fallback_mapping(1024, 4, r) for r in range(4)])


# ---------------------------------------------------------------------------
# derive_canonical_mappings end to end
# ---------------------------------------------------------------------------


def _vllm_config(tp=1, dcp=1, pcp=1, pp=1, interleave=1, total_kv_heads=2):
    config = MagicMock()
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    config.parallel_config.world_size = pp * tp * pcp
    config.parallel_config.rank = 0
    config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return config


def _kv_cache_config(groups):
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups
    )


@pytest.mark.parametrize("layout", list(KVCacheLayout))
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("mamba_state", [False, True])
def test_independent_mapping_reconstructs_last_group_page(
    layout, num_layers, mamba_state
):
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        VllmConfig,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import MambaSpec, get_kv_cache_regions
    from vllm.v1.worker.utils import allocate_kv_cache

    config = VllmConfig(
        model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B",
            dtype="float32",
            hf_overrides={
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "hidden_size": 128,
                "head_dim": 64,
            },
        ),
        cache_config=CacheConfig(),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            engine_id="mapping-test",
        ),
    )
    config.cache_config.kv_cache_layout = layout.name
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=64, dtype=torch.float32
    )
    assert config.model_config.get_total_num_kv_heads() == spec.num_kv_heads
    other_spec = (
        MambaSpec(
            block_size=4,
            shapes=((8, 3), (1, 4, 4)),
            dtypes=(torch.float16, torch.float32),
            page_size_padded=spec.page_size_bytes,
        )
        if mamba_state
        else spec
    )
    groups = [
        KVCacheGroupSpec(
            [f"{prefix}.{index}" for index in range(num_layers)], group_spec
        )
        for prefix, group_spec in (("attention", spec), ("other", other_spec))
    ]
    plan = get_kv_cache_config_from_groups(
        config, groups, 65536, group_num_blocks=(5, 3)
    )
    single_plan = get_kv_cache_config_from_groups(
        config,
        [
            KVCacheGroupSpec([name], group.kv_cache_spec)
            for group in groups
            for name in group.layer_names
        ],
        65536,
        group_num_blocks=(1,) * (2 * num_layers),
    )
    caches = allocate_kv_cache(plan, torch.device("cpu"), layout)
    single = allocate_kv_cache(single_plan, torch.device("cpu"), layout)
    regions, _ = get_kv_cache_regions(plan, caches, layout)
    assert build_offloading_config(config, plan).worker_kv_bytes_per_block == sum(
        region.shape[1] for region in regions.values()
    )
    for index, (name, cache) in enumerate(caches.items()):
        cache.copy_(torch.arange(cache.numel()).reshape(cache.shape) + index * 100000)
        single[name].copy_(cache[-1:])
    mappings = derive_canonical_mappings(config, plan, caches)
    assert mappings.keys() == caches.keys()
    for descriptor in single_plan.kv_cache_tensors:
        name = descriptor.layers[0]
        cache = caches[name]
        mapping = mappings[name]
        raw = torch.empty(0, dtype=torch.uint8).set_(cache.untyped_storage())
        base = (
            cache.storage_offset() + (cache.shape[0] - 1) * cache.stride(0)
        ) * cache.element_size()
        actual = torch.empty(mapping.canonical_page_size_bytes, dtype=torch.uint8)
        for run in mapping.runs:
            for fragment in range(run.num_fragments):
                source = base + run.local_offset + fragment * run.local_stride
                destination = run.canonical_offset + fragment * run.canonical_stride
                actual[destination : destination + run.fragment_size] = raw[
                    source : source + run.fragment_size
                ]
        expected = torch.empty(0, dtype=torch.uint8).set_(
            single[name].untyped_storage()
        )[descriptor.offset : descriptor.offset + mapping.canonical_page_size_bytes]
        assert torch.equal(actual, expected), (
            name,
            cache.shape,
            cache.stride(),
            mapping,
        )
        original = raw.clone()
        if name.startswith("other.") and mamba_state:
            raw[base : base + mapping.local_page_size_bytes].fill_(0)
        else:
            cache[-1].fill_(-1)
        for run in mapping.runs:
            for fragment in range(run.num_fragments):
                source = run.canonical_offset + fragment * run.canonical_stride
                destination = base + run.local_offset + fragment * run.local_stride
                raw[destination : destination + run.fragment_size].copy_(
                    actual[source : source + run.fragment_size]
                )
        assert torch.equal(raw, original)

    if not mamba_state:
        legacy_plan = get_kv_cache_config_from_groups(config, groups[:1], 65536)
        assert legacy_plan.group_num_blocks is None
        legacy_caches = allocate_kv_cache(legacy_plan, torch.device("cpu"), layout)
        legacy_mappings = derive_canonical_mappings(config, legacy_plan, legacy_caches)
        sparse = layout is KVCacheLayout.LHBNC or (
            layout is KVCacheLayout.BHLNC and num_layers > 1
        )
        page = spec.page_size_bytes
        expected_mapping = CanonicalPageMapping(
            canonical_page_size_bytes=page,
            local_page_size_bytes=page,
            runs=(CopyRun(0, 0, page, 1, page, page),),
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=not sparse,
        )
        assert legacy_mappings == dict.fromkeys(legacy_caches, expected_mapping)


def test_derive_mixed_model_with_dcp():
    attn_spec = _full_spec(num_kv_heads=1)
    mla_spec = _mla_spec()
    quant_spec = _full_spec(
        num_kv_heads=1, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD
    )
    kv_cache_config = _kv_cache_config(
        [
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=mla_spec),
            KVCacheGroupSpec(layer_names=["quant"], kv_cache_spec=quant_spec),
        ]
    )
    kv_caches = {
        "attn": _split_nhd_cache(attn_spec),
        "quant": _split_nhd_cache(quant_spec),
    }
    mappings = derive_canonical_mappings(
        _vllm_config(tp=4, dcp=2, total_kv_heads=2), kv_cache_config, kv_caches
    )
    assert set(mappings) == {"attn", "mla", "quant"}
    assert not mappings["attn"].parallelism_agnostic
    assert mappings["attn"].runs
    # Uncertifiable layers degrade to an opaque page, never disappear
    assert not mappings["quant"].parallelism_agnostic
    assert (
        mappings["quant"].canonical_page_size_bytes
        == 4 * quant_spec.unpadded_page_size_bytes
    )


def test_derive_refuses_foreign_worker_groups():
    attn_spec = _full_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec)]
    )
    kv_caches = {"attn": _split_nhd_cache(attn_spec)}
    assert (
        derive_canonical_mappings(
            _vllm_config(tp=2, pp=2, total_kv_heads=4), kv_cache_config, kv_caches
        )
        == {}
    )
