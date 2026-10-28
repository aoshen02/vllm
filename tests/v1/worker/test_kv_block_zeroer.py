# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.config import DeviceConfig, VllmConfig
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
    SlidingWindowSpec,
    create_kv_cache_views,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _zero_kv_blocks_kernel,
    allocate_kv_cache,
)


def _make_zeroer(*storages: torch.Tensor) -> KVBlockZeroer:
    groups = []
    context = {}
    for index, storage in enumerate(storages):
        name = f"layer.{index}"
        layer = torch.nn.Module()
        layer.register_buffer("kv_cache", storage)
        context[name] = layer
        spec = FullAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=storage.shape[1] // 2,
            dtype=torch.int32,
        )
        groups.append(AttentionGroup(None, [name], spec, 0))
    return KVBlockZeroer(
        storages[0].device,
        attn_groups_iter=groups,
        kernel_block_sizes=[1],
        static_forward_context=context,
        num_blocks=storages[0].shape[0],
    )


class BoundMambaStates(torch.nn.Module, MambaBase):
    def __init__(self, spec, pages):
        super().__init__()
        self.spec = spec
        self.bind_kv_cache(pages)
        for index, state in enumerate(self.kv_cache):
            self.register_buffer(f"state_{index}", state)

    def get_state_shape(self):
        return self.spec.shapes

    def get_state_dtype(self):
        return self.spec.dtypes

    @property
    def mamba_type(self):
        return self.spec.mamba_type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
@pytest.mark.parametrize("bound_states", [False, True])
def test_independent_mamba_zeroing_clears_only_selected_domain(layout, bound_states):
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups

    device = torch.device("cuda")
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=16,
        shapes=((8, 3), (2, 4, 4)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=192,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["attention"], attention),
        KVCacheGroupSpec(["state_a", "state_b"], state),
        KVCacheGroupSpec(["other_state"], state),
    ]
    config = VllmConfig(device_config=DeviceConfig("cuda"))
    config.cache_config.kv_cache_layout = layout.name
    counts = (7, 4, 5)
    plan = get_kv_cache_config_from_groups(
        config, groups, 2**20, group_num_blocks=counts
    )
    caches = allocate_kv_cache(plan, device, layout)
    raw = torch.empty(0, dtype=torch.uint8, device=device).set_(
        caches["attention"].untyped_storage()
    )
    raw.fill_(255)
    expected = raw.clone()
    context = {}
    for group_id, group in enumerate(groups):
        for name in group.layer_names:
            if group_id and bound_states:
                context[name] = BoundMambaStates(state, caches[name])
            else:
                layer = torch.nn.Module()
                layer.register_buffer("kv_cache", caches[name])
                context[name] = layer
    for descriptor in plan.kv_cache_tensors:
        if "state_a" in descriptor.layers:
            expected_views = create_kv_cache_views(
                expected, state, counts[1], layout, descriptor
            )
            for view in expected_views:
                view[3].zero_()
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, group.layer_names, group.kv_cache_spec, group_id)
            for group_id, group in enumerate(groups)
        ],
        kernel_block_sizes=[16, 16, 16],
        static_forward_context=context,
        num_blocks=counts,
    )
    zeroer.zero_block_ids({1: [3]})
    torch.accelerator.synchronize()
    assert torch.equal(raw, expected)
    assert int((raw == 0).sum().item()) == 2 * state.real_page_size_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("cuda")
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layers_in_one_group_may_use_different_kernel_pages_per_block():
    """Derive kernel pages per block from each layer's allocation."""
    device = torch.device("cuda")
    num_blocks = 4
    spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=2,
    )
    # Two kernel pages per logical block.
    wide = torch.ones((num_blocks * 2, 3), dtype=torch.int32, device=device)
    # One kernel page per logical block, carved out of a larger allocation so
    # an over-strided write lands in the guard region instead of faulting or
    # silently hitting another tensor.
    narrow_backing = torch.ones((num_blocks * 3, 5), dtype=torch.int32, device=device)
    narrow = narrow_backing[:num_blocks]

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                None,
                ["wide", "narrow"],
                spec,
                0,
            )
        ],
        kernel_block_sizes=[1],
        static_forward_context={
            "wide": SimpleNamespace(kv_cache=wide),
            "narrow": SimpleNamespace(kv_cache=narrow),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected_wide = torch.ones_like(wide)
    expected_wide[2 * (num_blocks - 1) :] = 0
    assert torch.equal(wide, expected_wide)

    expected_narrow = torch.ones_like(narrow_backing)
    expected_narrow[num_blocks - 1] = 0
    assert torch.equal(narrow_backing, expected_narrow)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = _make_zeroer(storage)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = _make_zeroer(storage_a, storage_b)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device("cuda")
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones(
        (num_blocks, block_stride_el), dtype=torch.int32, device=device
    )

    zeroer = _make_zeroer(backing[:, page_offset_el : page_offset_el + page_size_el])

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = SlidingWindowSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=1,
    )
    storages = {
        name: torch.ones((1, page_size), dtype=torch.int32)
        for name, page_size in zip(layer_names, page_sizes)
    }
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, [name], spec, group_id)
            for group_id, name in enumerate(layer_names)
        ],
        kernel_block_sizes=[1] * n_segs,
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=1,
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == page_sizes
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    assert captured_grids == [(n_blocks, n_segs, max_chunks)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_for_all_block_counts():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel.
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = _make_zeroer(storage)

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = _make_zeroer(storage)

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device("cuda")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
        num_blocks=num_blocks,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout
