# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import hashlib
import importlib
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.core.kv_cache_planning as kv_cache_planning
import vllm.v1.core.kv_cache_utils as kv_cache_utils
import vllm.v1.hisparse.runtime as hisparse_runtime_module
from tests.v1.core.utils import (
    new_kv_cache_spec,
    new_mamba_spec,
    new_mla_spec,
    new_sliding_window_spec,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.attention import HiSparseConfig
from vllm.config.kv_events import KVEventsConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor, xxhash, xxhash_cbor
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_groups,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    generate_scheduler_kv_cache_config,
    get_kv_cache_capacity,
    get_max_concurrency_for_kv_cache_config,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
    null_head_len,
    tensor_data,
    to_event_extra_keys,
)
from vllm.v1.hisparse.layout import (
    create_hisparse_layout,
    get_hisparse_kv_cache_groups,
    get_hisparse_steady_state_concurrency,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
    is_full_attention_spec,
    iter_layer_specs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.metrics.stats import CachingMetrics, PrefixCacheStats
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def test_hisparse_steady_state_concurrency_excludes_spilled_resident_pages():
    """Running requests pin only their resident tail; the newest still needs its
    full in-flight window to be admitted."""
    block_size = 16
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=10 * block_size),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=10 * block_size,
    )
    attn_spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=64, dtype=torch.float16
    )
    page_size = attn_spec.page_size_bytes
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], attn_spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], attn_spec),
            KVCacheGroupSpec(
                ["resident"],
                HiSparseResidentSpec(block_size=block_size, page_size=page_size),
            ),
            KVCacheGroupSpec(
                ["hot"],
                HiSparseHotSpec(
                    block_size=block_size, page_size=page_size, blocks_per_request=4
                ),
            ),
        ],
        hisparse_host_num_blocks=100,
    )

    # Admission takes indexer 10 + resident 10 + hot 4 = 24 blocks and steady
    # state 16, so 3 fit (2 x 16 + 24 <= 64) but not 4 (3 x 16 + 24 > 64).
    assert get_hisparse_steady_state_concurrency(config, kv_cache_config) == 3.5
    # The worst-case bound charges all 10 resident pages.
    assert get_max_concurrency_for_kv_cache_config(
        config, kv_cache_config
    ) == pytest.approx(64 / 24)


@pytest.mark.parametrize("extra_blocks,ok", [(1, False), (2, True)])
def test_hisparse_host_pool_must_fit_max_model_len(monkeypatch, extra_blocks, ok):
    """Every page needs a host block, so one max_model_len request must fit."""
    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    specs = {
        "model.layers.0.self_attn": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        ),
        "model.layers.0.self_attn.indexer": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
    }
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=HiSparseConfig()),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=128), max_model_len=64 * 4
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, world_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=7),
    )
    host_page = specs["model.layers.0.self_attn"].page_size_bytes
    host_budget = (4 + extra_blocks) * host_page
    groups = get_hisparse_kv_cache_groups(config, specs)

    if ok:
        layout = create_hisparse_layout(config, groups, host_budget=host_budget)
        assert layout.host_num_blocks == 4 + extra_blocks
    else:
        with pytest.raises(ValueError, match="increase host_pool_gib"):
            create_hisparse_layout(config, groups, host_budget=host_budget)


def test_hisparse_derived_specs_do_not_affect_kv_cache_layout(monkeypatch):
    """Workers report HiSparse's resident/hot caches, which lay out their own raw
    backing; KV cache layout resolution must skip them."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    specs: dict[str, KVCacheSpec] = {
        "model.layers.0.self_attn": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        ),
        "model.layers.0.self_attn.indexer": MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
    }
    config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    config.attention_config.hisparse_config = HiSparseConfig()
    config.model_config.hf_config.index_topk = 128
    reported = specs | {
        name: group.kv_cache_spec
        for group in get_hisparse_kv_cache_groups(config, specs) or []
        if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec))
        for name in group.layer_names
    }
    assert len(reported) > len(specs)

    layouts = []
    for worker_specs in (specs, reported):
        config.cache_config.kv_cache_layout = None
        layouts.append(
            resolve_kv_cache_layout(
                config, [["BLHNC", "LBNHC"]], list(worker_specs.values())
            )
        )
    assert layouts[0] == layouts[1]


def test_hisparse_rejects_deepseek_v4():
    full_specs = {
        "model.layers.0.attn": MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            model_version="deepseek_v4",
        )
    }
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=HiSparseConfig()),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_topk=512)),
        cache_config=SimpleNamespace(num_gpu_blocks_override=7),
    )

    with pytest.raises(ValueError, match="does not support DeepSeek V4"):
        get_hisparse_kv_cache_groups(config, full_specs)


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = request.getfixturevalue("hash_fn")
    else:
        hash_fn = sha256
    init_none_hash(hash_fn)


def make_request(
    request_id: str,
    prompt_token_ids: list[int] | None,
    block_size: int = 3,
    hash_fn: Callable = hash,
    mm_positions: list[PlaceholderRange] | None = None,
    mm_hashes: list[str] | None = None,
    cache_salt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
        prompt_embeds=prompt_embeds,
    )


@pytest.mark.parametrize("dcp", [1, 4])
def test_effective_attention_block_size_matches_events(dcp):
    from vllm.distributed.kv_events import BlockStored
    from vllm.v1.engine.core import EngineCore

    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], new_kv_cache_spec()),
        ],
    )
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([config]),
        max_model_len=256,
        scheduler_block_size=16 * dcp,
        hash_block_size=16 * dcp,
        dcp_world_size=dcp,
        enable_kv_cache_events=True,
    )
    core = EngineCore.__new__(EngineCore)
    core.vllm_config = SimpleNamespace(cache_config=CacheConfig(block_size=16))
    core.scheduler = SimpleNamespace(kv_cache_manager=manager)
    core._initialize_effective_attention_block_size()
    block_size = core.vllm_config.cache_config.effective_attention_block_size
    assert block_size == 16 * dcp

    request = make_request(
        "block-size", list(range(64)), block_size=16 * dcp, hash_fn=sha256
    )
    assert manager.allocate_slots(request, 64) is not None
    assert [
        event.block_size
        for event in manager.take_events()
        if isinstance(event, BlockStored)
    ] == [block_size]


def test_kv_cache_config_selects_only_transferable_groups():
    """Connectors must see a stable projection of transfer-eligible groups."""
    groups = [
        KVCacheGroupSpec(["layer.0"], new_kv_cache_spec()),
        KVCacheGroupSpec(["layer.1"], new_kv_cache_spec(), enable_kv_transfer=False),
        KVCacheGroupSpec(["layer.2"], new_kv_cache_spec()),
    ]
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )

    assert config.transfer_group_ids == (0, 2)
    assert config.transfer_groups == (groups[0], groups[2])
    assert config.transfer_group_index_by_layer == {"layer.0": 0, "layer.2": 1}
    first_blocks = [1, 2]
    third_blocks = [4]
    assert config.select_transfer_block_ids((first_blocks, [3], third_blocks)) == (
        first_blocks,
        third_blocks,
    )


def test_kv_cache_config_selects_prefix_cacheable_groups():
    """Prefix stores exclude scratch state without changing transfer groups."""
    full_group = KVCacheGroupSpec(["full"], new_kv_cache_spec())
    qsa_group = KVCacheGroupSpec(
        ["qsa"],
        CircularBufferSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=64,
            head_size_v=0,
            dtype=torch.float16,
        ),
    )
    disabled_group = KVCacheGroupSpec(
        ["disabled"], new_kv_cache_spec(), enable_kv_transfer=False
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[full_group, qsa_group, disabled_group],
    )
    assert config.transfer_group_ids == (0, 1)
    assert config.select_transfer_block_ids(([1], [2], [3])) == ([1], [2])
    assert config.prefix_cacheable_group_ids == (0,)
    assert config.prefix_cacheable_groups == (full_group,)


def test_kv_cache_blocks_selects_requested_groups():
    blocks = KVCacheBlocks(
        (
            [KVCacheBlock(1)],
            [KVCacheBlock(2)],
            [KVCacheBlock(3)],
        )
    )

    assert blocks.get_block_ids(group_ids=(0, 2)) == ([1], [3])


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_none_hash(monkeypatch, hash_fn):
    import vllm.v1.core.kv_cache_utils

    # case 1: PYTHONHASHSEED is not set -> deterministic default seed so that
    # independent processes compute identical block hashes for identical
    # content (e.g. for KV cache reuse across nodes).
    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        none_hash = reloaded_kv_cache_utils.NONE_HASH
        assert isinstance(none_hash, bytes)
        assert none_hash != b""
        assert none_hash == hash_fn(reloaded_kv_cache_utils.DEFAULT_NONE_HASH_SEED)
        # deterministic across re-initialization within the same environment
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert none_hash == reloaded_kv_cache_utils.NONE_HASH

    # case 2: PYTHONHASHSEED is set, use the seed and hash_fn
    with monkeypatch.context() as m:
        m.setenv("PYTHONHASHSEED", "python hash seed")
        reloaded_kv_cache_utils = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded_kv_cache_utils.init_none_hash(hash_fn)
        assert reloaded_kv_cache_utils.NONE_HASH is not None
        assert isinstance(reloaded_kv_cache_utils.NONE_HASH, bytes)
        assert hash_fn("python hash seed") == reloaded_kv_cache_utils.NONE_HASH


@pytest.mark.parametrize("non_crypto_fn", [xxhash, xxhash_cbor])
def test_none_hash_seed_random_for_non_crypto(monkeypatch, non_crypto_fn):
    """Non-cryptographic algorithms keep the per-process random seed.

    A deterministic seed is safe for SHA-256, whose collision resistance does
    not depend on a secret, but xxHash is not collision resistant: a known seed
    would let an attacker precompute colliding blocks offline. Keep the
    unpredictable seed there unless the operator opts into a shared one.
    """
    # PYTHONHASHSEED unset -> unpredictable, differs per resolution.
    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        seeds = {kv_cache_utils.resolve_none_hash_seed(non_crypto_fn) for _ in range(5)}
        assert len(seeds) == 5
        assert kv_cache_utils.DEFAULT_NONE_HASH_SEED not in seeds

    # PYTHONHASHSEED set -> operator opt-in wins, so peers can share a cache.
    with monkeypatch.context() as m:
        m.setenv("PYTHONHASHSEED", "12345")
        assert kv_cache_utils.resolve_none_hash_seed(non_crypto_fn) == "12345"


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_none_hash_seed_deterministic_for_crypto(monkeypatch, hash_fn):
    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        seed = kv_cache_utils.resolve_none_hash_seed(hash_fn)
        assert seed == kv_cache_utils.DEFAULT_NONE_HASH_SEED
        assert seed == kv_cache_utils.resolve_none_hash_seed(hash_fn)


def test_get_none_hash_seed_reports_effective_seed(monkeypatch):
    """P2P advertises the seed NONE_HASH was actually derived from.

    The P2P tier is constructed before init_none_hash runs, so it must read the
    resolved seed lazily rather than re-deriving it.
    """
    import vllm.v1.core.kv_cache_utils

    with monkeypatch.context() as m:
        m.delenv("PYTHONHASHSEED", raising=False)
        reloaded = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded.init_none_hash(sha256)
        assert reloaded.get_none_hash_seed() == reloaded.DEFAULT_NONE_HASH_SEED

    with monkeypatch.context() as m:
        m.setenv("PYTHONHASHSEED", "12345")
        reloaded = importlib.reload(vllm.v1.core.kv_cache_utils)
        reloaded.init_none_hash(sha256)
        assert reloaded.get_none_hash_seed() == "12345"


def test_kv_cache_block():
    # Test KVCacheBlock initialization
    block = KVCacheBlock(block_id=0)
    assert block.block_id == 0
    assert block.ref_cnt == 0
    assert block.block_hash is None

    # Test reference count manipulation
    block.ref_cnt += 1
    assert block.ref_cnt == 1
    block.ref_cnt -= 1
    assert block.ref_cnt == 0

    # Test block hash setting and resetting
    block_hash = make_block_hash_with_group_id(BlockHash(b"abc"), 0)
    block.set_block_hash(block_hash)
    assert block.block_hash == block_hash

    block.reset_hash()
    assert block.block_hash is None


def test_kv_cache_block_uses_slots():
    block = KVCacheBlock(block_id=0)

    # Slots eliminate per-instance __dict__, saving ~264 bytes per block.
    # At 100K+ blocks this avoids tens of MB of overhead and GC pressure.
    assert not hasattr(block, "__dict__")

    # Verify that slots actually prevent dynamic attribute assignment.
    with pytest.raises(AttributeError):
        block.unexpected_field = True


def test_free_kv_cache_block_queue_initialization():
    # Test with a single block
    block = KVCacheBlock(block_id=0)
    queue = FreeKVCacheBlockQueue([block])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is block
    assert queue.fake_free_list_tail.prev_free_block is block


def test_free_kv_cache_block_queue_operations():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check initial state
    assert queue.num_free_blocks == 5
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Pop the first block
    block1 = queue.popleft()
    assert block1 == blocks[0]
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert queue.fake_free_list_tail.prev_free_block is blocks[4]

    # Remove a block from the middle
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.num_free_blocks == 3
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]

    # Append a block back
    queue.append(block_to_remove)
    assert queue.num_free_blocks == 4
    assert queue.fake_free_list_tail.prev_free_block is block_to_remove
    assert block_to_remove.prev_free_block is blocks[4]
    assert block_to_remove.next_free_block is queue.fake_free_list_tail

    # Pop blocks until empty
    for _ in range(4):
        queue.popleft()
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head

    # Attempt to pop from an empty queue
    with pytest.raises(ValueError) as e:
        queue.popleft()
    assert str(e.value) == "No free blocks available"


def test_free_kv_cache_block_queue_append_n():
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue([])
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Append 0 block
    # fake_head->fake_tail
    queue.append_n([])
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head
    # Append 1 block
    # fake_head->b0->fake_tail
    queue.append_n(blocks[0:1])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[0]
    # Append 2 blocks
    # fake_head->b0->b4->b5->fake_tail
    queue.append_n(blocks[4:6])
    assert queue.num_free_blocks == 3
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[5]
    # Append 3 blocks
    # fake_head->b0->b4->b5->b1->b2->b3->fake_tail
    queue.append_n(blocks[1:4])
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[0]
    assert blocks[0].prev_free_block is queue.fake_free_list_head
    assert blocks[0].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[0]
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is blocks[1]
    assert blocks[1].prev_free_block is blocks[5]
    assert blocks[1].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[1]
    assert blocks[2].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[2]
    assert blocks[3].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[3]

    # Create an empty FreeKVCacheBlockQueue
    invalid_queue = FreeKVCacheBlockQueue([])
    # set prev_free_block to None and this will cause assertion in append_n
    invalid_queue.fake_free_list_tail.prev_free_block = None
    with pytest.raises(AssertionError):
        # Append 1 block
        # fake_head->fake_tail
        invalid_queue.append_n(blocks[0:1])
    assert invalid_queue.num_free_blocks == 0
    assert (
        invalid_queue.fake_free_list_head.next_free_block
        == invalid_queue.fake_free_list_tail
    )


def test_free_kv_cache_block_queue_prepend_n():
    # Seed the queue with one block so prepend has an existing head to splice
    # in front of (fake_head->b0->fake_tail).
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    queue = FreeKVCacheBlockQueue(blocks[0:1])

    # Prepend 0 blocks is a no-op.
    queue.prepend_n([])
    assert queue.num_free_blocks == 1
    assert queue.fake_free_list_head.next_free_block is blocks[0]

    # Prepend 2 blocks; they land in front of the existing head, in order.
    # fake_head->b4->b5->b0->fake_tail
    queue.prepend_n(blocks[4:6])
    assert queue.num_free_blocks == 3
    assert queue.fake_free_list_head.next_free_block is blocks[4]
    assert blocks[4].prev_free_block is queue.fake_free_list_head
    assert blocks[4].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[4]
    assert blocks[5].next_free_block is blocks[0]
    assert blocks[0].prev_free_block is blocks[5]
    assert blocks[0].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[0]

    # A second prepend goes ahead of everything previously prepended.
    # fake_head->b1->b2->b4->b5->b0->fake_tail
    queue.prepend_n(blocks[1:3])
    assert queue.num_free_blocks == 5
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert blocks[1].next_free_block is blocks[2]
    assert blocks[2].next_free_block is blocks[4]

    # The popleft order reflects the front-to-back queue order.
    assert [queue.popleft().block_id for _ in range(5)] == [1, 2, 4, 5, 0]
    assert queue.num_free_blocks == 0


def test_free_kv_cache_block_queue_popleft_n():
    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
    # Create an empty FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(
        [blocks[1], blocks[3], blocks[5], blocks[4], blocks[0], blocks[2]]
    )
    assert queue.num_free_blocks == 6
    assert queue.fake_free_list_head.next_free_block is blocks[1]
    assert blocks[1].prev_free_block is queue.fake_free_list_head
    assert blocks[1].next_free_block is blocks[3]
    assert blocks[3].prev_free_block is blocks[1]
    assert blocks[3].next_free_block is blocks[5]
    assert blocks[5].prev_free_block is blocks[3]
    assert blocks[5].next_free_block is blocks[4]
    assert blocks[4].prev_free_block is blocks[5]
    assert blocks[4].next_free_block is blocks[0]
    assert blocks[0].prev_free_block is blocks[4]
    assert blocks[0].next_free_block is blocks[2]
    assert blocks[2].prev_free_block is blocks[0]
    assert blocks[2].next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is blocks[2]

    # Pop 0 block
    # fake_head->b1->b3->b5->b4->b0->b2->fake_tail
    assert len(queue.popleft_n(0)) == 0
    assert queue.num_free_blocks == 6
    # Pop 1 block
    # fake_head->b3->b5->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(1)
    assert queue.num_free_blocks == 5
    assert len(result_blocks) == 1
    assert result_blocks[0] is blocks[1]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 2 blocks
    # fake_head->b4->b0->b2->fake_tail
    result_blocks = queue.popleft_n(2)
    assert len(result_blocks) == 2
    assert queue.num_free_blocks == 3
    assert result_blocks[0] is blocks[3]
    assert result_blocks[1] is blocks[5]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None
    # Pop 3 blocks
    # fake_head->fake_tail
    result_blocks = queue.popleft_n(3)
    assert len(result_blocks) == 3
    assert queue.num_free_blocks == 0
    assert result_blocks[0] is blocks[4]
    assert result_blocks[1] is blocks[0]
    assert result_blocks[2] is blocks[2]
    for block in result_blocks:
        assert block.prev_free_block is None
        assert block.next_free_block is None


@pytest.mark.parametrize("num_blocks", [1, 2, 16])
def test_free_kv_cache_block_queue_popleft_n_drain_and_refill(num_blocks):
    """Draining a batch must leave the sentinels ready for reuse."""
    blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
    queue = FreeKVCacheBlockQueue(blocks)

    assert queue.popleft_n(num_blocks) == blocks
    assert queue.num_free_blocks == 0
    assert queue.fake_free_list_head.next_free_block is queue.fake_free_list_tail
    assert queue.fake_free_list_tail.prev_free_block is queue.fake_free_list_head
    assert all(b.prev_free_block is None and b.next_free_block is None for b in blocks)
    assert queue.popleft_n(0) == []

    queue.append_n(blocks)
    assert queue.popleft_n(num_blocks) == blocks
    assert queue.num_free_blocks == 0


def test_free_kv_cache_block_queue_get_all_free_blocks():
    # Create a list of KVCacheBlock objects
    blocks = [KVCacheBlock(block_id=i) for i in range(5)]

    # Create a FreeKVCacheBlockQueue with these blocks
    queue = FreeKVCacheBlockQueue(blocks)

    # Check all blocks are correctly retrieved
    assert queue.get_all_free_blocks() == blocks

    # Pop a block and check again
    queue.popleft()
    assert queue.get_all_free_blocks() == blocks[1:]

    # Remove a block and check again
    block_to_remove = blocks[2]
    queue.remove(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:]

    # Append a block back and check again
    queue.append(block_to_remove)
    assert queue.get_all_free_blocks() == blocks[1:2] + blocks[3:] + [block_to_remove]


def test_generate_block_hash_extra_keys():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
            PlaceholderRange(offset=10, length=5),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys == (("mm", "hash1", 0),)
    assert next_mm_idx == 1

    # Test with partial overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 3, 8, 0)
    assert extra_keys == (("mm", "hash1", -3),)
    assert next_mm_idx == 1

    # Test with no overlap
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None
    assert next_mm_idx == 1

    # Test with multiple extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 15, 0)
    assert extra_keys == (("mm", "hash1", 0), ("mm", "hash2", 10))
    assert next_mm_idx == 2


def test_generate_block_hash_extra_keys_no_mm_inputs():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
    )

    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys is None
    assert next_mm_idx == 0


def test_generate_block_hash_extra_keys_cache_salt():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=None,
        mm_hashes=None,
        cache_salt="salt",
    )

    # salt is added for the first token
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 1, 0)
    assert extra_keys == (("cache_salt", "salt"),)
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 10, 0)
    assert extra_keys == (("cache_salt", "salt"),)

    # no salt added for other tokens
    extra_keys, _ = generate_block_hash_extra_keys(request, 1, 2, 0)
    assert extra_keys is None
    extra_keys, _ = generate_block_hash_extra_keys(request, 6, 10, 0)
    assert extra_keys is None

    # works together with other extra keys
    request_mm = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(20)],
        mm_positions=[
            PlaceholderRange(offset=0, length=5),
        ],
        mm_hashes=["hash1"],
        cache_salt="salt",
    )

    # Test with no extra keys
    extra_keys, next_mm_idx = generate_block_hash_extra_keys(request_mm, 0, 5, 0)
    assert extra_keys == (("mm", "hash1", 0), ("cache_salt", "salt"))
    assert next_mm_idx == 1


def test_generate_block_hash_extra_keys_prompt_embeds():
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
    )

    # Test with prompt embeds for the first block
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    expected_embeds = prompt_embeds[0:5]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (("prompt_embeds", expected_hash),)

    # Test with prompt embeds for the second block
    extra_keys, _ = generate_block_hash_extra_keys(request, 5, 10, 0)
    expected_embeds = prompt_embeds[5:10]
    expected_hash = hashlib.sha256(kv_cache_utils.tensor_data(expected_embeds)).digest()
    assert extra_keys == (("prompt_embeds", expected_hash),)


def test_generate_block_hash_extra_keys_prompt_embeds_cached(monkeypatch):
    prompt_embeds = torch.randn(10, 3)
    request = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds,
        block_size=20,
    )

    num_tensor_data_calls = 0
    original_tensor_data = kv_cache_utils.tensor_data

    def counting_tensor_data(tensor: torch.Tensor):
        nonlocal num_tensor_data_calls
        num_tensor_data_calls += 1
        return original_tensor_data(tensor)

    monkeypatch.setattr(kv_cache_utils, "tensor_data", counting_tensor_data)

    extra_keys_1, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    extra_keys_2, _ = generate_block_hash_extra_keys(request, 0, 5, 0)
    assert extra_keys_1 == extra_keys_2
    assert num_tensor_data_calls == 1


def test_generate_block_hash_extra_keys_different_prompt_embeds():
    prompt_embeds1 = torch.randn(10, 3)
    prompt_embeds2 = torch.randn(10, 3)
    request1 = make_request(
        request_id="0",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds1,
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=None,
        mm_positions=None,
        mm_hashes=None,
        prompt_embeds=prompt_embeds2,
    )

    extra_keys1, _ = generate_block_hash_extra_keys(request1, 0, 5, 0)
    extra_keys2, _ = generate_block_hash_extra_keys(request2, 0, 5, 0)
    assert extra_keys1 != extra_keys2


def test_generate_block_hash_extra_keys_lora():
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
    )

    request.lora_request = LoRARequest(
        lora_name="test_lora_adapter", lora_int_id=1, lora_path="/path/to/lora"
    )

    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys == (("lora", "test_lora_adapter", "/path/to/lora"),)

    request.lora_request = None
    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 3, 0)
    assert extra_keys is None


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_lora_name_and_cache_salt_block_hashes_do_not_collide(hash_fn):
    """A cache_salt equal to a LoRA name must not share that LoRA's blocks."""
    lora_request = LoRARequest(
        lora_name="foo", lora_int_id=1, lora_path="/path/to/lora"
    )
    lora_req = Request(
        request_id="lora",
        prompt_token_ids=[0, 1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        lora_request=lora_request,
        block_hasher=get_request_block_hasher(3, hash_fn),
    )
    salted_req = make_request("salted", [0, 1, 2], hash_fn=hash_fn, cache_salt="foo")

    assert lora_req.block_hashes[0] != salted_req.block_hashes[0]


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_lora_path_change_changes_block_hashes(hash_fn):
    """Re-pointing a LoRA name at another adapter must not reuse its blocks."""

    def make_lora_request(lora_path: str) -> Request:
        return Request(
            request_id=lora_path,
            prompt_token_ids=[0, 1, 2],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            lora_request=LoRARequest(
                lora_name="foo", lora_int_id=1, lora_path=lora_path
            ),
            block_hasher=get_request_block_hasher(3, hash_fn),
        )

    first = make_lora_request("/path/to/lora_a")
    second = make_lora_request("/path/to/lora_b")

    assert first.block_hashes[0] != second.block_hashes[0]


def test_to_event_extra_keys_keeps_untagged_event_format():
    """KV events keep publishing the extra-key shapes consumers already parse."""
    request = make_request(
        "0",
        list(range(10)),
        mm_positions=[PlaceholderRange(offset=2, length=3)],
        mm_hashes=["hash1"],
        cache_salt="salt",
    )
    request.lora_request = LoRARequest(
        lora_name="adapter", lora_int_id=1, lora_path="/path/to/lora"
    )

    extra_keys, _ = generate_block_hash_extra_keys(request, 0, 5, 0)

    assert extra_keys == (
        ("lora", "adapter", "/path/to/lora"),
        ("mm", "hash1", 2),
        ("cache_salt", "salt"),
    )
    event_keys = ("adapter", ("hash1", 2), "salt")
    assert to_event_extra_keys([extra_keys, None]) == [event_keys, None]
    assert to_event_extra_keys([]) is None
    assert to_event_extra_keys(None) is None


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_block_tokens(hash_fn):
    parent_block_hash = BlockHash(b"123")
    curr_block_token_ids = (1, 2, 3)
    extra_keys = ("key1", "key2")

    block_hash = hash_block_tokens(
        hash_fn, parent_block_hash, curr_block_token_ids, extra_keys
    )
    expected = hash_fn((parent_block_hash, curr_block_token_ids, extra_keys))
    assert block_hash == expected


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn(
        (kv_cache_utils.NONE_HASH, (0, 1, 2), (("mm", "hash1", 0),))
    )
    assert block_hashes[1] == hash_fn(
        (block_hashes[0], (3, 4, 5), (("mm", "hash2", 0),))
    )


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher_incremental_append_with_multiple_mm_features(hash_fn):
    mm_positions = [
        PlaceholderRange(offset=4, length=2),
        PlaceholderRange(offset=6, length=1),
    ]
    incremental = make_request(
        request_id="incremental",
        prompt_token_ids=list(range(7)),
        block_size=4,
        hash_fn=hash_fn,
        mm_positions=mm_positions,
        mm_hashes=["A", "B"],
    )
    incremental.append_output_token_ids(7)
    fresh = make_request(
        request_id="fresh",
        prompt_token_ids=list(range(8)),
        block_size=4,
        hash_fn=hash_fn,
        mm_positions=mm_positions,
        mm_hashes=["A", "B"],
    )

    expected_second_hash = hash_fn(
        (incremental.block_hashes[0], (4, 5, 6, 7), (("mm", "A", 0), ("mm", "B", 2)))
    )
    assert incremental.block_hashes[1] == expected_second_hash
    assert incremental.block_hashes == fresh.block_hashes


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_tokens_different_mm_input(hash_fn):
    request1 = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
    )
    request2 = make_request(
        request_id="1",
        prompt_token_ids=[_ for _ in range(6)],
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash3", "hash2"],
    )
    block_hashes1 = request1.block_hashes
    block_hashes2 = request2.block_hashes
    assert block_hashes1[0] != block_hashes2[0]
    assert block_hashes1[1] != block_hashes2[1]


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_hash_request_tokens_no_mm_inputs(hash_fn):
    request = make_request(
        request_id="0",
        prompt_token_ids=[_ for _ in range(6)],
        block_size=3,
        hash_fn=hash_fn,
        mm_positions=None,
        mm_hashes=None,
    )

    block_hashes = request.block_hashes

    assert len(block_hashes) == 2
    assert block_hashes[0] == hash_fn((kv_cache_utils.NONE_HASH, (0, 1, 2), None))
    assert block_hashes[1] == hash_fn((block_hashes[0], (3, 4, 5), None))


def _stats(requests: int, queries: int, hits: int) -> PrefixCacheStats:
    return PrefixCacheStats(requests=requests, queries=queries, hits=hits)


def test_metrics_empty_distinguishes_no_queries_from_no_hits():
    """`hit_rate` alone cannot tell the two apart; `empty` can.

    Both an unobserved window and a genuine all-miss window report a hit
    rate of 0.0, so anything surfacing that number to a human has to check
    `empty` first - which is what the prefix-cache log line does.
    """
    metrics = CachingMetrics(max_recent_requests=5)
    assert metrics.empty
    assert metrics.hit_rate == 0.0

    metrics.observe(_stats(1, 20, 0))
    assert not metrics.empty
    assert metrics.hit_rate == 0.0


def test_metrics():
    """Test the prefix caching metrics."""
    metrics = CachingMetrics(max_recent_requests=5)
    assert metrics.hit_rate == 0.0

    metrics.observe(_stats(1, 20, 9))
    # 9 / 20 = 0.45
    assert metrics.hit_rate == 0.45

    metrics.observe(_stats(4, 80, 16))

    # 25 / 100 = 0.25
    assert metrics.hit_rate == 0.25

    metrics.observe(_stats(1, 10, 2))

    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    metrics.reset()
    assert metrics.hit_rate == 0.0
    assert metrics.aggregated_requests == 0
    assert metrics.aggregated_query_total == 0
    assert metrics.aggregated_query_hit == 0
    assert not metrics.query_queue


def test_metrics_empty_stats():
    """Test the prefix caching metrics with empty stats."""
    metrics = CachingMetrics(max_recent_requests=5)
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 20, 9))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(4, 80, 16))
    metrics.observe(_stats(0, 0, 0))
    metrics.observe(_stats(1, 10, 2))
    # Remove (20, 9) and add (10, 2): 18 / 90 = 0.2
    assert metrics.aggregated_requests == 5
    assert metrics.aggregated_query_total == 90
    assert metrics.aggregated_query_hit == 18
    assert metrics.hit_rate == 0.2

    # Only the latest added stats preserved 10 / 20 = 0.5
    metrics.observe(_stats(11, 20, 10))
    assert metrics.aggregated_requests == 11
    assert metrics.aggregated_query_total == 20
    assert metrics.aggregated_query_hit == 10
    assert metrics.hit_rate == 0.5

    # Only the latest added stats preserved 30 / 40 = 0.75
    metrics.observe(_stats(22, 40, 30))
    assert metrics.aggregated_requests == 22
    assert metrics.aggregated_query_total == 40
    assert metrics.aggregated_query_hit == 30
    assert metrics.hit_rate == 0.75


@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("disable_hybrid", [False, True])
@pytest.mark.parametrize("pcp_size", [1, 4])
def test_dcp_target_allocates_replicated_draft_independently(
    monkeypatch, sliding_window, disable_hybrid, pcp_size
):
    """A draft must retain all positions even when the target shards them."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    config.parallel_config.decode_context_parallel_size = 4
    config.parallel_config.prefill_context_parallel_size = pcp_size
    config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hybrid
    config.cache_config.block_size = 16
    config.cache_config.kv_cache_layout = None
    draft_args = dict(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.bfloat16,
        dcp_sharded=False,
    )
    draft = (
        FullAttentionSpec(**draft_args)
        if sliding_window is None
        else SlidingWindowSpec(**draft_args, sliding_window=sliding_window)
    )
    specs = {"target": new_mla_spec(), "draft": draft}
    layout = resolve_kv_cache_layout(config, [["LBHNC", "BLHNC"]], specs.values())
    assert layout == KVCacheLayout.BLHNC
    groups = get_kv_cache_groups(config, specs)
    assert [group.layer_names for group in groups] == [["target"], ["draft"]]
    widths = [g.kv_cache_spec.max_num_blocks_per_req(config, 1024) for g in groups]
    assert widths == [16, 64]

    cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        config, groups, available_memory=16 * 1024 * 1024
    )
    scheduler_config = generate_scheduler_kv_cache_config([cache_config])
    assert get_max_concurrency_for_kv_cache_config(config, cache_config) > 0
    assert get_max_concurrency_for_kv_cache_config(config, cache_config) == (
        get_max_concurrency_for_kv_cache_config(config, scheduler_config)
    )
    manager = KVCacheManager(
        scheduler_config,
        max_model_len=1024,
        hash_block_size=16,
        scheduler_block_size=64,
        dcp_world_size=4,
        pcp_world_size=pcp_size,
        enable_caching=True,
    )
    assert manager.coordinator.group_block_sizes == (64, 16)
    request = make_request("replicated-draft", [1] * 65, block_size=16, hash_fn=sha256)
    blocks = manager.allocate_slots(request, 65)
    assert blocks is not None
    assert [len(group) for group in blocks.blocks] == [2, 5]
    manager.cache_blocks(request, 64)
    cached_request = make_request(
        "cached-draft", [1] * 65, block_size=16, hash_fn=sha256
    )
    cached_blocks, num_cached, _ = manager.get_computed_blocks(cached_request)
    assert num_cached == 64
    assert [len(group) for group in cached_blocks.blocks] == [1, 4]


def test_dcp_replicated_kpool_tail_keeps_block_interior_layout(monkeypatch):
    """A replicated kpool tail ring is per-request state, not a draft group."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    config.parallel_config.decode_context_parallel_size = 4
    config.cache_config.kv_cache_layout = None
    tail = CircularBufferSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        dcp_sharded=False,
    )
    layout = resolve_kv_cache_layout(config, [["LBHNC"]], [new_mla_spec(), tail])
    assert layout == KVCacheLayout.LBHNC


@pytest.mark.parametrize("use_mla", [False, True])
def test_full_attention_merge_preserves_replicated_cache_geometry(use_mla):
    config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    config.parallel_config.decode_context_parallel_size = 4
    draft = replace(
        new_mla_spec() if use_mla else new_kv_cache_spec(),
        block_size=16,
        dcp_sharded=False,
    )
    merged = type(draft).merge([draft, draft])
    wrapped = UniformTypeKVCacheSpecs.from_specs({"draft": merged})
    assert wrapped is not None and not wrapped.dcp_sharded
    assert merged.max_num_blocks_per_req(config, 1024) == 64
    assert merged.max_memory_usage_bytes(config) == 64 * draft.page_size_bytes
    assert kv_cache_utils.resolve_dcp_kv_block_size(merged, 4) == 16
    with pytest.raises(AssertionError):
        type(draft).merge([draft, replace(draft, dcp_sharded=True)])


@pytest.mark.parametrize(
    "target",
    [
        new_kv_cache_spec(),
        new_mla_spec(),
        SinkFullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float32, sink_len=4
        ),
    ],
)
def test_draft_layer_tag_is_not_part_of_the_cache_format(target):
    draft = replace(target, is_draft=True)
    assert draft == target and hash(draft) == hash(target)
    assert type(target).merge([target, draft]) == target


@pytest.mark.parametrize(
    "layer_type,dcp_size,expected_width",
    [
        ("mla", 1, 64),
        ("mla", 2, 32),
        # Mamba state is replicated, not DCP-sharded, and its width is the
        # resident state block count rather than cdiv(max_len, block_size).
        ("mamba", 2, 3),
    ],
)
def test_uniform_type_spec_block_table_width_matches_layer_spec(
    layer_type, dcp_size, expected_width
):
    # The runner sizes the block table from the group spec while the metadata
    # builders are constructed from the per-layer spec, so the aggregate must
    # report the same width as the layers it wraps.
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    layer_spec = new_mla_spec() if layer_type == "mla" else new_mamba_spec()
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=layer_spec.block_size,
        kv_cache_specs={"layer1": layer_spec, "layer2": layer_spec},
    )

    assert layer_spec.max_num_blocks_per_req(vllm_config, 1024) == expected_width
    assert uniform_spec.max_num_blocks_per_req(vllm_config, 1024) == expected_width


def test_merge_kv_cache_spec():
    same_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32),
    ]
    merged_layer_spec = same_layer_specs[0].merge(same_layer_specs)
    assert merged_layer_spec.block_size == 16
    assert merged_layer_spec.num_kv_heads == 32
    assert merged_layer_spec.head_size == 64
    assert merged_layer_spec.dtype == torch.float32
    assert merged_layer_spec.sliding_window is None

    different_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=16),
    ]
    with pytest.raises(AssertionError):
        different_layer_specs[0].merge(different_layer_specs)

    full_spec = new_kv_cache_spec(num_kv_heads=32)
    different_type_layer_specs = [
        full_spec,
        SlidingWindowSpec(
            block_size=full_spec.block_size,
            num_kv_heads=full_spec.num_kv_heads,
            head_size=full_spec.head_size,
            dtype=full_spec.dtype,
            sliding_window=1,
        ),
    ]
    with pytest.raises(AssertionError):
        different_type_layer_specs[0].merge(different_type_layer_specs)
    with pytest.raises(AssertionError):
        different_type_layer_specs[1].merge(different_type_layer_specs)

    different_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=2),
    ]
    with pytest.raises(ValueError):
        different_sliding_window_layer_specs[0].merge(
            different_sliding_window_layer_specs
        )

    same_sliding_window_layer_specs = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    ]
    merged_layer_spec = same_sliding_window_layer_specs[0].merge(
        same_sliding_window_layer_specs
    )
    assert merged_layer_spec.sliding_window == 1

    same_sliding_window_layer_spec_with_none = [
        new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
        new_kv_cache_spec(num_kv_heads=32, sliding_window=None),
    ]
    merged_layer_spec = same_sliding_window_layer_spec_with_none[0].merge(
        same_sliding_window_layer_spec_with_none
    )
    assert merged_layer_spec.sliding_window == 1


def test_get_max_concurrency_for_kv_cache_config():
    # Create a VllmConfig
    model_id = "Qwen/Qwen1.5-7B"
    max_model_len = 16384
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
        # Pin to sync: SWA per-request bounds grow with overlapping batches.
        async_scheduling=False,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )
    assert vllm_config.max_concurrent_batches == 1

    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
    )

    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        sliding_window=1024,
    )

    kv_cache_config_full_attention = KVCacheConfig(
        num_blocks=int(1024 * 1.5),
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
        ],
    )
    max_concurrency_full_attention = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_full_attention
    )
    assert max_concurrency_full_attention == 1.5

    kv_cache_config_sliding_window = KVCacheConfig(
        num_blocks=129 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], sliding_window_spec),
        ],
    )
    max_concurrency_sliding_window = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_sliding_window
    )
    assert max_concurrency_sliding_window == 3

    kv_cache_config_hybrid_model = KVCacheConfig(
        num_blocks=(1024 + 129) * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
            KVCacheGroupSpec(
                [f"layer_{i}" for i in range(32, 64)], sliding_window_spec
            ),
        ],
    )
    max_concurrency_hybrid_model = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_hybrid_model
    )
    assert max_concurrency_hybrid_model == 3
    num_tokens, max_concurrency = get_kv_cache_capacity(
        vllm_config, kv_cache_config_hybrid_model
    )
    assert num_tokens == max_concurrency_hybrid_model * max_model_len
    assert max_concurrency == max_concurrency_hybrid_model

    # Unequal group sizes in the standard layout: each group's pages cost
    # whole pool blocks, so a request needs 1024 + 129 = 1153 blocks — the
    # same as the equal-hybrid case above, regardless of the second group
    # holding only 2 layers.
    kv_cache_config_unequal_groups = KVCacheConfig(
        num_blocks=1153 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(32)], full_attention_spec),
            KVCacheGroupSpec(["layer_32", "layer_33"], sliding_window_spec),
        ],
    )
    assert (
        get_max_concurrency_for_kv_cache_config(
            vllm_config, kv_cache_config_unequal_groups
        )
        == 3
    )

    # UniformTypeKVCacheSpecs group (worker config shape): the aggregated
    # spec's memory/page ratio equals a single layer's page count, so the
    # group needs 1024 blocks and the request 1153 in total. The previous
    # formula normalized both groups' memory by the first group's page size,
    # reporting 3459/1057 = 3.27 here instead of 3 — and a different value
    # again for the scheduler-config shape below.
    uniform_full_spec = UniformTypeKVCacheSpecs(
        block_size=full_attention_spec.block_size,
        kv_cache_specs={f"layer_{i}": full_attention_spec for i in range(4)},
    )
    kv_cache_config_uniform_group = KVCacheConfig(
        num_blocks=1153 * 3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer_{i}" for i in range(4)], uniform_full_spec),
            KVCacheGroupSpec(["layer_4", "layer_5"], sliding_window_spec),
        ],
    )
    assert (
        get_max_concurrency_for_kv_cache_config(
            vllm_config, kv_cache_config_uniform_group
        )
        == 3
    )

    # Scheduler-config shape: generate_scheduler_kv_cache_config replaces the
    # uniform-type group's spec with a representative per-layer spec.
    # Capacity must not change between the two shapes (the engine computes
    # on the scheduler config, the worker loop on the worker config).
    kv_cache_config_scheduler_shape = generate_scheduler_kv_cache_config(
        [copy.deepcopy(kv_cache_config_uniform_group)]
    )
    assert get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_scheduler_shape
    ) == get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config_uniform_group
    )


def test_allocate_with_lookahead():
    """Verify that lookahead tokens correctly affect block allocation."""
    block_size = 4
    config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=100,
                layers=["layer1"],
                layer_stride=100,
                block_stride=10,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1"], new_kv_cache_spec(block_size=block_size)),
        ],
    )

    request = make_request(
        request_id="0",
        prompt_token_ids=[],
        block_size=block_size,
        mm_positions=None,
        mm_hashes=None,
    )

    # Test case 1: Requires additional lookahead tokens
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=100,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,  # Total required: 3+2=5 tokens
    )
    assert len(blocks.get_block_ids()[0]) == 2  # ceil(5/4)=2 blocks

    # Test case 2: With precomputed blocks
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=100,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    # required_blocks = ceil((3 + 2) /4) = 2
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=2,
    )
    assert len(blocks.get_block_ids()[0]) == 2

    # Test case 3: With precomputed blocks
    # required_blocks = ceil((3 + 4) / 4) = 2
    kv_cache_manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=100,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    blocks = kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=3,
        num_lookahead_tokens=4,
    )
    assert len(blocks.get_block_ids()[0]) == 2


def test_generate_uniform_type_kv_cache_specs():
    # All layers are full attention, can be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # Full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(sliding_window=1),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # different order of full attention + sliding window, cannot be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_kv_cache_spec(),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None

    # Same-size sliding window, can be merged
    kv_cache_specs = {
        "layer_1": new_sliding_window_spec(sliding_window=1),
        "layer_2": new_sliding_window_spec(sliding_window=1, head_size=128),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec == UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    # different block sizes, cannot be merged
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(block_size=16),
        "layer_2": new_kv_cache_spec(block_size=32),
    }
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(kv_cache_specs)
    assert uniform_spec is None


def test_generate_scheduler_kv_cache_config():
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(head_size=128),
    }
    kv_cache_configs = [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer_1", "layer_2"],
                    UniformTypeKVCacheSpecs(
                        block_size=16, kv_cache_specs=kv_cache_specs
                    ),
                ),
            ],
        )
    ]
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    assert scheduler_kv_cache_config == KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def test_resolve_dcp_kv_block_size_unwraps_uniform_type_specs():
    attention = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float16,),
    )
    wrapped_attention = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"attention.0": attention, "attention.1": attention},
    )
    wrapped_mamba = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"mamba.0": mamba, "mamba.1": mamba},
    )

    assert kv_cache_utils.resolve_dcp_kv_block_size(attention, 4) == 64
    assert kv_cache_utils.resolve_dcp_kv_block_size(wrapped_attention, 4) == 64
    assert kv_cache_utils.resolve_dcp_kv_block_size(mamba, 4) == 16
    assert kv_cache_utils.resolve_dcp_kv_block_size(wrapped_mamba, 4) == 16

    scaled_attention = kv_cache_utils.resolve_dcp_kv_cache_spec(wrapped_attention, 4)
    assert scaled_attention.block_size == 64
    assert isinstance(scaled_attention, UniformTypeKVCacheSpecs)
    assert all(
        spec.block_size == 64 for spec in scaled_attention.kv_cache_specs.values()
    )
    assert kv_cache_utils.resolve_dcp_kv_cache_spec(wrapped_mamba, 4) is wrapped_mamba


def test_get_kv_cache_spec_kind_prefers_specific_attention_subclasses():
    assert get_kv_cache_spec_kind(new_mla_spec()) == KVCacheSpecKind.MLA_ATTENTION

    sliding_window_mla_spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float32,
        sliding_window=128,
    )
    assert (
        get_kv_cache_spec_kind(sliding_window_mla_spec)
        == KVCacheSpecKind.SLIDING_WINDOW_MLA
    )

    sink_full_attention_spec = SinkFullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        sink_len=4,
    )
    assert (
        get_kv_cache_spec_kind(sink_full_attention_spec)
        == KVCacheSpecKind.SINK_FULL_ATTENTION
    )


def test_get_kv_cache_spec_kind_unwraps_uniform_type_specs():
    uniform_mla_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": new_mla_spec(),
            "layer_2": new_mla_spec(cache_dtype_str="fp8"),
        },
    )
    assert get_kv_cache_spec_kind(uniform_mla_spec) == KVCacheSpecKind.MLA_ATTENTION

    uniform_swa_mla_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.float32,
                sliding_window=128,
            ),
            "layer_2": SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1024,
                dtype=torch.float32,
                sliding_window=128,
            ),
        },
    )
    assert (
        get_kv_cache_spec_kind(uniform_swa_mla_spec)
        == KVCacheSpecKind.SLIDING_WINDOW_MLA
    )


def test_get_kv_cache_spec_kind_unknown_for_mixed_uniform_type_specs():
    uniform_mixed_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": new_mla_spec(),
            "layer_2": SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.float32,
                sliding_window=128,
            ),
        },
    )
    assert get_kv_cache_spec_kind(uniform_mixed_spec) == KVCacheSpecKind.UNKNOWN


def test_get_kv_cache_spec_sliding_window_reads_windowed_specs():
    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
    )
    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float32,
        sliding_window=128,
    )

    assert get_kv_cache_spec_sliding_window(full_attention_spec) is None
    assert get_kv_cache_spec_sliding_window(sliding_window_spec) == 128


def test_get_kv_cache_spec_sliding_window_unwraps_uniform_type_specs():
    uniform_window_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
                sliding_window=128,
            ),
            "layer_2": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.float32,
                sliding_window=128,
            ),
        },
    )
    mixed_window_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_1": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
                sliding_window=128,
            ),
            "layer_2": SlidingWindowSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.float32,
                sliding_window=256,
            ),
        },
    )

    assert get_kv_cache_spec_sliding_window(uniform_window_spec) == 128
    assert get_kv_cache_spec_sliding_window(mixed_window_spec) is None


def test_merge_mla_spec():
    kv_cache_specs = [
        new_mla_spec(),
        new_mla_spec(),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec()

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
    ]
    mla_spec = kv_cache_specs[0].merge(kv_cache_specs)
    assert mla_spec == new_mla_spec(cache_dtype_str="fp8_ds_mla")

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_mla_spec(cache_dtype_str=None),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_kv_cache_spec(),
        new_mla_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)

    kv_cache_specs = [
        new_mla_spec(cache_dtype_str="fp8_ds_mla"),
        new_kv_cache_spec(),
    ]
    with pytest.raises(AssertionError):
        kv_cache_specs[0].merge(kv_cache_specs)


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_block_hasher_with_prompt_embeds(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (("prompt_embeds", block1_embeds_hash),),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (("prompt_embeds", block2_embeds_hash),),
        )
    )
    assert block_hashes[1] == expected_hash2


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_request_with_prompt_embeds_and_mm_inputs(hash_fn: Callable[[Any], bytes]):
    block_size = 3
    num_tokens = 2 * block_size
    prompt_token_ids = [_ for _ in range(num_tokens)]
    hidden_size = 5
    prompt_embeds = torch.randn((num_tokens, hidden_size))

    request = make_request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        block_size=block_size,
        hash_fn=hash_fn,
        mm_positions=[
            PlaceholderRange(offset=0, length=3),
            PlaceholderRange(offset=3, length=3),
        ],
        mm_hashes=["hash1", "hash2"],
        prompt_embeds=prompt_embeds,
    )

    block_hashes = request.block_hashes
    assert len(block_hashes) == 2

    block1_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[:block_size])
    ).digest()
    expected_hash1 = hash_fn(
        (
            kv_cache_utils.NONE_HASH,
            tuple(prompt_token_ids[:block_size]),
            (("mm", "hash1", 0), ("prompt_embeds", block1_embeds_hash)),
        )
    )
    assert block_hashes[0] == expected_hash1

    block2_embeds_hash = hashlib.sha256(
        tensor_data(prompt_embeds[block_size:num_tokens])
    ).digest()
    expected_hash2 = hash_fn(
        (
            block_hashes[0],
            tuple(prompt_token_ids[block_size:num_tokens]),
            (("mm", "hash2", 0), ("prompt_embeds", block2_embeds_hash)),
        )
    )
    assert block_hashes[1] == expected_hash2


def test_unpadded_page_size_includes_per_token_head_scales():
    # Per-token-head quant carries inline fp32 scales that are carved from the
    # raw KV allocation, so they must be budgeted into the offload width. The
    # packing is published by the owning backend's customize_spec hook.
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    dense = new_kv_cache_spec(dtype=torch.uint8)
    spec = TritonAttentionBackend.customize_spec(
        new_kv_cache_spec(
            dtype=torch.uint8, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD
        )
    )
    scales = 2 * spec.block_size * spec.num_kv_heads * 4
    assert spec.unpadded_page_size_bytes == dense.unpadded_page_size_bytes + scales
    assert spec.page_size_bytes == spec.unpadded_page_size_bytes


def test_page_size_padded_wins():
    # An explicit padded page size takes precedence over the unpadded size.
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    spec = TritonAttentionBackend.customize_spec(
        new_kv_cache_spec(
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
            page_size_padded=65536,
        )
    )
    assert spec.page_size_bytes == 65536


def test_hma_not_disabled_when_kv_events_enabled():
    """Test enabling KV events must not force disable_hybrid_kv_cache_manager to True.

    This test guards against that regression by verifying that a VllmConfig
    with kv_events_config set still resolves disable_hybrid_kv_cache_manager
    to False (i.e. HMA remains enabled) when no other condition requires it
    to be disabled.
    """
    model_config = ModelConfig(max_model_len=16)
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="null",
    )

    # Leave disable_hybrid_kv_cache_manager as None (the default) so that
    # VllmConfig.__post_init__ resolves it automatically.
    vllm_config = VllmConfig(
        model_config=model_config,
        kv_events_config=kv_events_config,
    )

    assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is False, (
        "kv_events_config must not force-disable the hybrid KV cache manager."
    )


def test_resolve_block_hashes_gate():
    # Resolve symbols through the module so they stay consistent with each other
    # even after other tests reload ``kv_cache_utils``.
    resolve_block_hashes = kv_cache_utils.resolve_block_hashes
    BlockHashListWithBlockSize = kv_cache_utils.BlockHashListWithBlockSize
    # Raw, hash_block_size-granularity hashes (contents are opaque here).
    raw = [BlockHash(bytes([i])) for i in range(8)]

    # block_size == hash_block_size: always reuse the raw hashes.
    assert resolve_block_hashes(raw, 2, 2, alignment_tokens=2) is raw
    assert (
        resolve_block_hashes(
            raw, 2, 2, supports_fine_grained_hash_lookup=True, alignment_tokens=2
        )
        is raw
    )

    # Fine-grained manager, partial hits ON (alignment_tokens == hash_block_size
    # < block_size): keep raw hashes so the manager can scan at hash granularity.
    assert (
        resolve_block_hashes(
            raw, 2, 4, supports_fine_grained_hash_lookup=True, alignment_tokens=2
        )
        is raw
    )

    # Fine-grained manager, partial hits OFF (alignment_tokens ==
    # scheduler_block_size >= block_size): must fall back to a block-size view,
    # exactly like the pre-refactor coordinator did.
    off = resolve_block_hashes(
        raw, 2, 4, supports_fine_grained_hash_lookup=True, alignment_tokens=4
    )
    assert isinstance(off, BlockHashListWithBlockSize)
    assert off.scale_factor == 2

    # Non-fine-grained manager (e.g. sliding window): always a block-size view
    # when block_size != hash_block_size, regardless of alignment_tokens.
    swa = resolve_block_hashes(
        raw, 2, 4, supports_fine_grained_hash_lookup=False, alignment_tokens=2
    )
    assert isinstance(swa, BlockHashListWithBlockSize)
    assert swa.scale_factor == 2


def test_resolve_block_hashes_rejects_mismatched_view():
    resolve_block_hashes = kv_cache_utils.resolve_block_hashes
    BlockHashListWithBlockSize = kv_cache_utils.BlockHashListWithBlockSize
    raw = [BlockHash(bytes([i])) for i in range(8)]

    # A view built at this block_size is returned as-is (idempotent).
    view = BlockHashListWithBlockSize(raw, 2, 4)
    assert resolve_block_hashes(view, 2, 4) is view

    # A view built at a different block_size must fail loudly rather than be
    # silently reinterpreted at the wrong granularity.
    mismatched = BlockHashListWithBlockSize(raw, 2, 8)
    with pytest.raises(AssertionError):
        resolve_block_hashes(mismatched, 2, 4)


def test_is_full_attention_spec_unwraps_uniform_type_specs():
    """``UniformTypeKVCacheSpecs`` is not a ``FullAttentionSpec``, so callers
    scanning groups with a bare isinstance miss DeepSeek-V4-shaped configs."""
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    full = FullAttentionSpec(block_size=4, **common)
    mla = MLAAttentionSpec(block_size=4, **common)
    swa = SlidingWindowMLASpec(block_size=4, sliding_window=8, **common)

    def wrap(**specs):
        return UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs=specs)

    assert is_full_attention_spec(full)
    assert is_full_attention_spec(mla)
    assert not is_full_attention_spec(swa)

    assert is_full_attention_spec(wrap(a=mla, b=mla))
    assert not is_full_attention_spec(wrap(a=swa, b=swa))

    # Every layer must qualify: a group holding a recycling sliding-window layer
    # has no stable slot layout. Grouping forbids mixing today, but the wrapper
    # is also built directly (e.g. projecting groups onto a PP worker).
    assert not is_full_attention_spec(wrap(a=mla, b=swa))
    assert not is_full_attention_spec(wrap())


def test_iter_layer_specs_returns_group_members():
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    full = FullAttentionSpec(block_size=4, **common)
    mla = MLAAttentionSpec(block_size=4, **common)

    assert list(iter_layer_specs(full)) == [full]
    wrapped = UniformTypeKVCacheSpecs(
        block_size=4, kv_cache_specs={"a": full, "b": mla}
    )
    assert list(iter_layer_specs(wrapped)) == [full, mla]


def test_wrapped_mamba_group_requires_block_zeroing():
    mamba = MambaSpec(
        block_size=4,
        shapes=((4, 1),),
        dtypes=(torch.float32,),
    )
    wrapped = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"mamba": mamba})
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], wrapped)],
    )

    assert config.needs_kv_cache_zeroing


def test_null_head_len():
    null = KVCacheBlock(0, is_null=True)
    real = [KVCacheBlock(i) for i in range(1, 4)]
    tables = [
        [],
        real,
        [null] * 5,
        [null] * 5 + real,
        [null] * 2 + real[:1] + [null] * 3 + real[1:],  # not one contiguous head
    ]
    for table in tables:
        expected = next((i for i, b in enumerate(table) if not b.is_null), len(table))
        assert null_head_len(table) == expected
    blocks = KVCacheBlocks(([null] * 4 + real, [null] + real[:1] + [null]))
    assert blocks.get_unhashed_block_ids_all_groups() == [[1, 2, 3], [1]]


@pytest.fixture
def independent_mamba_groups():
    return [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
            ),
        ),
        KVCacheGroupSpec(
            ["state"],
            new_mamba_spec(
                shapes=((128, 64), (128, 64)),
                dtypes=(torch.bfloat16, torch.bfloat16),
                num_speculative_blocks=0,
            ),
        ),
    ]


@pytest.mark.parametrize("layout", [layout.name for layout in KVCacheLayout])
@pytest.mark.parametrize("counts", [(128, 128), (64, 128)])
def test_independent_group_capacity_views_do_not_alias(
    independent_mamba_groups, layout, counts
):
    from vllm.v1.worker.utils import allocate_kv_cache

    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.cache_config.kv_cache_layout = layout
    budget = counts[0] * 65536 + counts[1] * 32768
    plan = kv_cache_planning.get_kv_cache_config_from_groups(
        config, independent_mamba_groups, budget, group_num_blocks=counts
    )
    assert plan.group_num_blocks == counts
    assert [plan.num_blocks_of(tensor) for tensor in plan.kv_cache_tensors] == list(
        counts
    )
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {budget}
    views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout[layout])
    assert views["attention"].shape[0] == counts[0]
    assert views["state"].shape[0] == counts[1]
    assert (
        views["attention"].untyped_storage().data_ptr()
        == views["state"].untyped_storage().data_ptr()
    )
    assert views["attention"].untyped_storage().nbytes() == budget
    views["state"].fill_(7)
    assert torch.count_nonzero(views["attention"]) == 0
    views["attention"].fill_(11)
    assert torch.all(views["state"] == 7)


@pytest.mark.parametrize(
    "counts,budget,error",
    [
        ((128,), 12 * 1024**2, "positive integer"),
        ((128, 0), 12 * 1024**2, "positive integer"),
        ((128, True), 12 * 1024**2, "positive integer"),
        ((128, 128), 12 * 1024**2 - 1, "exceed the cache budget"),
    ],
)
def test_independent_group_capacity_rejects_invalid_placement(
    independent_mamba_groups, counts, budget, error
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.cache_config.kv_cache_layout = "BLHNC"
    with pytest.raises(ValueError, match=error):
        kv_cache_planning.get_kv_cache_config_from_groups(
            config, independent_mamba_groups, budget, group_num_blocks=counts
        )


def independent_capacity_manager(
    groups, counts, *, enable_caching=False, watermark=0, scheduler_block_size=16
):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.cache_config.kv_cache_layout = "BLHNC"
    plan = kv_cache_planning.get_kv_cache_config_from_groups(
        config, groups, 16 * 1024**2, group_num_blocks=counts
    )
    return KVCacheManager(
        plan,
        max_model_len=256,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=16,
        enable_caching=enable_caching,
        enable_kv_cache_events=True,
        watermark=watermark,
    )


@pytest.mark.parametrize("counts", [(8, 2), (2, 8)])
@pytest.mark.parametrize("full_sequence_must_fit", [False, True])
def test_independent_manager_denies_saturated_domain_atomically_and_reuses(
    independent_mamba_groups, counts, full_sequence_must_fit
):
    manager = independent_capacity_manager(independent_mamba_groups, counts)
    first = make_request("first", [1] * 16, block_size=16, hash_fn=sha256)
    allocated = manager.allocate_slots(first, 16)
    assert allocated is not None
    assert allocated.get_block_ids() == ([1], [1])
    pools = [single.block_pool for single in manager.coordinator.single_type_managers]
    assert pools[0] is not pools[1]
    assert all(
        block.pool is pool
        for pool, blocks in zip(pools, allocated.blocks)
        for block in blocks
    )
    before = [pool.get_num_free_blocks() for pool in pools]
    assert sum(before) > 2
    second = make_request("second", [2] * 16, block_size=16, hash_fn=sha256)
    assert (
        manager.allocate_slots(
            second, 16, full_sequence_must_fit=full_sequence_must_fit
        )
        is None
    )
    assert [pool.get_num_free_blocks() for pool in pools] == before
    assert manager.get_block_ids(first.request_id) == ([1], [1])
    assert manager.take_new_block_ids() == {0: [1], 1: [1]}
    assert manager.take_new_block_ids() == {0: [], 1: []}
    assert manager.take_kv_cache_block_copies() == ({0: [], 1: []}, [])
    assert manager.usage == pytest.approx(2 / sum(count - 1 for count in counts))
    retained = manager.pop_blocks_for_free(first)
    assert [pool.get_num_free_blocks() for pool in pools] == before
    manager.block_pool.free_blocks(reversed(retained))
    assert [pool.get_num_free_blocks() for pool in pools] == [
        count - 1 for count in counts
    ]
    assert manager.usage == 0
    reused = manager.allocate_slots(second, 16)
    assert reused is not None
    assert reused.get_block_ids() == ([1], [1])
    manager.free(second)
    assert manager.usage == 0


def test_independent_manager_applies_reservations_and_watermark_per_domain(
    independent_mamba_groups,
):
    manager = independent_capacity_manager(independent_mamba_groups, (8, 3))
    request = make_request("reserved", [1] * 16, block_size=16, hash_fn=sha256)
    before = [pool.get_num_free_blocks() for pool in manager.coordinator.block_pools]
    assert manager.allocate_slots(request, 16, reserved_blocks={1: 2}) is None
    assert [
        pool.get_num_free_blocks() for pool in manager.coordinator.block_pools
    ] == before
    with pytest.raises(ValueError, match="group-keyed"):
        manager.allocate_slots(request, 16, reserved_blocks=1)
    assert manager.allocate_slots(request, 16, reserved_blocks={0: 6, 1: 1}) is not None
    manager.free(request)
    manager = independent_capacity_manager(
        independent_mamba_groups, (8, 3), watermark=0.5
    )
    assert manager.allocate_slots(request, 16, has_scheduled_reqs=True) is not None
    other = make_request("watermark", [2] * 16, block_size=16, hash_fn=sha256)
    assert manager.allocate_slots(other, 16, has_scheduled_reqs=True) is None
    assert manager.allocate_slots(other, 16, has_scheduled_reqs=False) is not None


@pytest.mark.parametrize("mamba", [False, True])
def test_independent_manager_prefix_eviction_and_reset_are_domain_scoped(
    independent_mamba_groups, mamba
):
    from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored

    attention, state = independent_mamba_groups
    if mamba:
        state = replace(
            state,
            kv_cache_spec=replace(state.kv_cache_spec, mamba_cache_mode="align"),
        )
    else:
        state = replace(
            state,
            kv_cache_spec=replace(attention.kv_cache_spec, num_kv_heads=4),
        )
    manager = independent_capacity_manager(
        [attention, state], (5, 3), enable_caching=True
    )
    request = make_request("cached", list(range(17)), block_size=16, hash_fn=sha256)
    allocated = manager.allocate_slots(request, 16)
    assert allocated is not None
    manager.new_step_starts()
    hit, tokens, _ = manager.get_computed_blocks(request)
    assert tokens == 16
    assert hit.get_block_ids() == ([1], [1])
    stored = [
        event for event in manager.take_events() if isinstance(event, BlockStored)
    ]
    assert {event.group_idx for event in stored} == {0, 1}
    with pytest.raises(ValueError, match="group-keyed"):
        manager.evict_blocks({1})
    with pytest.raises(ValueError, match="outside"):
        manager.evict_blocks({0: {1}, 1: {3}})
    assert manager.get_computed_blocks(request)[1] == 16
    manager.evict_blocks({1: {1}})
    assert manager.get_computed_blocks(request)[1] == 0
    assert allocated.blocks[0][0].block_hash is not None
    assert allocated.blocks[1][0].block_hash is None
    removed = [
        event for event in manager.take_events() if isinstance(event, BlockRemoved)
    ]
    assert removed
    assert not manager.reset_prefix_cache()
    assert allocated.blocks[0][0].block_hash is not None
    manager.free(request)
    assert manager.reset_prefix_cache()
    assert manager.get_computed_blocks(request)[1] == 0
    cleared = [
        event for event in manager.take_events() if isinstance(event, AllBlocksCleared)
    ]
    assert len(cleared) == 1
    assert manager.usage == 0


def test_independent_manager_reclaims_only_its_domains_cached_prefix(
    independent_mamba_groups,
):
    attention, state = independent_mamba_groups
    state = replace(
        state, kv_cache_spec=replace(attention.kv_cache_spec, num_kv_heads=4)
    )
    manager = independent_capacity_manager(
        [attention, state], (6, 2), enable_caching=True
    )
    first = make_request("old", [1] * 17, block_size=16, hash_fn=sha256)
    allocated = manager.allocate_slots(first, 16)
    assert allocated is not None
    manager.free(first)
    assert manager.get_computed_blocks(first)[1] == 16
    second = make_request("new", [2] * 17, block_size=16, hash_fn=sha256)
    reused = manager.allocate_slots(second, 16)
    assert reused is not None
    assert reused.get_block_ids() == ([2], [1])
    assert allocated.blocks[0][0].block_hash is not None
    assert manager.get_computed_blocks(first)[1] == 0
    manager.free(second)
    assert manager.get_computed_blocks(second)[1] == 16


def test_independent_manager_cow_copies_and_retention_keep_group_ownership(
    independent_mamba_groups,
):
    attention, state = independent_mamba_groups
    attention = replace(
        attention, kv_cache_spec=replace(attention.kv_cache_spec, block_size=32)
    )
    state = replace(
        state,
        kv_cache_spec=replace(
            state.kv_cache_spec, block_size=32, mamba_cache_mode="align"
        ),
    )
    manager = independent_capacity_manager(
        [attention, state], (8, 5), enable_caching=True, scheduler_block_size=32
    )
    first = make_request("source", list(range(49)), block_size=16, hash_fn=sha256)
    assert manager.allocate_slots(first, 48) is not None
    manager.free(first)
    manager.new_step_starts()
    manager.take_new_block_ids()
    second = make_request("cow", list(range(64)), block_size=16, hash_fn=sha256)
    hit, num_computed, _ = manager.get_computed_blocks(second)
    assert num_computed == 48
    assert manager.allocate_slots(second, 16, num_computed, hit) is not None
    copies, retained = manager.take_kv_cache_block_copies()
    assert isinstance(copies, dict)
    assert set(copies) == {0, 1}
    assert all(copies.values())
    for group_id, pending in copies.items():
        pool = manager.coordinator.single_type_managers[group_id].block_pool
        for copy_record in pending:
            assert copy_record.src_block_id != copy_record.dst_block_id
            source = pool.blocks[copy_record.src_block_id]
            destination = pool.blocks[copy_record.dst_block_id]
            assert any(block is source for block in retained)
            assert any(block is destination for block in retained)
            assert source.pool is destination.pool is pool
    manager.free(second)
    assert all(block.ref_cnt > 0 for block in retained)
    manager.block_pool.free_blocks(retained)
    assert [pool.get_num_free_blocks() for pool in manager.coordinator.block_pools] == [
        7,
        4,
    ]
    assert manager.usage == 0
    assert manager.take_kv_cache_block_copies() == ({0: [], 1: []}, [])


def test_independent_manager_preserves_event_order_and_global_clear(
    independent_mamba_groups,
):
    from vllm.distributed.kv_events import (
        AllBlocksCleared,
        BlockRemoved,
        KVEventAggregator,
    )

    attention, state = independent_mamba_groups
    state = replace(
        state,
        kv_cache_spec=replace(state.kv_cache_spec, mamba_cache_mode="align"),
    )
    manager = independent_capacity_manager(
        [attention, state], (5, 3), enable_caching=True
    )
    request = make_request("events", list(range(17)), block_size=16, hash_fn=sha256)
    assert manager.allocate_slots(request, 16) is not None
    manager.take_events()
    manager.evict_blocks({1: {1}, 0: {1}})
    events = manager.take_events()
    assert [event.group_idx for event in events if isinstance(event, BlockRemoved)] == [
        1,
        0,
    ]
    assert manager.take_events() == []
    manager.free(request)
    assert manager.reset_prefix_cache()
    aggregator = KVEventAggregator(num_workers=1)
    aggregator.add_events(manager.take_events())
    cleared = aggregator.get_common_events()
    assert len(cleared) == 1
    assert isinstance(cleared[0], AllBlocksCleared)


@pytest.mark.parametrize("counts", [(5, 8), (9, 3), (6, 4)])
def test_independent_concurrency_matches_real_manager_request_capacity(
    independent_mamba_groups, counts
):
    attention, state = independent_mamba_groups
    state = replace(state, kv_cache_spec=replace(state.kv_cache_spec, block_size=32))
    config = VllmConfig(
        model_config=ModelConfig(max_model_len=32),
        device_config=DeviceConfig("cpu"),
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    plan = kv_cache_planning.get_kv_cache_config_from_groups(
        config, [attention, state], 16 * 1024**2, group_num_blocks=counts
    )
    scheduler_plan = generate_scheduler_kv_cache_config([plan, copy.deepcopy(plan)])
    assert scheduler_plan.group_num_blocks == counts
    expected = min((counts[0] - 1) / 2, counts[1] - 1)
    assert get_max_concurrency_for_kv_cache_config(config, plan) == expected
    assert get_max_concurrency_for_kv_cache_config(config, scheduler_plan) == expected
    assert get_kv_cache_capacity(config, scheduler_plan) == (
        int(expected * 32),
        expected,
    )
    manager = KVCacheManager(
        scheduler_plan,
        max_model_len=32,
        scheduler_block_size=32,
        hash_block_size=16,
        enable_caching=False,
    )
    requests = [
        make_request(str(index), [index] * 32, block_size=16, hash_fn=sha256)
        for index in range(int(expected) + 1)
    ]
    for request in requests[:-1]:
        allocated = manager.allocate_slots(request, 32, full_sequence_must_fit=True)
        assert allocated is not None
        assert [len(blocks) for blocks in allocated.blocks] == [2, 1]
    before = [pool.get_num_free_blocks() for pool in manager.coordinator.block_pools]
    assert manager.allocate_slots(requests[-1], 32, full_sequence_must_fit=True) is None
    assert [
        pool.get_num_free_blocks() for pool in manager.coordinator.block_pools
    ] == before
    for request in requests[:-1]:
        manager.free(request)
    assert manager.usage == 0
    with pytest.raises(AssertionError):
        generate_scheduler_kv_cache_config(
            [
                plan,
                replace(plan, group_num_blocks=(counts[0], counts[1] + 1)),
            ]
        )


def mamba_ratio_config(
    ratio,
    budget,
    *,
    max_model_len=32,
    pipeline_parallel_size=1,
    enable_prefix_caching=True,
    max_num_seqs=128,
):
    model_config = ModelConfig(max_model_len=max_model_len)
    config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(
            mamba_full_memory_ratio=ratio,
            kv_cache_memory_bytes=budget,
            enable_prefix_caching=enable_prefix_caching,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
            max_num_seqs=max_num_seqs,
        ),
        parallel_config=ParallelConfig(pipeline_parallel_size=pipeline_parallel_size),
        device_config=DeviceConfig("cpu"),
    )
    config.cache_config.mamba_fixed_num_seqs = max_num_seqs
    return config


@pytest.mark.parametrize("ratio", [0.5, 0.9])
def test_mamba_ratio_live_planner_uses_physical_native_pages(
    independent_mamba_groups, ratio
):
    from vllm.v1.worker.utils import allocate_kv_cache

    attention, state = independent_mamba_groups
    state = replace(
        state,
        kv_cache_spec=replace(
            state.kv_cache_spec, block_size=32, mamba_cache_mode="align"
        ),
    )
    specs = {"attention": attention.kv_cache_spec, "state": state.kv_cache_spec}
    budget = 12 * 1024**2
    config = mamba_ratio_config(ratio, budget)
    config.cache_config.kv_cache_layout = "BLHNC"
    (plan,) = kv_cache_planning.get_kv_cache_configs(
        config, [specs], [config.cache_config.kv_cache_memory_bytes]
    )
    state_count = int(budget * (ratio / (1 + ratio))) // 32768
    attention_count = (budget - state_count * 32768) // 65536
    assert plan.group_num_blocks == (attention_count, state_count)
    assert state.kv_cache_spec.page_size_bytes == 32768
    assert plan.kv_cache_groups[1].kv_cache_spec.page_size_bytes == 32768
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {
        attention_count * 65536 + state_count * 32768
    }
    assert all(tensor.size <= budget for tensor in plan.kv_cache_tensors)
    views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout.BLHNC)
    assert views["attention"].shape[0] == attention_count
    assert views["state"].shape[0] == state_count
    views["state"].fill_(3)
    assert torch.count_nonzero(views["attention"]) == 0
    scheduler_plan = generate_scheduler_kv_cache_config([plan])
    manager = KVCacheManager(
        scheduler_plan,
        max_model_len=32,
        scheduler_block_size=32,
        hash_block_size=16,
        enable_caching=True,
    )
    assert [pool.get_num_free_blocks() for pool in manager.coordinator.block_pools] == [
        attention_count - 1,
        state_count - 1,
    ]
    assert get_max_concurrency_for_kv_cache_config(config, scheduler_plan) == min(
        (attention_count - 1) / 2, (state_count - 1) / 2
    )
    request = make_request("ratio", list(range(32)), block_size=16, hash_fn=sha256)
    allocated = manager.allocate_slots(request, 32, full_sequence_must_fit=True)
    assert allocated is not None
    assert [len(blocks) for blocks in allocated.blocks] == [2, 1]
    manager.free(request)
    assert manager.usage == 0


@pytest.mark.parametrize("ratio", [0.5, 0.9])
def test_mamba_ratio_live_pp_projection_has_common_counts_and_zero_byte_empty_groups(
    independent_mamba_groups, ratio
):
    from vllm.v1.worker.utils import allocate_kv_cache

    attention, state = independent_mamba_groups
    state_spec = replace(state.kv_cache_spec, block_size=32, mamba_cache_mode="align")
    workers = [
        {
            "a0": attention.kv_cache_spec,
            "a1": attention.kv_cache_spec,
            "m0": state_spec,
        },
        {"m1": state_spec, "m2": state_spec},
    ]
    budgets = [12 * 1024**2, 10 * 1024**2]
    config = mamba_ratio_config(ratio, budgets[0], pipeline_parallel_size=2)
    config.cache_config.kv_cache_layout = "BLHNC"
    plans = kv_cache_planning.get_kv_cache_configs(config, workers, budgets)
    assert plans[0].group_num_blocks == plans[1].group_num_blocks
    assert any(not group.layer_names for group in plans[0].kv_cache_groups)
    assert any(not group.layer_names for group in plans[1].kv_cache_groups)
    for worker, plan, budget in zip(workers, plans, budgets):
        assert all(tensor.size <= budget for tensor in plan.kv_cache_tensors)
        assert {
            name for group in plan.kv_cache_groups for name in group.layer_names
        } == set(worker)
        views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout.BLHNC)
        assert set(views) == set(worker)
        assert views[next(iter(worker))].untyped_storage().nbytes() == (
            plan.kv_cache_tensors[0].size
        )
        expected_size = sum(
            len(group.layer_names) * group.kv_cache_spec.page_size_bytes * count
            for group, count in zip(plan.kv_cache_groups, plan.group_num_blocks)
        )
        assert plan.kv_cache_tensors[0].size == expected_size
    scheduler_plan = generate_scheduler_kv_cache_config(plans)
    assert scheduler_plan.group_num_blocks == plans[0].group_num_blocks
    assert (
        get_max_concurrency_for_kv_cache_config(config, plans[0])
        == (get_max_concurrency_for_kv_cache_config(config, plans[1]))
        == get_max_concurrency_for_kv_cache_config(config, scheduler_plan)
    )


def test_mamba_ratio_none_preserves_live_legacy_plan(independent_mamba_groups):
    specs = {
        group.layer_names[0]: group.kv_cache_spec for group in independent_mamba_groups
    }
    budget = 12 * 1024**2
    config = mamba_ratio_config(None, budget, enable_prefix_caching=False)
    config.cache_config.kv_cache_layout = "BLHNC"
    expected = kv_cache_planning.get_kv_cache_config_from_groups(
        config,
        kv_cache_planning.get_kv_cache_groups(config, specs),
        budget,
    )
    (actual,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    assert actual == expected
    assert actual.group_num_blocks is None
    assert (
        CacheConfig().compute_hash()
        == CacheConfig(mamba_full_memory_ratio=None).compute_hash()
    )


@pytest.mark.parametrize(
    "budget,error",
    [
        (65536, "null block"),
        (512 * 1024, "one request at max_model_len"),
    ],
)
def test_mamba_ratio_live_planner_rejects_insufficient_physical_capacity(
    independent_mamba_groups, budget, error
):
    attention, state = independent_mamba_groups
    specs = {
        "attention": attention.kv_cache_spec,
        "state": replace(state.kv_cache_spec, block_size=256, mamba_cache_mode="align"),
    }
    config = mamba_ratio_config(0.5, budget, max_model_len=256)
    config.cache_config.kv_cache_layout = "BLHNC"
    with pytest.raises(ValueError, match=error):
        kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    assert config.model_config.max_model_len == 256


def test_mamba_ratio_live_auto_fit_uses_usable_group_counts(independent_mamba_groups):
    attention, state = independent_mamba_groups
    specs = {
        "attention": attention.kv_cache_spec,
        "state": replace(state.kv_cache_spec, block_size=256, mamba_cache_mode="align"),
    }
    budget = 512 * 1024
    config = mamba_ratio_config(0.5, budget, max_model_len=-1)
    config.cache_config.kv_cache_layout = "BLHNC"
    assert config.model_config.original_max_model_len == -1
    (plan,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    assert config.model_config.max_model_len == 64
    assert plan.group_num_blocks == (5, 5)
    assert get_max_concurrency_for_kv_cache_config(config, plan) == 1


def test_mamba_ratio_live_minimum_includes_verification_pages(independent_mamba_groups):
    attention, state = independent_mamba_groups
    specs = {
        "attention": attention.kv_cache_spec,
        "state": replace(
            state.kv_cache_spec,
            block_size=256,
            num_speculative_blocks=8,
            mamba_cache_mode="align",
        ),
    }
    budget = 512 * 1024
    config = mamba_ratio_config(0.5, budget, max_model_len=-1)
    config.cache_config.kv_cache_layout = "BLHNC"
    with pytest.raises(ValueError, match="even one token"):
        kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])


@pytest.mark.parametrize("ratio", [0, -0.5, float("inf"), float("nan")])
def test_mamba_ratio_config_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="greater than 0|finite number"):
        CacheConfig(mamba_full_memory_ratio=ratio)


def test_mamba_ratio_engine_args_reach_native_cache_config():
    from vllm.engine.arg_utils import EngineArgs

    config = EngineArgs(
        mamba_full_memory_ratio=0.5,
        kv_cache_memory_bytes=12 * 1024**2,
        max_model_len=32,
        enforce_eager=True,
    ).create_engine_config()
    assert config.cache_config.mamba_full_memory_ratio == 0.5
    assert config.cache_config.kv_cache_memory_bytes == 12 * 1024**2


def test_mamba_ratio_live_alignment_charges_physical_stride(independent_mamba_groups):
    attention, state = independent_mamba_groups
    alignment = 128 * 1024
    specs = {
        "attention": replace(attention.kv_cache_spec, block_stride_alignment=alignment),
        "state": replace(
            state.kv_cache_spec,
            block_size=32,
            block_stride_alignment=alignment,
            mamba_cache_mode="align",
        ),
    }
    budget = 12 * 1024**2
    config = mamba_ratio_config(0.5, budget)
    config.cache_config.kv_cache_layout = "BLHNC"
    (plan,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    state_count = int((budget - 2 * (alignment - 1)) / 3) // alignment
    attention_count = (
        budget - 2 * (alignment - 1) - state_count * alignment
    ) // alignment
    assert plan.group_num_blocks == (attention_count, state_count)
    assert {tensor.block_stride for tensor in plan.kv_cache_tensors} == {alignment}
    assert all(tensor.offset % alignment == 0 for tensor in plan.kv_cache_tensors)
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {
        (attention_count + state_count) * alignment
    }
    assert plan.kv_cache_tensors[0].size <= budget


@pytest.mark.parametrize("ratio", [0.5, 0.9])
def test_mamba_ratio_without_prefix_reserves_native_request_demand(
    independent_mamba_groups, ratio
):
    attention, state = independent_mamba_groups
    state_spec = replace(state.kv_cache_spec, block_size=256, num_speculative_blocks=2)
    specs = {"attention": attention.kv_cache_spec, "state": state_spec}
    budget = 2 * 1024**2
    config = mamba_ratio_config(
        ratio, budget, enable_prefix_caching=False, max_num_seqs=3
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    (plan,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    assert plan.group_num_blocks == (27, 10)
    assert plan.kv_cache_tensors[0].size == 27 * 65536 + 10 * 32768
    assert get_max_concurrency_for_kv_cache_config(config, plan) == 3
    manager = KVCacheManager(
        plan,
        max_model_len=32,
        scheduler_block_size=256,
        hash_block_size=16,
        enable_caching=False,
    )
    requests = [
        make_request(str(index), [index] * 32, block_size=16, hash_fn=sha256)
        for index in range(4)
    ]
    for request in requests[:3]:
        allocated = manager.allocate_slots(request, 32, full_sequence_must_fit=True)
        assert allocated is not None
        assert [len(blocks) for blocks in allocated.blocks] == [2, 3]
    assert manager.allocate_slots(requests[3], 32) is None
    assert manager.coordinator.block_pools[0].get_num_free_blocks() == 20
    assert manager.coordinator.block_pools[1].get_num_free_blocks() == 0
    for request in requests[:3]:
        manager.free(request)
    assert manager.usage == 0


@pytest.mark.parametrize("ratio", [0.5, 0.9])
def test_mamba_ratio_without_explicit_request_bound_remains_automatic(
    independent_mamba_groups, ratio
):
    attention, state = independent_mamba_groups
    specs = {
        "attention": attention.kv_cache_spec,
        "state": replace(state.kv_cache_spec, block_size=256, num_speculative_blocks=2),
    }
    budget = 2 * 1024**2
    config = mamba_ratio_config(
        ratio, budget, enable_prefix_caching=False, max_num_seqs=3
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    (fixed,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    config.cache_config.mamba_fixed_num_seqs = None
    (automatic,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    assert fixed.group_num_blocks[1] == 10
    assert automatic.group_num_blocks[1] > fixed.group_num_blocks[1]
    assert automatic.kv_cache_tensors[0].size <= budget


def test_mamba_ratio_without_prefix_rejects_unfunded_request_state(
    independent_mamba_groups,
):
    attention, state = independent_mamba_groups
    specs = {
        "attention": attention.kv_cache_spec,
        "state": replace(state.kv_cache_spec, block_size=256, num_speculative_blocks=2),
    }
    budget = 512 * 1024
    config = mamba_ratio_config(
        0.5, budget, enable_prefix_caching=False, max_num_seqs=48
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    with pytest.raises(ValueError, match="fixed Mamba state for max_num_seqs"):
        kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])


def test_mamba_ratio_replay_cursors_are_charged_once_per_namespace(
    independent_mamba_groups,
):
    attention, state = independent_mamba_groups
    state_spec = replace(
        state.kv_cache_spec,
        block_size=32,
        mamba_cache_mode="align",
        replayssm_tracker_bytes_per_block=12,
    )
    specs = {
        "a0": attention.kv_cache_spec,
        "a1": attention.kv_cache_spec,
        "m0": state_spec,
        "m1": state_spec,
    }
    count = 64
    rebinding_headroom = 12 * 4
    budget = 3 * count * (2 * 32768 + 12) + rebinding_headroom
    config = mamba_ratio_config(0.5, budget, max_num_seqs=4)
    config.cache_config.kv_cache_layout = "BLHNC"
    (plan,) = kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])
    assert plan.group_num_blocks == (64, 64)
    assert [
        group.replayssm_tracker_bytes_per_block for group in plan.kv_cache_groups
    ] == [0, 12]
    backing_size = count * (2 * 65536 + 2 * 32768)
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {backing_size}
    assert backing_size + 12 * count + rebinding_headroom <= budget
    assert (
        plan.group_num_blocks[1] * (2 * 32768 + 12)
        == (budget - rebinding_headroom) // 3
    )


def test_mamba_ratio_replay_auxiliary_storage_projects_pp_namespaces(
    independent_mamba_groups,
):
    attention, state = independent_mamba_groups
    state_spec = replace(
        state.kv_cache_spec,
        block_size=32,
        mamba_cache_mode="align",
        replayssm_tracker_bytes_per_block=12,
    )
    workers = [
        {
            "a0": attention.kv_cache_spec,
            "a1": attention.kv_cache_spec,
            "m0": state_spec,
        },
        {"m1": state_spec, "m2": state_spec},
    ]
    budgets = [12 * 1024**2, 10 * 1024**2]
    config = mamba_ratio_config(
        0.5, budgets[0], pipeline_parallel_size=2, max_num_seqs=4
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    plans = kv_cache_planning.get_kv_cache_configs(config, workers, budgets)
    assert plans[0].group_num_blocks == plans[1].group_num_blocks
    for plan, budget in zip(plans, budgets):
        auxiliary = sum(
            group.replayssm_tracker_bytes_per_block * (count + 4)
            for group, count in zip(plan.kv_cache_groups, plan.group_num_blocks)
        )
        assert plan.kv_cache_tensors[0].size + auxiliary <= budget
        for group in plan.kv_cache_groups:
            if not group.layer_names:
                assert group.replayssm_tracker_bytes_per_block == 0
    assert (
        sum(
            group.replayssm_tracker_bytes_per_block
            for group in plans[0].kv_cache_groups
        )
        == 12
    )
    assert (
        sum(
            group.replayssm_tracker_bytes_per_block
            for group in plans[1].kv_cache_groups
        )
        == 24
    )


@pytest.mark.parametrize("layout", [layout.name for layout in KVCacheLayout])
def test_mamba_ratio_minimal_profiling_builder_preserves_independent_counts(
    independent_mamba_groups, layout
):
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.worker.utils import allocate_kv_cache

    config = mamba_ratio_config(0.5, 12 * 1024**2)
    config.cache_config.kv_cache_layout = layout
    builder = get_kv_cache_config_builder(config)
    plan = builder.get_kv_cache_config_from_groups(
        config, independent_mamba_groups, num_blocks=4
    )
    assert plan.group_num_blocks == (4, 4)
    assert plan.num_blocks == 4
    assert {tensor.size for tensor in plan.kv_cache_tensors} == {4 * (65536 + 32768)}
    views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout[layout])
    views["state"].fill_(9)
    assert torch.count_nonzero(views["attention"]) == 0


@pytest.mark.parametrize("cache_types", ["full", "sliding", "mixed"])
@pytest.mark.parametrize("disable_hybrid", [False, True])
def test_mamba_ratio_dense_preserves_legacy_plan_and_profiling(
    independent_mamba_groups, cache_types, disable_hybrid
):
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.worker.utils import allocate_kv_cache

    attention = independent_mamba_groups[0].kv_cache_spec
    sliding = SlidingWindowSpec(
        block_size=attention.block_size,
        num_kv_heads=attention.num_kv_heads,
        head_size=attention.head_size,
        dtype=attention.dtype,
        sliding_window=16,
    )
    specs = {}
    if cache_types != "sliding":
        specs["attention"] = attention
    if cache_types != "full":
        specs["sliding"] = sliding
    budget = 12 * 1024**2
    plans = []
    profiles = []
    for ratio in (None, 0.9):
        config = mamba_ratio_config(ratio, budget)
        config.cache_config.kv_cache_layout = "BLHNC"
        config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hybrid
        (plan,) = kv_cache_planning.get_kv_cache_configs(
            config, [dict(specs)], [budget]
        )
        plans.append(plan)
        groups = kv_cache_planning.get_kv_cache_groups(config, dict(specs))
        profiles.append(
            get_kv_cache_config_builder(config).get_kv_cache_config_from_groups(
                config, groups, num_blocks=4
            )
        )
        assert plan.group_num_blocks is None
        views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout.BLHNC)
        assert set(views) == set(specs)
    assert plans[0] == plans[1]
    assert profiles[0] == profiles[1]


@pytest.mark.parametrize("ratio", [0.5, 0.9])
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
@pytest.mark.parametrize("empty_pp_stage", [False, True])
def test_mamba_ratio_pure_state_uses_native_budget_and_nulls(
    independent_mamba_groups, ratio, enable_prefix_caching, empty_pp_stage
):
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.worker.utils import allocate_kv_cache

    state_spec = replace(
        independent_mamba_groups[1].kv_cache_spec,
        num_speculative_blocks=1,
        mamba_cache_mode="align" if enable_prefix_caching else "none",
        replayssm_tracker_bytes_per_block=12,
    )
    workers = [{"state": state_spec}]
    if empty_pp_stage:
        workers.insert(0, {})
    budget = 12 * 1024**2
    config = mamba_ratio_config(
        ratio,
        budget,
        pipeline_parallel_size=len(workers),
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=3,
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    config.cache_config.mamba_cache_mode = state_spec.mamba_cache_mode
    plans = kv_cache_planning.get_kv_cache_configs(
        config, workers, [budget] * len(workers)
    )
    if enable_prefix_caching:
        numerator, denominator = ratio.as_integer_ratio()
        state_budget = (budget - 12 * 3) * numerator // (numerator + denominator)
        expected_count = state_budget // (state_spec.page_size_bytes + 12)
    else:
        expected_count = 1 + 3 * 2
    for worker, plan in zip(workers, plans):
        assert plan.group_num_blocks == (expected_count,)
        views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout.BLHNC)
        assert set(views) == set(worker)
        if worker:
            assert views["state"].shape[0] == expected_count
            backing = expected_count * state_spec.page_size_bytes
            assert {tensor.size for tensor in plan.kv_cache_tensors} == {backing}
            assert backing + 12 * (expected_count + 3) <= budget
        else:
            assert plan.kv_cache_tensors == []
            assert plan.kv_cache_groups[0].replayssm_tracker_bytes_per_block == 0
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config(plans),
        max_model_len=32,
        scheduler_block_size=state_spec.block_size,
        hash_block_size=state_spec.block_size,
        enable_caching=enable_prefix_caching,
    )
    assert (
        manager.coordinator.block_pools[0].get_num_free_blocks() == expected_count - 1
    )
    builder = get_kv_cache_config_builder(config)
    profile = builder.get_kv_cache_config_from_groups(
        config, plans[-1].kv_cache_groups, num_blocks=4
    )
    assert profile.group_num_blocks == (4,)
    if empty_pp_stage:
        profile = builder.get_kv_cache_config_from_groups(
            config, plans[0].kv_cache_groups, num_blocks=4
        )
        assert profile.group_num_blocks is None
        assert profile.kv_cache_tensors == []


@pytest.mark.parametrize("ratio", [0.5, 0.9])
@pytest.mark.parametrize("include_full_attention", [False, True])
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
@pytest.mark.parametrize("projection", ["single", "uneven", "empty"])
def test_mamba_ratio_with_sliding_window_uses_physical_attention_budget(
    independent_mamba_groups,
    ratio,
    include_full_attention,
    enable_prefix_caching,
    projection,
):
    from vllm.v1.worker.utils import allocate_kv_cache

    attention, state = independent_mamba_groups
    state_spec = replace(state.kv_cache_spec, block_size=256)
    specs = {
        "sliding": SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=16,
        ),
        "state": state_spec,
    }
    if include_full_attention:
        specs["attention"] = attention.kv_cache_spec
    workers = [specs]
    if projection == "uneven":
        workers = [
            {name: spec for name, spec in specs.items() if name != "state"},
            {"state": state_spec, "state2": state_spec},
        ]
    elif projection == "empty":
        workers.append({})
    budgets = [2 * 1024**2 - index * 65536 for index in range(len(workers))]
    config = mamba_ratio_config(
        ratio,
        budgets[0],
        pipeline_parallel_size=len(workers),
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=3,
    )
    config.cache_config.kv_cache_layout = "BLHNC"
    plans = kv_cache_planning.get_kv_cache_configs(config, workers, budgets)
    costs = [
        (
            sum(
                spec.page_size_bytes
                for spec in worker.values()
                if isinstance(spec, MambaSpec)
            ),
            sum(
                spec.page_size_bytes
                for spec in worker.values()
                if not isinstance(spec, MambaSpec)
            ),
        )
        for worker in workers
    ]
    numerator, denominator = ratio.as_integer_ratio()
    state_count = (
        min(
            (budget * numerator // (numerator + denominator)) // state_bytes
            for (state_bytes, _), budget in zip(costs, budgets)
            if state_bytes
        )
        if enable_prefix_caching
        else 4
    )
    attention_count = min(
        (budget - state_count * state_bytes) // attention_bytes
        for (state_bytes, attention_bytes), budget in zip(costs, budgets)
        if attention_bytes
    )
    assert {spec.page_size_bytes for spec in specs.values()} == (
        {16384, 32768, 65536} if include_full_attention else {16384, 32768}
    )
    for worker, plan, budget in zip(workers, plans, budgets):
        assert plan.group_num_blocks is not None
        assert plan.group_num_blocks == plans[0].group_num_blocks
        expected_size = 0
        for group, count in zip(plan.kv_cache_groups, plan.group_num_blocks):
            is_state = isinstance(group.kv_cache_spec, MambaSpec)
            assert count == (state_count if is_state else attention_count)
            expected_size += sum(
                worker[name].page_size_bytes * count for name in group.layer_names
            )
        assert {tensor.size for tensor in plan.kv_cache_tensors} == (
            {expected_size} if worker else set()
        )
        views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout.BLHNC)
        assert set(views) == set(worker)
        backing_sizes = {
            view.untyped_storage().data_ptr(): view.untyped_storage().nbytes()
            for view in views.values()
        }
        assert len(backing_sizes) == (1 if worker else 0)
        assert sum(backing_sizes.values()) == expected_size
        assert sum(backing_sizes.values()) <= budget
        for name, view in views.items():
            assert view.shape[0] == (
                state_count if isinstance(worker[name], MambaSpec) else attention_count
            )
        if not worker:
            assert plan.kv_cache_tensors == []
    scheduler_plan = generate_scheduler_kv_cache_config(plans)
    requirements = [
        group.kv_cache_spec.max_memory_usage_bytes(config)
        // group.kv_cache_spec.page_size_bytes
        for group in scheduler_plan.kv_cache_groups
    ]
    assert all(
        count - 1 >= required
        for count, required in zip(scheduler_plan.group_num_blocks, requirements)
    )
    assert get_max_concurrency_for_kv_cache_config(config, scheduler_plan) == min(
        (count - 1) / required
        for count, required in zip(scheduler_plan.group_num_blocks, requirements)
    )


def test_mamba_ratio_sliding_window_rejects_unfunded_minimum_request(
    independent_mamba_groups,
):
    specs = {
        "sliding": SlidingWindowSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=16,
        ),
        "state": replace(independent_mamba_groups[1].kv_cache_spec, block_size=256),
    }
    budget = 6 * 32768
    config = mamba_ratio_config(0.5, budget, max_model_len=256)
    config.cache_config.kv_cache_layout = "BLHNC"
    assert specs["sliding"].max_memory_usage_bytes(config) == 17 * 16384
    with pytest.raises(ValueError, match="cannot serve one request at max_model_len"):
        kv_cache_planning.get_kv_cache_configs(config, [specs], [budget])


def test_mamba_ratio_sliding_window_manager_rollover_and_atomic_saturation(
    independent_mamba_groups,
):
    groups = [
        KVCacheGroupSpec(
            ["sliding"],
            SlidingWindowSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=128,
                dtype=torch.bfloat16,
                sliding_window=16,
            ),
        ),
        replace(
            independent_mamba_groups[1],
            kv_cache_spec=replace(
                independent_mamba_groups[1].kv_cache_spec, block_size=256
            ),
        ),
    ]
    manager = independent_capacity_manager(groups, (3, 3), scheduler_block_size=256)
    request = make_request("rollover", [1] * 64, block_size=16, hash_fn=sha256)
    assert manager.allocate_slots(request, 16) is not None
    first_ids = manager.get_block_ids(request.request_id)[0]
    request.num_computed_tokens = 16
    assert manager.allocate_slots(request, 16) is not None
    request.num_computed_tokens = 32
    assert manager.allocate_slots(request, 16) is not None
    rolled_ids = manager.get_block_ids(request.request_id)[0]
    assert rolled_ids[0] == 0
    assert rolled_ids[-1] == first_ids[0]
    pools = manager.coordinator.block_pools
    before = [pool.get_num_free_blocks() for pool in pools]
    assert before == [0, 1]
    other = make_request("blocked", [2] * 16, block_size=16, hash_fn=sha256)
    assert manager.allocate_slots(other, 16) is None
    assert [pool.get_num_free_blocks() for pool in pools] == before
    manager.free(request)
    assert manager.allocate_slots(other, 16) is not None
    manager.free(other)
    assert manager.usage == 0


def test_mamba_ratio_attention_only_pp_stage_profiles_legacy_but_plans_globally(
    independent_mamba_groups,
):
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.worker.utils import allocate_kv_cache

    attention, state = independent_mamba_groups
    workers = [
        {"attention": attention.kv_cache_spec},
        {"state": replace(state.kv_cache_spec, mamba_cache_mode="align")},
    ]
    budget = 12 * 1024**2
    config = mamba_ratio_config(0.9, budget, pipeline_parallel_size=2)
    config.cache_config.kv_cache_layout = "BLHNC"
    plans = kv_cache_planning.get_kv_cache_configs(config, workers, [budget, budget])
    assert plans[0].group_num_blocks == plans[1].group_num_blocks
    assert plans[0].group_num_blocks is not None
    assert len(plans[0].group_num_blocks) == 2
    profiles = []
    for ratio in (None, 0.9):
        profile_config = mamba_ratio_config(ratio, budget, pipeline_parallel_size=2)
        profile_config.cache_config.kv_cache_layout = "BLHNC"
        profiles.append(
            get_kv_cache_config_builder(profile_config).get_kv_cache_config_from_groups(
                profile_config, plans[0].kv_cache_groups, num_blocks=4
            )
        )
    assert profiles[0] == profiles[1]
    assert profiles[1].group_num_blocks is None
    for worker, plan in zip(workers, plans):
        views = allocate_kv_cache(plan, torch.device("cpu"), KVCacheLayout.BLHNC)
        assert set(views) == set(worker)
        for tensor in plan.kv_cache_tensors:
            assert views[tensor.layers[0]].shape[0] == plan.num_blocks_of(tensor)
