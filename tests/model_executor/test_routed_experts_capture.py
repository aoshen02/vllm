# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
    RoutedExpertsCapturer,
    RoutedExpertsCaptureState,
    RoutedExpertsTensors,
    RoutedExpertsWorkerWriter,
    RoutedExpertsWriteTask,
    bind_routed_experts_capturer,
    require_full_attn_group_id,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test

_CAPTURER_MODULE = (
    "vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer"
)


def test_worker_writer_rejects_unshared_engine_core_mmap(tmp_path, monkeypatch):
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        shared_region,
    )

    writer = RoutedExpertsWorkerWriter(
        instance_id="missing",
        dp_rank=0,
        slot_shape=(1, 1, 1),
        dtype="uint8",
    )
    writer._path = str(tmp_path / "missing.mmap")
    monkeypatch.setattr(shared_region, "_WAIT_TIMEOUT_S", -1.0)

    with pytest.raises(TimeoutError, match="rank 0.*same host"):
        writer.validate()


def test_multiple_full_attention_groups_are_rejected():
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], kv_cache_spec),
            KVCacheGroupSpec(["layer.1"], kv_cache_spec),
        ],
    )

    with pytest.raises(ValueError, match="exactly one full-attention"):
        require_full_attn_group_id(kv_cache_config)


def test_hybrid_groups_with_one_full_attention_anchor_are_supported():
    full_attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    sliding_window_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=128,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], full_attention_spec),
            KVCacheGroupSpec(["layer.1"], sliding_window_spec),
        ],
    )

    assert require_full_attn_group_id(kv_cache_config) == 0


def test_routed_experts_write_task_publishes_copied_tensors():
    routing_data = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
    slot_mapping = torch.tensor([5, 9], dtype=torch.int64)
    writer = Mock()
    write_task = RoutedExpertsWriteTask(
        routed_experts_tensors=RoutedExpertsTensors(routing_data, slot_mapping),
        writer=writer,
    )

    write_task.start_copy()
    write_task.finalize()

    stored_routing, stored_slots = writer.store_batch.call_args.args
    assert stored_routing.tolist() == routing_data.tolist()
    assert stored_slots.tolist() == slot_mapping.tolist()


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    moe_top_k: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.dp_rank = dp_rank
    capturer.tp_size = tp_size
    capturer.device_buffer = torch.full(
        (max_tokens, num_layers, moe_top_k),
        -1,
        dtype=torch.int32,
    )
    return capturer


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(expert_ids: torch.Tensor) -> None:
        captured.append(expert_ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(expert_ids: torch.Tensor) -> None:
        captured.append(expert_ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_bind_routed_experts_capturer(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()
            self.is_monolithic = False

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    model = SimpleNamespace(modules=lambda: [dummy_module])

    capturer = DummyCapturer()
    bind_routed_experts_capturer(model, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_capture_state_write_task_owns_immutable_snapshot():
    capturer = _capturer_with_buffer(max_tokens=4, num_layers=2)
    capturer.device_buffer.copy_(torch.arange(16).reshape(4, 2, 2))
    state = RoutedExpertsCaptureState(capturer, Mock(), full_attn_group_id=1)
    slot_mappings = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.int64
    )

    write_task = state.make_write_task(slot_mappings[1], 3)

    assert write_task is not None
    tensors = write_task.routed_experts_tensors
    assert torch.equal(tensors.routing_data, capturer.device_buffer[:3])
    assert torch.equal(tensors.slot_mapping, torch.tensor([21, 22, 23]))
    capturer.clear_buffer()
    slot_mappings.fill_(-1)
    assert torch.equal(
        tensors.routing_data,
        torch.arange(12, dtype=torch.int32).reshape(3, 2, 2),
    )
    assert torch.equal(tensors.slot_mapping, torch.tensor([21, 22, 23]))


def test_capture_state_close_releases_resources():
    capturer = Mock()
    writer = Mock()
    state = RoutedExpertsCaptureState(capturer, writer, full_attn_group_id=0)

    state.close()

    writer.close.assert_called_once_with()
    assert state.capturer is None
    assert state.writer is None


@pytest.mark.parametrize(("rank", "creates_writer"), [(0, True), (1, False)])
def test_capture_state_create_uses_explicit_dependencies(
    monkeypatch, rank, creates_writer
):
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        state as state_module,
    )

    model = Mock()
    capturer = Mock()
    writer = Mock()
    writer_factory = Mock(return_value=writer)
    bind = Mock()
    monkeypatch.setattr(state_module, "require_full_attn_group_id", lambda _: 2)
    monkeypatch.setattr(
        state_module, "RoutedExpertsCapturer", Mock(return_value=capturer)
    )
    monkeypatch.setattr(state_module, "bind_routed_experts_capturer", bind)
    monkeypatch.setattr(state_module, "get_routed_experts_output_rank", lambda: 0)
    monkeypatch.setattr(
        state_module,
        "get_routing_slot_shape_and_dtype",
        lambda *_: ((16, 4, 2), "uint8"),
    )
    monkeypatch.setattr(state_module, "RoutedExpertsWorkerWriter", writer_factory)
    vllm_config = SimpleNamespace(
        instance_id="instance",
        parallel_config=SimpleNamespace(rank=rank, data_parallel_rank=3),
    )
    kv_cache_config = Mock()

    state = RoutedExpertsCaptureState.create(
        model=model,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        max_num_batched_tokens=32,
    )

    assert state.capturer is capturer
    assert state.full_attn_group_id == 2
    assert state.can_write is creates_writer
    bind.assert_called_once_with(model, capturer)
    assert writer_factory.called is creates_writer


def test_bind_routed_experts_capturer_only_visits_target_model(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self.is_monolithic = False

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    model = SimpleNamespace(modules=lambda: [target_module])
    capturer = SimpleNamespace(capture=lambda *_: None)
    bind_routed_experts_capturer(model, capturer)

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None


def test_bind_routed_experts_capturer_rejects_monolithic_kernel(monkeypatch):
    class DummyFusedMoE:
        layer_id = 3
        router = _make_router()
        is_monolithic = True
        _quant_method = SimpleNamespace()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    model = SimpleNamespace(modules=lambda: [DummyFusedMoE()])

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            model,
            SimpleNamespace(capture=lambda *_: None),
        )


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert unsupported == []


@pytest.mark.parametrize(
    ("parallel_field", "error"),
    [
        ("pipeline_parallel_size", "pipeline parallelism"),
        ("decode_context_parallel_size", "context parallelism"),
        ("prefill_context_parallel_size", "context parallelism"),
    ],
)
def test_routed_experts_reject_unsupported_parallelism(parallel_field, error):
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
        ubatch_size=1,
        enable_elastic_ep=False,
    )
    setattr(parallel_config, parallel_field, 2)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            is_moe=True,
            runner_type="generate",
            is_encoder_decoder=False,
            is_multimodal_model=False,
            is_diffusion=False,
        ),
        parallel_config=parallel_config,
        device_config=SimpleNamespace(device_type="cuda"),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        ec_transfer_config=None,
        artifact_config=SimpleNamespace(shm_dir="/dev/shm"),
    )

    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_return_routed_experts_compatibility(config)


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    forward_context = SimpleNamespace(dp_metadata=None)
    with patch(
        f"{_CAPTURER_MODULE}.get_forward_context",
        return_value=forward_context,
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk_ids)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """Slice this rank's rows from routing concatenated across DP ranks."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(
        f"{_CAPTURER_MODULE}.get_forward_context",
        return_value=forward_context,
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    expected = topk_ids[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], expected)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """Capture routing that is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk_ids = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(
        f"{_CAPTURER_MODULE}.get_forward_context",
        return_value=forward_context,
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk_ids)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(
            f"{_CAPTURER_MODULE}.get_forward_context",
            return_value=forward_context,
        ),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    assert capturer.device_buffer[0, 0, 0].item() == -1
