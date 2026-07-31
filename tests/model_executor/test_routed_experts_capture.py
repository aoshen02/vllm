# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.distributed.kv_transfer.kv_connector.v1.sidecar import (
    KVConnectorSidecarBlockMap,
    KVConnectorSidecarConfig,
    KVConnectorSidecarTransferPlan,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
    RoutedExpertsCapturer,
    RoutedExpertsCaptureState,
    RoutedExpertsManager,
    RoutedExpertsTensors,
    RoutedExpertsWriteTask,
    bind_routed_experts_capturer,
    require_full_attn_group_id,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test

_CAPTURER_MODULE = (
    "vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer"
)


def test_multiple_full_attention_groups_use_hashable_warning_args():
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

    assert require_full_attn_group_id(kv_cache_config) == 0


def test_routed_experts_write_task_publishes_copied_tensors():
    routing_data = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
    slot_mapping = torch.tensor([5, 9], dtype=torch.int64)
    shm_writer = Mock()
    output = SimpleNamespace(routed_experts_slots=None)
    write_task = RoutedExpertsWriteTask(
        routed_experts_tensors=RoutedExpertsTensors(routing_data, slot_mapping),
        shm_writer=shm_writer,
    )

    write_task.start_copy()
    write_task.finalize(output)

    stored_routing, stored_slots = shm_writer.store_batch.call_args.args
    assert stored_routing.tolist() == routing_data.tolist()
    assert stored_slots.tolist() == slot_mapping.tolist()
    assert output.routed_experts_slots.tolist() == slot_mapping.tolist()


def test_routed_experts_manager_applies_public_sidecar_transfers():
    manager = RoutedExpertsManager.__new__(RoutedExpertsManager)
    manager.routed_experts_by_offload_block = np.zeros(
        (2, 2, 2, 1, 1),
        dtype=np.uint8,
    )
    manager._blocks_view = np.arange(6, dtype=np.uint8).reshape(3, 2, 1, 1)
    stores = KVConnectorSidecarBlockMap(
        gpu_block_ids=np.array([0, 1]),
        connector_block_ids=np.array([1, 1]),
        connector_block_offsets=np.array([0, 1]),
    )

    manager.apply_offload_transfers(KVConnectorSidecarTransferPlan(store=stores))

    np.testing.assert_array_equal(
        manager.routed_experts_by_offload_block[1, 0],
        np.array([0, 1], dtype=np.uint8).reshape(2, 1, 1),
    )
    np.testing.assert_array_equal(
        manager.routed_experts_by_offload_block[1, 1],
        np.array([2, 3], dtype=np.uint8).reshape(2, 1, 1),
    )

    manager._blocks_view[2].fill(0)
    loads = KVConnectorSidecarBlockMap(
        gpu_block_ids=np.array([2]),
        connector_block_ids=np.array([1]),
        connector_block_offsets=np.array([1]),
    )
    manager.apply_offload_transfers(KVConnectorSidecarTransferPlan(load=loads))
    np.testing.assert_array_equal(
        manager._blocks_view[2],
        np.array([2, 3], dtype=np.uint8).reshape(2, 1, 1),
    )


@pytest.mark.parametrize(
    "sidecar_config",
    [
        KVConnectorSidecarConfig(0, 1),
        KVConnectorSidecarConfig(1, 0),
    ],
)
def test_routed_experts_manager_rejects_invalid_sidecar_layout(
    sidecar_config: KVConnectorSidecarConfig,
):
    with pytest.raises(ValueError, match="sidecar block counts must be positive"):
        RoutedExpertsManager(Mock(), Mock(), sidecar_config)


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


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
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
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

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
    shm_writer = Mock()
    state = RoutedExpertsCaptureState(capturer, shm_writer, full_attn_group_id=0)

    state.close()

    shm_writer.close.assert_called_once_with()
    assert state.capturer is None
    assert state.shm_writer is None


@pytest.mark.parametrize(("rank", "creates_writer"), [(0, True), (1, False)])
def test_capture_state_create_uses_explicit_dependencies(
    monkeypatch, rank, creates_writer
):
    from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
        state as state_module,
    )

    model = Mock()
    capturer = Mock()
    shm_writer = Mock()
    writer_factory = Mock(return_value=shm_writer)
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
    monkeypatch.setattr(state_module, "RoutedExpertsShmWriter", writer_factory)
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
            self.routed_experts = _make_modular_routed_experts()
            self._quant_method = self.routed_experts.quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    model = SimpleNamespace(modules=lambda: [target_module])
    capturer = SimpleNamespace(capture=lambda *_: None)
    bind_routed_experts_capturer(model, capturer)

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None


def test_bind_routed_experts_capturer_rejects_unsupported_monolithic(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            SimpleNamespace(modules=lambda: [dummy_module]), DummyCapturer()
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
    )
    setattr(parallel_config, parallel_field, 2)
    config = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        parallel_config=parallel_config,
        kv_transfer_config=None,
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
