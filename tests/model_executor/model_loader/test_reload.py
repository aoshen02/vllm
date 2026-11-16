# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.machinery
import sys
import types
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
import torch

from vllm.config import CacheConfig
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLWeightTransferEngine,
)
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)
from vllm.model_executor.model_loader.online_processing import (
    initialize_online_processing,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    finalize_model_weights,
    load_attention_sink,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize(
    "module_name,class_name,sink_name",
    [
        ("mimo_v2_mtp", "MiMoV2MTP", "attention_sink_bias"),
        ("mimo_v2_mtp", "MiMoV2OmniMTP", "attention_sink_bias"),
        ("laguna", "LagunaModel", "sink"),
        ("qwen3_dflash", "DFlashQwen3Model", "attention_sink_bias"),
    ],
)
def test_attention_sink_model_loading_shards_once(
    monkeypatch, tp_rank, module_name, class_name, sink_name
):
    """Model loading must pass global sinks to the loader on cold load and reload."""
    module = import_module(f"vllm.model_executor.models.{module_name}")
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.models.utils import WeightsMapper

    for target in (module, weight_utils):
        monkeypatch.setattr(
            target, "get_tensor_model_parallel_rank", lambda: tp_rank, raising=False
        )
        monkeypatch.setattr(target, "get_tensor_model_parallel_world_size", lambda: 2)
    cls = getattr(module, class_name)
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.hf_to_vllm_mapper = WeightsMapper()
    model._context_kv_proj = None  # set by DFlashQwen3Model.__init__
    model.model = torch.nn.Module()
    model.model.mtp = torch.nn.Module()
    sink = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    def sink_loader(param, loaded_weight):
        assert loaded_weight.shape == (8,), "Sink was sharded before weight_loader"
        load_attention_sink(param, loaded_weight)

    sink.weight_loader = sink_loader
    model.model.mtp.register_parameter(sink_name, sink)
    name = f"model.mtp.{sink_name}"
    checkpoint = torch.arange(8, dtype=torch.float32)
    assert model.load_weights([(name, checkpoint)]) == {name}
    assert torch.equal(sink, checkpoint.chunk(2)[tp_rank])
    # a reload writes the live parameter through the same loader
    model.load_weights([(name, checkpoint + 10)])
    assert torch.equal(sink, (checkpoint + 10).chunk(2)[tp_rank])
    assert getattr(model.model.mtp, sink_name).data_ptr() == sink.data_ptr()


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize(
    "module_name,class_name,attention_name,sink_name",
    [
        ("deepseek_v4", "DeepseekV4Model", "attn", "attn_sink"),
        ("deepseek_v41", "DeepseekV4Model", "attn", "attn_sink"),
        ("hy_v4", "HYV4Model", "self_attn", "learnable_sink_param"),
    ],
)
def test_attention_sink_deepseek_hy_loading_uses_parameter_loader(
    monkeypatch, tp_rank, module_name, class_name, attention_name, sink_name
):
    """DeepSeek and HY-V4 pass global sinks to the parameter's current loader."""
    module = import_module(f"vllm.models.{module_name}.nvidia.model")
    from vllm.model_executor.model_loader import weight_utils

    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: tp_rank)
    cls = getattr(module, class_name)
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.quant_config = None
    model.get_expert_mapping = lambda: []
    attention = torch.nn.Module()
    layer = torch.nn.Module()
    setattr(layer, attention_name, attention)
    model.layers = torch.nn.ModuleList([layer])
    sink = torch.nn.Parameter(torch.zeros(6), requires_grad=False)
    attention.register_parameter(sink_name, sink)
    checkpoint = torch.arange(8, dtype=torch.float32)
    name = f"layers.0.{attention_name}.{sink_name}"

    for offset in (0, 10):
        weight = checkpoint + offset
        sink.weight_loader = Mock(side_effect=load_attention_sink)
        assert model.load_weights([(name, weight)]) == {name}
        sink.weight_loader.assert_called_once_with(sink, weight)
        assert torch.equal(sink[:4], weight.chunk(2)[tp_rank])
        assert torch.isneginf(sink[4:]).all()


def _gpt_oss_container(monkeypatch, quant_method=None):
    """A minimal GptOssModel whose loading paths can be driven directly."""
    from vllm.model_executor.models import gpt_oss

    group = SimpleNamespace(world_size=1, rank_in_group=0)
    for name in ("get_dp_group", "get_ep_group", "get_pcp_group"):
        monkeypatch.setattr(gpt_oss, name, lambda: group)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(gpt_oss, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(gpt_oss, "is_pp_missing_parameter", lambda *args: False)
    model = gpt_oss.GptOssModel.__new__(gpt_oss.GptOssModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_local_experts=1, intermediate_size=32)
    if quant_method is not None:
        model.config.quantization_config = {"quant_method": quant_method}
    model.parallel_config = SimpleNamespace(enable_expert_parallel=True)
    model.layers = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                experts=SimpleNamespace(
                    _quant_method=SimpleNamespace(weight_dtype=None)
                )
            )
        )
    ]
    model.get_expert_mapping = lambda: []  # type: ignore[method-assign]
    return model


@pytest.mark.parametrize("quant_method", ["mxfp4", "quark", None])
def test_gpt_oss_attention_sink_uses_parameter_loader(monkeypatch, quant_method):
    """All quantization dispatch paths honor the sink's current weight loader."""
    model = _gpt_oss_container(monkeypatch, quant_method)
    model.sinks = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    # A replacement loader must see the unsliced checkpoint even after reload
    # wraps the parameter; bare model-specific copies would bypass this spy.
    model.sinks.weight_loader = Mock(side_effect=default_weight_loader)
    checkpoint = torch.arange(4, dtype=torch.float32)
    assert model.load_weights([("sinks", checkpoint)]) == {"sinks"}
    model.sinks.weight_loader.assert_called_once_with(model.sinks, checkpoint)
    assert torch.equal(model.sinks, checkpoint)


def test_gpt_oss_attention_sink_skips_missing_parameter(monkeypatch):
    """A sink owned by another pipeline stage is skipped, not a KeyError."""
    from vllm.model_executor.models import gpt_oss

    model = _gpt_oss_container(monkeypatch)
    monkeypatch.setattr(gpt_oss, "is_pp_missing_parameter", lambda *args: True)
    # The parameter does not exist on this rank, so loading must not touch it.
    sinks = [("sinks", torch.arange(4, dtype=torch.float32))]
    assert model.load_weights(sinks) == set()


def _fp8_reload_unsupported() -> bool:
    """Whether the FP8 reload/online-quantize tests should be skipped.

    `supports_fp8()` returns True on MI250 (gfx90a) and RDNA3 (gfx11)
    because the general quantization paths upcast FP8 weights, but neither has
    native FP8 and cannot run these reload models, so treat them as
    unsupported here.
    """
    if not current_platform.supports_fp8():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx11, on_gfx90a

        return on_gfx90a() or on_gfx11()
    return False


class _ReloadableAttentionLayer(
    torch.nn.Module,
    AttentionLayerBase,
):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_config(self, vllm_config):
        return None

    def process_weights_after_loading(self) -> None:
        self.post_load_called = True


_MARLIN_SIZE_K, _MARLIN_SIZE_N, _MARLIN_GROUP_SIZE = 128, 64, 64


def _stub_marlin_ops(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda w, size_k, size_n, num_bits: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )


def _make_marlin_kernel():
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    kernel = object.__new__(MarlinLinearKernel)
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        partition_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=_MARLIN_GROUP_SIZE,
        zero_points=False,
    )
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    return kernel


def _load_marlin_checkpoint_format_weights(layer):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    layer.qweight = PackedvLLMParameter(
        data=torch.zeros(_MARLIN_SIZE_K // 8, _MARLIN_SIZE_N, dtype=torch.int32),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=8,
        weight_loader=default_weight_loader,
    )
    layer.scales = GroupQuantScaleParameter(
        data=torch.ones(
            _MARLIN_SIZE_K // _MARLIN_GROUP_SIZE, _MARLIN_SIZE_N, dtype=torch.float16
        ),
        input_dim=0,
        output_dim=1,
        weight_loader=default_weight_loader,
    )


def test_marlin_post_load_does_not_own_workspace(monkeypatch, dist_init):
    """Weight reload must not create layer-owned Marlin lock storage."""
    _stub_marlin_ops(monkeypatch)
    kernel = _make_marlin_kernel()

    layer = torch.nn.Module()
    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")

    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")


@pytest.mark.parametrize("variant", ["fp8", "mxfp8", "nvfp4"])
def test_marlin_prepare_layer_does_not_own_workspace(monkeypatch, variant):
    """Weight preparation must not attach runtime workspace to model layers."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import (
        marlin_utils,
        marlin_utils_fp4,
        marlin_utils_fp8,
    )

    size_k, size_n = 128, 64

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda b_q_weight, size_k, size_n, num_bits: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = torch.float16
    layer.params_dtype = torch.float16

    if variant == "fp8":
        prepare = marlin_utils_fp8.prepare_fp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_k, size_n, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )
    elif variant == "mxfp8":
        prepare = marlin_utils_fp8.prepare_mxfp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.full((size_n, size_k // 32), 127, dtype=torch.uint8),
                requires_grad=False,
            )
    else:
        prepare = marlin_utils_fp4.prepare_fp4_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(size_n, size_k // 16, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_global_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    load_checkpoint_format_weights()
    prepare(layer)
    assert not hasattr(layer, "workspace")

    # Reload: fresh checkpoint-format tensors, prepare runs again
    load_checkpoint_format_weights()
    prepare(layer)

    assert not hasattr(layer, "workspace")


def test_marlin_workspace_uses_persistent_workspace_manager(monkeypatch):
    """Calls reuse initialized locks; independent streams get separate storage."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    device = torch.device("cpu")
    manager = workspace_module.WorkspaceManager(device)
    monkeypatch.setattr(workspace_module, "_manager", manager)
    stream = "main"
    monkeypatch.setattr(torch_utils, "current_stream", lambda: stream)

    workspace = marlin_utils.get_marlin_workspace(device)
    assert workspace.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert workspace.dtype == torch.int32
    assert torch.count_nonzero(workspace) == 0

    workspace.fill_(1)
    assert marlin_utils.get_marlin_workspace(device) is workspace
    assert torch.all(workspace == 1)

    stream = "aux"
    aux_workspace = marlin_utils.get_marlin_workspace(device)
    assert aux_workspace.data_ptr() != workspace.data_ptr()
    assert torch.count_nonzero(aux_workspace) == 0

    manager.lock()
    assert marlin_utils.get_marlin_workspace(device) is aux_workspace


def test_marlin_workspace_without_manager(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(workspace_module, "_manager", None)
    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    first = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    second = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert torch.count_nonzero(first) == 0


class _RecordingQuantMethod(QuantizeMethodBase):
    """Records the layer's bias at the moment processing runs."""

    uses_meta_device = True

    def __init__(self):
        self.bias_at_process = None

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.bias_at_process = layer.bias.detach().clone()


class _LateBiasLayer(torch.nn.Module):
    """Mimics an online-quantized linear: `weight` is created on meta by
    `create_weights()`, which wraps the loaders, and the linear base registers
    `bias` afterwards."""

    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        weight = torch.nn.Parameter(torch.empty(4, 2, device="meta"))
        weight.weight_loader = default_weight_loader
        self.register_parameter("weight", weight)
        initialize_online_processing(self)
        bias = torch.nn.Parameter(torch.zeros(4))
        bias.weight_loader = default_weight_loader
        self.register_parameter("bias", bias)


def test_online_processing_waits_for_late_registered_bias():
    # Regression test: `bias` is skipped by the meta device paths, but it is
    # still loaded by a weight loader. Excluding it from the processing trigger
    # finalized the layer one load early, so the trailing bias was written into
    # an already-processed layer (e.g. over FP8 Marlin's permuted bias).
    quant_method = _RecordingQuantMethod()
    layer = _LateBiasLayer(quant_method)
    loaded_bias = torch.full((4,), 3.0)

    layer.weight.weight_loader(layer.weight, torch.full((4, 2), 2.0))
    assert quant_method.bias_at_process is None

    layer.bias.weight_loader(layer.bias, loaded_bias)
    assert quant_method.bias_at_process is not None
    assert torch.equal(quant_method.bias_at_process, loaded_bias)


@pytest.mark.parametrize("num_heads", [96, 128])
@pytest.mark.parametrize("tp_rank", [0, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_kda_online_loading_releases_sources_and_zeros_padding(
    monkeypatch, default_vllm_config, num_heads, tp_rank, reverse
):
    """Finish KDA's merged projection after six inputs, before model finalization."""
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=tp_rank),
    )
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.float32)
    config = KimiLinearConfig(
        hidden_size=16,
        linear_attn_config={
            "kda_layers": [0],
            "full_attn_layers": [],
            "head_dim": 128,
            "num_heads": num_heads,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    )
    attention = KimiK3DeltaAttention(
        config, default_vllm_config, prefix="model.layers.0.self_attn"
    )
    projection = attention.in_proj_qkvgfab
    # Exercise online loading on CPU without selecting a GPU quantization kernel.
    meta = projection.weight.data.to("meta")
    meta.__class__ = projection.weight.__class__
    meta.__dict__ = projection.weight.__dict__.copy()
    projection.weight = meta
    projection.quant_method = None
    initialize_online_processing(projection)

    empty_strided = torch.empty_strided

    def poisoned_empty(*args, **kwargs):
        return empty_strided(*args, **kwargs).fill_(float("nan"))

    monkeypatch.setattr(torch, "empty_strided", poisoned_empty)
    widths = [128 * num_heads] * 4 + [128, num_heads]
    expected = {}
    sources = []
    for shard in reversed(range(6)) if reverse else range(6):
        source = torch.arange(widths[shard] * 16, dtype=torch.float32).reshape(-1, 16)
        sources.append(ref(source))
        local_width = widths[shard] if shard == 4 else widths[shard] // 8
        start = 0 if shard == 4 else tp_rank * local_width
        expected[shard] = source[start : start + local_width].clone()
        assert projection.weight.is_meta
        projection.weight.weight_loader(projection.weight, source, shard)
        del source

    assert not projection.weight.is_meta
    assert all(source() is None for source in sources)
    expected_weight = torch.cat([expected[shard] for shard in range(6)])
    torch.testing.assert_close(
        projection.weight[: len(expected_weight)], expected_weight
    )
    padding = projection.weight[len(expected_weight) :]
    assert torch.equal(padding, torch.zeros_like(padding))


@pytest.fixture
def hpc_rope_norm(monkeypatch, default_vllm_config):
    """Import HpcRopeNorm with the external `hpc` package stubbed out."""
    if "hpc" not in sys.modules:
        stub = types.ModuleType("hpc")
        stub.__spec__ = importlib.machinery.ModuleSpec("hpc", loader=None)
        stub.QuantType = types.SimpleNamespace(  # type: ignore[attr-defined]
            QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR=types.SimpleNamespace(value=0)
        )
        monkeypatch.setitem(sys.modules, "hpc", stub)
    from vllm.model_executor.layers.hpc import rope_norm

    monkeypatch.setattr(rope_norm, "_hpc_rope_norm_instances", {})
    return rope_norm


def test_hpc_rope_norm_kernel_sees_refit_norm_weights(monkeypatch, hpc_rope_norm):
    """The fused HPC kernel is handed the live QK-norm weights after a refit.

    Drives the production `_forward_impl` with a recording `hpc` stub. The
    Q/K norm weights it receives must be the model's own float32 parameters,
    so a reload that rewrites them in place (same storage) is what
    the kernel sees. Previously the kernel read separate mirrors that no
    reload path refreshed.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm

    head_dim, num_heads, num_kv_heads, block_size = 128, 8, 1, 4
    layer = torch.nn.Module()
    layer.q_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.k_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.hpc_rope_norm = hpc_rope_norm.HpcRopeNorm(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cos_sin_cache=torch.ones(16, head_dim),
        use_qk_norm=True,
        fallback_qnorm=layer.q_norm,
        fallback_knorm=layer.k_norm,
        kv_cache_dtype="auto",
        layer_name="hpc_test_layer",
    )
    rnorm = layer.hpc_rope_norm

    calls: list[dict] = []

    def record(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sys.modules["hpc"], "rope_norm_store_kv", record, raising=False)

    def kernel_norm_weights():
        q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
        qkv = torch.zeros(1, q_size + 2 * kv_size, dtype=torch.bfloat16)
        kv_cache = torch.zeros(
            2, num_kv_heads, block_size, 2 * head_dim, dtype=torch.bfloat16
        )
        attn_layer = types.SimpleNamespace(
            _k_scale=torch.ones(1), _v_scale=torch.ones(1)
        )
        attn_metadata = types.SimpleNamespace(
            num_actual_tokens=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            max_query_len=1,
            decode_query_len=1,
            qo_indptr=None,
            qo_indptr_decode=None,
            slot_mapping=torch.tensor([4]),
            seq_lens=torch.tensor([1]),
            block_table_tensor=torch.tensor([[1]]),
            hpc_kv_written=False,
        )
        output = torch.zeros(1, q_size, dtype=torch.bfloat16)
        rnorm._forward_impl(qkv, kv_cache, attn_metadata, attn_layer, output)
        return calls[-1]["q_norm_weight"], calls[-1]["k_norm_weight"]

    def loaded(value):
        return torch.full((head_dim,), value, dtype=torch.bfloat16)

    default_weight_loader(layer.q_norm.weight, loaded(0.5))
    default_weight_loader(layer.k_norm.weight, loaded(0.25))
    q, k = kernel_norm_weights()
    assert torch.equal(q, loaded(0.5).float())
    assert torch.equal(k, loaded(0.25).float())
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()

    # a reload writes the live parameters in place
    layer.q_norm.weight.weight_loader(layer.q_norm.weight, loaded(2.0))
    layer.k_norm.weight.weight_loader(layer.k_norm.weight, loaded(3.0))

    q, k = kernel_norm_weights()
    assert q.dtype == k.dtype == torch.float32
    assert torch.equal(q, loaded(2.0).float())
    assert torch.equal(k, loaded(3.0).float())
    assert (q.data_ptr(), k.data_ptr()) == (q_ptr, k_ptr)
    assert not hasattr(rnorm, "qnorm_weight")


def _device_tensor_ptrs(worker) -> dict[str, int]:
    """Addresses of every device tensor held by a module; graphs capture them."""
    ptrs = {}
    for name, module in worker.get_model().named_modules():
        tensors = {**vars(module), **module._parameters, **module._buffers}
        for attr, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                ptrs[f"{name}.{attr}"] = tensor.data_ptr()
    return ptrs


def _moved_tensors(llm, ptrs: list[dict[str, int]]) -> list[list[str]]:
    after = llm.collective_rpc(_device_tensor_ptrs)
    return [
        sorted(n for n, p in old.items() if new.get(n) != p)
        for old, new in zip(ptrs, after)
    ]


@pytest.mark.parametrize(
    "use_aiter",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(), reason="AITER is ROCm-only"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-add-FP8_BLOCK",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-multiply-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-add-W4A16-G128",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-add-FP8_DYNAMIC",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-multiply-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-add-NVFP4A16",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_reload_weights(
    base_model, mul_model, add_model, tp_size, use_aiter, vllm_runner, monkeypatch
):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if "FP8" in base_model and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if use_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    with vllm_runner(
        model_name=base_model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
        # Reloading without a pause that empties the KV cache needs headroom.
        gpu_memory_utilization=CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION,
    ) as llm:
        # Reloading must update tensors in place: captured graphs keep pointers.
        ptrs = llm.collective_rpc(_device_tensor_ptrs)
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        assert _moved_tensors(llm, ptrs) == [[]] * len(ptrs)
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        assert _moved_tensors(llm, ptrs) == [[]] * len(ptrs)
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


def _reload_with_lora_expert_names(worker) -> None:
    """Reload the checkpoint under the names that LoRA-wrapped experts load."""

    def lora_name(name: str) -> str:
        parts = name.split(".")
        if "experts" in parts[:-2] and parts[parts.index("experts") + 1].isdigit():
            parts.insert(parts.index("experts") + 3, "base_layer")
        return ".".join(parts)

    path = Path(hf_api().snapshot_download(worker.vllm_config.model_config.model))
    files = sorted(str(f) for f in path.glob("*.safetensors"))
    weights = safetensors_weights_iterator(files, use_tqdm_on_load=False)
    worker.reload_weights(
        weights_iterator=((lora_name(name), weight) for name, weight in weights)
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="LoRA on quantized MoE requires CUDA"
)
def test_reload_weights_with_lora_moe(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    lora_path = hf_api().snapshot_download(
        repo_id="vllm-project/qwen3-moe-text2sql-spider"
    )
    prompts = [
        (
            "<|im_start|>user\nCount the rows of table candidate.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "SELECT count(*) FROM candidate"
        )
    ]

    def prompt_logprobs(llm, lora_id):
        lora_request = LoRARequest("sql", lora_id, lora_path) if lora_id else None
        output = llm.generate_greedy_logprobs(
            prompts,
            max_tokens=1,
            num_logprobs=None,
            num_prompt_logprobs=0,
            lora_request=lora_request,
        )[0]
        return [next(iter(token.values())).logprob for token in output[3][1:]]

    with vllm_runner(
        model_name="RedHatAI/Qwen3-30B-A3B-quantized.w4a16",
        enable_lora=True,
        max_lora_rank=8,
        max_loras=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=64,
        max_num_seqs=1,
        # Reloading without a pause that empties the KV cache needs headroom.
        gpu_memory_utilization=CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION,
    ) as llm:
        before = [prompt_logprobs(llm, 0), prompt_logprobs(llm, 1)]
        assert before[0] != before[1]
        llm.collective_rpc(_reload_with_lora_expert_names)
        after = [prompt_logprobs(llm, 0), prompt_logprobs(llm, 2)]
    assert after == before


def test_kv_scale_reload(vllm_runner):
    """Test reloading a checkpoint that contains k_scale/v_scale weights."""
    if _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    model = "nm-testing/Llama-3.2-1B-Instruct-FP8-KV"

    # Load dummy weights, then reload real checkpoint
    with vllm_runner(
        model_name=model,
        load_format="dummy",
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc(
            "update_config",
            kwargs={"overrides": {"load_config": {"load_format": "auto"}}},
        )
        llm.collective_rpc("reload_weights", kwargs={"weights_path": model})
        reloaded_perp = llm.generate_prompt_perplexity(
            ["The capital of France is the city of Paris"],
            mask=["The capital of France is"],
        )[0]

    assert reloaded_perp < 10


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model,quantization",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "fp8_per_tensor",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "fp8_per_tensor",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "mxfp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "mxfp8",
            marks=[
                pytest.mark.slow_test,
                pytest.mark.xfail(reason="mxfp4 & mla is not supported yet"),
            ],
        ),
    ],
)
def test_online_quantize_reload(
    base_model, mul_model, add_model, quantization, tp_size, vllm_runner
):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough GPU devices")

    if quantization == "fp8_per_tensor" and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
        # Reloading without a pause that empties the KV cache needs headroom.
        gpu_memory_utilization=CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


class _TwoLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(2, 2, bias=False)
        self.b = torch.nn.Linear(2, 2, bias=False)
        self.hook_saw: list[tuple[float, float]] = []

    def load_weights(self, weights) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            loaded.add(name)
        return loaded

    def process_weights_after_loading(self) -> None:
        a, b = self.a.weight.mean().item(), self.b.weight.mean().item()
        self.hook_saw.append((a, b))


def _full(value: float) -> torch.Tensor:
    return torch.full((2, 2), value)


class _LoggedHookModule(torch.nn.Module):
    def __init__(self, name: str, log: list[str], *children: torch.nn.Module):
        super().__init__()
        self.name, self.log = name, log
        self.kids = torch.nn.ModuleList(children)

    def process_weights_after_loading(self) -> None:
        self.log.append(self.name)


class _QuantMethodModule(torch.nn.Module, QuantizeMethodBase):
    def create_weights(self, *args, **kwargs) -> None:
        pass

    def apply(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise AssertionError("quant methods run in the per-layer pass")


def test_finalize_runs_every_module_hook_after_its_submodules():
    log: list[str] = []
    shared = _LoggedHookModule("shared", log)
    model = _LoggedHookModule(
        "root",
        log,
        _LoggedHookModule("a", log, _LoggedHookModule("a.x", log), shared),
        _LoggedHookModule("b", log, shared),
    )
    model.attn = _ReloadableAttentionLayer()
    model.quantized = torch.nn.Module()
    model.quantized.quant_method = _QuantMethodModule()
    # owned by the quant method, which processes it in the per-layer pass
    model.quantized.transform = _LoggedHookModule("transform", log)

    finalize_model_weights(model)

    assert sorted(log) == ["a", "a.x", "b", "root", "shared"]
    for child, parent in [("a.x", "a"), ("shared", "a"), ("shared", "b")]:
        assert log.index(child) < log.index(parent)
    assert log[-1] == "root"
    # attention-like layers run in the same traversal
    assert model.attn.post_load_called


class _TransformReadingAttention(_ReloadableAttentionLayer):
    def __init__(self, read) -> None:
        super().__init__()
        self.read, self.seen = read, None

    def process_weights_after_loading(self) -> None:
        self.seen = self.read()


def test_attention_post_load_reads_normalized_transform(dist_init):
    """MLA absorbs kv_b_proj, transforms included, in the attention pass, so a
    quantized layer's Hadamard transform must be normalized before it."""
    # transform.linear is imported by compressed_tensors and imports it back
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors  # noqa: F401, E501
    from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
        CompressedTensorsLinearTransformMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.transform.module import (  # noqa: E501
        HadamardTransform,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.parameter import SharedWeightParameter

    transform = HadamardTransform.__new__(HadamardTransform)
    torch.nn.Module.__init__(transform)
    transform.weight = SharedWeightParameter(weight_loader=default_weight_loader)
    transform.weight.add_partition(0, object(), (4, 4))
    transform.weight.partitions[0].data.fill_(1.0)

    method = CompressedTensorsLinearTransformMethod.__new__(
        CompressedTensorsLinearTransformMethod
    )
    method.quant_method = SimpleNamespace(process_weights_after_loading=lambda _: None)
    linear = torch.nn.Module()
    linear.quant_method = method
    linear.transform = transform

    model = torch.nn.Module()
    model.kv_b_proj = linear
    model.attn = _TransformReadingAttention(
        lambda: transform.weight.partitions[0].data[0, 0].item()
    )
    config = SimpleNamespace(
        dtype=torch.float32,
        quantization=None,
        word_embeddings_untied_by_checkpoint=False,
    )

    process_weights_after_loading(model, config, torch.device("cpu"))

    # normalized by 1/sqrt(4) before the attention pass, and only once
    assert model.attn.seen == 0.5
    assert transform.weight.partitions[0].data[0, 0].item() == 0.5


def _loaded_model() -> _TwoLayerModel:
    model = _TwoLayerModel()
    model.load_weights([("a.weight", _full(1.0)), ("b.weight", _full(2.0))])
    return model


class TestModelHookOnSplitLoad:
    def test_sparse_update(self):
        model = _loaded_model()
        engine = object.__new__(SparseNCCLWeightTransferEngine)
        engine.model = model

        engine.start_weight_update()
        for name, value in (("a.weight", 5.0), ("b.weight", 6.0)):
            values, indices = torch.full((4,), value), torch.arange(4)
            patch = CheckpointWeightPatch(name, (2, 2), torch.float32, values, indices)
            load_checkpoint_weight_patches(model, [patch])
        assert model.hook_saw == []
        engine.finish_weight_update()
        assert model.hook_saw == [(5.0, 6.0)]
