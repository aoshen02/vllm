# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-layout normalization for the FlashInfer CuTeDSL NVFP4 MoE backend."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    reorder_w13_to_w31_for_flashinfer_cutedsl,
)

_GATE = torch.tensor([[[1], [2], [3], [4]]])
_UP = torch.tensor([[[10], [20], [30], [40]]])
_EXPECTED = torch.cat([_UP, _GATE], dim=1)


@pytest.mark.parametrize(
    "change", ["none", "auto", "bi", "model", "tp", "eplb", "dtype"]
)
def test_w4a16_experiment_is_explicit_and_scoped(change):
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
        nemotron_w4a16_support,
    )

    config = SimpleNamespace(
        moe_backend="flashinfer_cutedsl",
        in_dtype=torch.bfloat16,
        activation=MoEActivation.RELU2_NO_MUL,
        hidden_dim=2688,
        intermediate_size_per_partition=1856,
        num_experts=128,
        experts_per_token=6,
        num_local_experts=32,
        moe_parallel_config=SimpleNamespace(
            tp_size=1, ep_size=4, enable_eplb=False, use_batched_activation_format=False
        ),
    )
    if change == "auto":
        config.moe_backend = "auto"
    elif change == "tp":
        config.moe_parallel_config.tp_size = 2
    elif change == "eplb":
        config.moe_parallel_config.enable_eplb = True
    elif change == "dtype":
        config.in_dtype = torch.float16
    supported, _ = nemotron_w4a16_support(
        config, "other" if change == "model" else "nemotron_h", change == "bi"
    )
    assert supported == (change == "none")


def test_w4a16_adapter_preserves_operands_and_does_not_quantize():
    """Adapter forwards original bytes/scales and writes the supplied output."""
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
        FlashInferCuteDSLW4A16Experts,
    )

    x = torch.randn(1, 2688, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 31, 32, 64, 96, -1]], dtype=torch.int32)
    weights = torch.ones(1, 6, dtype=torch.float32) / 6
    w1, w2 = torch.zeros(1, dtype=torch.uint8), torch.ones(1, dtype=torch.uint8)
    sf1, sf2 = torch.ones(1), torch.ones(1) * 2
    alpha1, alpha2 = torch.ones(32), torch.ones(32) * 3
    output = torch.empty_like(x)

    def forward(inputs, tactic):
        for actual, expected in zip(
            inputs, [x, ids, weights, w1, sf1, alpha1, w2, sf2, alpha2, output]
        ):
            assert actual.data_ptr() == expected.data_ptr()
        assert tactic[0][0] == (128, 8, 256)
        inputs[-1].copy_(inputs[0])

    adapter = SimpleNamespace(
        global_num_experts=128,
        w1_scale=sf1,
        w2_scale=sf2,
        g1_alphas=alpha1,
        g2_alphas=alpha2,
        runner=SimpleNamespace(forward=forward),
    )
    FlashInferCuteDSLW4A16Experts.apply(
        adapter,
        output,
        x,
        w1,
        w2,
        weights,
        ids,
        MoEActivation.RELU2_NO_MUL,
        128,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    assert torch.equal(output, x)


def test_w4a16_scale_storage_preserves_interleaver_bytes(monkeypatch):
    """Only test adapter view/padding, not the GPU interleaver implementation."""
    from flashinfer.quantization import fp4_quantization

    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
        prepare_w4a16_scales,
    )

    raw = torch.arange(2 * 65 * 7, dtype=torch.int64).to(torch.uint8).reshape(2, 65, 7)
    sf = raw.view(torch.float8_e4m3fn)
    packed = torch.arange(2 * 128 * 8, dtype=torch.int64).to(torch.uint8)

    def interleave(value):
        assert value.dtype == torch.uint8 and torch.equal(value, raw)
        return packed

    monkeypatch.setattr(fp4_quantization, "block_scale_interleave", interleave)
    actual = prepare_w4a16_scales(sf)
    assert actual.shape == (2, 128, 8)
    assert actual.data_ptr() == packed.data_ptr()
    assert torch.equal(actual.reshape(-1).view(torch.uint8), packed)
    assert torch.equal(sf.view(torch.uint8), raw)


def test_w4a16_oracle_keeps_auto_class_and_uses_a16_descriptor(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import (
        flashinfer_cutedsl_w4a16_moe as adapter,
    )
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4

    backend = nvfp4.NvFp4MoeBackend.FLASHINFER_CUTEDSL
    assert adapter.FlashInferCuteDSLW4A16Experts not in nvfp4.backend_to_kernel_cls(
        backend
    )
    scale1, scale2 = torch.ones(1, 2, 1), torch.ones(1, 2, 1) * 2
    weight1, weight2 = (
        torch.zeros(1, 2, 8, dtype=torch.uint8),
        torch.ones(1, 2, 8, dtype=torch.uint8),
    )
    alpha1, alpha2 = torch.ones(1) * 3, torch.ones(1) * 4
    monkeypatch.setattr(adapter, "prepare_w4a16_scales", lambda value: value)
    values = nvfp4.convert_to_nvfp4_moe_kernel_format(
        backend,
        None,
        weight1,
        scale1,
        alpha1,
        None,
        weight2,
        scale2,
        alpha2,
        None,
        False,
        use_a16=True,
    )
    for actual, expected in zip(
        values, [weight1, scale1, alpha1, None, weight2, scale2, alpha2, None]
    ):
        assert actual is expected
    config = nvfp4.make_nvfp4_moe_quant_config(
        backend, scale1, scale2, alpha1, alpha2, None, None, use_a16=True
    )
    assert config.quant_dtype is None
    assert config.g1_alphas is alpha1 and config.g2_alphas is alpha2


def test_w4a16_workspace_uses_actual_modular_problem_size():
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
        FlashInferCuteDSLW4A16Experts,
    )

    adapter = SimpleNamespace(hidden_dim=2688)
    x = torch.empty(64, 2688, dtype=torch.bfloat16, device="meta")
    w1 = torch.empty(32, 1856, 1344, dtype=torch.uint8, device="meta")
    w2 = torch.empty(32, 2688, 928, dtype=torch.uint8, device="meta")
    ids = torch.empty(64, 6, dtype=torch.int32, device="meta")
    e, m, n, k, topk = FlashInferCuteDSLW4A16Experts.moe_problem_size(
        adapter, x, w1, w2, ids
    )
    assert k == 2688
    shapes = FlashInferCuteDSLW4A16Experts.workspace_shapes(
        adapter, m, n, k, topk, 128, e, None, MoEActivation.RELU2_NO_MUL
    )
    assert shapes == ((0,), (0,), (64, 2688))


def test_w4a16_rejects_reprocessing_padded_scale_storage():
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4

    w1 = torch.empty(32, 1856, 1344, device="meta", dtype=torch.uint8)
    w2 = torch.empty(32, 2688, 928, device="meta", dtype=torch.uint8)
    s1 = torch.empty(32, 1920, 168, device="meta", dtype=torch.float8_e4m3fn)
    s2 = torch.empty(32, 2688, 116, device="meta", dtype=torch.float8_e4m3fn)
    alpha = torch.empty(32, device="meta")
    with pytest.raises(ValueError, match="fresh unswizzled scales"):
        nvfp4.convert_to_nvfp4_moe_kernel_format(
            nvfp4.NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            None,
            w1,
            s1,
            alpha,
            None,
            w2,
            s2,
            alpha,
            None,
            False,
            use_a16=True,
        )


def test_w4a16_reload_restore_preserves_registered_scale_alias():
    """Exercise the real storage-restoration helper with A→B→A packed bytes."""
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_w4a16_moe import (  # noqa: E501
        FlashInferCuteDSLW4A16Experts,
    )
    from vllm.model_executor.model_loader.reload.layerwise import (
        _copy_and_restore_kernel_tensors,
    )

    layer = torch.nn.Module()
    layer.activation = MoEActivation.RELU2_NO_MUL
    layer.w13_input_scale = None
    layer.expert_map = None
    original = torch.nn.Parameter(torch.zeros(2, 128, 4), requires_grad=False)
    layer.register_parameter("w13_weight_scale", original)
    kernel_view = original.view(-1)
    info = SimpleNamespace(
        kernel_tensors=({"w13_weight_scale": original}, {}),
        kernel_non_persistent_buffers=set(),
        loaded_weights=[],
    )
    for value in (1.0, 2.0, 1.0):
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.full_like(original, value), requires_grad=False
        )
        experts = object.__new__(FlashInferCuteDSLW4A16Experts)
        experts.quant_config = SimpleNamespace(w1_scale=layer.w13_weight_scale)
        experts.process_weights_after_loading(layer)
        _copy_and_restore_kernel_tensors(layer, info)
        assert layer.w13_weight_scale is original
        assert experts.w1_scale is original
        assert experts.quant_config.w1_scale is not original
        assert layer.w13_weight_scale.data_ptr() == kernel_view.data_ptr()
        assert torch.equal(kernel_view, torch.full_like(kernel_view, value))


def test_reorder_w13_swigluoai_interleaved():
    """gpt-oss w13 is [gate0, up0, gate1, ...] rather than packed [gate; up]."""
    w13 = torch.empty(1, 8, 1, dtype=_GATE.dtype)
    w13[:, 0::2] = _GATE
    w13[:, 1::2] = _UP

    out, out_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
        MoEActivation.SWIGLUOAI, w13, w13 + 100
    )

    torch.testing.assert_close(out, _EXPECTED)
    torch.testing.assert_close(out_scale, _EXPECTED + 100)


@pytest.mark.parametrize(
    "activation", [MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE]
)
def test_reorder_w13_packed_layouts(activation: MoEActivation):
    w13 = torch.cat([_GATE, _UP], dim=1)

    out, out_scale = reorder_w13_to_w31_for_flashinfer_cutedsl(
        activation, w13, w13 + 100
    )

    torch.testing.assert_close(out, _EXPECTED)
    torch.testing.assert_close(out_scale, _EXPECTED + 100)
