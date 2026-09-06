# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the FP8 weight strategy helpers' output contract."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8,
)
from vllm.model_executor.layers.quantization.modelopt import WEIGHT, KFp8StaticChannel
from vllm.model_executor.layers.quantization.utils import fp8_utils, w8a8_utils
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
)

FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _ocp_fp8(monkeypatch):
    """Test the layout contract with OCP E4M3 semantics on every platform;
    FNUZ normalization is a separate transform with its own coverage."""
    monkeypatch.setattr(fp8_utils.current_platform, "is_fp8_fnuz", lambda: False)


@pytest.fixture
def _torch_scaled_fp8_quant(monkeypatch):
    """Stand-in for the accelerator quant op so requantization runs on CPU."""

    def scaled_fp8_quant(x, scale):
        return (x / scale).clamp(-448.0, 448.0).to(FP8), scale

    monkeypatch.setattr(w8a8_utils.ops, "scaled_fp8_quant", scaled_fp8_quant)


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.contiguous(), b.contiguous()
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_tensor_strategy_returns_kernel_layout_and_fused_scales():
    """Checkpoint ``(N, K)`` in, kernel ``(K, N)`` out; a checkpoint already
    quantized with one scale is not requantized; the static input scale is
    collapsed to the max across fused shards."""
    weight = torch.randn(6, 4).to(FP8)
    weight_scale = torch.tensor(0.5)
    input_scale = torch.tensor([0.1, 0.3, 0.2])

    out, out_scale, out_input_scale = process_fp8_weight_tensor_strategy(
        weight, weight_scale, [2, 2, 2], input_scale
    )

    assert out.shape == (4, 6)
    assert _same_bytes(out.t(), weight)
    assert out_scale == 0.5
    assert out_input_scale.ndim == 0 and out_input_scale == 0.3


@pytest.mark.usefixtures("_torch_scaled_fp8_quant")
def test_tensor_strategy_requantizes_fused_shards_in_place():
    """Shards quantized with different scales are requantized to the max scale,
    written into ``weight`` itself; the returned ``(K, N)`` weight is a view."""
    dequant = torch.tensor([[1.0, 2.0], [4.0, 8.0]] * 2)  # two shards, two rows each
    shard_scales = torch.tensor([0.25, 1.0])
    weight = torch.cat(
        [dequant[:2] / shard_scales[0], dequant[2:] / shard_scales[1]]
    ).to(FP8)

    out, out_scale, _ = process_fp8_weight_tensor_strategy(
        weight, shard_scales, [2, 2], None
    )

    assert out_scale == 1.0
    assert out.data_ptr() == weight.data_ptr()
    assert torch.equal(out.t().float() * out_scale, dequant)


def test_tensor_strategy_keeps_dynamic_input_scale_absent():
    weight = torch.randn(6, 4).to(FP8)
    _, _, out_input_scale = process_fp8_weight_tensor_strategy(
        weight, torch.tensor(0.5), [6], None
    )
    assert out_input_scale is None


def test_channel_strategy_returns_kernel_layout():
    weight = torch.randn(6, 4).to(FP8)
    weight_scale = torch.rand(6, 1)
    input_scale = torch.tensor([0.2, 0.7])

    out, out_scale, out_input_scale = process_fp8_weight_channel_strategy(
        weight, weight_scale, input_scale
    )

    assert out.shape == (4, 6)
    assert _same_bytes(out.t(), weight)
    assert torch.equal(out_scale, weight_scale)
    assert out_input_scale.ndim == 0 and out_input_scale == 0.7


@pytest.mark.parametrize(
    "strategy", [QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL]
)
@pytest.mark.parametrize("static", [True, False])
def test_compressed_tensors_fp8_stores_kernel_layout(
    default_vllm_config, strategy, static
):
    """The compressed-tensors scheme stores what the helpers return: a ``(K, N)``
    weight tagged as such, and a scalar static input scale or ``None``."""
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    scheme = CompressedTensorsW8A8Fp8(
        QuantizationArgs(num_bits=8, type="float", strategy=strategy), static
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.randn(6, 4).to(FP8), requires_grad=False)
    layer.weight_scale = Parameter(
        torch.tensor(0.5)
        if strategy is QuantizationStrategy.TENSOR
        else torch.rand(6, 1),
        requires_grad=False,
    )
    layer.logical_widths = [2, 2, 2]
    if static:
        layer.input_scale = Parameter(
            torch.tensor([0.1, 0.3, 0.2]), requires_grad=False
        )
    checkpoint = layer.weight.data.clone()

    scheme.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 6)
    assert (layer.weight.input_dim, layer.weight.output_dim) == (0, 1)
    assert _same_bytes(layer.weight.t(), checkpoint)
    if static:
        assert layer.input_scale.ndim == 0 and layer.input_scale == 0.3
    else:
        assert layer.input_scale is None


def test_modelopt_fp8_channel_stores_kernel_layout():
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.randn(6, 4).to(FP8), requires_grad=False)
    layer.weight_scale = Parameter(torch.rand(6), requires_grad=False)
    checkpoint = layer.weight.data.clone()

    KFp8StaticChannel().process(layer, WEIGHT)

    assert layer.weight.shape == (4, 6)
    assert _same_bytes(layer.weight.t(), checkpoint)
