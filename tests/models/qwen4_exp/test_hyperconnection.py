# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import (
    GatedResidual,
    _ReplicatedZeroPaddedMergedColumnParallelLinear,
)


@pytest.mark.parametrize("loader_name", ["weight_loader", "weight_loader_v2"])
@pytest.mark.parametrize(
    ("hc_count", "hc_lowrank", "padding_size"),
    [(2, 4, 10), (4, 12, 0)],
)
def test_gated_residual_loader_zeroes_runtime_padding(
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    hc_count: int,
    hc_lowrank: int,
    padding_size: int,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 1
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    residual = GatedResidual(
        HyperConnectionConfig(
            hc_count=hc_count,
            hidden_size=8,
            hc_lowrank=hc_lowrank,
            params_dtype=torch.float32,
        )
    )
    layer = residual.input_mix_weight_down_block_inject

    assert isinstance(layer, _ReplicatedZeroPaddedMergedColumnParallelLinear)
    assert layer.tp_size == 1
    assert layer.tp_rank == 0
    assert layer.padding_size == padding_size

    weight = layer.weight
    assert weight.tp_size == 1
    assert weight.tp_rank == 0
    weight.data.fill_(-1)
    loader = getattr(layer, loader_name)
    loader(weight, torch.ones((hc_lowrank, layer.input_size)), 0)
    loader(weight, torch.full((hc_count, layer.input_size), 2.0), 1)

    torch.testing.assert_close(
        weight[:hc_lowrank], torch.ones_like(weight[:hc_lowrank])
    )
    torch.testing.assert_close(
        weight[hc_lowrank : hc_lowrank + hc_count],
        torch.full_like(weight[hc_lowrank : hc_lowrank + hc_count], 2.0),
    )
    if padding_size:
        torch.testing.assert_close(
            weight[-padding_size:], torch.zeros_like(weight[-padding_size:])
        )
