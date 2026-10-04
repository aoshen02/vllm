# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


def test_nemotron_h_lm_head_receives_quant_config():
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.dtype = None
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch("vllm.model_executor.models.nemotron_h.NemotronHModel") as MockModel,
        patch("vllm.model_executor.models.nemotron_h.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.nemotron_h.LogitsProcessor"),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockModel.return_value.has_moe = False

        NemotronHForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_relu2_fp8_fusion_uses_registry():
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = NemotronHMLP.__new__(NemotronHMLP)
    torch.nn.Module.__init__(mlp)
    mlp.up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.nemotron_h.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused


@pytest.mark.parametrize(
    "custom_ops,fuse_norm_quant,expected_ops,expected_fuse",
    [
        (["none"], None, ["none", "+quant_fp8"], False),
        (["none"], True, ["none", "+quant_fp8"], True),
        (["-quant_fp8"], None, ["-quant_fp8"], None),
    ],
)
def test_nemotron_h_batch_invariant_uses_cuda_fp8_quant(
    custom_ops, fuse_norm_quant, expected_ops, expected_fuse
):
    """Batch-invariant Nemotron-H quantizes FP8 activations with the standalone
    CUDA kernel that training replays, without enabling quant fusions by
    default, and leaves an explicit user choice alone."""
    from vllm.model_executor.models.config import NemotronHForCausalLMConfig

    pass_config = SimpleNamespace(fuse_norm_quant=fuse_norm_quant, fuse_act_quant=None)
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            custom_ops=list(custom_ops), pass_config=pass_config
        )
    )
    NemotronHForCausalLMConfig.use_cuda_fp8_quant(vllm_config)
    assert vllm_config.compilation_config.custom_ops == expected_ops
    assert pass_config.fuse_norm_quant == expected_fuse
    assert pass_config.fuse_act_quant == (None if expected_fuse is None else False)


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_gate_skips_small_m_cute_gemm_under_bi(monkeypatch, batch_invariant):
    """The small-M CuTe router GEMM reduces differently from the large-M path,
    so batch invariance must not select it."""
    from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
    from vllm.model_executor.layers.fused_moe.router import gate_linear

    monkeypatch.setattr(gate_linear.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(ll_bf16, "is_available", lambda: True)
    gate = gate_linear.GateLinear.__new__(gate_linear.GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.empty(128, 2688, dtype=torch.bfloat16))
    gate.allow_specialized_router_gemm = True
    gate.allow_cublas_router_gemm = False
    gate._router_gemm_cublas_capable = True
    gate.out_dtype = None
    gate.set_out_dtype(torch.float32)
    assert gate.allow_ll_bf16_gemm is not batch_invariant
    # cuBLAS bf16 -> fp32 stays: row invariance is covered by
    # tests/v1/determinism/test_nemotron_h_batch_invariance.py.
    assert gate.allow_cublas_router_gemm


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_cuda_rms_norm_dispatch_under_bi(monkeypatch, batch_invariant):
    from vllm.model_executor.layers import layernorm

    calls = []

    def shared(*args):
        calls.append("rms")
        return "shared"

    monkeypatch.setattr(layernorm.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(layernorm, "cuda_rms_norm", shared)
    monkeypatch.setattr(
        layernorm.RMSNorm, "forward", lambda self, x, residual=None: "default"
    )
    norm = layernorm.CudaRMSNorm.__new__(layernorm.CudaRMSNorm)
    torch.nn.Module.__init__(norm)
    norm.weight = torch.nn.Parameter(torch.ones(4))
    norm.variance_epsilon = 1e-5
    out = layernorm.CudaRMSNorm.forward(norm, torch.ones(2, 4))
    assert out == ("shared" if batch_invariant else "default")
    assert calls == (["rms"] if batch_invariant else [])


@pytest.mark.parametrize("tp_size", [1, 2])
def test_grouped_gated_norm_only_at_tp1(monkeypatch, tp_size):
    """The grouped gated-norm kernel normalizes whole groups on one rank; with
    TP the default (batch-invariant, all-reducing) path is used instead of
    failing at runtime."""
    from vllm.model_executor.layers.mamba import mamba_mixer2

    monkeypatch.setattr(mamba_mixer2.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        torch.ops.vllm, "grouped_gated_rms_norm", lambda *args: "grouped"
    )
    monkeypatch.setattr(
        mamba_mixer2.Mixer2RMSNormGated, "forward", lambda self, x, gate: "default"
    )
    cls = mamba_mixer2.GroupedMixer2RMSNormGated
    norm = cls.__new__(cls)
    torch.nn.Module.__init__(norm)
    norm.tp_size = tp_size
    norm.use_rms_norm = True
    norm.weight = torch.nn.Parameter(torch.ones(4))
    norm.group_size = 4
    norm.variance_epsilon = 1e-5
    out = cls.forward(norm, torch.ones(2, 4), None)
    assert out == ("grouped" if tp_size == 1 else "default")
