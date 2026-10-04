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
def test_nemotron_h_gate_skips_small_m_cute_gemm_under_bi(monkeypatch, batch_invariant):
    from vllm.model_executor.models import nemotron_h

    def parent_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.allow_ll_bf16_gemm = True
        self.allow_cublas_router_gemm = True

    monkeypatch.setattr(nemotron_h.GateLinear, "__init__", parent_init)
    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    gate = nemotron_h.NemotronHGateLinear(2688, 128, out_dtype=torch.float32)
    assert gate.allow_ll_bf16_gemm is not batch_invariant
    # cuBLAS bf16 -> fp32 stays: row invariance is covered by
    # tests/v1/determinism/test_nemotron_h_batch_invariance.py.
    assert gate.allow_cublas_router_gemm


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_nemotron_h_norms_dispatch_to_shared_kernels_under_bi(
    monkeypatch, batch_invariant
):
    from vllm.model_executor.models import nemotron_h, nemotron_h_alignment

    calls = []
    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(nemotron_h, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        nemotron_h_alignment,
        "rms_forward",
        lambda *args: calls.append("rms") or "shared",
    )
    monkeypatch.setattr(
        nemotron_h.RMSNorm, "forward", lambda self, x, residual=None: "default"
    )
    norm = nemotron_h.NemotronHRMSNorm.__new__(nemotron_h.NemotronHRMSNorm)
    torch.nn.Module.__init__(norm)
    norm.weight = torch.nn.Parameter(torch.ones(4))
    norm.variance_epsilon = 1e-5
    out = nemotron_h.NemotronHRMSNorm.forward(norm, torch.ones(2, 4))
    assert out == ("shared" if batch_invariant else "default")
    assert calls == (["rms"] if batch_invariant else [])


@pytest.mark.parametrize("tp_size", [1, 2])
def test_nemotron_h_gated_norm_uses_shared_kernel_only_at_tp1(monkeypatch, tp_size):
    """The shared gated-norm kernel normalizes whole groups on one rank; with
    TP the default (batch-invariant, all-reducing) path is used instead of
    failing at runtime."""
    from vllm.model_executor.models import nemotron_h, nemotron_h_alignment

    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(nemotron_h_alignment, "gated_forward", lambda *args: "shared")
    monkeypatch.setattr(
        nemotron_h.Mixer2RMSNormGated, "forward", lambda self, x, gate: "default"
    )
    norm = nemotron_h.NemotronHGatedRMSNorm.__new__(nemotron_h.NemotronHGatedRMSNorm)
    torch.nn.Module.__init__(norm)
    norm.tp_size = tp_size
    norm.use_rms_norm = True
    norm.weight = torch.nn.Parameter(torch.ones(4))
    norm.group_size = 4
    norm.variance_epsilon = 1e-5
    out = nemotron_h.NemotronHGatedRMSNorm.forward(norm, torch.ones(2, 4), None)
    assert out == ("shared" if tp_size == 1 else "default")
