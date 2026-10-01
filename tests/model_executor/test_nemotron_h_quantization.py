# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_nemotron_copy_specs_cover_all_replay_buffers(monkeypatch, batch_invariant):
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncCalculator,
    )
    from vllm.model_executor.models import nemotron_h

    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    funcs = nemotron_h.NemotronHForCausalLM.get_mamba_state_copy_func()
    base = MambaStateCopyFuncCalculator.mamba2_state_copy_func()
    assert funcs[:2] == base
    assert len(funcs) == (5 if batch_invariant else 2)
    state = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    for copy_func in funcs[2:]:
        spec = copy_func(state, [0, 2, 1], 1, 1)
        assert spec.start_addr == state[2].data_ptr()
        assert spec.num_elements == state[2].numel()
    assert MambaStateCopyFuncCalculator.mamba2_state_copy_func() == base


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("ll_eligible", [False, True])
def test_nemotron_gate_bi_keeps_default_and_other_model_dispatch_isolated(
    monkeypatch, batch_invariant, ll_eligible
):
    """CPU constructor policy only; GPU arithmetic is a separate router gate."""
    from vllm.model_executor.models import nemotron_h

    inherited = {
        "allow_ll_bf16_gemm": ll_eligible,
        "allow_specialized_router_gemm": True,
        "allow_fp32_router_gemm": False,
        "allow_bf16x3_router_gemm": False,
        "allow_cublas_router_gemm": True,
    }
    constructor_calls = []

    def fake_parent_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        constructor_calls.append((args, kwargs))
        for name, value in inherited.items():
            setattr(self, name, value)

    monkeypatch.setattr(nemotron_h.GateLinear, "__init__", fake_parent_init)
    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    arguments = dict(
        out_dtype=torch.float32,
        force_fp32_compute=True,
        prefix="backbone.layers.1.mixer.gate",
    )
    before = nemotron_h.GateLinear(2688, 128, **arguments)
    candidate = nemotron_h.NemotronHGateLinear(2688, 128, **arguments)
    after = nemotron_h.GateLinear(2688, 128, **arguments)
    assert constructor_calls == [((2688, 128), arguments)] * 3
    expected = dict(inherited)
    if batch_invariant:
        expected["allow_ll_bf16_gemm"] = False
    for name in inherited:
        assert getattr(candidate, name) == expected[name]
        assert getattr(before, name) == inherited[name]
        assert getattr(after, name) == inherited[name]


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
