# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

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


def test_nemotron_h_stacked_experts_use_per_expert_loader():
    """HF exports must load every non-gated expert without changing its bytes."""
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    tensors = {
        "model.layers.1.mixer.experts.up_proj": torch.arange(24).reshape(2, 3, 4),
        "model.layers.1.mixer.experts.down_proj": torch.arange(24).reshape(2, 4, 3),
        "model.norm_f.weight": torch.ones(4),
    }
    with patch("vllm.model_executor.models.nemotron_h.AutoWeightsLoader") as loader:
        loader.return_value.load_weights.side_effect = (
            lambda weights, **_: dict(weights)
        )
        actual = NemotronHForCausalLM.load_weights(Mock(), iter(tensors.items()))

    assert len(actual) == 5
    for projection in ("up_proj", "down_proj"):
        prefix = "model.layers.1.mixer.experts"
        original = tensors[f"{prefix}.{projection}"]
        for expert in range(2):
            value = actual[f"{prefix}.{expert}.{projection}.weight"]
            assert torch.equal(value, original[expert])
            assert value.data_ptr() == original[expert].data_ptr()
    assert actual["model.norm_f.weight"] is tensors["model.norm_f.weight"]
