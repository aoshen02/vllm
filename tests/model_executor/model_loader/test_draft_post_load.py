# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-layer draft weights are refreshed after reload, with stable storage."""

from types import MethodType

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gemma4_dspark import (
    Gemma4DSparkForCausalLM,
    Gemma4DSparkModel,
)
from vllm.model_executor.models.laguna_dflash import (
    DFlashLagunaForCausalLM,
    DFlashLagunaModel,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import update_derived_buffer
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def _norm():
    norm = nn.Module()
    norm.weight = nn.Parameter(torch.ones(2))
    norm.variance_epsilon = 1e-6
    return norm


def _draft(model_cls, backbone_cls, bias):
    model = nn.Module()
    model.supports_model_post_load_reload = model_cls.supports_model_post_load_reload
    backbone = model.model = nn.Module()
    backbone.hidden_norm = _norm()
    backbone.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.input_layernorm = _norm()
        attn = layer.self_attn = nn.Module()
        attn.qkv_proj = nn.Linear(2, 6, bias=bias)
        attn.k_proj = nn.Linear(2, 2, bias=bias)
        attn.k_norm, attn.q_norm, attn.kv_a_layernorm = _norm(), _norm(), _norm()
        attn.q_size = attn.kv_size = attn.head_dim = 2
        attn.num_kv_heads = 1
        attn.q_lora_rank = attn.kv_lora_rank = attn.qk_rope_head_dim = 2
        attn.use_k_eq_v = True
        attn.rotary_emb = nn.Module()
        attn.rotary_emb.head_size = 2
        attn.rotary_emb.is_neox_style = True
        attn.rotary_emb.register_buffer("cos_sin_cache", torch.ones(4, 2))
        attn.attn = nn.Module()
        backbone.layers.append(layer)
    for name in (
        "_build_fused_kv_buffers",
        "_build_context_kv_buffers",
        "_build_fused_context_kv_metadata",
    ):
        if hasattr(backbone_cls, name):
            setattr(backbone, name, MethodType(getattr(backbone_cls, name), backbone))
    model.process_weights_after_loading = MethodType(
        model_cls.process_weights_after_loading, model
    )
    return model


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "model_cls,backbone_cls",
    [
        (DFlashQwen3ForCausalLM, DFlashQwen3Model),
        (DFlash2Qwen3ForCausalLM, DFlashQwen3Model),
        (Qwen3DSparkForCausalLM, DFlashQwen3Model),
        (Gemma4DSparkForCausalLM, Gemma4DSparkModel),
        (DFlashLagunaForCausalLM, DFlashLagunaModel),
        (K3DSparkForCausalLM, K3DSparkModel),
    ],
)
def test_draft_partial_reload_matches_fresh_build(model_cls, backbone_cls, bias):
    model = _draft(model_cls, backbone_cls, bias)
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()
    buffers = dict(model.model.named_buffers(recurse=False))
    assert buffers
    for layer_idx in (0, 1):
        fresh = _draft(model_cls, backbone_cls, bias)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        updated = f"model.layers.{layer_idx}."
        for name, weight in model.named_parameters():
            if name.startswith(updated):
                weights[name].add_(1)
        fresh.load_state_dict(weights)
        fresh.process_weights_after_loading()

        initialize_layerwise_reload(model)
        for name, weight in list(model.named_parameters()):
            if name.startswith(updated):
                getattr(weight, "weight_loader", default_weight_loader)(
                    weight, weights[name]
                )
        finalize_layerwise_reload(model, model_config=None)
        for name, buffer in buffers.items():
            assert getattr(model.model, name) is buffer
            torch.testing.assert_close(buffer, getattr(fresh.model, name))
        assert buffers.keys().isdisjoint(model.model.state_dict())


def test_derived_buffer_rejects_meta_and_layout_changes():
    model = nn.Module()
    update_derived_buffer(model, "fused", torch.ones(2))
    for value in (torch.empty(2, device="meta"), torch.ones(3), torch.ones(2).double()):
        with pytest.raises(RuntimeError):
            update_derived_buffer(model, "fused", value)
        torch.testing.assert_close(model.fused, torch.ones(2))
