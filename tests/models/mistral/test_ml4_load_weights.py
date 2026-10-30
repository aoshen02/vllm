# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 checkpoint loading (CPU only)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.models.mistral_large4.nvidia.model import _load_weights

NUM_EXPERTS = 2
ROTARY_DIM = 4


class _Experts(nn.Module):
    """Two experts split over two expert-parallel ranks; this rank owns expert 0."""

    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(torch.zeros(1, 4, 3), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.zeros(1, 3, 2), requires_grad=False)
        for param in (self.w13_weight, self.w2_weight):
            param.weight_loader = self._load

    @staticmethod
    def _load(param, weight, name, shard_id, expert_id, return_success=False):
        if expert_id != 0:
            return False
        if shard_id == "w1":
            param.data[0, :2].copy_(weight)
        elif shard_id == "w3":
            param.data[0, 2:].copy_(weight)
        else:
            param.data[0].copy_(weight)
        return True


class _Rotary(nn.Module):
    def __init__(self, cache_device: str) -> None:
        super().__init__()
        self.lerope_alpha = torch.zeros(ROTARY_DIM // 2)
        self.register_buffer(
            "cos_sin_cache", torch.zeros(8, device=cache_device), persistent=False
        )
        self.alphas_set: list[torch.Tensor] = []

    def set_lerope_alpha(self, lerope_alpha: torch.Tensor) -> None:
        self.alphas_set.append(lerope_alpha)


def _model(lerope: bool = False, cache_device: str = "cpu") -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.config = SimpleNamespace(
        n_routed_experts=NUM_EXPERTS, num_hidden_layers=1, lerope=lerope
    )
    model.model.rotary_emb = _Rotary(cache_device)
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    layer.mlp.experts.routed_experts = _Experts()
    model.model.layers = nn.ModuleList([layer])
    return model


def _expert_weights(experts: range) -> list[tuple[str, torch.Tensor]]:
    weights = []
    for e in experts:
        weights += [
            (f"layers.0.experts.{e}.w1.weight", torch.full((2, 3), 1.0 + e)),
            (f"layers.0.experts.{e}.w3.weight", torch.full((2, 3), 2.0 + e)),
            (f"layers.0.experts.{e}.w2.weight", torch.full((3, 2), 3.0 + e)),
        ]
    return weights


class TestExpertParallelLoad:
    def test_non_local_expert_is_skipped(self):
        model = _model()
        loaded = _load_weights(model, model.model, _expert_weights(range(2)))
        experts = model.model.layers[0].mlp.experts.routed_experts
        assert loaded == {
            "model.layers.0.mlp.experts.routed_experts.w13_weight",
            "model.layers.0.mlp.experts.routed_experts.w2_weight",
        }
        assert torch.equal(experts.w13_weight[0, :2], torch.full((2, 3), 1.0))
        assert torch.equal(experts.w2_weight[0], torch.full((3, 2), 3.0))

    def test_unknown_expert_raises(self):
        model = _model()
        weights = _expert_weights(range(1)) + [
            ("layers.0.experts.7.w1.weight", torch.ones(2, 3))
        ]
        with pytest.raises(ValueError, match="Unexpected expert weight"):
            _load_weights(model, model.model, weights)


class TestLeropeAlphaReload:
    @pytest.mark.parametrize(
        "alpha,raises",
        [(torch.zeros(ROTARY_DIM // 2), False), (torch.ones(ROTARY_DIM // 2), True)],
        ids=["unchanged", "changed"],
    )
    def test_reload_cannot_drop_a_new_alpha(self, alpha, raises):
        # During a layerwise reload the cos/sin cache is on the meta device and
        # is restored from the previous load afterwards.
        model = _model(lerope=True, cache_device="meta")
        weights = _expert_weights(range(1)) + [("lerope.alpha", alpha)]
        with disable_mtp_completeness_check():
            if raises:
                with pytest.raises(ValueError, match="lerope.alpha"):
                    _load_weights(model, model.model, weights)
            else:
                _load_weights(model, model.model, weights)
        assert model.model.rotary_emb.alphas_set == []

    def test_cold_load_sets_alpha(self):
        model = _model(lerope=True)
        alpha = torch.ones(ROTARY_DIM // 2)
        _load_weights(
            model, model.model, _expert_weights(range(1)) + [("lerope.alpha", alpha)]
        )
        assert [torch.equal(a, alpha) for a in model.model.rotary_emb.alphas_set] == [
            True
        ]
