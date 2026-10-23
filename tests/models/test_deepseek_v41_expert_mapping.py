# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch


def test_expert_mapping_index_matches_linear_scan():
    """The indexed lookup returns exactly the entries the linear scan would match."""
    from vllm.models.deepseek_v4.nvidia.model import (
        make_deepseek_v4_expert_params_mapping,
    )
    from vllm.models.deepseek_v41.nvidia.model import _expert_mapping_index

    mapping = make_deepseek_v4_expert_params_mapping(384)
    candidates = _expert_mapping_index(mapping)
    names = [
        f"layers.{layer}.ffn.experts.{expert}.{w}.{kind}"
        for layer in (0, 39)
        for expert in (0, 1, 11, 38, 111, 383)
        for w in ("w1", "w2", "w3")
        for kind in ("weight", "scale")
    ] + ["layers.0.ffn.shared_experts.w1.weight", "layers.0.attn.wq_a.weight"]
    for name in names:
        assert candidates(name) == [m for m in mapping if m[1] in name], name

    fallback = [("experts.w13_", "experts.0.gate_proj.", 0, "w1")]
    assert _expert_mapping_index(fallback)("experts.0.gate_proj.weight") == fallback


def test_routed_expert_shards_load_as_one_run_per_parameter_shard():
    """Interleaved expert shards (w1, w2, w3 of each expert in turn) reach the
    MegaMoE batched loader as one run per (parameter, shard); a gap in expert
    ids starts a new run and other tensors pass through."""
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

    class Experts:
        def __init__(self):
            self.calls = []

        def load_routed_experts(self, param, name, shard_id, first, weights):
            self.calls.append((name, shard_id, first, [int(w) for w in weights]))

    experts = Experts()
    model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(ffn=SimpleNamespace(experts=experts))],
    )
    params = {
        "layers.0.ffn.experts.w13_weight": None,
        "layers.0.ffn.experts.w2_weight": None,
    }
    weights = [
        (f"layers.0.ffn.experts.{e}.{w}.weight", torch.tensor(e))
        for e in (4, 5, 6, 9)
        for w in ("w1", "w2", "w3")
    ] + [("layers.0.attn.wq_a.weight", torch.tensor(-1))]
    loaded: set[str] = set()
    passed = list(
        DeepseekV4Model._load_routed_expert_runs(model, iter(weights), params, loaded)
    )
    assert [name for name, _ in passed] == ["layers.0.attn.wq_a.weight"]
    assert loaded == set(params)
    w13, w2 = "layers.0.ffn.experts.w13_weight", "layers.0.ffn.experts.w2_weight"
    assert sorted(experts.calls) == sorted(
        [
            (w13, "w1", 4, [4, 5, 6]),
            (w2, "w2", 4, [4, 5, 6]),
            (w13, "w3", 4, [4, 5, 6]),
            (w13, "w1", 9, [9]),
            (w2, "w2", 9, [9]),
            (w13, "w3", 9, [9]),
        ]
    )
