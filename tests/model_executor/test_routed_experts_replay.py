# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.model_executor.layers.fused_moe.routed_experts_replay as replay_module
from vllm.distributed.aux_output_connector.connector import (
    check_routed_experts_replay,
)
from vllm.model_executor.layers.fused_moe.routed_experts_replay import (
    RoutedExpertsReplay,
    replay_topk_weights,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused routers need CUDA"
)


@pytest.mark.parametrize(
    "rows,prompt_len,ok",
    [
        (np.zeros((3, 2, 4), np.uint8), 3, True),
        (np.zeros((0, 2, 4), np.int16), 3, True),
        (np.zeros((4, 2, 4), np.uint8), 3, False),
        (np.zeros((3, 2, 2), np.uint8), 3, False),
        (np.zeros((3, 8), np.uint8), 3, False),
        (np.zeros((3, 2, 4), np.float32), 3, False),
        ([[[0] * 4] * 2] * 3, 3, False),
    ],
    ids=["full", "empty", "past-prompt", "top-k", "rank", "float", "list"],
)
def test_check_routed_experts_replay(rows, prompt_len, ok):
    reason = check_routed_experts_replay(rows, prompt_len, (2, 4))
    assert (reason is None) == ok


def _routers():
    bias = torch.randn(64, device="cuda")
    return [
        FusedTopKRouter(top_k=4, global_num_experts=64, scoring_func="softmax"),
        FusedTopKRouter(
            top_k=4, global_num_experts=64, scoring_func="softmax", renormalize=False
        ),
        FusedTopKRouter(top_k=4, global_num_experts=64, scoring_func="sigmoid"),
        FusedTopKBiasRouter(
            top_k=6,
            global_num_experts=64,
            e_score_correction_bias=bias,
            routed_scaling_factor=2.5,
        ),
        FusedTopKBiasRouter(
            top_k=6,
            global_num_experts=64,
            e_score_correction_bias=bias,
            routed_scaling_factor=1.5,
            scoring_func="sqrtsoftplus",
        ),
        GroupedTopKRouter(
            top_k=8,
            global_num_experts=64,
            num_expert_group=8,
            topk_group=4,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            e_score_correction_bias=bias,
        ),
    ]


@requires_cuda
@pytest.mark.parametrize(
    "index",
    range(6),
    ids=[
        "softmax",
        "softmax-no-renorm",
        "sigmoid",
        "sigmoid-bias",
        "sqrtsoftplus-bias",
        "grouped-sigmoid-bias",
    ],
)
def test_replayed_own_choice_reproduces_router_weights(index):
    torch.manual_seed(index)
    router = _routers()[index]
    hidden = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(32, 64, device="cuda", dtype=torch.float32)

    weights, ids = router._compute_routing(hidden, logits, None)

    torch.testing.assert_close(
        replay_topk_weights(router, logits, ids),
        weights.float(),
        rtol=1e-4,
        atol=1e-5,
    )


def _replay(monkeypatch, tp_size=1, tp_rank=0, num_local_tokens=None):
    monkeypatch.setattr(
        replay_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank),
    )
    monkeypatch.setattr(
        replay_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            batch_descriptor=None
            if num_local_tokens is None
            else SimpleNamespace(num_tokens=num_local_tokens)
        ),
    )
    monkeypatch.setattr(replay_module, "dbo_current_ubatch_token_offset", lambda: 0)
    return RoutedExpertsReplay(8, (2, 2), "cuda")


@requires_cuda
def test_replay_forces_only_recorded_prefill_positions(monkeypatch):
    replay = _replay(monkeypatch)
    router = FusedTopKRouter(top_k=2, global_num_experts=8, scoring_func="softmax")
    recorded = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[7, 6], [5, 4]]])
    replay.add_request("a", recorded)
    logits = torch.randn(6, 8, device="cuda")
    weights = torch.full((6, 2), 0.5, device="cuda")
    ids = torch.full((6, 2), 3, dtype=torch.int32, device="cuda")

    replay.prepare(["b", "a"], np.array([0, 1]), np.array([0, 2, 6]))
    forced_weights, forced_ids = replay.apply(1, router, logits, weights, ids)

    expected_ids = ids.clone()
    expected_ids[2:4] = torch.tensor(recorded[1:3, 1], device="cuda")
    torch.testing.assert_close(forced_ids, expected_ids)
    expected_weights = weights.clone()
    expected_weights[2:4] = replay_topk_weights(router, logits[2:4], expected_ids[2:4])
    torch.testing.assert_close(forced_weights, expected_weights)

    replay.prepare(["b", "a"], np.array([0, 3]), np.array([0, 2, 6]))
    unforced_weights, unforced_ids = replay.apply(1, router, logits, weights, ids)
    torch.testing.assert_close(unforced_ids, ids)
    torch.testing.assert_close(unforced_weights, weights)

    replay.remove_request("a")
    replay.prepare(["a"], np.array([0]), np.array([0, 3]))
    _, removed_ids = replay.apply(0, router, logits[:3], weights[:3], ids[:3])
    torch.testing.assert_close(removed_ids, ids[:3])


@requires_cuda
@pytest.mark.parametrize(
    "num_rows,tp_rank,first_row",
    [(2, 1, 2), (2, 0, 0), (4, 1, 0)],
    ids=["shard-rank1", "shard-rank0", "full-batch"],
)
def test_replay_rows_follow_sequence_parallel_shards(
    monkeypatch, num_rows, tp_rank, first_row
):
    replay = _replay(monkeypatch, tp_size=2, tp_rank=tp_rank, num_local_tokens=4)
    router = FusedTopKRouter(top_k=2, global_num_experts=8, scoring_func="softmax")
    recorded = np.arange(16).reshape(4, 2, 2) % 8
    replay.add_request("a", recorded)
    replay.prepare(["a"], np.array([0]), np.array([0, 4]))
    ids = torch.zeros((num_rows, 2), dtype=torch.int32, device="cuda")
    weights = torch.zeros((num_rows, 2), device="cuda")
    logits = torch.zeros((num_rows, 8), device="cuda")

    _, forced_ids = replay.apply(0, router, logits, weights, ids)

    expected = recorded[first_row : first_row + num_rows, 0]
    assert forced_ids.cpu().numpy().tolist() == expected.tolist()
