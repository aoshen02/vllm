# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A multi-tensor weight may be split across load_weights calls (RL weight
updates send one bucket per call); its parts pair up across calls."""

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload import finalize_layerwise_reload
from vllm.model_executor.model_loader.weight_utils import (
    finalize_model_weights,
    stash_weight_part,
)


class TestStashWeightPart:
    @pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["first", "second"])
    def test_parts_pair_across_calls_with_reused_buffer(self, order):
        tensors = [torch.ones(2, 2), torch.full((1,), 0.5)]
        names = ("weight", "scale")
        module = nn.Module()
        bucket = tensors[order[0]].clone()
        assert stash_weight_part(module, "k", names[order[0]], bucket) is None
        bucket.zero_()  # The transfer bucket is reused after the call.
        parts = stash_weight_part(module, "k", names[order[1]], tensors[order[1]])

        assert parts is not None
        torch.testing.assert_close(parts["weight"], tensors[0])
        torch.testing.assert_close(parts["scale"], tensors[1])
        finalize_model_weights(module)

    def test_unpaired_part_raises_once_at_finish(self):
        module = nn.Module()
        stash_weight_part(module, "layer.k", "weight", torch.ones(1))

        with pytest.raises(ValueError, match="layer.k"):
            finalize_model_weights(module)
        finalize_model_weights(module)


class _HookCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def process_weights_after_loading(self) -> None:
        self.calls += 1


def test_reload_finish_runs_model_hook_once():
    model = _HookCounter()
    finalize_layerwise_reload(model, None)
    assert model.calls == 1
