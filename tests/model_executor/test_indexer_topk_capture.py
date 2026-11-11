# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.indexer_topk_capturer import (
    bind_indexer_topk_capturer,
)

pytestmark = pytest.mark.cpu_test


class FakeIndexer(torch.nn.Module):
    def __init__(self, layer_id, topk=2):
        super().__init__()
        self.layer_id = layer_id
        self.topk_tokens = topk
        self.topk_capture_fn = None


class TestIndexerTopkCapture:
    def test_layers_capture_into_their_slots(self):
        model = torch.nn.Sequential(
            FakeIndexer(5), torch.nn.Linear(1, 1), FakeIndexer(2)
        )
        capturer = bind_indexer_topk_capturer(model, 4, "cpu")
        assert capturer.layer_ids == [2, 5]
        model[0].topk_capture_fn(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
        model[2].topk_capture_fn(torch.tensor([[5, 6], [7, -1]], dtype=torch.int32))
        assert capturer.snapshot(2).tolist() == [
            [[5, 6], [1, 2]],
            [[7, -1], [3, 4]],
        ]

    @pytest.mark.parametrize(
        "layers,error",
        [([], None), ([FakeIndexer(0, 2), FakeIndexer(1, 4)], "different top-k")],
        ids=["no-indexer", "mixed-topk"],
    )
    def test_unsupported_models(self, layers, error):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), *layers)
        if error is None:
            assert bind_indexer_topk_capturer(model, 4, "cpu") is None
        else:
            with pytest.raises(ValueError, match=error):
                bind_indexer_topk_capturer(model, 4, "cpu")
