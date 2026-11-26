# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.aux_output_connector.fields import AuxFieldCapturer
from vllm.model_executor.layers.indexer_topk_capturer import (
    bind_indexer_topk_capturer,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_test


class FakeIndexer(torch.nn.Module):
    def __init__(self, layer_id, topk=2, width=None):
        super().__init__()
        self.layer_id = layer_id
        self.topk_tokens = topk
        self.topk_indices_buffer = torch.full(
            (4, width or topk),
            -7,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.topk_capture_fn = None


class TestIndexerTopkCapture:
    def test_layers_capture_into_their_slots(self):
        # Layers bind in any order and capture by layer id; a buffer wider
        # than top-k keeps its first top-k columns.
        model = torch.nn.Sequential(
            FakeIndexer(5, width=3), torch.nn.Linear(1, 1), FakeIndexer(2)
        )
        capturer = bind_indexer_topk_capturer(model, 4)
        model[0].topk_indices_buffer[:2] = torch.tensor([[1, 2, 9], [3, 4, 9]])
        model[0].topk_capture_fn(2)
        model[2].topk_indices_buffer[:2] = torch.tensor([[5, 6], [7, -1]])
        model[2].topk_capture_fn(2)
        assert capturer.layer_ids == [2, 5]
        assert capturer.snapshot(2).tolist() == [
            [[5, 6], [1, 2]],
            [[7, -1], [3, 4]],
        ]

    @pytest.mark.parametrize(
        "layers,error",
        [
            ([], r"top-k size, got \[\]"),
            ([FakeIndexer(0, 2), FakeIndexer(1, 4)], r"\[2, 4\]"),
        ],
        ids=["no-indexer", "mixed-topk"],
    )
    def test_models_without_one_indexer_top_k_are_refused(self, layers, error):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), *layers)
        with pytest.raises(ValueError, match=error):
            bind_indexer_topk_capturer(model, 4)


def test_field_capturer_snapshots_every_enabled_field():
    model = torch.nn.Sequential(FakeIndexer(0))
    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(aux_fields={"indexer_topk"}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
    )
    capturer = AuxFieldCapturer(model, config)
    model[0].topk_indices_buffer[:3] = torch.tensor([[1, 2], [3, 4], [5, 6]])
    model[0].topk_capture_fn(3)
    snapshot = capturer.snapshot(2)
    assert {name: rows.tolist() for name, rows in snapshot.items()} == {
        "indexer_topk": [[[1, 2]], [[3, 4]]]
    }
