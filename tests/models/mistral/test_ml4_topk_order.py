# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import _sort_topk_indices_


class TestSortTopkIndices:
    @pytest.mark.parametrize(
        "rows,expected",
        [
            ([[5, 1, 3, -1, -1]], [[1, 3, 5, -1, -1]]),
            ([[-1, 4, -1, 0, 2]], [[0, 2, 4, -1, -1]]),
            ([[-1, -1, -1]], [[-1, -1, -1]]),
            ([[2, 0, 1], [9, -1, 7]], [[0, 1, 2], [7, 9, -1]]),
        ],
        ids=["tail-padding", "interleaved-padding", "all-padding", "two-rows"],
    )
    def test_rows_sorted_padding_last(self, rows, expected):
        t = torch.tensor(rows, dtype=torch.int32)
        _sort_topk_indices_(t)
        assert t.tolist() == expected

    def test_permuted_selection_gives_identical_rows(self):
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(4096, generator=g)[:2048].to(torch.int32)
        a = sel[torch.randperm(2048, generator=g)].unsqueeze(0).clone()
        b = sel[torch.randperm(2048, generator=g)].unsqueeze(0).clone()
        assert not torch.equal(a, b)
        _sort_topk_indices_(a)
        _sort_topk_indices_(b)
        assert torch.equal(a, b)

    def test_writes_through_buffer_view(self):
        buf = torch.full((4, 8), -1, dtype=torch.int32)
        buf[1:3, :3] = torch.tensor([[2, 0, 1], [5, 4, 3]], dtype=torch.int32)
        _sort_topk_indices_(buf[1:3, :6])
        assert buf[1, :3].tolist() == [0, 1, 2]
        assert buf[2, :3].tolist() == [3, 4, 5]
        assert (buf[0] == -1).all() and (buf[3] == -1).all()
