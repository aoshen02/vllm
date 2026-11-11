# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol, runtime_checkable

import torch

from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset


@runtime_checkable
class IndexerTopkCaptureSource(Protocol):
    layer_id: int
    topk_tokens: int
    topk_capture_fn: Callable[[torch.Tensor], None] | None


class IndexerTopkCapturer:
    """Copy each sparse-attention indexer layer's top-k rows into one GPU buffer.

    Layers call `capture` with the step's `[num_tokens, topk]` indices; every op
    is a copy into a persistent buffer, so it also replays in CUDA graphs.
    """

    def __init__(
        self,
        layer_ids: list[int],
        topk: int,
        max_num_batched_tokens: int,
        device: torch.device | str,
    ) -> None:
        self.layer_ids = sorted(layer_ids)
        self._slots = {layer_id: slot for slot, layer_id in enumerate(self.layer_ids)}
        self.shape_per_token = (len(self.layer_ids), topk)
        self.device_buffer = torch.full(
            (max_num_batched_tokens, *self.shape_per_token),
            -1,
            dtype=torch.int32,
            device=device,
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor) -> None:
        offset = dbo_current_ubatch_token_offset()
        rows = topk_indices[:, : self.shape_per_token[1]]
        self.device_buffer[offset : offset + len(rows), self._slots[layer_id]] = rows

    def snapshot(self, num_tokens: int) -> torch.Tensor:
        """Return a stable snapshot of the current step's rows."""
        return self.device_buffer[:num_tokens].clone()


def bind_indexer_topk_capturer(
    model: torch.nn.Module, max_num_batched_tokens: int, device: torch.device | str
) -> IndexerTopkCapturer | None:
    """Attach a capturer to the model's indexer layers; None if it has none."""
    sources = [
        module
        for module in model.modules()
        if isinstance(module, IndexerTopkCaptureSource)
    ]
    if not sources:
        return None
    topks = {source.topk_tokens for source in sources}
    if len(topks) != 1:
        raise ValueError(f"indexer layers have different top-k sizes: {topks}")
    capturer = IndexerTopkCapturer(
        [source.layer_id for source in sources],
        topks.pop(),
        max_num_batched_tokens,
        device,
    )
    for source in sources:
        source.topk_capture_fn = partial(capturer.capture, source.layer_id)
    return capturer
