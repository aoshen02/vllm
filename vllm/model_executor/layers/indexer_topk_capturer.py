# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture of the top-k positions that sparse-attention indexers select."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol, runtime_checkable

import torch

from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset


@runtime_checkable
class IndexerTopkCaptureSource(Protocol):
    layer_id: int
    topk_tokens: int
    topk_indices_buffer: torch.Tensor
    topk_capture_fn: Callable[[int], None] | None


class IndexerTopkCapturer:
    """Copy each indexer layer's top-k positions of a step into one GPU
    buffer of shape `[num_tokens, num_indexer_layers, topk]`. Every op is a
    copy into a persistent buffer, so it also replays in CUDA graphs."""

    def __init__(
        self, layer_ids: list[int], topk: int, max_num_batched_tokens: int
    ) -> None:
        self.layer_ids = sorted(layer_ids)
        self._slots = {layer_id: slot for slot, layer_id in enumerate(self.layer_ids)}
        self.device_buffer = torch.full(
            (max_num_batched_tokens, len(self.layer_ids), topk),
            -1,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

    def capture(self, source: IndexerTopkCaptureSource, num_tokens: int) -> None:
        offset = dbo_current_ubatch_token_offset()
        rows = source.topk_indices_buffer[:num_tokens, : self.device_buffer.shape[2]]
        self.device_buffer[
            offset : offset + num_tokens, self._slots[source.layer_id]
        ] = rows

    def snapshot(self, num_tokens: int) -> torch.Tensor:
        """A stable copy of the current step's rows."""
        return self.device_buffer[:num_tokens].clone()


def bind_indexer_topk_capturer(
    model: torch.nn.Module, max_num_batched_tokens: int
) -> IndexerTopkCapturer:
    """Attach a capturer to the indexer layers of `model`."""
    sources = [
        module
        for module in model.modules()
        if isinstance(module, IndexerTopkCaptureSource)
    ]
    topks = {source.topk_tokens for source in sources}
    if len(topks) != 1:
        raise ValueError(
            "enable_return_indexer_topk needs sparse-attention indexer layers "
            f"of one top-k size, got {sorted(topks)}."
        )
    capturer = IndexerTopkCapturer(
        [source.layer_id for source in sources], topks.pop(), max_num_batched_tokens
    )
    for source in sources:
        source.topk_capture_fn = partial(capturer.capture, source)
    return capturer
