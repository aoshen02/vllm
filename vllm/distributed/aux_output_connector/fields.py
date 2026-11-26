# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture of the auxiliary outputs that `aux_namespace` objects store besides
the routed experts, as `AuxOutputConfig` enables them."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.indexer_topk_capturer import (
    bind_indexer_topk_capturer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ROUTED_EXPERTS = "routed_experts"


class AuxFieldCapturer:
    """Capture the fields that `AuxOutputConfig.aux_fields` enables."""

    def __init__(self, model: torch.nn.Module, vllm_config: VllmConfig) -> None:
        fields = vllm_config.aux_output_config.aux_fields
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._snapshots: dict[str, Callable[[int], torch.Tensor]] = {}
        if "indexer_topk" in fields:
            capturer = bind_indexer_topk_capturer(model, max_num_tokens)
            self._snapshots["indexer_topk"] = capturer.snapshot

    def snapshot(self, num_rows: int) -> dict[str, torch.Tensor]:
        """Stable copies of the current step's first `num_rows` rows of every
        enabled field."""
        return {field: take(num_rows) for field, take in self._snapshots.items()}
