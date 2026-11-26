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
        self._snapshots: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        if "indexer_topk" in fields:
            capturer = bind_indexer_topk_capturer(model, max_num_tokens)
            self._snapshots["indexer_topk"] = lambda rows: capturer.snapshot(len(rows))
        if "hidden_states" in fields:
            self._snapshots["hidden_states"] = _hidden_states

    def snapshot(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stable copies of every enabled field for the current step's rows
        whose LM-head inputs are `hidden_states`."""
        return {field: take(hidden_states) for field, take in self._snapshots.items()}


def _hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """NumPy has no bfloat16: its rows keep their bit patterns as int16."""
    if hidden_states.dtype == torch.bfloat16:
        return hidden_states.view(torch.int16).clone()
    return hidden_states.clone()
