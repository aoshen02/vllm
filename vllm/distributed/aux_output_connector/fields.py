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
        self._topk = (
            vllm_config.aux_output_config.return_topk_logprobs
            if "topk_logprobs" in fields
            else 0
        )

    def snapshot(
        self,
        hidden_states: torch.Tensor,
        compute_logits: Callable[[torch.Tensor], torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        """Stable copies of every enabled field for the current step's rows
        whose LM-head inputs are `hidden_states`. Call on every rank: the
        top-k logprobs gather the logits across tensor parallelism."""
        snapshots = {
            field: take(hidden_states) for field, take in self._snapshots.items()
        }
        if self._topk:
            snapshots.update(topk_logprobs(hidden_states, compute_logits, self._topk))
        return snapshots


def topk_logprobs(
    hidden_states: torch.Tensor,
    compute_logits: Callable[[torch.Tensor], torch.Tensor | None],
    k: int,
    chunk_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """The `k` most likely next tokens of every row (int32) and their
    log-probabilities (float32, log-softmax of the logits); nothing on the
    ranks that do not receive the logits."""
    token_ids, logprobs = [], []
    for start in range(0, len(hidden_states), chunk_size):
        logits = compute_logits(hidden_states[start : start + chunk_size])
        if logits is None:
            continue
        values, indices = torch.log_softmax(logits.float(), dim=-1).topk(k, dim=-1)
        token_ids.append(indices.to(torch.int32))
        logprobs.append(values)
    if not token_ids:
        return {}
    return {
        "topk_token_ids": torch.cat(token_ids),
        "topk_logprobs": torch.cat(logprobs),
    }


def _hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """NumPy has no bfloat16: its rows keep their bit patterns as int16."""
    if hidden_states.dtype == torch.bfloat16:
        return hidden_states.view(torch.int16).clone()
    return hidden_states.clone()
