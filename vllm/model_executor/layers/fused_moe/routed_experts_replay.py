# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay recorded routed experts during prefill."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


def replay_topk_weights(
    router: BaseRouter, router_logits: torch.Tensor, topk_ids: torch.Tensor
) -> torch.Tensor:
    """Weights the router assigns to `topk_ids`, given its scoring function."""
    logits = router_logits.float()
    if router.scoring_func == "softmax":
        scores = logits.softmax(dim=-1)
    elif router.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    elif router.scoring_func == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function {router.scoring_func!r}")
    weights = scores.gather(1, topk_ids.long())
    if router.renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * getattr(router, "routed_scaling_factor", 1.0)


class RoutedExpertsReplay:
    """Device rows the routers take for replayed prefill positions."""

    def __init__(
        self,
        max_num_batched_tokens: int,
        shape_per_token: tuple[int, int],
        device: torch.device | str,
    ) -> None:
        tp_group = get_tp_group()
        self._tp_size = tp_group.world_size
        self._tp_rank = tp_group.rank_in_group
        num_rows = cdiv(max_num_batched_tokens, self._tp_size) * self._tp_size
        self.ids = torch.zeros(
            (num_rows, *shape_per_token), dtype=torch.int32, device=device
        )
        self.mask = torch.zeros(num_rows, dtype=torch.bool, device=device)
        pin_memory = is_pin_memory_available()
        self._host = [
            (
                torch.zeros(self.ids.shape, dtype=torch.int32, pin_memory=pin_memory),
                torch.zeros(num_rows, dtype=torch.bool, pin_memory=pin_memory),
                torch.Event() if pin_memory else None,
            )
            for _ in range(2)
        ]
        self._next_host = 0
        self._num_staged = 0
        self._rows: dict[str, np.ndarray] = {}

    def add_request(self, request_id: str, rows: np.ndarray) -> None:
        self._rows[request_id] = rows

    def remove_request(self, request_id: str) -> None:
        self._rows.pop(request_id, None)

    def prepare(
        self,
        request_ids: list[str],
        num_computed_tokens: np.ndarray,
        query_start_loc: np.ndarray,
    ) -> None:
        """Stage this step's replayed rows in batch token order."""
        jobs = []
        for i, request_id in enumerate(request_ids):
            rows = self._rows.get(request_id)
            if rows is None:
                continue
            start = int(num_computed_tokens[i])
            token_start = int(query_start_loc[i])
            end = min(start + int(query_start_loc[i + 1]) - token_start, len(rows))
            if end > start:
                jobs.append((token_start, rows[start:end]))
        if not jobs and not self._num_staged:
            return
        host_ids, host_mask, event = self._host[self._next_host]
        self._next_host ^= 1
        if event is not None:
            event.synchronize()
        num_rows = max((start + len(rows) for start, rows in jobs), default=0)
        num_clear = max(num_rows, self._num_staged)
        host_mask[:num_clear] = False
        for start, rows in jobs:
            host_ids[start : start + len(rows)].numpy()[...] = rows
            host_mask[start : start + len(rows)] = True
        self.mask[:num_clear].copy_(host_mask[:num_clear], non_blocking=True)
        self.ids[:num_rows].copy_(host_ids[:num_rows], non_blocking=True)
        if event is not None:
            event.record()
        self._num_staged = num_rows

    def _local_rows(self, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = dbo_current_ubatch_token_offset()
        batch_descriptor = get_forward_context().batch_descriptor
        num_local_tokens = (
            num_rows if batch_descriptor is None else batch_descriptor.num_tokens
        )
        shard = cdiv(num_local_tokens, self._tp_size)
        if num_rows != num_local_tokens and num_rows == shard and self._tp_size > 1:
            offset += self._tp_rank * shard
        return (
            self.ids[offset : offset + num_rows],
            self.mask[offset : offset + num_rows],
        )

    def apply(
        self,
        layer_id: int,
        router: BaseRouter,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self._local_rows(topk_ids.shape[0])
        forced_ids = ids[:, layer_id]
        forced_weights = replay_topk_weights(router, router_logits, forced_ids)
        mask = mask[:, None]
        return (
            torch.where(mask, forced_weights.to(topk_weights.dtype), topk_weights),
            torch.where(mask, forced_ids.to(topk_ids.dtype), topk_ids),
        )


def bind_routed_experts_replay(
    model: torch.nn.Module, replay: RoutedExpertsReplay
) -> None:
    """Let every routed-experts router of `model` take replayed rows."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopKRouter,
    )

    routers = []
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        router = module.router
        if module._quant_method.is_monolithic:
            raise ValueError(
                "Routed-experts replay is not supported with the monolithic MoE "
                f"kernel of layer {module.layer_id}; choose a modular moe_backend."
            )
        if (
            not isinstance(
                router, (FusedTopKRouter, FusedTopKBiasRouter, GroupedTopKRouter)
            )
            or getattr(router, "num_fused_shared_experts", 0)
            or getattr(router, "bias_vl", None) is not None
        ):
            raise ValueError(
                "Routed-experts replay is not supported for MoE layer "
                f"{module.layer_id} with router {type(router).__name__}."
            )
        routers.append((module.layer_id, router))
    if not routers:
        raise ValueError("No MoE router found for routed-experts replay.")
    for layer_id, router in routers:
        router.replay_fn = partial(replay.apply, layer_id)
