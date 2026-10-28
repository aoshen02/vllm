# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared sizing helpers for SimpleCPUOffloadConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def derive_independent_cpu_config(
    vllm_config: VllmConfig,
    gpu_config: KVCacheConfig,
    capacity_bytes: int,
    aligned_counts: tuple[int, ...] | None = None,
) -> KVCacheConfig:
    """Place CPU domains using GPU usable-capacity proportions and CPU bytes."""
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups

    gpu_counts = gpu_config.group_num_blocks
    assert gpu_counts is not None

    def place(counts: tuple[int, ...]) -> KVCacheConfig:
        return get_kv_cache_config_from_groups(
            vllm_config,
            gpu_config.kv_cache_groups,
            capacity_bytes,
            group_num_blocks=counts,
        )

    if aligned_counts is not None:
        return place(aligned_counts)
    null_counts = (1,) * len(gpu_counts)
    null_plan = place(null_counts)
    owned = [
        group_id
        for group_id, group in enumerate(gpu_config.kv_cache_groups)
        if group.layer_names
    ]
    if not owned:
        return place((torch.iinfo(torch.int64).max,) * len(gpu_counts))
    usable = tuple(count - 1 for count in gpu_counts)
    denominator = max(usable)
    if not denominator:
        return null_plan
    lower = 0
    upper = capacity_bytes * denominator
    best = null_plan
    while lower <= upper:
        scale = (lower + upper) // 2
        counts = tuple(1 + scale * count // denominator for count in usable)
        if max(counts) > torch.iinfo(torch.int64).max:
            upper = scale - 1
            continue
        try:
            candidate = place(counts)
        except ValueError:
            upper = scale - 1
        else:
            best = candidate
            lower = scale + 1
    return best


def sync_cpu_group_counts(counts: tuple[int, ...]) -> tuple[int, ...]:
    """Converge independent CPU capacities with the existing world CPU group."""
    if not dist.is_initialized():
        return counts
    from vllm.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return counts
    values = torch.tensor(counts, dtype=torch.int64, device="cpu")
    dist.all_reduce(values, group=world_group.cpu_group, op=dist.ReduceOp.MIN)
    return tuple(values.tolist())


def sync_num_offload_blocks_across_workers(num_offload_blocks: int) -> int:
    """All-reduce MIN so every rank allocates the same offload pool size."""
    if not dist.is_initialized():
        return num_offload_blocks

    from vllm.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return num_offload_blocks

    blocks_tensor = torch.tensor([num_offload_blocks], dtype=torch.int64, device="cpu")
    dist.all_reduce(blocks_tensor, group=world_group.cpu_group, op=dist.ReduceOp.MIN)
    synced = int(blocks_tensor.item())
    if synced != num_offload_blocks:
        logger.info(
            "SimpleCPUOffload: aligned num_offload_blocks from %d to %d "
            "across %d workers",
            num_offload_blocks,
            synced,
            world_group.world_size,
        )
    return synced


def gpu_total_bytes(gpu_config: KVCacheConfig) -> int:
    if gpu_config.group_num_blocks is not None and not any(
        group.layer_names for group in gpu_config.kv_cache_groups
    ):
        return 0
    assert len(gpu_config.kv_cache_tensors) > 0
    # Every KVCacheTensor describes placement within the same backing allocation,
    # so its size is the total GPU KV cache size.
    return gpu_config.kv_cache_tensors[0].size
