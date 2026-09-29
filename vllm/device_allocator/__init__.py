# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# py_device, py_size_or_aligned_size, py_ptr, py_handle
# py_handle has type list[int] on ROCm and int otherwise
HandleType: TypeAlias = tuple[int, int, int, list[int] | int]


def cumem_cudagraph_pool_enabled(vllm_config: "VllmConfig") -> bool:
    """Whether decoder CUDA graphs are captured into cuMem pools that sleep
    mode offloads: Model Runner V2 with the cumem sleep backend on CUDA.

    NCCL buffer registration must then stay off (``NCCL_GRAPH_REGISTER=0``,
    no ``TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK``): it retains the cuMem
    handles of graph buffers, pinning them through sleep, and keeps stale
    registrations after wake remaps the pool, which causes hangs or wrong
    results. An explicit opt-in is refused, not warned about.
    """
    return (
        vllm_config.use_v2_model_runner
        and vllm_config.model_config.enable_sleep_mode
        and vllm_config.model_config.sleep_mode_backend == "cumem"
        and current_platform.is_cuda()
    )


def is_capturing_into_cumem_pool() -> bool:
    """Whether the ongoing CUDA graph capture allocates from a cuMem pool.

    Legacy CUDA IPC (``cudaIpcGetMemHandle``) rejects cuMem memory, so custom
    allreduce copies such graph inputs into its pre-registered IPC buffer
    instead of registering them after capture.
    """
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.instance
    return (
        allocator is not None and allocator.current_tag == CuMemAllocator.cudagraph_tag
    )


_plain_cudagraph_pool: tuple[int, int] | None = None


def plain_cudagraph_pool_handle() -> tuple[int, int]:
    """A CUDA graph pool that is never routed through cuMem. Memory profiling
    captures into it so that destroying its graphs frees it normally. Only the
    latest one is kept; pool handles are never reused, so none can collide."""
    global _plain_cudagraph_pool
    _plain_cudagraph_pool = current_platform.graph_pool_handle()
    return _plain_cudagraph_pool


def use_cudagraph_pool(
    pool: tuple[int, int] | None, vllm_config: "VllmConfig"
) -> AbstractContextManager[tuple[int, int] | None]:
    """Route CUDA graph allocations through cuMem when sleep mode is enabled."""
    if (
        pool is not None
        and pool != _plain_cudagraph_pool
        and cumem_cudagraph_pool_enabled(vllm_config)
    ):
        from vllm.device_allocator.cumem import CuMemAllocator

        return CuMemAllocator.get_instance().use_cudagraph_pool(pool)
    return nullcontext(pool)


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    is_asleep: bool = False


class MemAllocator(Protocol):
    def use_memory_pool(self, tag: str | None = None) -> AbstractContextManager: ...

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None: ...

    def discard(self, tags: tuple[str, ...] | str) -> None: ...

    def wake_up(self, tags: list[str] | None = None) -> None: ...

    def get_current_usage(self) -> int: ...


def get_mem_allocator_instance() -> MemAllocator:
    if current_platform.is_cuda_alike():
        from vllm.device_allocator.cumem import CuMemAllocator

        return CuMemAllocator.get_instance()

    if current_platform.is_xpu():
        from vllm.device_allocator.xpumem import XpuMemAllocator

        return XpuMemAllocator.get_instance()

    raise RuntimeError(
        "Sleep mode allocator is not available on platform "
        f"{type(current_platform).__name__} "
        f"(device_type={current_platform.device_type})."
    )
