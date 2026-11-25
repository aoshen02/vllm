# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.device_allocator import AllocationData, HandleType
from vllm.device_allocator._base import BaseMemAllocator
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)

MEMCPY_HOST_TO_DEVICE = 0
MEMCPY_DEVICE_TO_HOST = 1
MEMCPY_DEVICE_TO_DEVICE = 2

xpumem_available = False
xpumem_allocator: Any = None

try:
    from vllm_xpu_kernels import xpumem_allocator as _xpumem_allocator

    xpumem_allocator = _xpumem_allocator
    xpumem_available = True
except ImportError:
    xpumem_allocator = None


def _xpu_memory_module() -> Any:
    mem_mod = getattr(torch.xpu, "memory", None)
    if mem_mod is None:
        raise RuntimeError("torch.xpu.memory is not available")
    return mem_mod


def _supports_xpu_mem_pool(mem_mod: Any) -> bool:
    return hasattr(mem_mod, "MemPool") and hasattr(mem_mod, "use_mem_pool")


def _xpu_memcpy_sync(
    dst_ptr: int,
    src_ptr: int,
    n_bytes: int,
    kind: int,
    device: int,
) -> None:
    def _to_i64_ptr(ptr: int) -> int:
        # torch custom-op `int` arguments are signed int64.
        # data_ptr() may return a uint64 value above 2^63-1, so normalize it.
        return ptr if ptr < (1 << 63) else ptr - (1 << 64)

    torch.ops._C.xpu_memcpy_sync(
        _to_i64_ptr(dst_ptr),
        _to_i64_ptr(src_ptr),
        n_bytes,
        kind,
        device,
    )


class XpuMemAllocator(BaseMemAllocator):
    """A singleton pluggable allocator helper for XPU.

    Note:
    Sleep will offload selected payloads to CPU or discard and unmap XPU
    physical memory. Wake-up remaps physical memory back to the same
    reserved virtual address and restores payload.

    """

    instance: "XpuMemAllocator | None" = None

    @classmethod
    def _check_available(cls) -> None:
        assert xpumem_available, "xpumem allocator is not available"

    def __init__(self) -> None:
        super().__init__()
        self._pending_allocators: list[Any] = []

    @staticmethod
    def _create_and_map(allocation_handle: HandleType) -> None:
        if not xpumem_available or xpumem_allocator is None:
            raise RuntimeError("xpumem allocator extension is not available")
        xpumem_allocator.python_create_and_allocate(*allocation_handle)

    @staticmethod
    def _unmap_and_release(allocation_handle: HandleType) -> None:
        if not xpumem_available or xpumem_allocator is None:
            raise RuntimeError("xpumem allocator extension is not available")
        xpumem_allocator.python_unmap_and_release(*allocation_handle)

    @staticmethod
    def _create_cpu_backup(size_in_bytes: int) -> torch.Tensor:
        return torch.empty(
            size_in_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

    def _copy_to_cpu(self, data: AllocationData, ptr: int, cpu_ptr: int) -> None:
        device, size_in_bytes, _, _ = data.handle
        _xpu_memcpy_sync(cpu_ptr, ptr, size_in_bytes, MEMCPY_DEVICE_TO_HOST, device)

    def _copy_from_cpu(self, data: AllocationData, cpu_ptr: int, ptr: int) -> None:
        device, size_in_bytes, _, _ = data.handle
        _xpu_memcpy_sync(ptr, cpu_ptr, size_in_bytes, MEMCPY_HOST_TO_DEVICE, device)

    def _get_pluggable_allocator(self) -> Any:
        if not xpumem_available or xpumem_allocator is None:
            raise RuntimeError("xpumem allocator extension is not available")

        xpumem_allocator.init_module(
            self.python_malloc_callback, self.python_free_callback
        )
        mem_mod = _xpu_memory_module()
        alloc_cls = getattr(mem_mod, "XPUPluggableAllocator", None)
        if alloc_cls is None:
            raise RuntimeError(
                "torch.xpu.memory.XPUPluggableAllocator is not available"
            )

        lib_name = xpumem_allocator.__file__
        return alloc_cls(lib_name, "my_malloc", "my_free")

    @contextmanager
    def _use_memory_pool_with_allocator(self) -> Generator[tuple[Any, Any]]:
        mem_mod = _xpu_memory_module()
        if not _supports_xpu_mem_pool(mem_mod):
            raise RuntimeError(
                "torch.xpu.memory MemPool APIs are not available "
                "(need MemPool and use_mem_pool)."
            )
        new_alloc = self._get_pluggable_allocator()
        mem_pool = mem_mod.MemPool(new_alloc._allocator)
        with mem_mod.use_mem_pool(mem_pool):
            yield mem_pool, new_alloc

    def _python_free_callback(self, ptr: int) -> HandleType:
        data = self.pointer_to_data.pop(ptr)
        data.cpu_backup_tensor = None
        logger.debug("Freed %s bytes for %s at %s", data.handle[1], data.tag, ptr)
        return data.handle

    def release_pools(self) -> None:
        """Release pools, retaining allocators until their allocations are freed."""
        if not self.allocator_and_pools and not self._pending_allocators:
            return

        pool_entries = [e for es in self.allocator_and_pools.values() for e in es]
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        # Live tensors can outlast their MemPool wrappers. Keep allocators alive
        # across calls so deferred frees cannot invoke a destroyed allocator.
        self._pending_allocators.extend(entry[1] for entry in pool_entries)
        pool_entries.clear()

        xpu_sync = getattr(torch.xpu, "synchronize", None)
        if callable(xpu_sync):
            try:
                xpu_sync()
            except Exception:
                logger.debug("torch.xpu.synchronize() failed during release_pools")

        mem_pools.clear()
        gc.collect()
        torch.xpu.empty_cache()

        # Free callbacks remove allocation records, including cached blocks.
        # If any remain, a later release_pools() call must retry this cleanup.
        if not self.pointer_to_data:
            self._pending_allocators.clear()

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        if tag is None:
            tag = self.default_tag

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            with self._use_memory_pool_with_allocator() as data:
                # Keep every entry's pool alive (pytorch/pytorch#146431).
                self.allocator_and_pools.setdefault(tag, []).append(data)
                yield
        finally:
            self.current_tag = old_tag
