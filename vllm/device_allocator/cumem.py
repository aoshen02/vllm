# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import gc
import mmap
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch

from vllm.device_allocator import HandleType
from vllm.device_allocator._base import BaseMemAllocator
from vllm.device_allocator.alloc_conf import (
    EXPANDABLE_SEGMENTS,
    conf_flag_enabled,
    current_alloc_conf,
    set_alloc_conf,
    with_conf_flag,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import find_loaded_library
from vllm.utils.torch_utils import PIN_MEMORY

logger = init_logger(__name__)


cumem_available = False
libcudart: Any = None
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
except ModuleNotFoundError:
    # only cuda and rocm platforms support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None


class CuMemAllocator(BaseMemAllocator):
    """A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory (zeroed on ROCm).

    The `_create_and_map`, `_unmap_and_release`, `_create_cpu_backup`,
    `_release_cpu_backup`, `_copy`, `_get_pluggable_allocator`, and
    `_use_memory_pool_with_allocator` methods are subclass hooks for
    out-of-tree device allocators. Their names and signatures form the
    subclassing contract; renaming them or changing their signatures is
    a breaking change.

    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance: "CuMemAllocator | None" = None

    @classmethod
    def _check_available(cls) -> None:
        assert cumem_available, "cumem allocator is not available"

    @staticmethod
    def _create_and_map(allocation_handle: HandleType) -> None:
        python_create_and_map(*allocation_handle)

    @staticmethod
    def _unmap_and_release(allocation_handle: HandleType) -> None:
        python_unmap_and_release(*allocation_handle)

    @staticmethod
    def _create_cpu_backup(size_in_bytes: int) -> torch.Tensor:
        backup = torch.frombuffer(
            mmap.mmap(-1, size_in_bytes, flags=mmap.MAP_PRIVATE), dtype=torch.uint8
        )
        if PIN_MEMORY:
            error = libcudart.cudaHostRegister(backup.data_ptr(), size_in_bytes)
            if error:
                libcudart.cudaGetLastError()
                logger.warning_once(
                    "cudaHostRegister failed (error %d); sleep mode CPU backups "
                    "fall back to pageable memory.",
                    error,
                )
        return backup

    @staticmethod
    def _release_cpu_backup(backup: torch.Tensor) -> None:
        if backup.is_pinned() and libcudart.cudaHostUnregister(backup.data_ptr()):
            libcudart.cudaGetLastError()

    @staticmethod
    def _copy(src_ptr: int, dst_ptr: int, nbytes: int) -> None:
        """Copy `nbytes` bytes from `src_ptr` to `dst_ptr`."""
        libcudart.cudaMemcpy(dst_ptr, src_ptr, nbytes)

    def _post_wake_discarded(self, ptr: int, handle: HandleType) -> None:
        if current_platform.is_rocm():
            # amdgpu <6.14 may return stale VRAM (#44972); drop with its support
            libcudart.cudaMemset(ptr, 0, handle[1])
            torch.accelerator.synchronize()

    def _get_pluggable_allocator(self) -> torch.cuda.memory.CUDAPluggableAllocator:
        init_module(self.python_malloc_callback, self.python_free_callback)
        return torch.cuda.memory.CUDAPluggableAllocator(
            lib_name, "my_malloc", "my_free"
        )

    @contextmanager
    def _use_memory_pool_with_allocator(
        self,
    ) -> Generator[
        tuple[torch.cuda.memory.MemPool, torch.cuda.memory.CUDAPluggableAllocator]
    ]:
        new_alloc = self._get_pluggable_allocator()
        mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
        with torch.cuda.memory.use_mem_pool(mem_pool):
            yield mem_pool, new_alloc

    def release_pools(self) -> None:
        """Drop Python references to MemPool/pluggable allocators eagerly.

        A cumem `MemPool` outlives the `use_memory_pool` context (a strong
        reference is kept in `allocator_and_pools` to work around
        pytorch/pytorch#146431), and a captured CUDA graph can keep it alive
        longer still. `MemPool` only holds a non-owning pointer to the
        allocator, whose owning reference lives in the Python
        `CUDAPluggableAllocator`. If both are instead dropped during
        interpreter shutdown, GC may finalize the allocator first; the eventual
        `~MemPool` -> `emptyCache` -> `release_block` then makes a virtual
        call into the freed allocator -- aborting the process with "pure virtual
        method called" (pytorch/pytorch#145168).

        Release the kept-alive pools before interpreter finalization, and keep
        the pluggable allocator wrappers alive while MemPool destructors run.
        This is safe to call more than once.
        """
        if not self.allocator_and_pools:
            return

        pool_entries = [e for es in self.allocator_and_pools.values() for e in es]
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        allocators = [entry[1] for entry in pool_entries]
        pool_entries.clear()

        # Phase 1: drop MemPool refs while allocators are still strongly held.
        mem_pools.clear()
        gc.collect()

        # Phase 2: now it is safe to release allocator wrappers.
        allocators.clear()

    def close(self) -> None:
        """Compatibility alias for deterministic pool release."""
        self.release_pools()

    def _python_free_callback(self, ptr: int) -> HandleType:
        """Internal method to look up the allocation data
        when memory is freed in the memory pool.
        """
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            self._release_cpu_backup(data.cpu_backup_tensor)
            data.cpu_backup_tensor = None
        if data.is_asleep and current_platform.is_rocm():
            # On ROCm, sleep() already unmapped and released this allocation's
            # physical chunks and holds its virtual address as a placeholder
            # reservation. Return a handle with an empty chunk list so the C
            # extension skips unmap/release (avoiding a double-free) while
            # still freeing the placeholder address.
            device, size, d_mem, _ = data.handle
            return (device, size, d_mem, [])
        # Drain pending kernels before the C extension's cuMemUnmap.
        # The pluggable allocator path doesn't defer reclaim like the
        # regular caching allocator, so without this, in-flight work
        # (e.g. quant helpers' transient tensors during weight loading)
        # races the unmap and surfaces as CUDA_ERROR_ILLEGAL_ADDRESS.
        torch.accelerator.synchronize(data.handle[0])
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        Args:
            tag: The tag of the memory allocation. If None, the default tag
                will be used.

        """
        if tag is None:
            tag = self.default_tag

        assert isinstance(tag, str)

        # Expandable segments are incompatible with the memory pool used for
        # sleep mode (see https://github.com/pytorch/pytorch/issues/147851).
        # If the user has enabled expandable segments, temporarily disable
        # them for the duration of the memory pool context and restore on
        # exit. The whole config string is rewritten, not just this one
        # field: torch resets every option that is absent from the string it
        # is handed, so a bare "expandable_segments:False" would also drop
        # max_split_size_mb, garbage_collection_threshold and
        # roundup_power2_divisions for the rest of the process.
        prev_conf = current_alloc_conf()
        expandable_was_enabled = conf_flag_enabled(prev_conf, EXPANDABLE_SEGMENTS)
        if expandable_was_enabled:
            set_alloc_conf(with_conf_flag(prev_conf, EXPANDABLE_SEGMENTS, False))

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            if tag != old_tag:
                # Older pools of this tag are never reused, so trim them.
                for pool, _ in self.allocator_and_pools.get(tag, []):
                    self._trim(pool)
            with self._use_memory_pool_with_allocator() as data:
                # start to hit another PyTorch bug in PyTorch 2.6,
                # possibly because of gc-related issue w.r.t. the allocator
                # and the memory pool.
                # to avoid the issue, we keep a reference of the data.
                # see https://github.com/pytorch/pytorch/issues/146431 .
                self.allocator_and_pools.setdefault(tag, []).append(data)
                yield
                # PyTorch's bug, torch.accelerator.empty_cache() does not
                # release memory held by a pluggable allocator, see
                # https://github.com/pytorch/pytorch/issues/145168 .
                # if we have some memory allocated and then freed,
                # the memory will not be released, e.g. in online
                # quantization, where the model is created in higher
                # precision, and then quantized in lower precision.
                # Find all unused allocations and manually release them.
                # TODO: we should expose `empty_cache` method in the memory
                # pool.
                # TODO: ask for help from PyTorch team to expose this method.
                self._trim(data[0])
        finally:
            self.current_tag = old_tag
            if expandable_was_enabled:
                set_alloc_conf(prev_conf)

    def _trim(self, pool: Any) -> None:
        """Release free segments of a pool that is never allocated from again."""
        for allocation in pool.snapshot():
            data = self.pointer_to_data.get(allocation["address"])
            if allocation["allocated_size"] == 0 and data and not data.is_asleep:
                self._unmap_and_release(
                    self._python_free_callback(allocation["address"])
                )

    @contextmanager
    def cudagraph_pool(self) -> Generator[tuple[int, int]]:
        """Tag CUDA graph capture allocations and yield the graph pool id."""
        if "cudagraph" not in self.allocator_and_pools:
            allocator = self._get_pluggable_allocator()
            mem_pool = torch.cuda.memory.MemPool(allocator._allocator)
            self.allocator_and_pools["cudagraph"] = [(mem_pool, allocator)]
        old_tag = self.current_tag
        self.current_tag = "cudagraph"
        try:
            # capture_begin routes allocations to this pool; no use_mem_pool.
            yield self.allocator_and_pools["cudagraph"][0][0].id
        finally:
            self.current_tag = old_tag
