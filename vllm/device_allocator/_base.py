# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import gc
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, ClassVar, Self

import torch

from vllm.device_allocator import (
    DEFERRABLE_TAGS,
    AllocationData,
    HandleType,
    discardable_allocations,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaseMemAllocator(ABC):
    """Shared base for singleton pluggable sleep-mode device memory allocators."""

    instance: ClassVar[Any] = None
    default_tag: ClassVar[str] = "default"

    @classmethod
    @abstractmethod
    def _check_available(cls) -> None:
        """Assert that the platform allocator extension is available."""
        raise NotImplementedError

    @classmethod
    def get_instance(cls) -> Self:
        """Return the process-wide singleton instance of this allocator."""
        cls._check_available()
        if cls.instance is None:
            cls.instance = cls()
            # Ensure MemPool/allocator wrappers are released before interpreter
            # finalization tears down PyTorch allocator internals.
            atexit.register(cls._shutdown_singleton)
        return cls.instance

    @classmethod
    def _shutdown_singleton(cls) -> None:
        instance = cls.instance
        if instance is None:
            return
        try:
            instance.release_pools()
        except Exception:
            logger.exception("%s singleton shutdown failed", cls.__name__)

    def __init__(self) -> None:
        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = self.default_tag
        self.allocator_and_pools: dict[str, list[Any]] = {}
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    @abstractmethod
    def release_pools(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _python_free_callback(self, ptr: int) -> HandleType:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _create_and_map(allocation_handle: HandleType) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _unmap_and_release(allocation_handle: HandleType) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _create_cpu_backup(size_in_bytes: int) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod  # noqa: B027
    def _release_cpu_backup(backup: torch.Tensor) -> None:
        """Release resources held by a CPU backup tensor (optional hook)."""

    @staticmethod
    def _copy(src_ptr: int, dst_ptr: int, nbytes: int) -> None:
        """Copy `nbytes` bytes from `src_ptr` to `dst_ptr`."""
        raise NotImplementedError

    def _copy_to_cpu(self, data: AllocationData, ptr: int, cpu_ptr: int) -> None:
        self._copy(ptr, cpu_ptr, data.handle[1])

    def _copy_from_cpu(self, data: AllocationData, cpu_ptr: int, ptr: int) -> None:
        self._copy(cpu_ptr, ptr, data.handle[1])

    def _post_wake_discarded(self, ptr: int, handle: HandleType) -> None:  # noqa: B027
        """Optional hook run after remapping a discarded allocation."""

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """Internal method to store the allocation data
        when memory is allocated in the memory pool.
        """
        ptr = allocation_handle[2]
        self.pointer_to_data[ptr] = AllocationData(allocation_handle, self.current_tag)
        logger.debug(
            "Allocated %s bytes for %s with address %s from %s",
            allocation_handle[1],
            self.current_tag,
            ptr,
            type(self).__name__,
        )

    def sleep(
        self,
        offload_tags: tuple[str, ...] | str | None = None,
        discardable: Collection[int] = (),
    ) -> None:
        """Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        Args:
            offload_tags: The tags of the memory allocation that will be
                offloaded. The rest of the memory allocation will be discarded.
            discardable: Addresses of tensors the caller rewrites after waking.
                Weights allocations holding nothing else are discarded too.

        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (self.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0
        has_policy_conflict = False
        discarded = discardable_allocations(self.allocator_and_pools, discardable)

        for ptr, data in self.pointer_to_data.items():
            offload = data.tag in offload_tags and ptr not in discarded
            if data.is_asleep:
                if offload != (data.cpu_backup_tensor is not None):
                    has_policy_conflict = True
                continue
            total_bytes += data.handle[1]
            if offload:
                backup_bytes += data.handle[1]
            self._put_to_sleep(ptr, data, offload)

        logger.info(
            "%s: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            type(self).__name__,
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        if has_policy_conflict:
            logger.warning(
                "%s: sleep cannot change the policy of "
                "already-asleep allocations; the existing policy was kept.",
                type(self).__name__,
            )

        gc.collect()
        torch.accelerator.empty_cache()

    def _put_to_sleep(self, ptr: int, data: AllocationData, offload: bool) -> None:
        if offload:
            if data.cpu_backup_tensor is None:
                data.cpu_backup_tensor = self._create_cpu_backup(data.handle[1])
            cpu_ptr = data.cpu_backup_tensor.data_ptr()
            self._copy_to_cpu(data, ptr, cpu_ptr)
        elif data.cpu_backup_tensor is not None:
            self._release_cpu_backup(data.cpu_backup_tensor)
            data.cpu_backup_tensor = None
        try:
            self._unmap_and_release(data.handle)
        finally:
            data.is_asleep = True

    def discard(self, tags: tuple[str, ...] | str) -> None:
        """Discard mapped allocations with the given tags without CPU backup."""
        if isinstance(tags, str):
            tags = (tags,)

        discarded_bytes = 0
        has_policy_conflict = False
        for ptr, data in self.pointer_to_data.items():
            if data.tag not in tags:
                continue
            if data.is_asleep:
                if data.cpu_backup_tensor is not None:
                    has_policy_conflict = True
                continue
            torch.accelerator.synchronize(data.handle[0])
            self._put_to_sleep(ptr, data, offload=False)
            discarded_bytes += data.handle[1]

        logger.info(
            "%s: discarded %.2f GiB for tags %s.",
            type(self).__name__,
            discarded_bytes / 1024**3,
            tags,
        )

        if has_policy_conflict:
            logger.warning(
                "%s: discard cannot change the policy of "
                "already-asleep allocations; the existing policy was kept.",
                type(self).__name__,
            )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to device
        memory, and the rest will have empty memory (zeroed on ROCm).

        Args:
            tags: The deferrable tags (weights, kv_cache) to load back to device
                memory; every other tag is always loaded back. If None, all
                memory allocation will be loaded back to device memory.

        """
        gc.collect()
        torch.accelerator.empty_cache()

        for ptr, data in self.pointer_to_data.items():
            if not data.is_asleep:
                continue
            if tags is None or data.tag in tags or data.tag not in DEFERRABLE_TAGS:
                handle = data.handle
                self._create_and_map(handle)
                data.is_asleep = False
                if data.cpu_backup_tensor is not None:
                    cpu_ptr = data.cpu_backup_tensor.data_ptr()
                    self._copy_from_cpu(data, cpu_ptr, ptr)
                else:
                    self._post_wake_discarded(ptr, handle)

    def get_current_usage(self) -> int:
        """Get the total number of bytes allocated in the memory pool."""
        return sum(data.handle[1] for data in self.pointer_to_data.values())
