# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for BaseMemAllocator, CuMemAllocator, and XpuMemAllocator."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm.device_allocator.cumem as cumem
import vllm.device_allocator.xpumem as xpumem
from vllm.device_allocator._base import BaseMemAllocator
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.device_allocator.xpumem import XpuMemAllocator


def _make_handle(
    ptr: int, size: int = 64, device: int = 0
) -> tuple[int, int, int, int]:
    return (device, size, ptr, ptr + 1)


def test_singleton_isolation_and_shutdown(monkeypatch):
    """CuMemAllocator and XpuMemAllocator keep independent singletons."""
    monkeypatch.setattr(cumem, "cumem_available", True)
    monkeypatch.setattr(xpumem, "xpumem_available", True)
    monkeypatch.setattr(CuMemAllocator, "instance", None)
    monkeypatch.setattr(XpuMemAllocator, "instance", None)

    registered = []
    monkeypatch.setattr(
        "vllm.device_allocator._base.atexit.register", registered.append
    )

    cu = CuMemAllocator.get_instance()
    xpu = XpuMemAllocator.get_instance()
    assert isinstance(cu, CuMemAllocator)
    assert isinstance(cu, BaseMemAllocator)
    assert isinstance(xpu, XpuMemAllocator)
    assert isinstance(xpu, BaseMemAllocator)
    assert CuMemAllocator.get_instance() is cu
    assert XpuMemAllocator.get_instance() is xpu
    assert cu is not xpu
    assert len(registered) == 2

    released = []
    monkeypatch.setattr(cu, "release_pools", lambda: released.append("cu"))
    monkeypatch.setattr(xpu, "release_pools", lambda: released.append("xpu"))
    CuMemAllocator._shutdown_singleton()
    XpuMemAllocator._shutdown_singleton()
    assert released == ["cu", "xpu"]


def test_get_instance_unavailable_asserts(monkeypatch):
    monkeypatch.setattr(cumem, "cumem_available", False)
    monkeypatch.setattr(xpumem, "xpumem_available", False)
    with pytest.raises(AssertionError, match="cumem allocator is not available"):
        CuMemAllocator.get_instance()
    with pytest.raises(AssertionError, match="xpumem allocator is not available"):
        XpuMemAllocator.get_instance()


def test_cumem_subclass_hooks_sleep_discard_wake(monkeypatch):
    """Out-of-tree CuMemAllocator subclasses overriding the 7 documented hooks
    continue to receive exact hook invocations across sleep/discard/wake_up."""
    events: list[tuple] = []

    class CustomCuMemAllocator(CuMemAllocator):
        instance = None

        @staticmethod
        def _create_and_map(allocation_handle) -> None:
            events.append(("create_and_map", allocation_handle))

        @staticmethod
        def _unmap_and_release(allocation_handle) -> None:
            events.append(("unmap_and_release", allocation_handle))

        @staticmethod
        def _create_cpu_backup(size_in_bytes: int) -> torch.Tensor:
            events.append(("create_cpu_backup", size_in_bytes))
            return torch.zeros(size_in_bytes, dtype=torch.uint8)

        @staticmethod
        def _release_cpu_backup(backup: torch.Tensor) -> None:
            events.append(("release_cpu_backup", backup.nbytes))

        @staticmethod
        def _copy(src_ptr: int, dst_ptr: int, nbytes: int) -> None:
            events.append(("copy", src_ptr, dst_ptr, nbytes))

        def _get_pluggable_allocator(self):
            events.append(("get_pluggable_allocator",))
            return SimpleNamespace(_allocator="fake-alloc")

        @contextmanager
        def _use_memory_pool_with_allocator(self):
            alloc = self._get_pluggable_allocator()
            pool = SimpleNamespace(snapshot=lambda: [])
            yield pool, alloc

    empty_cache_calls = 0
    sync_calls = []

    def fake_empty_cache():
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.accelerator, "empty_cache", fake_empty_cache)
    monkeypatch.setattr(torch.accelerator, "synchronize", sync_calls.append)
    monkeypatch.setattr(cumem.current_platform, "is_rocm", lambda: False)

    alloc = CustomCuMemAllocator()
    h_weights = _make_handle(1000, 128)
    h_kv = _make_handle(2000, 256)
    h_runtime = _make_handle(3000, 64)

    with alloc.use_memory_pool("weights"):
        alloc.python_malloc_callback(h_weights)
    with alloc.use_memory_pool("kv_cache"):
        alloc.python_malloc_callback(h_kv)
    with alloc.use_memory_pool("runtime"):
        alloc.python_malloc_callback(h_runtime)

    assert alloc.get_current_usage() == 128 + 256 + 64

    # Sleep offloading weights and runtime, discarding kv_cache.
    alloc.sleep(offload_tags=("weights", "runtime"))
    assert empty_cache_calls == 1
    assert all(d.is_asleep for d in alloc.pointer_to_data.values())
    assert alloc.pointer_to_data[1000].cpu_backup_tensor is not None
    assert alloc.pointer_to_data[2000].cpu_backup_tensor is None
    assert alloc.pointer_to_data[3000].cpu_backup_tensor is not None

    # Selective wake_up(tags=["weights"]) wakes weights and non-deferrable
    # runtime, leaving kv_cache asleep.
    alloc.wake_up(tags=["weights"])
    assert empty_cache_calls == 2
    assert not alloc.pointer_to_data[1000].is_asleep
    assert alloc.pointer_to_data[2000].is_asleep
    assert not alloc.pointer_to_data[3000].is_asleep

    # Discard weights releases its CPU backup and marks it asleep.
    alloc.discard("weights")
    assert alloc.pointer_to_data[1000].is_asleep
    assert alloc.pointer_to_data[1000].cpu_backup_tensor is None
    assert sync_calls == [0]
    assert ("release_cpu_backup", 128) in events

    # Wake remaining tags.
    alloc.wake_up()
    assert empty_cache_calls == 3
    assert all(not d.is_asleep for d in alloc.pointer_to_data.values())


def test_xpumem_sleep_wake_and_unmap_failure_marks_asleep(monkeypatch):
    """XpuMemAllocator uses _xpu_memcpy_sync, runs empty_cache in wake_up,
    and marks is_asleep=True even if unmap raises."""
    ops: list[tuple] = []
    empty_cache_calls = 0

    def fake_empty_cache():
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    fake_ext = SimpleNamespace(
        python_create_and_allocate=lambda *a: ops.append(("create", a)),
        python_unmap_and_release=lambda *a: ops.append(("unmap", a)),
    )
    monkeypatch.setattr(xpumem, "xpumem_available", True)
    monkeypatch.setattr(xpumem, "xpumem_allocator", fake_ext)
    monkeypatch.setattr(
        xpumem,
        "_xpu_memcpy_sync",
        lambda dst, src, nbytes, kind, dev: ops.append(("memcpy", kind, nbytes, dev)),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", fake_empty_cache)
    monkeypatch.setattr(xpumem, "PIN_MEMORY", False)

    alloc = XpuMemAllocator()
    alloc.current_tag = "weights"
    h1 = _make_handle(4000, 32, device=1)
    alloc.python_malloc_callback(h1)

    alloc.sleep(offload_tags=("weights",))
    assert empty_cache_calls == 1
    assert alloc.pointer_to_data[4000].is_asleep is True
    assert ("memcpy", xpumem.MEMCPY_DEVICE_TO_HOST, 32, 1) in ops
    assert ("unmap", h1) in ops

    alloc.wake_up()
    assert empty_cache_calls == 2
    assert alloc.pointer_to_data[4000].is_asleep is False
    assert ("create", h1) in ops
    assert ("memcpy", xpumem.MEMCPY_HOST_TO_DEVICE, 32, 1) in ops

    # Verify try/finally sets is_asleep=True even when unmap raises.
    def boom(*args):
        raise RuntimeError("unmap failure")

    monkeypatch.setattr(fake_ext, "python_unmap_and_release", boom)
    with pytest.raises(RuntimeError, match="unmap failure"):
        alloc.sleep(offload_tags=())
    assert alloc.pointer_to_data[4000].is_asleep is True
