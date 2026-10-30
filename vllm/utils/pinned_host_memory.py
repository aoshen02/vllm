# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned host memory whose registration can be dropped and restored in place.

PyTorch pins host memory with ``cudaHostAlloc`` and rounds each block up to a
power of two. Memory allocated here is instead an exact-size (page-rounded)
anonymous mapping registered with ``cudaHostRegister``. Its owner keeps the
pages and their address for its whole lifetime, so the registration can be
released, e.g. around a CUDA process checkpoint, and restored at the same
address; device views and CUDA graphs that read it through UVA stay valid.
"""

import mmap
import weakref

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class _HostRegistration:
    """One registered host range; released and restored as a group."""

    def __init__(self, pointer: int, num_bytes: int):
        self.pointer = pointer
        self.num_bytes = num_bytes
        self.registered = False

    def register(self) -> None:
        result = torch.cuda.cudart().cudaHostRegister(self.pointer, self.num_bytes, 0)
        if result.value != 0:
            raise RuntimeError(
                f"cudaHostRegister of {self.num_bytes} bytes failed: {result}"
            )
        self.registered = True

    def unregister(self) -> None:
        result = torch.cuda.cudart().cudaHostUnregister(self.pointer)
        if result.value != 0:
            raise RuntimeError(f"cudaHostUnregister failed: {result}")
        self.registered = False


# Registrations die with their owner array; the set only enumerates them.
_registrations: "weakref.WeakSet[_HostRegistration]" = weakref.WeakSet()
_released = False


def _close(registration: _HostRegistration, keepalive: object) -> None:
    # `keepalive` (the mapping) outlives the registration it backs.
    if registration.registered:
        try:
            registration.unregister()
        except RuntimeError:
            logger.warning("Dropping a host registration failed", exc_info=True)


def pin_host_array(array: np.ndarray, keepalive: object = None) -> torch.Tensor:
    """Register ``array``'s pages with the device for its lifetime.

    ``array`` must cover whole pages that no other registration shares, and
    ``keepalive`` must own its memory (e.g. the mapping it views) if the array
    does not. Returns a uint8 CPU tensor sharing the array's storage.
    """
    if _released:
        raise RuntimeError("Host registrations are released; restore them first")
    registration = _HostRegistration(array.ctypes.data, array.nbytes)
    registration.register()
    _registrations.add(registration)
    # Torch storage retains the array (and through it the mapping), including
    # through cached views; unregister only once the array itself is gone.
    weakref.finalize(array, _close, registration, keepalive).atexit = False
    return torch.from_numpy(array.view(np.uint8).reshape(-1))


def pinned_host_empty(
    shape: int | tuple[int, ...] | list[int] | torch.Size, dtype: torch.dtype
) -> torch.Tensor:
    """Zero-filled pinned CPU tensor of exactly the requested size.

    On CUDA the memory is a registered anonymous mapping (see module
    docstring); elsewhere it comes from PyTorch's pinned allocator.
    """
    size = torch.Size([shape] if isinstance(shape, int) else shape)
    if not current_platform.is_cuda():
        return torch.zeros(size, dtype=dtype, pin_memory=True)
    num_bytes = size.numel() * dtype.itemsize
    if num_bytes == 0:
        return torch.empty(size, dtype=dtype)
    mapping = mmap.mmap(-1, num_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    raw = pin_host_array(np.frombuffer(mapping, dtype=np.uint8), mapping)
    return raw.view(dtype).view(size)


def release_pinned_host_memory() -> None:
    """Leave the process holding no pinned host memory, or raise.

    Drops every registration made here (pages and addresses stay) and frees
    PyTorch's cached pinned blocks. PyTorch-pinned blocks that are still in use
    cannot be released in place, so they fail the call.
    """
    global _released
    if _released:
        return
    torch._C._host_emptyCache()
    stats = torch.cuda.host_memory_stats()
    if stats.get("allocations.current", 0):
        raise RuntimeError(
            "Cannot release pinned host memory: PyTorch still holds "
            f"{stats['allocations.current']} pinned blocks "
            f"({stats.get('allocated_bytes.current', 0)} bytes) in use. "
            "Allocate long-lived pinned buffers with "
            "vllm.utils.pinned_host_memory.pinned_host_empty."
        )
    registrations = [r for r in _registrations if r.registered]
    for registration in registrations:
        registration.unregister()
    _released = True
    logger.info(
        "Released %d pinned host ranges (%.2f GiB)",
        len(registrations),
        sum(r.num_bytes for r in registrations) / 2**30,
    )


def restore_pinned_host_memory() -> None:
    """Register again, at the same addresses, every range released above."""
    global _released
    if not _released:
        return
    for registration in list(_registrations):
        if not registration.registered:
            registration.register()
    _released = False
