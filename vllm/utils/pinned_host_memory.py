# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-size pinned host memory.

PyTorch pins host memory with ``cudaHostAlloc`` and rounds each block up to a
power of two. Memory allocated here is instead an exact-size (page-rounded)
anonymous mapping registered with ``cudaHostRegister`` for its owner's
lifetime and returned to the OS when the owner frees it.
"""

import mmap
import weakref

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class _HostRegistration:
    """One registered host range."""

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
    registration = _HostRegistration(array.ctypes.data, array.nbytes)
    registration.register()
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
