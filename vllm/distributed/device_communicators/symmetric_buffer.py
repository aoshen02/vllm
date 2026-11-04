# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric device buffers: one per rank, reachable from every rank.

`TorchSymmetricBuffer` uses PyTorch symmetric memory. Its peer mappings and
NVLink multicast object live as long as the buffer, so a process holding one
cannot be checkpointed by ``cuCheckpointProcess``: a live multicast object
hangs the checkpoint, and a peer import fails the importer's restore.

`VmmSymmetricBuffer` owns the same layout as cuMem allocations shared by
POSIX file descriptor (`cuda_vmm_ipc`). ``suspend`` releases the multicast
object and every peer import but keeps all address reservations and this
rank's memory, and ``resume`` maps everything back at the same addresses, so
kernels and captured CUDA graphs that hold any of them stay valid.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_vmm_ipc import (
    CuMemDriver,
    MulticastMapping,
    PeerAllocations,
    VmmBuffer,
    all_ranks_ok,
    multicast_granularity,
)


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a device address."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class SymmetricBuffer:
    """``tensor`` holds this rank's ``nbytes`` bytes. ``peer_ptrs[r]`` is the
    address of rank r's bytes in this process (this rank's own address at r ==
    rank), and ``multicast_ptr`` an NVLink multicast address that stores to
    every rank's bytes at once (0 when not requested or unavailable).

    Allocation is collective over the group, and so are ``suspend`` and
    ``resume``, which require the buffer to be idle.
    """

    tensor: torch.Tensor
    peer_ptrs: list[int]
    multicast_ptr: int

    def suspend(self) -> None:
        raise RuntimeError(f"{type(self).__name__} cannot be suspended")

    def resume(self) -> None:
        raise RuntimeError(f"{type(self).__name__} cannot be suspended")

    def close(self) -> None:
        pass


class TorchSymmetricBuffer(SymmetricBuffer):
    """PyTorch symmetric memory; mapped for the buffer's lifetime."""

    def __init__(
        self, nbytes: int, device: torch.device, group: ProcessGroup | str
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.tensor = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
        self._handle = symm_mem.rendezvous(self.tensor, group)
        self.peer_ptrs = list(self._handle.buffer_ptrs)
        self.multicast_ptr = int(self._handle.multicast_ptr or 0)

    def close(self) -> None:
        self._handle = None


class VmmSymmetricBuffer(SymmetricBuffer):
    """cuMem memory whose peer and multicast mappings can be suspended.

    ``group`` must be a CPU (gloo) group of ranks on one node. Peer mappings
    are made only with ``peers``; ``peer_ptrs`` is then indexed by rank, and
    otherwise holds only this rank's address. The multicast object is made
    only with ``multicast``. This rank's memory is zeroed and never released
    before ``close``, so it keeps its contents across a suspend.
    """

    def __init__(
        self,
        nbytes: int,
        device: int,
        group: ProcessGroup,
        *,
        peers: bool,
        multicast: bool,
    ) -> None:
        driver = CuMemDriver()
        granularity = (
            multicast_granularity(driver, device, dist.get_world_size(group))
            if multicast
            else None
        )
        buffer: VmmBuffer | None = None
        error: Exception | None = None
        try:
            buffer = VmmBuffer(nbytes, device, driver, granularity)
        except Exception as e:
            error = e
        if not all_ranks_ok(group, buffer is not None):
            if buffer is not None:
                buffer.close()
            raise RuntimeError(
                "A rank failed to allocate its symmetric buffer"
            ) from error
        assert buffer is not None
        self._buffer = buffer
        self._peers: PeerAllocations | None = None
        self._multicast: MulticastMapping | None = None
        try:
            self.peer_ptrs = [buffer.ptr]
            if peers:
                self._peers = PeerAllocations(group, device, driver)
                shared = self._peers.share([buffer.ptr])
                self.peer_ptrs = [rank_ptrs[0][0] for rank_ptrs in shared]
            self.multicast_ptr = 0
            if multicast:
                self._multicast = MulticastMapping(group, buffer, device, driver)
                self.multicast_ptr = self._multicast.ptr
        except BaseException:
            self.close()
            raise
        self.tensor = torch.as_tensor(
            _DeviceBytes(buffer.ptr, nbytes), device=torch.device("cuda", device)
        )

    def suspend(self) -> None:
        if self._multicast is not None:
            self._multicast.release()
        if self._peers is not None:
            self._peers.release()

    def resume(self) -> None:
        if self._peers is not None:
            self._peers.reshare()
        if self._multicast is not None:
            self._multicast.rebind()

    def close(self) -> None:
        if self._multicast is not None:
            self._multicast.close()
            self._multicast = None
        if self._peers is not None:
            self._peers.close()
            self._peers = None
        self._buffer.close()
