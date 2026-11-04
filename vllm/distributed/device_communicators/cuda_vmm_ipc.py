# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer access to cuMem allocations shared by POSIX file descriptor.

Legacy CUDA IPC cannot be suspended: once ``cudaIpcGetMemHandle`` exports an
allocation, ``cuCheckpointProcess`` refuses to checkpoint its owner until the
allocation is freed, and a re-opened handle lands at a new address. cuMem
allocations exported by POSIX file descriptor have neither problem. A peer
drops its imports while suspended, keeps the virtual address reservations, and
maps the re-exported allocations at the same addresses on resume. Device-side
pointer tables and captured CUDA graphs that hold peer addresses therefore stay
valid without being rewritten.

File descriptors move between ranks over ``SCM_RIGHTS`` on abstract Unix
sockets, so ranks must share a node and a network namespace. ``pidfd_getfd``
would need ptrace permission.
"""

import array
import ctypes
import os
import socket
import struct
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch.distributed import ProcessGroup

_CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MULTICAST_GRANULARITY_MINIMUM = 0
# Linux SCM_MAX_FD is 253 descriptors per message.
_FDS_PER_MESSAGE = 64
_HEADER = struct.Struct("<iii")  # sender rank, first index, descriptor count


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class _MulticastProp(ctypes.Structure):
    _fields_ = [
        ("numDevices", ctypes.c_uint),
        ("size", ctypes.c_size_t),
        ("handleTypes", ctypes.c_ulonglong),
        ("flags", ctypes.c_ulonglong),
    ]


_u64 = ctypes.c_uint64
_size = ctypes.c_size_t
_SIGNATURES: dict[str, list[Any]] = {
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuMemCreate": [
        ctypes.POINTER(_u64),
        _size,
        ctypes.POINTER(_AllocationProp),
        _u64,
    ],
    "cuMemRelease": [_u64],
    "cuMemAddressReserve": [ctypes.POINTER(_u64), _size, _size, _u64, _u64],
    "cuMemAddressFree": [_u64, _size],
    "cuMemMap": [_u64, _size, _size, _u64, _u64],
    "cuMemUnmap": [_u64, _size],
    "cuMemSetAccess": [_u64, _size, ctypes.POINTER(_AccessDesc), _size],
    "cuMemExportToShareableHandle": [ctypes.c_void_p, _u64, ctypes.c_int, _u64],
    "cuMemImportFromShareableHandle": [
        ctypes.POINTER(_u64),
        ctypes.c_void_p,
        ctypes.c_int,
    ],
    "cuMemRetainAllocationHandle": [ctypes.POINTER(_u64), ctypes.c_void_p],
    "cuMemGetAddressRange_v2": [ctypes.POINTER(_u64), ctypes.POINTER(_size), _u64],
    "cuMemGetAllocationGranularity": [
        ctypes.POINTER(_size),
        ctypes.POINTER(_AllocationProp),
        ctypes.c_int,
    ],
    "cuMemsetD8_v2": [_u64, ctypes.c_ubyte, _size],
    "cuCtxSynchronize": [],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuMulticastCreate": [ctypes.POINTER(_u64), ctypes.POINTER(_MulticastProp)],
    "cuMulticastGetGranularity": [
        ctypes.POINTER(_size),
        ctypes.POINTER(_MulticastProp),
        ctypes.c_int,
    ],
    "cuMulticastAddDevice": [_u64, ctypes.c_int],
    "cuMulticastBindMem": [_u64, _size, _u64, _size, _size, _u64],
    "cuMulticastUnbind": [_u64, ctypes.c_int, _size, _size],
}


class CuMemDriver:
    """The cuMem driver calls used to share allocations by file descriptor."""

    _lib: Any = None

    def __init__(self) -> None:
        if CuMemDriver._lib is None:
            lib = ctypes.CDLL("libcuda.so.1")
            for name, argtypes in _SIGNATURES.items():
                fn = getattr(lib, name)
                fn.argtypes = argtypes
                fn.restype = ctypes.c_int
            CuMemDriver._lib = lib
        self._lib = CuMemDriver._lib

    def _check(self, name: str, *args) -> None:
        rc = getattr(self._lib, name)(*args)
        if rc:
            err = ctypes.c_char_p()
            self._lib.cuGetErrorName(rc, ctypes.byref(err))
            label = err.value.decode() if err.value else str(rc)
            raise RuntimeError(f"{name} failed: {label}")

    @staticmethod
    def _device_prop(device: int) -> _AllocationProp:
        prop = _AllocationProp()
        prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location = _Location(_CU_MEM_LOCATION_TYPE_DEVICE, device)
        return prop

    def granularity(self, device: int) -> int:
        size = _size()
        prop = self._device_prop(device)
        self._check(
            "cuMemGetAllocationGranularity",
            ctypes.byref(size),
            ctypes.byref(prop),
            _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        return size.value

    def multicast_granularity(self, num_devices: int) -> int:
        size = _size()
        prop = _MulticastProp(
            num_devices, 0, _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
        )
        self._check(
            "cuMulticastGetGranularity",
            ctypes.byref(size),
            ctypes.byref(prop),
            _CU_MULTICAST_GRANULARITY_MINIMUM,
        )
        return size.value

    def multicast_create(self, size: int, num_devices: int) -> int:
        handle = _u64()
        prop = _MulticastProp(
            num_devices, size, _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0
        )
        self._check("cuMulticastCreate", ctypes.byref(handle), ctypes.byref(prop))
        return handle.value

    def _cu_device(self, device: int) -> int:
        cu_device = ctypes.c_int()
        self._check("cuDeviceGet", ctypes.byref(cu_device), device)
        return cu_device.value

    def multicast_add_device(self, handle: int, device: int) -> None:
        self._check("cuMulticastAddDevice", handle, self._cu_device(device))

    def multicast_bind(self, handle: int, memory: int, size: int) -> None:
        self._check("cuMulticastBindMem", handle, 0, memory, 0, size, 0)

    def multicast_unbind(self, handle: int, device: int, size: int) -> None:
        self._check("cuMulticastUnbind", handle, self._cu_device(device), 0, size)

    def synchronize(self) -> None:
        self._check("cuCtxSynchronize")

    def create(self, size: int, device: int) -> int:
        handle = _u64()
        prop = self._device_prop(device)
        self._check("cuMemCreate", ctypes.byref(handle), size, ctypes.byref(prop), 0)
        return handle.value

    def release(self, handle: int) -> None:
        self._check("cuMemRelease", handle)

    def reserve(self, size: int, alignment: int) -> int:
        va = _u64()
        self._check("cuMemAddressReserve", ctypes.byref(va), size, alignment, 0, 0)
        return va.value

    def free_address(self, va: int, size: int) -> None:
        self._check("cuMemAddressFree", va, size)

    def map(self, va: int, size: int, handle: int, device: int) -> None:
        self._check("cuMemMap", va, size, 0, handle, 0)
        access = _AccessDesc(
            _Location(_CU_MEM_LOCATION_TYPE_DEVICE, device),
            _CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        )
        try:
            self._check("cuMemSetAccess", va, size, ctypes.byref(access), 1)
        except Exception:
            self._lib.cuMemUnmap(va, size)
            raise

    def unmap(self, va: int, size: int) -> None:
        self._check("cuMemUnmap", va, size)

    def zero(self, va: int, size: int) -> None:
        self._check("cuMemsetD8_v2", va, 0, size)
        self._check("cuCtxSynchronize")

    def address_range(self, ptr: int) -> tuple[int, int]:
        """Return the base and size of the allocation mapped at ``ptr``."""
        base, size = _u64(), _size()
        self._check(
            "cuMemGetAddressRange_v2", ctypes.byref(base), ctypes.byref(size), ptr
        )
        return base.value, size.value

    def is_cumem(self, ptr: int) -> bool:
        """Whether ``ptr`` lies in a cuMem mapping rather than cudaMalloc memory."""
        handle = _u64()
        if self._lib.cuMemRetainAllocationHandle(ctypes.byref(handle), ptr):
            return False
        self.release(handle.value)
        return True

    def export_handle_fd(self, handle: int) -> int:
        """Export an allocation or multicast handle; the caller owns the fd."""
        fd = ctypes.c_int(-1)
        self._check(
            "cuMemExportToShareableHandle",
            ctypes.byref(fd),
            handle,
            _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
        return fd.value

    def export_fd(self, ptr: int) -> int:
        """Export the allocation mapped at ``ptr``; the caller owns the fd."""
        handle = _u64()
        self._check("cuMemRetainAllocationHandle", ctypes.byref(handle), ptr)
        try:
            fd = self.export_handle_fd(handle.value)
        except RuntimeError as error:
            raise RuntimeError(
                "Allocation is not exportable by POSIX file descriptor; peer "
                "buffers must come from cuMem allocations that request it "
                f"({error})"
            ) from error
        finally:
            self.release(handle.value)
        return fd

    def import_fd(self, fd: int) -> int:
        handle = _u64()
        self._check(
            "cuMemImportFromShareableHandle",
            ctypes.byref(handle),
            ctypes.c_void_p(fd),
            _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
        return handle.value


def all_gather_object(group: ProcessGroup, obj: Any) -> list[Any]:
    # dist.all_gather_object does not work with gloo under inference mode
    # (pytorch/pytorch#126032); broadcast each rank's object instead.
    ranks = sorted(dist.get_process_group_ranks(group))
    rank = dist.get_rank(group)
    gathered: list[list[Any]] = [
        [obj if i == rank else None] for i in range(len(ranks))
    ]
    for i, src in enumerate(ranks):
        dist.broadcast_object_list(gathered[i], src=src, group=group, device="cpu")
    return [item[0] for item in gathered]


def exchange_fds(group: ProcessGroup, fds: list[int]) -> list[list[int]]:
    """Send ``fds`` to every peer; return the descriptors received from each.

    The entry for this rank is empty. The caller owns every returned descriptor.
    """
    rank = dist.get_rank(group)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.bind(f"\0vllm-cuda-vmm-ipc-{uuid.uuid4().hex}")
        peers = all_gather_object(group, (sock.getsockname(), len(fds)))
        sent: list[BaseException] = []

        def send() -> None:
            try:
                for peer, (address, _) in enumerate(peers):
                    if peer == rank:
                        continue
                    for start in range(0, len(fds), _FDS_PER_MESSAGE):
                        chunk = fds[start : start + _FDS_PER_MESSAGE]
                        header = _HEADER.pack(rank, start, len(chunk))
                        rights = array.array("i", chunk)
                        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
                        sock.sendmsg([header], ancillary, 0, address)
            except BaseException as error:
                sent.append(error)

        sender = threading.Thread(target=send, name="cuda-vmm-ipc-send")
        sender.start()
        received: list[list[int]] = [[-1] * n for _, n in peers]
        received[rank] = []
        missing = sum(n for peer, (_, n) in enumerate(peers) if peer != rank)
        try:
            while missing:
                msg, got, _, _ = socket.recv_fds(sock, _HEADER.size, _FDS_PER_MESSAGE)
                src, start, count = _HEADER.unpack(msg)
                if len(got) != count:
                    raise RuntimeError("Lost file descriptors in transfer")
                received[src][start : start + count] = got
                missing -= count
        finally:
            sender.join()
        if sent:
            raise RuntimeError("Failed to send file descriptors") from sent[0]
        return received
    finally:
        sock.close()


class VmmBuffer:
    """Zeroed device memory owned by this rank and exportable by POSIX fd.

    ``granularity`` overrides the allocation granularity, for example with the
    multicast granularity of a buffer that a `MulticastMapping` binds.
    """

    def __init__(
        self,
        size: int,
        device: int,
        driver: CuMemDriver,
        granularity: int | None = None,
    ) -> None:
        if granularity is None:
            granularity = driver.granularity(device)
        self.size = -(-size // granularity) * granularity
        self._driver = driver
        self._handle = driver.create(self.size, device)
        try:
            self.ptr = driver.reserve(self.size, granularity)
            try:
                driver.map(self.ptr, self.size, self._handle, device)
            except Exception:
                driver.free_address(self.ptr, self.size)
                raise
        except Exception:
            driver.release(self._handle)
            raise
        driver.zero(self.ptr, self.size)

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("VmmBuffer is closed")
        return self._handle

    def close(self) -> None:
        if self._handle is None:
            return
        self._driver.unmap(self.ptr, self.size)
        self._driver.release(self._handle)
        self._driver.free_address(self.ptr, self.size)
        self._handle = None


@dataclass
class _PeerMapping:
    va: int
    size: int
    handle: int | None


class PeerAllocations:
    """Peer allocations mapped into this process at stable addresses.

    ``share`` exports allocations of this rank and maps those of every peer.
    ``release`` drops all imports but keeps their address reservations, and
    ``reshare`` maps the re-exported allocations at the same addresses. The
    owners' allocations must keep their addresses across the cycle.
    """

    def __init__(self, group: ProcessGroup, device: int, driver: CuMemDriver):
        self.group = group
        self.rank = dist.get_rank(group)
        self.device = device
        self._driver = driver
        self._granularity = driver.granularity(device)
        # (peer rank, peer base) -> mapping in this process.
        self._mappings: dict[tuple[int, int], _PeerMapping] = {}
        # Bases of this rank's allocations shared so far, in order.
        self._shared: list[int] = []
        self.released = False

    def share(self, ptrs: list[int]) -> list[list[tuple[int, int]]]:
        """Share the allocations holding ``ptrs``.

        Returns, for each rank, the (local address, allocation base) of each
        of that rank's pointers. Local addresses keep the pointer's offset.
        """
        if self.released:
            raise RuntimeError("Peer allocations are released; reshare first")
        ranges = [self._driver.address_range(ptr) for ptr in ptrs]
        bases = list(dict.fromkeys(base for base, _ in ranges))
        located = [
            (base, ptr - base) for ptr, (base, _) in zip(ptrs, ranges, strict=True)
        ]
        local = self._exchange(bases, dict(ranges), allow_new=True)
        self._shared.extend(b for b in bases if b not in self._shared)
        per_rank = all_gather_object(self.group, located)
        return [
            [(local[peer][base] + offset, base) for base, offset in peer_located]
            for peer, peer_located in enumerate(per_rank)
        ]

    def release(self) -> None:
        """Unmap and release every import; keep the address reservations."""
        for mapping in self._mappings.values():
            if mapping.handle is not None:
                self._driver.unmap(mapping.va, mapping.size)
                self._driver.release(mapping.handle)
                mapping.handle = None
        self.released = True

    def reshare(self) -> None:
        """Re-import every shared allocation at its previous address."""
        if not self.released:
            return
        sizes = {base: self._driver.address_range(base)[1] for base in self._shared}
        self._exchange(self._shared, sizes, allow_new=False)
        self.released = False

    def close(self) -> None:
        self.release()
        for mapping in self._mappings.values():
            self._driver.free_address(mapping.va, mapping.size)
        self._mappings.clear()

    def _exchange(
        self, bases: list[int], sizes: dict[int, int], allow_new: bool
    ) -> list[dict[int, int]]:
        fds = [self._driver.export_fd(base) for base in bases]
        try:
            meta = all_gather_object(self.group, [(b, sizes[b]) for b in bases])
            received = exchange_fds(self.group, fds)
        finally:
            for fd in fds:
                os.close(fd)
        local: list[dict[int, int]] = []
        try:
            for peer, peer_meta in enumerate(meta):
                if peer == self.rank:
                    local.append({base: base for base, _ in peer_meta})
                    continue
                local.append(
                    {
                        base: self._map(peer, base, size, fd, allow_new)
                        for (base, size), fd in zip(
                            peer_meta, received[peer], strict=True
                        )
                    }
                )
        finally:
            for peer_fds in received:
                for fd in peer_fds:
                    os.close(fd)
        return local

    def _map(self, peer: int, base: int, size: int, fd: int, allow_new: bool) -> int:
        mapping = self._mappings.get((peer, base))
        if mapping is None:
            if not allow_new:
                raise RuntimeError(
                    f"Rank {peer} shared an allocation at {base:#x} that was "
                    "not mapped before the suspend"
                )
            va = self._driver.reserve(size, self._granularity)
            mapping = self._mappings[(peer, base)] = _PeerMapping(va, size, None)
        elif mapping.size != size:
            raise RuntimeError(
                f"Rank {peer}'s allocation at {base:#x} changed size from "
                f"{mapping.size} to {size}"
            )
        if mapping.handle is None:
            handle = self._driver.import_fd(fd)
            try:
                self._driver.map(mapping.va, size, handle, self.device)
            except Exception:
                self._driver.release(handle)
                raise
            mapping.handle = handle
        return mapping.va


def all_ranks_ok(group: ProcessGroup, ok: bool) -> bool:
    """Whether ``ok`` holds on every rank of ``group``; also a barrier."""
    return all(all_gather_object(group, ok))


def multicast_granularity(driver: CuMemDriver, device: int, num_devices: int) -> int:
    """Granularity of a `VmmBuffer` bound to a multicast object."""
    return max(driver.granularity(device), driver.multicast_granularity(num_devices))


class MulticastMapping:
    """An NVLink multicast object over one `VmmBuffer` of every rank, mapped at
    an address that survives `release` and `rebind`.

    A live multicast object blocks ``cuCheckpointProcessCheckpoint`` even when
    nothing is bound to it or mapped from it, so it cannot be kept across a
    checkpoint. ``release`` unbinds this rank's buffer, unmaps and releases the
    object, and returns once every rank has released it. The address
    reservation and the buffers stay, so ``rebind`` can collectively create an
    object of the same size, bind the same buffers and map it at the same
    address: multicast addresses held by kernels and captured CUDA graphs stay
    valid, and the buffers keep their contents.

    Each collective step ends with an agreement across ranks, so a failure on
    one rank fails every rank instead of leaving the others waiting for it.
    """

    def __init__(
        self,
        group: ProcessGroup,
        buffer: VmmBuffer,
        device: int,
        driver: CuMemDriver,
    ) -> None:
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = device
        self._driver = driver
        self._buffer = buffer
        self.size = buffer.size
        granularity = driver.multicast_granularity(self.world_size)
        if self.size % granularity:
            raise ValueError(
                f"Buffer size {self.size} is not a multiple of the multicast "
                f"granularity {granularity}"
            )
        self.ptr = driver.reserve(self.size, granularity)
        self._handle: int | None = None
        try:
            self.rebind()
        except BaseException:
            driver.free_address(self.ptr, self.size)
            raise

    @property
    def bound(self) -> bool:
        return self._handle is not None

    def rebind(self) -> None:
        """Create, bind and map a new multicast object (collective)."""
        if self._handle is not None:
            return
        driver, device, size = self._driver, self.device, self.size
        handle = self._create_or_import()
        undo = [lambda: driver.release(handle)]
        steps = (
            (
                "add its device to",
                lambda: driver.multicast_add_device(handle, device),
                None,
            ),
            # The agreement above orders every rank's device before any bind.
            (
                "bind its buffer to",
                lambda: driver.multicast_bind(handle, self._buffer.handle, size),
                lambda: driver.multicast_unbind(handle, device, size),
            ),
            (
                "map",
                lambda: driver.map(self.ptr, size, handle, device),
                lambda: driver.unmap(self.ptr, size),
            ),
        )
        try:
            for step, run, revert in steps:
                error: Exception | None = None
                try:
                    run()
                except Exception as e:
                    error = e
                else:
                    if revert is not None:
                        undo.append(revert)
                if not all_ranks_ok(self.group, error is None):
                    raise RuntimeError(
                        f"A rank failed to {step} the multicast object"
                    ) from error
        except BaseException:
            for revert in reversed(undo):
                revert()
            raise
        self._handle = handle

    def release(self) -> None:
        """Unbind, unmap and release the multicast object but keep its address
        reservation (collective; the buffers must be idle)."""
        if self._handle is None:
            return
        # No rank may still access the object when another one unbinds.
        self._driver.synchronize()
        all_ranks_ok(self.group, True)
        self._drop()
        # No rank holds the object once this returns.
        all_ranks_ok(self.group, True)

    def close(self) -> None:
        """Free everything without synchronizing with other ranks."""
        self._drop()
        self._driver.free_address(self.ptr, self.size)

    def _drop(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        self._driver.multicast_unbind(handle, self.device, self.size)
        self._driver.unmap(self.ptr, self.size)
        self._driver.release(handle)

    def _create_or_import(self) -> int:
        """Return a handle to an object rank 0 creates (collective)."""
        handle = fd = None
        error: Exception | None = None
        if self.rank == 0:
            try:
                handle = self._driver.multicast_create(self.size, self.world_size)
                fd = self._driver.export_handle_fd(handle)
            except Exception as e:
                error = e
        try:
            if not all_ranks_ok(self.group, error is None):
                raise RuntimeError("Rank 0 failed to create a multicast object") from (
                    error
                )
            received = exchange_fds(self.group, [] if fd is None else [fd])
        except BaseException:
            if handle is not None:
                self._driver.release(handle)
            raise
        finally:
            if fd is not None:
                os.close(fd)
        if self.rank != 0:
            (peer_fd,) = received[0]
            try:
                handle = self._driver.import_fd(peer_fd)
            except Exception as e:
                error = e
            finally:
                os.close(peer_fd)
        if not all_ranks_ok(self.group, error is None):
            if handle is not None:
                self._driver.release(handle)
            raise RuntimeError("A rank failed to import the multicast object") from (
                error
            )
        assert handle is not None
        return handle
