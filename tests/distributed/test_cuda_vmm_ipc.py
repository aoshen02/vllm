# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer cuMem mappings keep their addresses across release and reshare."""

import os
import tempfile

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators import cuda_vmm_ipc
from vllm.distributed.device_communicators.cuda_vmm_ipc import PeerAllocations

# The simulated peer's allocations sit at this distance from this rank's.
PEER_SHIFT = 0x1000_0000
GRANULARITY = 0x20_0000


class FakeDriver:
    """Records cuMem calls; this rank owns ``allocations`` (base -> size)."""

    def __init__(self, allocations: dict[int, int]):
        self.allocations = allocations
        self.next_va = 0x7000_0000_0000
        self.next_handle = 1
        self.reserved: dict[int, int] = {}
        self.mapped: dict[int, int] = {}  # va -> handle
        self.live_handles: set[int] = set()
        self.imports = 0

    def granularity(self, device):
        return GRANULARITY

    def address_range(self, ptr):
        for base, size in self.allocations.items():
            if base <= ptr < base + size:
                return base, size
        raise RuntimeError(f"{ptr:#x} is not mapped")

    def export_fd(self, ptr):
        self.address_range(ptr)
        return os.open(os.devnull, os.O_RDONLY)

    def import_fd(self, fd):
        os.fstat(fd)  # a real descriptor
        self.imports += 1
        handle = self.next_handle
        self.next_handle += 1
        self.live_handles.add(handle)
        return handle

    def reserve(self, size, alignment):
        va = self.next_va
        self.next_va += size
        self.reserved[va] = size
        return va

    def free_address(self, va, size):
        assert self.reserved.pop(va) == size

    def map(self, va, size, handle, device):
        assert self.reserved[va] == size and va not in self.mapped
        self.mapped[va] = handle

    def unmap(self, va, size):
        del self.mapped[va]

    def release(self, handle):
        self.live_handles.remove(handle)


@pytest.fixture
def peers(monkeypatch):
    """Rank 0 of two; rank 1 mirrors rank 0's allocations at PEER_SHIFT."""
    peer_view = {"extra": [], "size_scale": 1}

    def mirror(obj):
        if isinstance(obj, tuple):  # (socket address, fd count)
            return obj
        shifted = [(base + PEER_SHIFT, value) for base, value in obj]
        if obj and all(value >= GRANULARITY for _, value in obj):  # (base, size)
            shifted = [(b, v * peer_view["size_scale"]) for b, v in shifted]
            shifted += peer_view["extra"]
        return shifted

    def all_gather_object(group, obj):
        return [obj, mirror(obj)]

    def exchange_fds(group, fds):
        count = len(fds) + len(peer_view["extra"])
        return [[], [os.open(os.devnull, os.O_RDONLY) for _ in range(count)]]

    monkeypatch.setattr(cuda_vmm_ipc, "all_gather_object", all_gather_object)
    monkeypatch.setattr(cuda_vmm_ipc, "exchange_fds", exchange_fds)
    monkeypatch.setattr(cuda_vmm_ipc.dist, "get_rank", lambda group: 0)
    driver = FakeDriver(
        {0x1000_0000_0000: GRANULARITY, 0x1000_0040_0000: 2 * GRANULARITY}
    )
    return PeerAllocations(None, 0, driver), driver, peer_view


B1, B2 = 0x1000_0000_0000, 0x1000_0040_0000


class TestPeerAllocations:
    def test_share_maps_each_peer_allocation_once(self, peers):
        allocations, driver, _ = peers
        ptrs = [B1, B1 + 256, B2 + 64]
        own, peer = allocations.share(ptrs)
        assert [ptr for ptr, _ in own] == ptrs
        assert driver.imports == 2
        va1, va2 = sorted(driver.mapped)
        assert peer == [
            (va1, B1 + PEER_SHIFT),
            (va1 + 256, B1 + PEER_SHIFT),
            (va2 + 64, B2 + PEER_SHIFT),
        ]
        # Sharing an allocation again reuses its mapping.
        allocations.share([B1 + 512])
        assert driver.imports == 2

    def test_reshare_maps_at_the_same_addresses(self, peers):
        allocations, driver, _ = peers
        allocations.share([B1, B2])
        before = dict(driver.mapped)
        reserved = dict(driver.reserved)

        allocations.release()
        assert not driver.mapped and not driver.live_handles
        assert driver.reserved == reserved

        allocations.reshare()
        assert driver.mapped.keys() == before.keys()
        assert set(driver.mapped.values()).isdisjoint(before.values())
        assert driver.reserved == reserved
        assert driver.imports == 4

    @pytest.mark.parametrize(
        "extra,size_scale,match",
        [
            ([(B1 + 2 * PEER_SHIFT, GRANULARITY)], 1, "not mapped before"),
            ([], 2, "changed size"),
        ],
        ids=["new-peer-allocation", "resized-peer-allocation"],
    )
    def test_reshare_rejects_moved_peer_memory(self, peers, extra, size_scale, match):
        allocations, _, peer_view = peers
        allocations.share([B1])
        allocations.release()
        peer_view.update(extra=extra, size_scale=size_scale)
        with pytest.raises(RuntimeError, match=match):
            allocations.reshare()

    def test_share_while_released_is_refused(self, peers):
        allocations, _, _ = peers
        allocations.share([B1])
        allocations.release()
        with pytest.raises(RuntimeError, match="reshare first"):
            allocations.share([B2])

    def test_close_frees_every_reservation(self, peers):
        allocations, driver, _ = peers
        allocations.share([B1, B2])
        allocations.close()
        assert not driver.mapped and not driver.reserved and not driver.live_handles


def _exchange_worker(rank, world_size, init_file, count):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    pipes = [os.pipe() for _ in range(count)]
    received = cuda_vmm_ipc.exchange_fds(dist.group.WORLD, [r for r, _ in pipes])
    for i, (_, write_end) in enumerate(pipes):
        os.write(write_end, f"{rank}:{i};".encode() * (world_size - 1))
        os.close(write_end)
    dist.barrier()
    assert received[rank] == []
    for peer, fds in enumerate(received):
        if peer == rank:
            continue
        assert len(fds) == count
        for i, fd in enumerate(fds):
            # Every receiver reads the same pipe; check only the first tag.
            assert os.read(fd, len(f"{peer}:{i};")).decode() == f"{peer}:{i};"
            os.close(fd)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,count",
    [(2, 3), (3, 0), (3, 2 * cuda_vmm_ipc._FDS_PER_MESSAGE + 5)],
    ids=["two-ranks", "no-descriptors", "several-messages"],
)
def test_exchange_fds_delivers_each_descriptor_in_order(world_size, count):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _exchange_worker,
            args=(world_size, os.path.join(tmp, "init"), count),
            nprocs=world_size,
        )
