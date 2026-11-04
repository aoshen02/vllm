# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer cuMem mappings and multicast objects keep their addresses across
release and reshare (rebind)."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators import cuda_vmm_ipc
from vllm.distributed.device_communicators.cuda_vmm_ipc import (
    MulticastMapping,
    PeerAllocations,
)

# The simulated peer's allocations sit at this distance from this rank's.
PEER_SHIFT = 0x1000_0000
GRANULARITY = 0x20_0000


class FakeDriver:
    """Records cuMem calls; this rank owns ``allocations`` (base -> size).

    Multicast calls log to ``events``; a call named in ``fail`` raises.
    """

    def __init__(self, allocations: dict[int, int]):
        self.allocations = allocations
        self.next_va = 0x7000_0000_0000
        self.next_handle = 1
        self.reserved: dict[int, int] = {}
        self.mapped: dict[int, int] = {}  # va -> handle
        self.live_handles: set[int] = set()
        self.imports = 0
        self.bound: dict[int, int] = {}  # multicast handle -> memory handle
        self.devices: dict[int, set[int]] = {}  # multicast handle -> devices
        self.events: list[str] = []
        self.fail: set[str] = set()

    def _call(self, name):
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        self.events.append(name)

    def _new_handle(self):
        handle = self.next_handle
        self.next_handle += 1
        self.live_handles.add(handle)
        return handle

    def multicast_granularity(self, num_devices):
        return GRANULARITY

    def multicast_create(self, size, num_devices):
        self._call("create")
        return self._new_handle()

    def export_handle_fd(self, handle):
        assert handle in self.live_handles
        self._call("export")
        return os.open(os.devnull, os.O_RDONLY)

    def multicast_add_device(self, handle, device):
        self._call("add_device")
        self.devices.setdefault(handle, set()).add(device)

    def multicast_bind(self, handle, memory, size):
        assert self.devices.get(handle), "bound before adding the device"
        self._call("bind")
        self.bound[handle] = memory

    def multicast_unbind(self, handle, device, size):
        self._call("unbind")
        del self.bound[handle]

    def synchronize(self):
        self._call("synchronize")

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
        self._call("import")
        self.imports += 1
        return self._new_handle()

    def reserve(self, size, alignment):
        va = self.next_va
        self.next_va += size
        self.reserved[va] = size
        return va

    def free_address(self, va, size):
        assert self.reserved.pop(va) == size

    def map(self, va, size, handle, device):
        assert self.reserved[va] == size and va not in self.mapped
        self._call("map")
        self.mapped[va] = handle

    def unmap(self, va, size):
        self.events.append("unmap")
        del self.mapped[va]

    def release(self, handle):
        self.events.append("release")
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


# The agreements MulticastMapping.rebind makes, in order.
REBIND_STEPS = ["create", "import", "add_device", "bind", "map"]


class TwoRanks:
    """This rank and one peer: the peer agrees unless ``peer_fails`` names the
    step, and ``barriers`` counts agreements outside rebind."""

    def __init__(self, monkeypatch, rank):
        self.rank = rank
        self.peer_fails: str | None = None
        self.steps = iter(())
        self.received_fds: list[int] = []
        monkeypatch.setattr(cuda_vmm_ipc, "all_gather_object", self.all_gather)
        monkeypatch.setattr(cuda_vmm_ipc, "exchange_fds", self.exchange_fds)
        monkeypatch.setattr(cuda_vmm_ipc.dist, "get_rank", lambda group: rank)
        monkeypatch.setattr(cuda_vmm_ipc.dist, "get_world_size", lambda group: 2)

    def rebinding(self):
        self.steps = iter(REBIND_STEPS)

    def all_gather(self, group, ok):
        step = next(self.steps, "barrier")
        if step == "barrier":
            self.driver.events.append("barrier")
        peer = step != self.peer_fails
        return [ok, peer] if self.rank == 0 else [peer, ok]

    def exchange_fds(self, group, fds):
        assert len(fds) == (1 if self.rank == 0 else 0)
        if self.rank == 0:
            return [[], []]
        fd = os.open(os.devnull, os.O_RDONLY)
        self.received_fds.append(fd)
        return [[fd], []]


BUFFER_HANDLE = 0xB0FF


@pytest.fixture(params=[0, 1], ids=["rank0-creates", "rank1-imports"])
def ranks(request, monkeypatch):
    ranks = TwoRanks(monkeypatch, request.param)
    ranks.driver = FakeDriver({})
    ranks.driver.live_handles.add(BUFFER_HANDLE)
    return ranks


def _mapping(ranks, size=2 * GRANULARITY):
    buffer = SimpleNamespace(size=size, handle=BUFFER_HANDLE)
    ranks.rebinding()
    return MulticastMapping(None, buffer, 0, ranks.driver)


def _assert_released(driver, mapping):
    """Only the buffer is live and nothing is bound or mapped."""
    assert driver.live_handles == {BUFFER_HANDLE}
    assert not driver.bound and mapping.ptr not in driver.mapped
    assert not mapping.bound


class TestMulticastMapping:
    def test_bind_maps_this_ranks_buffer(self, ranks):
        mapping = _mapping(ranks)
        driver = ranks.driver
        handle = driver.mapped[mapping.ptr]
        assert driver.bound == {handle: BUFFER_HANDLE}
        assert driver.devices[handle] == {0}
        assert driver.live_handles == {BUFFER_HANDLE, handle}
        # Rank 0 creates and exports the object; the other rank imports it.
        made = ["create", "export"] if ranks.rank == 0 else ["import"]
        assert driver.events == [*made, "add_device", "bind", "map"]
        for fd in ranks.received_fds:
            with pytest.raises(OSError):
                os.fstat(fd)

    def test_release_and_rebind_keep_the_address(self, ranks):
        mapping = _mapping(ranks)
        driver = ranks.driver
        ptr, old = mapping.ptr, driver.mapped[mapping.ptr]
        reserved = dict(driver.reserved)

        driver.events.clear()
        mapping.release()
        # Every rank is idle before any unbinds, and has released the object
        # before release returns.
        assert driver.events == [
            "synchronize",
            "barrier",
            "unbind",
            "unmap",
            "release",
            "barrier",
        ]
        _assert_released(driver, mapping)
        assert driver.reserved == reserved
        mapping.release()  # idempotent

        ranks.rebinding()
        mapping.rebind()
        new = driver.mapped[ptr]
        assert mapping.ptr == ptr and new != old
        assert driver.bound == {new: BUFFER_HANDLE}
        assert driver.reserved == reserved

    @pytest.mark.parametrize("step", REBIND_STEPS)
    @pytest.mark.parametrize("failing", ["this-rank", "peer"])
    def test_a_failed_step_fails_and_undoes_every_rank(self, ranks, step, failing):
        driver = ranks.driver
        if failing == "peer":
            ranks.peer_fails = step
        elif (step, ranks.rank) in {("create", 1), ("import", 0)}:
            pytest.skip(f"rank {ranks.rank} makes no {step} call")
        else:
            driver.fail.add(step)
        with pytest.raises(RuntimeError):
            _mapping(ranks)
        assert driver.live_handles == {BUFFER_HANDLE}
        assert not driver.bound and not driver.mapped
        # A failed construction frees its reservation.
        assert not driver.reserved

    def test_failed_rebind_keeps_the_reservation(self, ranks):
        mapping = _mapping(ranks)
        driver = ranks.driver
        ptr = mapping.ptr
        mapping.release()
        ranks.peer_fails = "map"
        ranks.rebinding()
        with pytest.raises(RuntimeError, match="map"):
            mapping.rebind()
        _assert_released(driver, mapping)
        assert ptr in driver.reserved
        ranks.peer_fails = None
        ranks.rebinding()
        mapping.rebind()
        assert ptr in driver.mapped

    def test_size_must_fit_the_multicast_granularity(self, ranks):
        with pytest.raises(ValueError, match="granularity"):
            _mapping(ranks, size=GRANULARITY + 4096)
        assert not ranks.driver.reserved

    def test_close_frees_without_synchronizing(self, ranks):
        mapping = _mapping(ranks)
        ranks.driver.events.clear()
        mapping.close()
        assert "barrier" not in ranks.driver.events
        assert ranks.driver.live_handles == {BUFFER_HANDLE}
        assert not ranks.driver.reserved and not ranks.driver.bound
