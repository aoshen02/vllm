# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Suspendable symmetric buffers: collective allocation without leaks."""

from types import SimpleNamespace

import pytest

from vllm.distributed.device_communicators import symmetric_buffer as sb

GRANULARITY = 0x20_0000
MULTICAST_GRANULARITY = 4 * GRANULARITY
OWN, PEER, MULTICAST = 0x1000_0000, 0x2000_0000, 0x3000_0000


class Parts:
    """Fake parts of a VmmSymmetricBuffer; ``fail`` names one that raises
    on this rank, ``peer_fails`` makes the peer report an allocation failure."""

    def __init__(self, monkeypatch, fail=None, peer_fails=False):
        self.fail, self.peer_fails = fail, peer_fails
        self.open: list[str] = []
        self.granularity = "unset"
        parts = self

        class Buffer:
            def __init__(self, size, device, driver, granularity=None):
                parts.granularity = granularity
                parts.check("buffer")
                self.ptr, self.size, self.handle = OWN, size, 1
                parts.open.append("buffer")

            def close(self):
                parts.open.remove("buffer")

        class Peers:
            def __init__(self, group, device, driver):
                parts.check("peers")
                parts.open.append("peers")

            def share(self, ptrs):
                return [[(ptrs[0], ptrs[0])], [(PEER, 0x5000_0000)]]

            def close(self):
                parts.open.remove("peers")

        class Multicast:
            def __init__(self, group, buffer, device, driver):
                parts.check("multicast")
                self.ptr = MULTICAST
                parts.open.append("multicast")

            def close(self):
                parts.open.remove("multicast")

        driver = SimpleNamespace(
            granularity=lambda device: GRANULARITY,
            multicast_granularity=lambda n: MULTICAST_GRANULARITY,
        )
        monkeypatch.setattr(sb, "CuMemDriver", lambda: driver)
        monkeypatch.setattr(sb, "VmmBuffer", Buffer)
        monkeypatch.setattr(sb, "PeerAllocations", Peers)
        monkeypatch.setattr(sb, "MulticastMapping", Multicast)
        monkeypatch.setattr(
            sb, "all_ranks_ok", lambda group, ok: ok and not self.peer_fails
        )
        monkeypatch.setattr(sb.dist, "get_world_size", lambda group: 2)
        monkeypatch.setattr(sb.torch, "as_tensor", lambda data, device: data)

    def check(self, name):
        if name == self.fail:
            raise RuntimeError(f"{name} failed")


class TestVmmSymmetricBuffer:
    @pytest.mark.parametrize(
        "peers,multicast,granularity,peer_ptrs,multicast_ptr",
        [
            (True, True, MULTICAST_GRANULARITY, [OWN, PEER], MULTICAST),
            (False, True, MULTICAST_GRANULARITY, [OWN], MULTICAST),
            (True, False, None, [OWN, PEER], 0),
        ],
        ids=["peers+multicast", "multicast-only", "peers-only"],
    )
    def test_maps_only_what_is_asked(
        self, monkeypatch, peers, multicast, granularity, peer_ptrs, multicast_ptr
    ):
        parts = Parts(monkeypatch)
        buffer = sb.VmmSymmetricBuffer(
            100, 0, None, peers=peers, multicast=multicast
        )
        # A buffer bound to a multicast object has its granularity.
        assert parts.granularity == granularity
        assert buffer.peer_ptrs == peer_ptrs
        assert buffer.multicast_ptr == multicast_ptr
        buffer.close()
        assert parts.open == []

    @pytest.mark.parametrize(
        "fail,peer_fails",
        [
            ("buffer", False),
            (None, True),
            ("peers", False),
            ("multicast", False),
        ],
        ids=["allocation", "peer-allocation", "peer-mapping", "multicast"],
    )
    def test_a_failure_leaves_nothing_behind(self, monkeypatch, fail, peer_fails):
        parts = Parts(monkeypatch, fail=fail, peer_fails=peer_fails)
        with pytest.raises(RuntimeError):
            sb.VmmSymmetricBuffer(100, 0, None, peers=True, multicast=True)
        assert parts.open == []
