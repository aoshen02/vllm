# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifetime and checkpoint release of registered pinned host memory, against a
recording cudart and PyTorch host-allocator stats."""

import gc
from types import SimpleNamespace

import pytest
import torch

from vllm.utils import pinned_host_memory as phm


class FakeCudart:
    def __init__(self):
        self.registered: dict[int, int] = {}
        self.calls: list[tuple[str, int]] = []

    def cudaHostRegister(self, pointer, num_bytes, flags):
        assert pointer not in self.registered, "range registered twice"
        self.registered[pointer] = num_bytes
        self.calls.append(("register", pointer))
        return SimpleNamespace(value=0)

    def cudaHostUnregister(self, pointer):
        del self.registered[pointer]
        self.calls.append(("unregister", pointer))
        return SimpleNamespace(value=0)


@pytest.fixture
def cudart(monkeypatch):
    fake = FakeCudart()
    fake.torch_blocks = 0  # PyTorch-pinned blocks still in use after emptyCache
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake)
    monkeypatch.setattr(phm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "host_memory_stats",
        lambda: {"allocations.current": fake.torch_blocks},
    )
    monkeypatch.setattr(phm, "_released", False)
    yield fake
    phm.restore_pinned_host_memory()
    gc.collect()


class TestPinnedHostEmpty:
    @pytest.mark.parametrize(
        "shape,dtype,num_bytes",
        [
            (5000, torch.uint8, 5000),
            ((3, 7), torch.int32, 84),
            ([(1 << 30) + 4096], torch.uint8, (1 << 30) + 4096),
        ],
        ids=["odd-bytes", "2d-int32", "just-over-power-of-two"],
    )
    def test_registers_exact_size_at_tensor_address(
        self, cudart, shape, dtype, num_bytes
    ):
        t = phm.pinned_host_empty(shape, dtype)
        assert t.dtype == dtype and t.numel() * dtype.itemsize == num_bytes
        assert cudart.registered == {t.data_ptr(): num_bytes}
        assert not t.any()

    def test_unregisters_once_last_view_is_gone(self, cudart):
        t = phm.pinned_host_empty(4096, torch.uint8)
        pointer = t.data_ptr()
        view = t[10:20]
        del t
        gc.collect()
        assert pointer in cudart.registered
        view.fill_(1)  # the mapping is still alive
        del view
        gc.collect()
        assert cudart.calls == [("register", pointer), ("unregister", pointer)]

    def test_empty_tensor_registers_nothing(self, cudart):
        assert phm.pinned_host_empty(0, torch.float32).numel() == 0
        assert cudart.calls == []


class TestCheckpointRelease:
    def test_release_then_restore_keeps_addresses_and_data(self, cudart):
        tensors = [phm.pinned_host_empty(n, torch.uint8) for n in (4096, 70000)]
        tensors[1][-1] = 7
        pointers = {t.data_ptr(): t.numel() for t in tensors}
        phm.release_pinned_host_memory()
        assert cudart.registered == {}
        phm.restore_pinned_host_memory()
        assert cudart.registered == pointers
        assert [t.data_ptr() for t in tensors] == list(pointers)
        assert tensors[1][-1] == 7

    def test_live_torch_pinned_blocks_fail_before_any_release(self, cudart):
        t = phm.pinned_host_empty(4096, torch.uint8)
        cudart.torch_blocks = 3
        with pytest.raises(RuntimeError, match="3 pinned blocks"):
            phm.release_pinned_host_memory()
        assert cudart.registered == {t.data_ptr(): 4096}

    def test_no_new_registration_while_released(self, cudart):
        phm.release_pinned_host_memory()
        with pytest.raises(RuntimeError, match="released"):
            phm.pinned_host_empty(4096, torch.uint8)
        assert cudart.registered == {}

    def test_freed_while_released_is_not_reregistered(self, cudart):
        t = phm.pinned_host_empty(4096, torch.uint8)
        phm.release_pinned_host_memory()
        del t
        gc.collect()
        phm.restore_pinned_host_memory()
        assert cudart.registered == {}
