# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which weight updates and sleep/wake transitions notify weight listeners."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.model_loader.reload.inplace as inplace
import vllm.model_executor.model_loader.reload.lifecycle as lifecycle
from vllm.model_executor.model_loader import weight_listeners


class Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def weights_invalidated(self) -> None:
        self.events.append("invalidated")

    def weights_ready(self) -> None:
        self.events.append("ready")


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(weight_listeners, "_listeners", [recorder])
    return recorder


def _vllm_config(quantization=None):
    return SimpleNamespace(
        lora_config=None,
        load_config=SimpleNamespace(load_format="auto", device=None),
        device_config=SimpleNamespace(device="cpu"),
        model_config=SimpleNamespace(quantization=quantization),
    )


class TestWeightUpdates:
    def test_checkpoint_format_update(self, recorder, monkeypatch):
        steps: list[str] = []
        monkeypatch.setattr(lifecycle, "direct_reload_enabled", lambda: True)
        monkeypatch.setattr(
            lifecycle,
            "begin_direct_reload",
            lambda model: steps.append(f"begin after {recorder.events}"),
        )
        monkeypatch.setattr(
            lifecycle,
            "finish_direct_reload",
            lambda *args: steps.append(f"finish after {recorder.events}"),
        )

        lifecycle.start_reload(nn.Module())
        lifecycle.finish_reload(nn.Module(), None, torch.device("cpu"))

        assert steps == ["begin after ['invalidated']", "finish after ['invalidated']"]
        assert recorder.events == ["invalidated", "ready"]

    def test_kernel_format_update(self, recorder):
        model = nn.Linear(2, 2, bias=False).requires_grad_(False)
        new_weight = torch.ones(2, 2)

        inplace.reload_weights(
            _vllm_config(),
            model,
            [("weight", new_weight)],
            weights_path=None,
            is_checkpoint_format=False,
        )

        assert torch.equal(model.weight.data, new_weight)
        assert recorder.events == ["invalidated", "ready"]

    def test_reload_from_loader(self, recorder):
        model = nn.Module()
        vllm_config = _vllm_config()

        class Loader:
            def reload_weights(self, config, reloaded):
                assert (config, reloaded) == (vllm_config, model)
                with weight_listeners.weights_updating():
                    recorder.events.append("written")
                return set()

        inplace.reload_weights(
            vllm_config,
            model,
            None,
            weights_path=None,
            is_checkpoint_format=True,
            model_loader=Loader(),
        )

        assert recorder.events == ["invalidated", "written", "ready"]


class TestSleepWake:
    @pytest.mark.parametrize(
        "discardable,wake_tags,expected",
        [
            (set(), None, ["invalidated", "ready"]),
            (set(), ["weights"], ["invalidated", "ready"]),
            (set(), ["kv_cache"], ["invalidated"]),
            ({1}, None, ["invalidated"]),
        ],
        ids=[
            "weights-preserved",
            "wake-weights",
            "wake-kv-only",
            "weights-discarded-reload-makes-ready",
        ],
    )
    def test_events(self, recorder, monkeypatch, discardable, wake_tags, expected):
        import vllm.device_allocator as device_allocator
        from vllm.device_allocator.sleep_mode_backend import CuMemBackend

        class Allocator:
            def sleep(self, offload_tags, discardable) -> None:
                assert recorder.events == ["invalidated"]

            def wake_up(self, tags) -> None:
                pass

        monkeypatch.setattr(
            device_allocator, "get_mem_allocator_instance", lambda: Allocator()
        )
        backend = CuMemBackend()

        backend.suspend(discardable)
        backend.resume(wake_tags)

        assert recorder.events == expected
