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
        model = nn.Linear(2, 2, bias=False)
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


class TestWorkerSleepWake:
    @pytest.mark.parametrize(
        "level,discardable,wake_tags,expected",
        [
            (1, set(), None, ["invalidated", "ready"]),
            (1, set(), ["weights"], ["invalidated", "ready"]),
            (1, set(), ["kv_cache"], ["invalidated"]),
            (2, {1}, None, ["invalidated"]),
            (2, set(), None, ["invalidated", "ready"]),
        ],
        ids=[
            "level1",
            "level1-wake-weights",
            "level1-wake-kv-only",
            "level2-reload-makes-ready",
            "level2-all-preserved",
        ],
    )
    def test_events(
        self, recorder, monkeypatch, level, discardable, wake_tags, expected
    ):
        from vllm.v1.worker.gpu_worker import Worker

        class Backend:
            def suspend(self, discardable=()) -> None:
                assert recorder.events == ["invalidated"]

            def resume(self, tags=None) -> None:
                pass

        worker = object.__new__(Worker)
        worker._sleep_mode_backend = Backend()
        worker.model_runner = SimpleNamespace(extensible_kv_cache=None, model=None)
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(enable_nccl_comm_suspend=False)
        )
        monkeypatch.setattr(
            Worker, "_discardable_parameters", lambda self, model: discardable
        )
        monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
        monkeypatch.setattr("torch.accelerator.get_memory_info", lambda: (0, 0))
        monkeypatch.setattr(Worker, "synchronize_device", lambda self: None)

        worker.sleep(level=level)
        worker.wake_up(tags=wake_tags)

        assert recorder.events == expected
