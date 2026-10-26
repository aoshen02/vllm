# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUWorker weight-transfer pass-through behavior.

The worker no longer contains transport, layerwise, or sparse logic: it only
delegates to the configured weight transfer engine and tracks whether an update
session is active. These tests verify that delegation and the session guard.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.model_loader.reload.inplace import get_parameter_for_reload
from vllm.v1.worker.gpu_worker import Worker


class _RecordingEngine:
    """Minimal stand-in for a weight transfer engine."""

    def __init__(self, raise_on_update: bool = False):
        self.raise_on_update = raise_on_update
        self.raise_on_finish = False
        self.started = False
        self.finished = False
        self.closed = False
        self.reset_count = 0
        self.supports_draft_weight_update = False
        self.update_calls: list[dict] = []
        self.seen_configs: list[VllmConfig] = []

    def _record_config(self) -> None:
        self.seen_configs.append(get_current_vllm_config())

    def start_weight_update(self) -> None:
        self._record_config()
        self.started = True

    def update_weights(self, update_info: dict) -> None:
        self._record_config()
        self.update_calls.append(update_info)
        if self.raise_on_update:
            raise ValueError("boom")

    def finish_weight_update(self) -> None:
        self._record_config()
        if self.raise_on_finish:
            raise ValueError("boom")
        self.finished = True

    def reset_weight_update_target(self) -> None:
        self.reset_count += 1

    def shutdown(self) -> None:
        self.closed = True


class _RecordingModelRunner:
    def __init__(self) -> None:
        self.seen_config: VllmConfig | None = None
        self.reset_lora_calls = 0

    def reload_weights(self) -> None:
        self.seen_config = get_current_vllm_config()

    def reset_lora_state(self) -> None:
        self.reset_lora_calls += 1

    def get_draft_model(self) -> nn.Module | None:
        return None


def _make_worker(engine: _RecordingEngine | None) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = VllmConfig()
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.model_runner = _RecordingModelRunner()
    return worker


def test_reload_weights_sets_current_config():
    worker = _make_worker(None)
    model_runner = _RecordingModelRunner()
    worker.model_runner = model_runner

    Worker.reload_weights(worker)

    assert model_runner.seen_config is worker.vllm_config


@pytest.mark.parametrize(
    "enabled,active", [(True, False), (True, True), (False, False)]
)
def test_shutdown_weight_transfer_engine_preserves_inference(enabled, active):
    engine = _RecordingEngine() if enabled else None
    worker = _make_worker(engine)
    worker._weight_update_active = active
    model_runner = worker.model_runner
    if active:
        with pytest.raises(RuntimeError, match="during a weight update"):
            worker.shutdown_weight_transfer_engine()
    else:
        worker.shutdown_weight_transfer_engine()
    if engine is not None:
        assert engine.closed is not active
    assert worker.weight_transfer_engine is engine
    assert worker.model_runner is model_runner


def test_reload_parameter_lookup_preserves_lora_module_names():
    base_layer = nn.Module()
    qweight = nn.Parameter(torch.ones(1))
    base_layer.register_parameter("qweight", qweight)
    wrapper = BaseLayerWithLoRA()
    wrapper.base_layer = base_layer
    model = nn.Module()
    model.proj = wrapper

    named_parameters = dict(model.named_parameters())
    assert set(named_parameters) == {"proj.base_layer.qweight"}
    assert named_parameters["proj.base_layer.qweight"] is qweight
    assert model.get_parameter("proj.base_layer.qweight") is qweight
    assert get_parameter_for_reload(model, "proj.qweight") is qweight


def test_start_update_finish_delegates_to_engine():
    engine = _RecordingEngine()
    worker = _make_worker(engine)

    Worker.start_weight_update(worker)
    assert engine.started is True
    assert worker._weight_update_active is True

    Worker.update_weights(worker, {"names": ["w"]})
    assert engine.update_calls == [{"names": ["w"]}]
    assert worker._weight_update_active is True

    Worker.finish_weight_update(worker)
    assert engine.finished is True
    assert engine.reset_count == 1
    assert worker._weight_update_active is False
    assert engine.seen_configs == [worker.vllm_config] * 3
    assert worker.model_runner.reset_lora_calls == 1


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, {"names": ["rank-1"]}), (2, {"names": []})],
)
def test_rank_local_update_selects_worker_payload(rank, expected):
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = rank
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker, [{"names": ["rank-0"]}, {"names": ["rank-1"]}, {"names": []}]
    )

    assert engine.update_calls == [expected]
    assert worker._weight_update_active is True


def test_rank_local_update_uses_data_parallel_index_after_reconfigure():
    engine = _RecordingEngine()
    worker = _make_worker(engine)
    worker.rank = 0
    parallel_config = ParallelConfig(
        data_parallel_size=4,
        data_parallel_rank=2,
    )
    assert parallel_config.data_parallel_rank == 2
    assert parallel_config.data_parallel_index == 2

    parallel_config.reconfigure_for_independent_dp_rank()
    assert parallel_config.data_parallel_rank == 0
    assert parallel_config.data_parallel_index == 2

    worker.vllm_config.parallel_config = parallel_config
    Worker.start_weight_update(worker)

    Worker.update_weights(
        worker,
        [
            {"names": ["dp-0"]},
            {"names": ["dp-1"]},
            {"names": ["dp-2"]},
            {"names": ["dp-3"]},
        ],
    )

    assert engine.update_calls == [{"names": ["dp-2"]}]
    assert worker._weight_update_active is True


def test_finish_draft_session_keeps_lora_state(monkeypatch):
    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    # Stub the bound method on a hand-built worker: retargeting needs a real model.
    monkeypatch.setattr(worker, "_set_draft_weight_update_target", lambda: None)

    Worker.start_draft_weight_update(worker)
    Worker.finish_weight_update(worker)

    assert isinstance(worker.model_runner, _RecordingModelRunner)
    assert worker.model_runner.reset_lora_calls == 0


def test_double_start_raises():
    worker = _make_worker(_RecordingEngine())
    Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="already"):
        Worker.start_weight_update(worker)


def test_update_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        Worker.update_weights(worker, {"names": ["w"]})


def test_finish_without_start_raises():
    worker = _make_worker(_RecordingEngine())
    with pytest.raises(RuntimeError, match="without a matching"):
        Worker.finish_weight_update(worker)


def test_update_resets_active_on_error():
    engine = _RecordingEngine(raise_on_update=True)
    worker = _make_worker(engine)
    Worker.start_weight_update(worker)

    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})

    # A failed update ends the session so the next start is clean.
    assert engine.reset_count == 1
    assert worker._weight_update_active is False


def test_missing_engine_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        Worker.start_weight_update(worker)


@pytest.mark.parametrize("fail", ["update", "finish"])
def test_failed_update_refuses_to_run_until_an_update_completes(fail):
    # Updates write the weights in place without a rollback (direct reload),
    # so after a failure the model may hold a mix of old and new weights.
    engine = _RecordingEngine(raise_on_update=fail == "update")
    engine.raise_on_finish = fail == "finish"
    worker = _make_worker(engine)
    Worker._check_weights_complete(worker)

    Worker.start_weight_update(worker)
    with pytest.raises(ValueError, match="boom"):
        Worker.update_weights(worker, {"names": ["w"]})
        Worker.finish_weight_update(worker)
    assert worker._weight_update_active is False
    with pytest.raises(RuntimeError, match="refuses to run the model"):
        Worker._check_weights_complete(worker)

    # a retry that completes makes the worker usable again
    engine.raise_on_update = engine.raise_on_finish = False
    Worker.start_weight_update(worker)
    Worker.update_weights(worker, {"names": ["w"]})
    Worker.finish_weight_update(worker)
    Worker._check_weights_complete(worker)


def test_refused_start_leaves_the_model_runnable(monkeypatch):
    # direct reload checks everything that can refuse before modifying the model
    from vllm.model_executor.model_loader.reload.direct import DirectReloadRefused

    engine = _RecordingEngine()
    error: Exception = DirectReloadRefused("direct reload: refused")

    def start():
        raise error

    monkeypatch.setattr(engine, "start_weight_update", start)
    worker = _make_worker(engine)
    with pytest.raises(DirectReloadRefused):
        Worker.start_weight_update(worker)
    assert worker._weight_update_active is False
    Worker._check_weights_complete(worker)

    error = ValueError("boom")  # may have modified the model
    with pytest.raises(ValueError, match="boom"):
        Worker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="refuses to run the model"):
        Worker._check_weights_complete(worker)


def test_failed_draft_head_restore_keeps_the_draft_incomplete(monkeypatch):
    import vllm.v1.worker.gpu_worker as gpu_worker

    engine = _RecordingEngine()
    engine.supports_draft_weight_update = True
    worker = _make_worker(engine)
    draft = nn.Module()
    monkeypatch.setattr(worker.model_runner, "get_draft_model", lambda: draft)
    monkeypatch.setattr(worker, "_set_draft_weight_update_target", lambda: None)
    monkeypatch.setattr(gpu_worker, "share_source_lm_heads", lambda model: None)

    def restore(model, refresh=True):
        if refresh:
            raise torch.OutOfMemoryError("quantized head copy")

    monkeypatch.setattr(gpu_worker, "restore_quantized_lm_heads", restore)
    Worker.start_draft_weight_update(worker)
    with pytest.raises(torch.OutOfMemoryError):
        Worker.finish_weight_update(worker)
    with pytest.raises(RuntimeError, match="draft model did not complete"):
        Worker._check_weights_complete(worker)


@pytest.mark.parametrize(
    "from_disk,checkpoint_format,clears",
    [(True, True, True), (False, True, False), (True, False, False)],
    ids=["from-disk", "iterator", "kernel-format-copy"],
)
def test_reload_from_disk_clears_a_failed_update(from_disk, checkpoint_format, clears):
    # only a checkpoint-format reload from disk runs the reload lifecycle over
    # the whole model; a kernel-format copy skips it
    class Runner(_RecordingModelRunner):
        def reload_weights(self, weights_iterator=None, **_) -> None:
            pass

    worker = _make_worker(None)
    worker.model_runner = Runner()
    worker._incomplete_updates = frozenset({"target"})
    Worker.reload_weights(
        worker,
        None if from_disk else iter([]),
        is_checkpoint_format=checkpoint_format,
    )
    assert worker._incomplete_updates == (set() if clears else {"target"})
