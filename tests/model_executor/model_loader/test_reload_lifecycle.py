# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`weight_update_active` tells a model's `load_weights` whether it runs in
a weight update, where one call may hold only part of the checkpoint, or in a
cold start, where completeness checks must run."""

import gc
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.model_loader.reload import (
    abort_reload,
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
    weight_update_active,
)
from vllm.model_executor.model_loader.reload.layerwise import max_layer_reload_bytes
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.float32
)
_DEVICE = torch.device("cpu")


class _TinyModel(torch.nn.Module):
    """Records `weight_update_active` at every weight it loads."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.observed: list[bool] = []

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, weight in weights:
            self.observed.append(weight_update_active())
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
        return set(params)


def _checkpoint(scale: float = 1.0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {
        "embed.weight": torch.rand(8, 4, generator=gen) * scale,
        "proj.weight": torch.rand(4, 4, generator=gen) * scale,
    }


@pytest.mark.parametrize("mode", ["direct", "layerwise"], ids=["direct", "layerwise"])
def test_active_only_between_start_and_finish(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _TinyModel()
    record_metadata_for_reloading(model)  # as the loader does at cold start
    model.load_weights(_checkpoint().items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    assert model.observed == [False, False]  # cold start keeps its checks
    assert not weight_update_active()

    model.observed.clear()
    start_reload(model)
    assert weight_update_active()
    model.load_weights(_checkpoint(1.01).items())
    assert model.observed == [True, True]
    finish_reload(model, _MODEL_CONFIG, _DEVICE)
    assert not weight_update_active()

    torch.testing.assert_close(model.proj.weight, _checkpoint(1.01)["proj.weight"])


class _Outer(torch.nn.Module):
    """A wrapper like Step5ForConditionalGeneration around its language model."""

    def __init__(self):
        super().__init__()
        self.language_model = _TinyModel()

    def load_weights(self, weights):
        return self.language_model.load_weights(
            (name.removeprefix("language_model."), w) for name, w in weights
        )


@pytest.mark.parametrize("mode", ["direct", "layerwise"], ids=["direct", "layerwise"])
def test_nested_module_sees_the_update(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _Outer()
    record_metadata_for_reloading(model)
    model.load_weights(
        (f"language_model.{name}", w) for name, w in _checkpoint().items()
    )
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    inner = model.language_model
    inner.observed.clear()

    start_reload(model)
    # one weight per call, as a bucketed update sends them
    for name, w in _checkpoint(1.01).items():
        model.load_weights([(f"language_model.{name}", w)])
    finish_reload(model, _MODEL_CONFIG, _DEVICE)

    assert inner.observed == [True, True]
    assert not weight_update_active()


class _CPUOffloadedEngram(torch.nn.Module):
    """A host table whose loader and post-load hook must keep its storage."""

    _skip_layerwise_reload = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 4), requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(
            torch.zeros(8, 1), requires_grad=False
        )
        self.loads = 0
        self.finalized: list[dict[str, torch.Tensor]] = []
        for param in self.parameters():
            param.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        self.loads += 1
        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self):
        self.finalized.append(
            {name: param.clone() for name, param in self.named_parameters()}
        )


@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_cpu_offloaded_table_keeps_storage_and_loader_across_updates(monkeypatch, mode):
    """A -> B -> A updates load host weights once and still finalize every layer."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    table = _CPUOffloadedEngram()
    linear = torch.nn.Linear(4, 4, bias=False)
    model = torch.nn.Sequential(table, linear)
    record_metadata_for_reloading(model)
    assert max_layer_reload_bytes(table) == 0
    assert max_layer_reload_bytes(model) == linear.weight.nbytes
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    table.finalized.clear()
    originals = {
        name: (param, param.data_ptr(), param.weight_loader)
        for name, param in table.named_parameters()
    }
    linear_weight = linear.weight

    for update, value in enumerate((1, 2, 1), start=1):
        start_reload(model)
        if mode == "layerwise":
            assert linear.weight.is_meta
            assert linear.weight.weight_loader.__name__ == "online_process_loader"

        expected = {}
        for name, param in table.named_parameters():
            assert param.device.type == "cpu"
            assert param.weight_loader is originals[name][2]
            expected[name] = torch.full_like(param, value)
            calls_before = table.loads
            param.weight_loader(param, expected[name])
            assert table.loads == calls_before + 1
            assert torch.equal(param, expected[name])
        new_linear_weight = torch.full_like(linear_weight, value)
        getattr(linear.weight, "weight_loader", default_weight_loader)(
            linear.weight, new_linear_weight
        )
        finish_reload(model, _MODEL_CONFIG, _DEVICE)

        assert table.loads == update * len(originals)
        assert len(table.finalized) == update
        for name, param in table.named_parameters():
            original, pointer, loader = originals[name]
            assert param is original
            assert param.data_ptr() == pointer
            assert param.weight_loader is loader
            assert torch.equal(param, expected[name])
            assert torch.equal(table.finalized[-1][name], expected[name])
        assert linear.weight is linear_weight
        assert torch.equal(linear.weight, new_linear_weight)


@pytest.mark.parametrize("mode", ["direct", "layerwise"], ids=["direct", "layerwise"])
def test_abort_reload_clears_active_state_and_allows_retry(monkeypatch, mode):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _TinyModel()
    record_metadata_for_reloading(model)
    model.load_weights(_checkpoint().items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)

    start_reload(model)
    assert weight_update_active()
    abort_reload(model)
    assert not weight_update_active()

    if mode == "direct":
        with pytest.raises(RuntimeError, match="finish without a begin"):
            finish_reload(model, _MODEL_CONFIG, _DEVICE)

    # A fresh update can start and complete cleanly after abort.
    start_reload(model)
    assert weight_update_active()
    model.load_weights(_checkpoint(1.02).items())
    finish_reload(model, _MODEL_CONFIG, _DEVICE)
    assert not weight_update_active()
    torch.testing.assert_close(model.proj.weight, _checkpoint(1.02)["proj.weight"])


@pytest.mark.parametrize("mode", ["direct", "layerwise"], ids=["direct", "layerwise"])
def test_inplace_reload_clears_active_on_load_error(monkeypatch, mode):
    from vllm.model_executor.model_loader.reload.inplace import (
        reload_weights_from_iterator,
    )

    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _TinyModel()
    record_metadata_for_reloading(model)
    model.load_weights(_checkpoint().items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)

    def failing_weights():
        yield "embed.weight", torch.ones(8, 4)
        raise RuntimeError("simulated mid-transfer failure")

    vllm_config = SimpleNamespace(
        model_config=_MODEL_CONFIG,
        load_config=SimpleNamespace(device="cpu"),
        device_config=SimpleNamespace(device="cpu"),
        lora_config=None,
    )
    with pytest.raises(RuntimeError, match="simulated mid-transfer failure"):
        reload_weights_from_iterator(
            vllm_config, model, failing_weights(), is_checkpoint_format=True
        )
    assert not weight_update_active()


def test_unfinished_model_gc_clears_active_state(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model = _TinyModel()
    record_metadata_for_reloading(model)
    model.load_weights(_checkpoint().items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)

    start_reload(model)
    assert weight_update_active()
    del model
    gc.collect()
    assert not weight_update_active()


def test_multiple_models_track_active_independently(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    m1 = _TinyModel()
    m2 = _TinyModel()
    for m in (m1, m2):
        record_metadata_for_reloading(m)
        m.load_weights(_checkpoint().items())
        process_weights_after_loading(m, _MODEL_CONFIG, _DEVICE)

    start_reload(m1)
    start_reload(m2)
    assert weight_update_active()

    m1.load_weights(_checkpoint(1.01).items())
    finish_reload(m1, _MODEL_CONFIG, _DEVICE)
    # m2 is still mid-update.
    assert weight_update_active()

    abort_reload(m2)
    assert not weight_update_active()


def test_layerwise_abort_restores_params_and_retry_does_not_corrupt(monkeypatch):
    """Aborting a layerwise reload mid-layer must restore live tensors off meta,
    clear buffered state, and allow a subsequent retry to load every parameter."""
    from vllm.model_executor.model_loader.reload.layerwise import (
        LAYERWISE_INFO,
        LOADING_LAYERS,
    )

    class _TwoParamLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.full((4, 4), 1.0), requires_grad=False)
            self.w2 = torch.nn.Parameter(torch.full((4, 4), 2.0), requires_grad=False)

    class _TwoParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _TwoParamLayer()

        def load_weights(self, weights):
            params = dict(self.named_parameters())
            for name, weight in weights:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            return set(params)

    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    model = _TwoParamModel()
    record_metadata_for_reloading(model)
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)

    # Start a layerwise update, partially load w1, then abort before w2 arrives.
    start_reload(model)
    assert model.layer.w1.is_meta and model.layer.w2.is_meta
    model.load_weights([("layer.w1", torch.full((4, 4), 9.0))])
    info = LAYERWISE_INFO[model.layer]
    assert info.can_load()
    assert len(info.loaded_weights) == 1

    abort_reload(model)
    assert not weight_update_active()
    assert not model.layer.w1.is_meta and not model.layer.w2.is_meta
    assert not info.can_load()
    assert info.kernel_tensors is None
    assert info.loaded_weights == []
    assert model.layer not in LOADING_LAYERS
    # Original live weights are intact after abort.
    assert torch.equal(model.layer.w1, torch.full((4, 4), 1.0))
    assert torch.equal(model.layer.w2, torch.full((4, 4), 2.0))

    # Retrying the full update must load both w1 and w2 cleanly (no double-count).
    start_reload(model)
    model.load_weights(
        [
            ("layer.w1", torch.full((4, 4), 5.0)),
            ("layer.w2", torch.full((4, 4), 6.0)),
        ]
    )
    finish_reload(model, _MODEL_CONFIG, _DEVICE)
    assert not weight_update_active()
    assert torch.equal(model.layer.w1, torch.full((4, 4), 5.0))
    assert torch.equal(model.layer.w2, torch.full((4, 4), 6.0))
