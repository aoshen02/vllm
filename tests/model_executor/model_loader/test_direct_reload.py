# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload (``VLLM_RELOAD_MODE=direct``) on a toy model whose
post-load processing has the hazards found on real models:

(a) a parent derives a separate tensor from a child's weight (MLA W_UK/W_UV
    from a quantized kv_b_proj);
(b) post-load processing changes a weight's layout;
(c) the model's loader writes a weight without its ``weight_loader`` (MiMo);
(d) post-load processing consumes a checkpoint tensor and deletes it (NVFP4
    ``weight_scale_2``);
(e) a tensor the trainer does not sync (Engram), named by
    ``VLLM_DIRECT_RELOAD_EXCLUDE_REGEX``.

Invariant: a reload equals a cold start on the new weights, at the startup
addresses.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    stash_weight_part,
)
from vllm.model_executor.utils import replace_parameter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.bfloat16
)
_DEVICE = torch.device("cuda")


class _Method(QuantizeMethodBase):
    """Post-load processing given as a function of the layer; counts runs."""

    def __init__(self, fn):
        self.fn = fn
        self.runs = 0

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.runs += 1
        self.fn(layer)


def _derive_w_uk(layer):  # (a): a separate tensor derived from the child
    layer.W_UK = (layer.kv_b.weight.float() * layer.k_scale).t().contiguous()


def _transpose(layer):  # (b): kernel layout differs from the checkpoint's
    replace_parameter(layer, "weight", layer.weight.t().contiguous())


def _requantize(layer):  # (c): fused-FP8-like rescale
    scale = layer.weight.float().abs().amax()
    replace_parameter(layer, "weight", (layer.weight / scale).to(layer.weight.dtype))
    replace_parameter(layer, "weight_scale", scale.reshape(1))


def _consume_scale_2(layer):  # (d): NVFP4 weight_scale_2 folded in, then deleted
    ws2 = layer.weight_scale_2
    replace_parameter(layer, "global_scale", (1.0 / ws2).reshape(1))
    replace_parameter(layer, "weight", (layer.weight * ws2).to(layer.weight.dtype))
    del layer.weight_scale_2


def _pack(layer):  # (e): not idempotent, must never run on converted bytes
    replace_parameter(layer, "weight", (layer.weight * 2).t().contiguous())


def _rebind(layer):  # new storage on every run
    layer.weight.data = layer.weight.data + 1


_SHAPES = {
    "attn.kv_b.weight": ((8, 4), torch.bfloat16),
    "proj.weight": ((4, 6), torch.bfloat16),
    "mimo.weight": ((4, 4), torch.bfloat16),
    "nvfp4.weight": ((4, 4), torch.bfloat16),
    "nvfp4.weight_scale_2": ((), torch.float32),
    "engram.weight": ((3, 5), torch.bfloat16),
    "rebind.weight": ((2, 2), torch.float32),
}
_PWAL = {
    "attn.kv_b": None,
    "proj": _transpose,
    "mimo": _requantize,
    "nvfp4": _consume_scale_2,
    "engram": _pack,
    "rebind": _rebind,
}


class _Layer(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.quant_method = _Method(fn) if fn is not None else None


class _ToyModel(torch.nn.Module):
    def __init__(self, rebind: bool = False):
        super().__init__()
        self.attn = _Layer(_derive_w_uk)
        self.attn.register_buffer("k_scale", torch.tensor(0.5))  # not in checkpoint
        for name, (shape, dtype) in _SHAPES.items():
            if name.startswith("rebind.") and not rebind:
                continue
            *path, local = name.split(".")
            parent = self
            for i, part in enumerate(path):
                if not hasattr(parent, part):
                    parent.add_module(part, _Layer(_PWAL[".".join(path[: i + 1])]))
                parent = getattr(parent, part)
            param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), False)
            parent.register_parameter(local, param)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            param = params[name]
            if name.startswith("mimo."):
                param.data.copy_(weight)  # (c): bypasses param.weight_loader
            else:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded


def _checkpoint(scale: float = 1.0, rebind: bool = False) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {
        name: ((torch.rand(shape, generator=gen) + 0.5) * scale).to(dtype)
        for name, (shape, dtype) in _SHAPES.items()
        if rebind or not name.startswith("rebind.")
    }


def _cold_start(weights: dict[str, torch.Tensor], **kwargs) -> _ToyModel:
    """As ``BaseModelLoader.load_model`` does."""
    with _DEVICE:
        model = _ToyModel(**kwargs)
    model.load_weights(weights.items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    return model


def _reload(model: _ToyModel, weights: dict[str, torch.Tensor]) -> None:
    begin_direct_reload(model)
    model.load_weights(weights.items())
    finish_direct_reload(model, _MODEL_CONFIG)


def _state(model: torch.nn.Module) -> dict[str, tuple]:
    """Every parameter, buffer and tensor attribute: layout, bytes, address."""
    tensors: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                tensors[prefix + attr] = value
    tensors.update(model.named_parameters())
    tensors.update(model.named_buffers())
    return {
        name: (
            (tuple(t.shape), t.stride(), t.dtype),
            t.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
            t.data_ptr(),
        )
        for name, t in tensors.items()
    }


def _assert_same(actual: dict, expected: dict, names=None, addresses=False) -> None:
    """Layout and bytes (and addresses) equal, for ``names`` or every tensor."""
    if names is None:
        assert sorted(actual) == sorted(expected)
        names = sorted(expected)
    for name in names:
        (layout, data, ptr), (exp_layout, exp_data, exp_ptr) = (
            actual[name],
            expected[name],
        )
        assert layout == exp_layout, name
        assert torch.equal(data, exp_data), f"{name} differs"
        assert not addresses or ptr == exp_ptr, f"{name} moved"


def _ptrs(state: dict, names=None) -> dict[str, int]:
    return {n: state[n][2] for n in (state if names is None else names)}


def _runs(model: _ToyModel) -> dict[str, int]:
    return {
        name: m.quant_method.runs
        for name, m in model.named_modules()
        if getattr(m, "quant_method", None) is not None
    }


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.delenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", raising=False)
    monkeypatch.delenv("VLLM_DIRECT_RELOAD_ALLOW_MOVED", raising=False)


class TestReloadEqualsColdStart:
    def test_same_checkpoint_is_bit_identical(self):
        model = _cold_start(_checkpoint())
        before = _state(model)
        _reload(model, _checkpoint())
        _assert_same(_state(model), before, addresses=True)
        assert set(_runs(model).values()) == {2}

    def test_new_checkpoint_matches_fresh_cold_start(self):
        model = _cold_start(_checkpoint())
        before = _state(model)
        _reload(model, _checkpoint(1.01))
        after = _state(model)
        _assert_same(after, _state(_cold_start(_checkpoint(1.01))))
        assert _ptrs(after) == _ptrs(before)
        assert "nvfp4.weight_scale_2" not in after  # deleted again, as cold start

    @pytest.mark.parametrize(
        "names",
        [
            ["attn.W_UK"],
            ["proj.weight"],
            ["mimo.weight", "mimo.weight_scale"],
            ["nvfp4.weight", "nvfp4.global_scale"],
            ["engram.weight"],
        ],
        ids=[
            "a-parent-derived",
            "b-layout-change",
            "c-loader-bypass",
            "d-deleted-scale",
            "e-unexcluded-engram",
        ],
    )
    def test_new_checkpoint_hazard(self, names):
        model = _cold_start(_checkpoint())
        before = _state(model)
        _reload(model, _checkpoint(1.01))
        after = _state(model)
        assert any(not torch.equal(after[n][1], before[n][1]) for n in names)
        _assert_same(after, _state(_cold_start(_checkpoint(1.01))), names=names)
        assert _ptrs(after, names) == _ptrs(before, names)


class TestExclusion:
    def test_excluded_module_keeps_kernel_state(self, monkeypatch):
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^engram\.")
        model = _cold_start(_checkpoint())
        before = _state(model)
        new = _checkpoint(1.01)
        _reload(model, {n: t for n, t in new.items() if not n.startswith("engram.")})
        after = _state(model)

        _assert_same(after, before, names=["engram.weight"], addresses=True)
        assert model.engram.quant_method.runs == 1
        fresh = _state(_cold_start(new))
        _assert_same(after, fresh, names=[n for n in fresh if "engram" not in n])
        assert sorted(after) == sorted(fresh)


class TestMovedAddress:
    @pytest.mark.parametrize("allow", [False, True], ids=["raises", "allowed"])
    def test_rebinding_pwal(self, monkeypatch, allow):
        if allow:
            monkeypatch.setenv("VLLM_DIRECT_RELOAD_ALLOW_MOVED", "1")
        model = _cold_start(_checkpoint(rebind=True), rebind=True)
        new = _checkpoint(1.01, rebind=True)
        if not allow:
            with pytest.raises(RuntimeError, match=r"moved: \['rebind.weight'\]"):
                _reload(model, new)
            return
        _reload(model, new)
        fresh = _state(_cold_start(new, rebind=True))
        _assert_same(_state(model), fresh)


class TestAbortedUpdate:
    def test_parked_part_does_not_pair_with_next_update(self):
        """A part parked by an update that never finished is dropped when the
        next update begins, so the next update's other part waits for its own."""
        model = _cold_start(_checkpoint())
        initialize_layerwise_reload(model)  # update N parks a weight, then aborts
        stale = torch.full((2, 2), 7.0, device=_DEVICE)
        assert stash_weight_part(model.attn, "wk", "weight", stale) is None

        initialize_layerwise_reload(model)  # update N + 1
        scale = torch.ones(1, device=_DEVICE)
        assert stash_weight_part(model.attn, "wk", "scale", scale) is None
        fresh = torch.full((2, 2), 3.0, device=_DEVICE)
        entry = stash_weight_part(model.attn, "wk", "weight", fresh)
        assert entry is not None and torch.equal(entry["weight"], fresh)
