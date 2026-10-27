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
import regex as re
import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import initialize_layerwise_reload
from vllm.model_executor.model_loader.reload.direct import (
    DirectReloadRefused,
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.reload.side_state import _MAX_DEPTH
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    set_cached_tensor,
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
    w_uk = (layer.kv_b.weight.float() * layer.k_scale).t().contiguous()
    set_cached_tensor(layer, "W_UK", w_uk)


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
    @pytest.mark.parametrize(
        "module", ["engram", "nvfp4"], ids=["engram", "dropped-tensor"]
    )
    def test_excluded_module_keeps_kernel_state(self, monkeypatch, module):
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", rf"^{module}\.")
        model = _cold_start(_checkpoint())
        before = _state(model)
        new = _checkpoint(1.01)
        _reload(model, {n: t for n, t in new.items() if not n.startswith(module)})
        after = _state(model)

        mine = [n for n in before if n.startswith(f"{module}.")]
        _assert_same(after, before, names=mine, addresses=True)
        assert getattr(model, module).quant_method.runs == 1
        fresh = _state(_cold_start(new))
        _assert_same(after, fresh, names=[n for n in fresh if n not in mine])
        assert sorted(after) == sorted(fresh)

    def test_write_into_excluded_module_raises(self, monkeypatch):
        # e.g. a regex naming the fused qkv_proj, while the update delivers
        # the checkpoint's q/k/v_proj, which the regex does not match
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^engram\.")
        model = _cold_start(_checkpoint())
        begin_direct_reload(model)
        param = model.engram.weight
        with pytest.raises(ValueError, match=r"writes engram.weight, which"):
            param.weight_loader(param, _checkpoint()["engram.weight"])

    def test_delivered_excluded_name_is_dropped_then_refused(self, monkeypatch):
        # an engine has received it (raising there strands the trainer)
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^engram\.")
        model = _cold_start(_checkpoint())
        before = _state(model)
        with pytest.raises(ValueError, match=r"not written: \['engram.weight'\]"):
            _reload(model, _checkpoint(1.01))
        assert model.engram.quant_method.runs == 1
        _assert_same(_state(model), before, names=["engram.weight"], addresses=True)

    @pytest.mark.parametrize(
        "regex,module",
        [
            (r"^nvfp4\.weight_scale_2$", "nvfp4"),
            (r"^attn\.k_scale$", "attn"),
            (r"^attn\.kv_b\.", "attn"),
        ],
        ids=[
            "own-tensors-partly",  # rerun would convert the kept scale again
            "children-synced",  # skipping attn would leave W_UK stale
            "quantized-parent",  # attn's processing read kv_b's ckpt layout
        ],
    )
    def test_partial_exclusion_raises(self, monkeypatch, regex, module):
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", regex)
        model = _cold_start(_checkpoint())
        with pytest.raises(RuntimeError, match=rf"reads them: \['{module}'\]"):
            begin_direct_reload(model)


def _with_weight(module: torch.nn.Module, shape=(2, 2)) -> torch.nn.Module:
    module.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(shape), requires_grad=False)
    )
    return module


class _Attn(torch.nn.Module):
    """A deferred attention layer without a quant_method (unquantized MLA):
    derives W from its children."""

    def __init__(self):
        super().__init__()
        self.kv_b = _with_weight(_Layer(lambda layer: None))
        self.q = _with_weight(_Layer(lambda layer: None))

    def process_weights_after_loading(self, dtype):
        set_cached_tensor(self, "W", self.kv_b.weight * 3 + self.q.weight)


class _Tree(torch.nn.Module):
    """Modules given by dotted name; ``hook`` is a model-level hook."""

    def __init__(self, modules: dict[str, torch.nn.Module], hook=None):
        super().__init__()
        for name, module in modules.items():
            *path, local = name.split(".")
            parent = self.get_submodule(".".join(path)) if path else self
            parent.add_module(local, module)
        self.hook = hook

    def process_weights_after_loading(self):
        if self.hook is not None:
            self.hook(self)

    def load_weights(self, weights):
        for name, weight in weights:
            default_weight_loader(self.get_parameter(name), weight)
        return {name for name, _ in weights}


def _tree_cold_start(build, weights) -> "_Tree":
    with _DEVICE:
        model = build()
    model.load_weights(weights.items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    return model


class TestExclusionRule:
    """Canonical names, whole subtrees, and no post-load processing that
    reads a skipped module."""

    @pytest.fixture
    def deferred(self, monkeypatch):
        import vllm.model_executor.layers.attention as attention
        import vllm.model_executor.model_loader.utils as loader_utils

        real = attention.is_deferred_attention_layer

        def is_deferred(module):
            return isinstance(module, _Attn) or real(module)

        monkeypatch.setattr(attention, "is_deferred_attention_layer", is_deferred)
        monkeypatch.setattr(loader_utils, "is_deferred_attention_layer", is_deferred)

    def test_deferred_attention_over_skipped_module_raises(self, monkeypatch, deferred):
        # MLA absorbing an excluded kv_b_proj
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^attn\.kv_b\.")
        model = _tree_cold_start(
            lambda: _Tree({"attn": _Attn()}),
            {"attn.kv_b.weight": torch.rand(2, 2), "attn.q.weight": torch.rand(2, 2)},
        )
        with pytest.raises(RuntimeError, match=r"reads them: \['attn'\]"):
            begin_direct_reload(model)

    def test_alias_is_matched_on_its_canonical_name(self, monkeypatch):
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^sub\.alias\.")

        def build():
            proj = _with_weight(_Layer(lambda layer: None))
            return _Tree({"proj": proj, "sub": _Layer(None), "sub.alias": proj})

        model = _tree_cold_start(build, {"proj.weight": torch.rand(2, 2)})
        new = torch.rand(2, 2)
        _reload(model, {"proj.weight": new})
        assert model.proj.quant_method.runs == 2
        assert torch.equal(model.proj.weight, new.to(_DEVICE))

    def test_excluding_a_tied_head_refuses(self, monkeypatch):
        # the embedding's update writes the tensor the excluded head shares
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^lm_head\.")

        def build():
            embed, head = _with_weight(_Layer(None)), _Layer(None)
            head.weight = embed.weight
            return _Tree({"embed": embed, "lm_head": head})

        model = _tree_cold_start(build, {"embed.weight": torch.rand(2, 2)})
        before = _state(model)
        with pytest.raises(DirectReloadRefused, match=r"shares.*\['lm_head.weight'"):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)

    def test_disk_reload_leaves_excluded_modules_untouched(self, monkeypatch):
        # the documented repair of a failed update: the checkpoint on disk
        # holds the excluded tensors, which the reload must not write
        import vllm.model_executor.model_loader.reload.inplace as inplace

        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^engram\.")
        model = _cold_start(_checkpoint())
        before = _state(model)
        new = _checkpoint(1.01)
        loader = SimpleNamespace(get_all_weights=lambda *_: iter(new.items()))
        monkeypatch.setattr(inplace, "get_model_loader", lambda _: loader)
        config = SimpleNamespace(
            model_config=_MODEL_CONFIG, load_config=None, lora_config=None
        )
        inplace.reload_weights(config, model, None, None, True)
        after = _state(model)
        _assert_same(after, before, names=["engram.weight"], addresses=True)
        assert model.engram.quant_method.runs == 1
        fresh = _state(_cold_start(new))
        _assert_same(after, fresh, names=[n for n in fresh if n != "engram.weight"])

    def test_skipped_module_skips_its_descendants(self, monkeypatch):
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^engram\.")
        model = _tree_cold_start(
            lambda: _Tree(
                {
                    "engram": _with_weight(_Layer(None)),
                    "engram.sub": _Layer(lambda layer: None),  # no tensors
                    "other": _with_weight(_Layer(None)),
                }
            ),
            {"engram.weight": torch.rand(2, 2), "other.weight": torch.rand(2, 2)},
        )
        _reload(model, {"other.weight": torch.rand(2, 2)})
        assert model.engram.sub.quant_method.runs == 1

    def test_engram_like_model(self, monkeypatch):
        """DeepSeek-V4.1's shape under VLLM_DIRECT_RELOAD_EXCLUDE_REGEX=
        '[.]engram[.]': a decoder layer (no post-load processing of its own)
        holds the Engram module beside quantized layers; the model keeps
        a batch object listing the Engram modules outside the module tree;
        the model-level hook reads a synced weight and rebuilds a cache on
        the Engram module."""

        def hook(model):
            engram = model.layers[0].engram  # in place, as the model does
            set_cached_tensor(model, "broadcast", model.layers[0].attn.weight.sum(0))
            set_cached_tensor(engram, "cache", engram.weight * 2)

        def build():
            model = _Tree(
                {
                    "layers": torch.nn.ModuleList([torch.nn.Module()]),
                    "layers.0.attn": _with_weight(_Layer(_transpose)),
                    "layers.0.engram": _with_weight(_Layer(None)),
                    "layers.0.engram.proj": _with_weight(_Layer(_transpose)),
                },
                hook,
            )
            model.batch = SimpleNamespace(layers=[model.layers[0].engram])
            return model

        weights = {
            "layers.0.attn.weight": torch.rand(2, 2),
            "layers.0.engram.weight": torch.rand(2, 2),
            "layers.0.engram.proj.weight": torch.rand(2, 2),
        }
        monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"[.]engram[.]")
        model = _tree_cold_start(build, weights)
        before = _state(model)
        new = dict(weights, **{"layers.0.attn.weight": torch.rand(2, 2)})
        _reload(model, {"layers.0.attn.weight": new["layers.0.attn.weight"]})
        after = _state(model)
        monkeypatch.delenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX")
        _assert_same(after, _state(_tree_cold_start(build, new)))
        assert _ptrs(after) == _ptrs(before)
        assert model.layers[0].engram.proj.quant_method.runs == 1


class TestCompleteness:
    """The reload lifecycle checks that an update delivered every checkpoint
    name cold start fed to ``load_weights``, whoever feeds it: an engine, or
    ``reload_weights`` with an iterator."""

    @staticmethod
    def _cold_start() -> _ToyModel:
        from vllm.model_executor.model_loader.base_loader import (
            record_checkpoint_weight_names,
        )

        with _DEVICE:
            model = _ToyModel()
        with record_checkpoint_weight_names(model):
            model.load_weights(_checkpoint().items())
        process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
        return model

    def test_partial_iterator_reload_raises_before_post_load(self):
        from vllm.model_executor.model_loader.reload.inplace import reload_weights

        model = self._cold_start()
        before, runs = _state(model), _runs(model)
        new = _checkpoint(1.01)
        config = SimpleNamespace(
            model_config=_MODEL_CONFIG, load_config=None, lora_config=None
        )
        partial = (item for item in new.items() if item[0] != "proj.weight")
        with pytest.raises(ValueError, match=r"did not deliver 1 .*'proj.weight'"):
            reload_weights(config, model, partial, None, True)
        assert _runs(model) == runs  # nothing converted again
        reload_weights(config, model, iter(new.items()), None, True)
        after = _state(model)
        _assert_same(after, _state(_cold_start(new)))
        assert _ptrs(after) == _ptrs(before)


class TestMovedAddress:
    def test_rebinding_pwal_raises(self):
        model = _cold_start(_checkpoint(rebind=True), rebind=True)
        with pytest.raises(RuntimeError, match=r"moved: \['rebind.weight'\]"):
            _reload(model, _checkpoint(1.01, rebind=True))


class _OneLayer(torch.nn.Module):
    """One layer ``s`` with a (4, 4) weight and the given post-load
    processing; ``hook`` is an optional model-level post-load hook."""

    def __init__(self, fn, hook=None):
        super().__init__()
        self.s = _Layer(fn)
        self.s.register_parameter(
            "weight", torch.nn.Parameter(torch.empty(4, 4), requires_grad=False)
        )
        self.hook = hook

    def process_weights_after_loading(self):
        if self.hook is not None:
            self.hook(self)

    def load_weights(self, weights):
        for name, weight in weights:
            default_weight_loader(self.get_parameter(name), weight)
        return {name for name, _ in weights}


def _one_layer(fn, weight, hook=None) -> _OneLayer:
    with _DEVICE:
        model = _OneLayer(fn, hook)
    model.load_weights([("s.weight", weight)])
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    return model


def _side(model: torch.nn.Module) -> dict[str, tuple]:
    from vllm.model_executor.model_loader.reload.side_state import side_tensors

    return {
        n: (t.data_ptr(), t.shape, t.stride()) for n, t in side_tensors(model).items()
    }


def _views(layer):  # side refs and views of a weight refilled in place
    replace_parameter(layer, "weight", layer.weight * 1)
    layer.cfg = {"w": layer.weight, "w_t": layer.weight.t()}
    layer.cfg["w_u8"] = layer.weight.view(torch.uint8)


def _relaid(layer):  # a scale whose layout processing changes (moe_quant_config)
    replace_parameter(layer, "weight", layer.weight.reshape(2, 8) * 2)
    layer.cfg = {"w": layer.weight, "w_t": layer.weight.t()}


def _cached(layer):
    set_cached_tensor(layer, "cache", layer.weight * 2)
    layer.cfg = {"c": layer.cache}


def _root_cached(model):  # model-level hook, root has no quant_method
    set_cached_tensor(model, "derived", model.s.weight * 2)


def _plain(layer):
    layer.cache = layer.weight * 2


def _in_dict(layer):
    layer.cfg = {"d": layer.weight * 3}


def _data_rebind(layer):  # `x.data = fresh`: same object, new storage
    fresh = layer.weight * 2
    if getattr(layer, "cache", None) is None:
        layer.cache = fresh
    else:
        layer.cache.data = fresh


def _resized(layer):
    runs = layer.quant_method.runs
    set_cached_tensor(layer, "cache", torch.zeros(runs, device=_DEVICE))


def _dropped(layer):
    runs = layer.quant_method.runs
    layer.cache = torch.zeros(2, device=_DEVICE) if runs == 1 else None


def _new_view(layer):  # graphs read the startup view, the layer a transposed one
    if layer.quant_method.runs == 1:
        layer.cache = layer.weight * 2
    else:
        layer.cache = layer.cache.t()


def _registered(layer):
    layer.register_buffer("kc", layer.weight.sum().reshape(1))


def _root_plain(model):
    model.derived = model.s.weight * 2


def _noop(layer):
    pass


class TestSideTensors:
    """A derived tensor outside the checkpoint must be refilled in place: the
    reload raises, naming it, when post-load processing moved it."""

    @pytest.mark.parametrize(
        "fn,hook",
        [
            (_views, None),
            (_relaid, None),
            (_cached, None),
            (_noop, _root_cached),
        ],
        ids=["views", "relaid-weight", "cached", "model-root"],
    )
    def test_refilled_in_place(self, fn, hook):
        model = _one_layer(fn, torch.rand(4, 4), hook)
        before = _side(model)
        new = torch.rand(4, 4)
        _reload(model, {"s.weight": new})
        assert before and _side(model) == before
        _assert_same(_state(model), _state(_one_layer(fn, new, hook)))

    @pytest.mark.parametrize(
        "fn,hook,moved",
        [
            (_plain, None, "s.cache"),
            (_in_dict, None, "s.cfg['d']"),
            (_data_rebind, None, "s.cache"),
            (_resized, None, "s.cache"),
            (_dropped, None, "s.cache"),
            (_new_view, None, "s.cache"),
            (_registered, None, "s.kc"),
            (_noop, _root_plain, "derived"),
        ],
        ids=[
            "plain",
            "in-a-dict",
            "data-rebind",
            "cached-new-layout",
            "gone",
            "new-view",
            "registered",
            "model-root",
        ],
    )
    def test_moved_raises(self, fn, hook, moved):
        model = _one_layer(fn, torch.rand(4, 4), hook)
        with pytest.raises(RuntimeError, match=re.escape(f"moved: {[moved]}")):
            _reload(model, {"s.weight": torch.rand(4, 4)})


class _CountedDict(dict):
    walks = 0

    def items(self):
        type(self).walks += 1
        return super().items()


def _nested(o: object, depth: int) -> object:
    for _ in range(depth):
        o = [o]
    return o


def _walk(**attrs) -> dict[str, tuple]:
    """Side tensors of a two-layer model; ``l<i>__<attr>=value`` sets
    ``attr`` on layer ``i``."""
    model = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for name, value in attrs.items():
        layer, attr = name.split("__")
        setattr(model[int(layer[1:])], attr, value)
    return _side(model)


class TestSideWalk:
    def test_shared_object_walked_once(self):
        # a quantization config with a dict per checkpoint layer, referenced
        # by every layer
        _CountedDict.walks = 0
        config = _CountedDict({f"layers.{i}": {"bits": 4} for i in range(8)})
        holder = {"t": torch.ones(2, device=_DEVICE)}
        found = _walk(l0__config=config, l0__holder=holder, l1__config=config)
        assert list(found) == ["0.holder['t']"]
        assert _CountedDict.walks == 1

    @pytest.mark.parametrize("same_layer", [False, True], ids=["two-layers", "one"])
    def test_shallowest_path(self, same_layer):
        shared = {"t": torch.ones(2, device=_DEVICE)}
        deep = _nested(shared, _MAX_DEPTH)  # the tensor is past the limit here
        layer = 0 if same_layer else 1
        found = _walk(l0__deep=deep, **{f"l{layer}__shared": shared})
        assert list(found) == [f"{layer}.shared['t']"]

    @pytest.mark.parametrize(
        "depth,found", [(_MAX_DEPTH - 1, True), (_MAX_DEPTH, False)]
    )
    def test_depth_limit(self, depth, found):
        t = torch.ones(2, device=_DEVICE)
        assert bool(_walk(l0__deep=_nested(t, depth))) is found

    @pytest.mark.parametrize(
        "attr,module",
        [
            ("kv_cache", __name__),
            ("_workspace_buffer", __name__),
            ("parallel_group", "vllm.distributed.device_communicators.x"),
            ("holder", __name__),
        ],
        ids=["kv-cache", "workspace", "communicator", "walked"],
    )
    def test_runtime_state(self, attr, module):
        holder = type("Holder", (), {"__module__": module})()
        holder.buffer = torch.ones(2, device=_DEVICE)
        found = _walk(**{f"l0__{attr}": holder})
        assert bool(found) is (attr == "holder")

    @pytest.mark.parametrize(
        "base", ["tokenizer", "config", "vllm-subclass", "remote-code"]
    )
    def test_library_objects(self, base):
        # the cached tokenizer of a multimodal root subclasses a transformers
        # tokenizer; any other subclass, or a trust_remote_code class, may
        # hold a derived tensor
        import transformers

        parent = {
            "tokenizer": transformers.PreTrainedTokenizerBase,
            "config": transformers.PreTrainedConfig,
            "vllm-subclass": transformers.GenerationConfig,
            "remote-code": object,
        }[base]
        module = "transformers_modules.m.x" if base == "remote-code" else __name__
        obj: object = object.__new__(type("Holder", (parent,), {"__module__": module}))
        object.__setattr__(obj, "t", torch.ones(2, device=_DEVICE))
        found = _walk(l0__holder=obj)
        assert bool(found) is (base in ("vllm-subclass", "remote-code"))


class TestLoudChecks:
    def test_same_address_new_layout_moves(self):
        def fn(layer):  # same storage, flattened only on reloads
            if layer.quant_method.runs > 1:
                layer.weight.data = layer.weight.data.view(16)

        model = _one_layer(fn, torch.rand(4, 4))
        with pytest.raises(RuntimeError, match=r"moved: \['s.weight'\]"):
            _reload(model, {"s.weight": torch.rand(4, 4)})

    def test_weight_derived_host_scalar_change_raises(self):
        # an fp8 KV-cache scale read to a Python float, as kv_cache.py does
        def fn(layer):
            layer._k_scale_float = layer.weight.abs().amax().item()

        model = _one_layer(fn, torch.rand(4, 4))
        _reload(model, {"s.weight": model.s.weight.clone()})  # same: fine
        with pytest.raises(RuntimeError, match=r"s._k_scale_float \(host scalar"):
            _reload(model, {"s.weight": torch.rand(4, 4) + 2})

    def test_nan_host_scalar_is_unchanged(self):
        def fn(layer):  # e.g. an unset optional scale
            layer._o_scale = float("nan")

        model = _one_layer(fn, torch.rand(4, 4))
        _reload(model, {"s.weight": torch.rand(4, 4)})

    def test_cpu_offloaded_weights_refused(self):
        # post-processing would move them to the device and back into new
        # host tensors on every reload
        model = _tree_cold_start(
            lambda: _Tree({"s": _with_weight(_Layer(lambda layer: None))}),
            {"s.weight": torch.rand(2, 2)},
        )
        model.s.weight.data = model.s.weight.data.cpu()
        with pytest.raises(DirectReloadRefused, match="s: CPU-offloaded weights"):
            begin_direct_reload(model)

    def test_pre_processed_weights_refused(self):
        from vllm.model_executor.utils import weights_already_processed

        with weights_already_processed():
            model = _one_layer(None, torch.rand(4, 4))
        with pytest.raises(DirectReloadRefused, match="pre-processed"):
            begin_direct_reload(model)

    def test_eplb_refused(self, monkeypatch):
        # EPLB moves experts after cold start; loading places them as cold
        # start did while routing keeps the rearranged map
        import vllm.config

        model = _one_layer(None, torch.rand(4, 4))
        config = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=True))
        monkeypatch.setattr(
            vllm.config, "get_current_vllm_config_or_none", lambda: config
        )
        with pytest.raises(DirectReloadRefused, match="EPLB"):
            begin_direct_reload(model)

    @pytest.mark.parametrize("mode", ["direct", "layerwise"])
    def test_draft_keeps_its_own_copy_equal_to_the_target(self, monkeypatch, mode):
        # an EAGLE draft head shared by equality would take the draft's
        # updates into the target
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

        monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
        head = SimpleNamespace(weight=torch.ones(2))
        draft = SimpleNamespace(has_own_lm_head=True)
        shared = _should_share(draft, "has_own_lm_head", head, head)
        assert shared is (mode == "layerwise")

    @pytest.mark.parametrize("refusal", ["exclusion", "bad-regex", "storage-size"])
    def test_refused_begin_leaves_the_model_untouched(self, monkeypatch, refusal):
        # the worker keeps serving a model whose update was refused
        model = _cold_start(_checkpoint())
        if refusal == "exclusion":  # nvfp4: a dropped scale, re-registered
            monkeypatch.setenv(
                "VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", r"^nvfp4\.weight_scale_2$"
            )
        elif refusal == "bad-regex":
            monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", "[")
        else:  # proj is viewed back first, in module order
            shape, stride, dtype, offset = model.mimo.weight._ckpt_layout
            model.mimo.weight._ckpt_layout = ((64, 64), (64, 1), dtype, offset)
        before = _state(model)
        with pytest.raises(DirectReloadRefused):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)
        assert not hasattr(model, "_direct_skipped")


def test_layerwise_reload_ignores_the_direct_regex(monkeypatch):
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        record_metadata_for_reloading,
    )

    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    monkeypatch.setenv("VLLM_DIRECT_RELOAD_EXCLUDE_REGEX", "[")
    model = torch.nn.Linear(2, 2)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))


def test_retry_after_aborted_update_matches_cold_start():
    # Direct reload has no rollback: a failed update leaves the model partly
    # updated (the worker refuses to run it), and the next full update repairs it.
    model = _cold_start(_checkpoint())
    begin_direct_reload(model)
    model.load_weights(list(_checkpoint(1.01).items())[:3])  # update aborts
    _reload(model, _checkpoint(1.01))
    _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))


class _TiedModel(torch.nn.Module):
    """A config that ties the word embeddings over a checkpoint that ships
    an lm_head: built untied (outside direct mode, re-tied if the two are
    equal)."""

    def __init__(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        super().__init__()
        self.embed = VocabParallelEmbedding(16, 4)
        self.lm_head = ParallelLMHead(16, 4)

    def load_weights(self, weights):
        for name, weight in weights:
            param = self.get_parameter(name)
            param.weight_loader(param, weight)
        return {name for name, _ in weights}


_TIED_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=True,
    quantization=None,
    dtype=torch.float32,
    model="tied",
)


@pytest.mark.usefixtures("dist_init")
class TestTiedEmbeddings:
    """Direct reload never re-ties by equality, cold start included: equal
    copies in one checkpoint say nothing about the next one."""

    @staticmethod
    def _cold_start(embed, head) -> _TiedModel:
        with _DEVICE:
            model = _TiedModel()
        model.load_weights([("embed.weight", embed), ("lm_head.weight", head)])
        process_weights_after_loading(model, _TIED_CONFIG, _DEVICE)
        return model

    @pytest.mark.parametrize("equal", [True, False], ids=["equal", "different"])
    def test_copies_stay_untied(self, equal):
        w = torch.rand(16, 4)
        model = self._cold_start(w, w)
        assert model.lm_head.weight is not model.embed.weight
        ptrs = model.embed.weight.data_ptr(), model.lm_head.weight.data_ptr()
        embed, head = torch.rand(16, 4), torch.rand(16, 4)
        if equal:
            head = embed
        begin_direct_reload(model)
        model.load_weights([("embed.weight", embed), ("lm_head.weight", head)])
        finish_direct_reload(model, _TIED_CONFIG)
        fresh = self._cold_start(embed, head)
        assert (model.embed.weight.data_ptr(), model.lm_head.weight.data_ptr()) == ptrs
        assert torch.equal(model.embed.weight, fresh.embed.weight)
        assert torch.equal(model.lm_head.weight, fresh.lm_head.weight)


def test_draft_wired_to_target_modules_after_cold_start():
    # the proposer replaces a draft's embedding with the target's after the
    # draft's cold start; updates compare against the addresses it serves with
    draft = _tree_cold_start(
        lambda: _Tree(
            {"emb": _with_weight(_Layer(None)), "s": _with_weight(_Layer(None))}
        ),
        {"emb.weight": torch.rand(2, 2), "s.weight": torch.rand(2, 2)},
    )
    target_emb = _tree_cold_start(
        lambda: _Tree({"emb": _with_weight(_Layer(None))}),
        {"emb.weight": torch.rand(2, 2)},
    ).emb
    del draft.emb
    draft.emb = target_emb
    ptr = draft.emb.weight.data_ptr()
    new = torch.rand(2, 2)
    for _ in range(2):
        _reload(draft, {"s.weight": new})
    assert draft.emb.weight.data_ptr() == ptr
    assert torch.equal(draft.s.weight, new.to(_DEVICE))


class TestAbortedUpdate:
    def test_retry_after_failed_finish_compares_with_startup(self):
        """The first rerun replaces a plain cache, then raises; the retry
        still compares with the storage graphs captured at cold start."""

        def fn(layer):
            layer.cache = layer.weight * 2
            if layer.quant_method.runs == 2:
                raise RuntimeError("failed rerun")

        model = _one_layer(fn, torch.rand(4, 4))
        with pytest.raises(RuntimeError, match="failed rerun"):
            _reload(model, {"s.weight": torch.rand(4, 4)})
        with pytest.raises(RuntimeError, match=re.escape("moved: ['s.cache']")):
            _reload(model, {"s.weight": torch.rand(4, 4)})

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
