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
    DirectReloadRefused,
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
        with pytest.raises(ValueError, match=r"writes engram.weight, which"):
            model.load_weights([("engram.weight", _checkpoint()["engram.weight"])])

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
        self.W = self.kv_b.weight * 3 + self.q.weight


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
            model.broadcast = model.layers[0].attn.weight.sum(0)
            model.layers[0].engram.cache = model.layers[0].engram.weight * 2

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


def _side(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    from vllm.model_executor.model_loader.reload.side_state import (
        collect_side_tensors,
    )

    tensors = {str(p): t for p, t in collect_side_tensors(module).items()}
    tensors.update(module.named_buffers(recurse=False))
    return tensors


def _side_refs(layer):  # registered derived buffer, side refs and views of it
    replace_parameter(layer, "weight", layer.weight * 1)
    layer.register_buffer("kc", layer.weight.sum().reshape(1) * 2)
    layer.cfg = {
        "kc": layer.kc,
        "kc_u8": layer.kc.view(torch.uint8),
        "w": layer.weight,
        "d": layer.weight * 3,
    }


def _data_rebind(layer):  # `x.data = fresh` on a side tensor (J)
    fresh = layer.weight * 2
    if getattr(layer, "cache", None) is None:
        layer.cache = fresh
    else:
        layer.cache.data = fresh


def _root_cache(model):  # model-level hook, root has no quant_method (B)
    model.derived = model.s.weight * 2


class TestSideState:
    """Derived tensors outside the checkpoint stay at their cold-start
    storage and hold the values a cold start on the new weights computes."""

    @pytest.mark.parametrize(
        "fn,hook,module",
        [
            (_side_refs, None, "s"),
            (_data_rebind, None, "s"),
            (lambda layer: None, _root_cache, ""),
        ],
        ids=["buffer-and-views", "data-rebind", "model-root"],
    )
    def test_pinned(self, fn, hook, module):
        model = _one_layer(fn, torch.rand(4, 4), hook)
        before = {
            n: t.data_ptr() for n, t in _side(model.get_submodule(module)).items()
        }
        new = torch.rand(4, 4)
        _reload(model, {"s.weight": new})
        after = _side(model.get_submodule(module))
        fresh = _side(_one_layer(fn, new, hook).get_submodule(module))
        assert sorted(after) == sorted(before) == sorted(fresh)
        for name, t in after.items():
            assert t.data_ptr() == before[name], f"{name} moved"
            assert torch.equal(t, fresh[name]), f"{name} differs"

    def test_reference_to_a_relaid_checkpoint_weight(self):
        # e.g. moe_quant_config holding a scale whose layout processing
        # changes: begin views the scale back, the rerun lays it out again
        def fn(layer):
            replace_parameter(layer, "weight", layer.weight.reshape(2, 8) * 2)
            layer.cfg = {"w": layer.weight, "w_t": layer.weight.t()}

        model = _one_layer(fn, torch.rand(4, 4))
        before = {n: t.data_ptr() for n, t in _side(model.s).items()}
        new = torch.rand(4, 4)
        _reload(model, {"s.weight": new})
        after, fresh = _side(model.s), _side(_one_layer(fn, new).s)
        assert sorted(after) == sorted(before) == sorted(fresh)
        for name, t in after.items():
            assert t.data_ptr() == before[name], f"{name} moved"
            assert torch.equal(t, fresh[name]), f"{name} differs"

    @staticmethod
    def _resized(layer):
        layer.cache = torch.zeros(layer.quant_method.runs, device=_DEVICE)

    @staticmethod
    def _dropped(layer):
        runs = layer.quant_method.runs
        layer.cache = torch.zeros(2, device=_DEVICE) if runs == 1 else None

    @pytest.mark.parametrize("fn", ["_resized", "_dropped"])
    def test_unpinnable_raises(self, fn):
        model = _one_layer(getattr(self, fn), torch.rand(4, 4))
        with pytest.raises(RuntimeError, match=r"moved: \['s.cache'\]"):
            _reload(model, {"s.weight": torch.rand(4, 4)})


class _CountedDict(dict):
    walks = 0

    def items(self):
        type(self).walks += 1
        return super().items()


class TestSideWalk:
    """One pass over the modules walks a tensor-free object they all reference
    (a quantization config) once, and still finds every tensor."""

    def test_shared_tensor_free_object_walked_once(self):
        from vllm.model_executor.model_loader.reload.side_state import (
            SideWalk,
            collect_side_tensors,
        )

        _CountedDict.walks = 0
        config = _CountedDict({f"layers.{i}": {"bits": 4} for i in range(8)})
        holder = {"t": torch.ones(2)}
        side_walk = SideWalk()
        for _ in range(3):
            layer = torch.nn.Module()
            layer.quant_config, layer.holder = config, holder
            found = collect_side_tensors(layer, side_walk)
            assert list(found) == [(("a", "holder"), ("k", "t"))]
        assert _CountedDict.walks == 1

    @staticmethod
    def _nested(o: object, depth: int) -> object:
        for _ in range(depth):
            o = [o]
        return o

    @pytest.mark.parametrize("same_layer", [False, True], ids=["two-layers", "one"])
    def test_reached_at_the_depth_limit_then_shallower(self, same_layer):
        from vllm.model_executor.model_loader.reload.side_state import (
            _MAX_DEPTH,
            SideWalk,
            collect_side_tensors,
        )

        shared = {"k": {"t": torch.ones(2)}}
        deep = self._nested(shared, _MAX_DEPTH - 1)  # the tensor is past the limit
        first, second = torch.nn.Module(), torch.nn.Module()
        first.a_deep = deep
        (first if same_layer else second).b_shared = shared
        side_walk = SideWalk()
        found = collect_side_tensors(first, side_walk)
        if not same_layer:
            assert found == {}
            found = collect_side_tensors(second, side_walk)
        assert list(found) == [(("a", "b_shared"), ("k", "k"), ("k", "t"))]
        assert not side_walk.truncated  # walked after all: nothing unpinned

    def test_depth_limit_is_not_a_proof(self):
        from vllm.model_executor.model_loader.reload.side_state import (
            _MAX_DEPTH,
            SideWalk,
            collect_side_tensors,
        )

        layer = torch.nn.Module()
        layer.deep = self._nested(torch.ones(2), _MAX_DEPTH + 1)
        side_walk = SideWalk()
        assert collect_side_tensors(layer, side_walk) == {}
        assert len(side_walk.truncated) == 1
        assert not any(isinstance(o, list) for o in side_walk.clean.values())

    def test_truncated_walk_refuses_the_update(self):
        from vllm.model_executor.model_loader.reload.side_state import _MAX_DEPTH

        def fn(layer):  # a derived tensor below the depth limit
            layer.deep = self._nested(layer.weight * 2, _MAX_DEPTH + 1)

        model = _one_layer(fn, torch.rand(4, 4))
        before = _state(model)
        with pytest.raises(DirectReloadRefused, match="stopped at depth"):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)

    @pytest.mark.parametrize(
        "base", ["tokenizer", "config", "vllm-subclass", "remote-code"]
    )
    def test_library_objects(self, base):
        # the cached tokenizer of a multimodal root subclasses a transformers
        # tokenizer; any other subclass, or a trust_remote_code class, may
        # hold a derived tensor
        import transformers

        from vllm.model_executor.model_loader.reload.side_state import (
            collect_side_tensors,
        )

        parent = {
            "tokenizer": transformers.PreTrainedTokenizerBase,
            "config": transformers.PreTrainedConfig,
            "vllm-subclass": transformers.GenerationConfig,
            "remote-code": object,
        }[base]
        module = "transformers_modules.m.x" if base == "remote-code" else __name__
        obj: object = object.__new__(type("Holder", (parent,), {"__module__": module}))
        object.__setattr__(obj, "t", torch.ones(2))
        layer = torch.nn.Module()
        layer.holder = obj
        found = collect_side_tensors(layer)
        assert bool(found) is (base in ("vllm-subclass", "remote-code"))


class TestLoudChecks:
    def test_cached_tensor_cannot_change_layout(self):
        from vllm.model_executor.model_loader.weight_utils import set_cached_tensor

        def fn(layer):
            runs = layer.quant_method.runs
            set_cached_tensor(layer, "cache", torch.zeros(runs, device=_DEVICE))

        model = _one_layer(fn, torch.rand(4, 4))
        with pytest.raises(RuntimeError, match=r"cannot refill _Layer.cache"):
            _reload(model, {"s.weight": torch.rand(4, 4)})

    def test_same_address_new_layout_moves(self):
        def fn(layer):  # same storage, flattened only on reloads
            if layer.quant_method.runs > 1:
                layer.weight.data = layer.weight.data.view(16)

        model = _one_layer(fn, torch.rand(4, 4))
        with pytest.raises(RuntimeError, match=r"moved: \['s.weight'\]"):
            _reload(model, {"s.weight": torch.rand(4, 4)})

    @pytest.mark.parametrize("split", [True, False], ids=["split", "merged"])
    def test_side_tensor_aliasing_change_raises(self, split):
        # a cold start on the new weights gives a and b separate values; one
        # startup storage cannot hold both (or the reverse: one tensor pinned
        # into two storages ends up reading only the last)
        def fn(layer):
            shared = (layer.quant_method.runs == 1) == split
            layer.a = layer.weight * 2
            layer.b = layer.a if shared else layer.weight * 3

        model = _one_layer(fn, torch.rand(4, 4))
        with pytest.raises(RuntimeError, match=r"s.b \(aliasing changed\)"):
            _reload(model, {"s.weight": torch.rand(4, 4)})

    def test_weight_derived_host_scalar_change_raises(self):
        # an fp8 KV-cache scale read to a Python float, as kv_cache.py does
        def fn(layer):
            layer._k_scale_float = layer.weight.abs().amax().item()

        model = _one_layer(fn, torch.rand(4, 4))
        _reload(model, {"s.weight": model.s.weight.clone()})  # same: fine
        with pytest.raises(RuntimeError, match=r"s._k_scale_float \(host scalar"):
            _reload(model, {"s.weight": torch.rand(4, 4) + 2})

    def test_side_tensor_new_view_of_same_storage_moves(self):
        def fn(layer):  # graphs read the startup view, the layer a transposed one
            if layer.quant_method.runs == 1:
                layer.cache = layer.weight * 2
            else:
                layer.cache = layer.cache.t()

        model = _one_layer(fn, torch.rand(4, 4))
        with pytest.raises(RuntimeError, match=r"moved: \['s.cache'\]"):
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
    def test_retry_after_failed_finish_pins_into_startup_storage(self):
        """The first rerun replaces a plain cache, then raises; the retry
        still leaves the value in the storage graphs captured at cold start."""

        def fn(layer):
            layer.cache = layer.weight * 2
            if layer.quant_method.runs == 2:
                raise RuntimeError("failed rerun")

        model = _one_layer(fn, torch.rand(4, 4))
        startup = model.s.cache.data_ptr()
        new = torch.rand(4, 4)
        with pytest.raises(RuntimeError, match="failed rerun"):
            _reload(model, {"s.weight": new})
        assert model.s.cache.data_ptr() != startup
        _reload(model, {"s.weight": new})
        assert model.s.cache.data_ptr() == startup
        assert torch.equal(model.s.cache, _one_layer(fn, new).s.cache)

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
