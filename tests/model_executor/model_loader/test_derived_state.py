# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level derived state is rebuilt after a reload, or the reload is refused."""

import pytest
import torch

from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_derived_buffer

MODES = ["layerwise", "direct"]


class _Fused(torch.nn.Module):
    """Fuses its layers' weights into a buffer in the model-level post-load
    hook, the way Kimi K3 fuses its MegaMoE experts."""

    fused: torch.Tensor
    reload_safe_post_load = True

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.b = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        for p in self.parameters():
            p.weight_loader = default_weight_loader
        self.register_buffer(
            "fused", torch.zeros(2, 4, 4, dtype=torch.bfloat16), persistent=False
        )
        self.seen: list[torch.Tensor] = []

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)

    def process_weights_after_loading(self) -> None:
        self.seen.append(self.a.weight.clone())
        self.fused.copy_(torch.stack([self.a.weight, self.b.weight]))


class _RebindsStorage(_Fused):
    """ROCm DeepSeek-V4 calls replace_parameter in this hook."""

    def process_weights_after_loading(self) -> None:
        self.fused = torch.zeros_like(self.fused)


class _ChangesLayout(_Fused):
    """A transpose keeps data_ptr() but a captured graph reads the old strides."""

    def process_weights_after_loading(self) -> None:
        self.fused = self.fused.transpose(1, 2)


class _FinalizesDuringLoad(_Fused):
    """DeepSeek-V4 calls its own hook at the end of load_weights, and says so."""

    finalizes_weights_during_load = True


class _RebindsPlainAttribute(_Fused):
    """Kimi K3 DSpark stacks its layers' norm weights into a bare attribute."""

    def process_weights_after_loading(self) -> None:
        self.stacked = torch.stack([self.a.weight, self.b.weight])


class _NotDeclaredSafe(_Fused):
    """The default: a model-level hook nobody has reviewed for re-entrancy."""

    reload_safe_post_load = False


class _ConsumesInput(torch.nn.Module):
    """Fuses its raw parameter into a buffer and drops it, as the MegaMoE
    finalizers do, so the hook's own guard has to be the raw parameter."""

    raw: torch.nn.Parameter | None
    fused: torch.Tensor | None
    reload_safe_post_load = True

    def __init__(self) -> None:
        super().__init__()
        self.raw = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
        self.raw.weight_loader = default_weight_loader
        self.register_buffer("fused", None, persistent=False)

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)

    def process_weights_after_loading(self) -> None:
        if self.raw is None:
            return
        set_derived_buffer(self, "fused", self.raw.data * 2)
        self.raw = None


class _NoHook(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.a.weight.weight_loader = default_weight_loader

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)


def _update(model: torch.nn.Module, value: float, mode: str) -> None:
    # Build the payload first: layerwise puts the parameters on meta at start.
    payload = [(n, torch.full_like(p, value)) for n, p in model.named_parameters()]
    start_reload(model, mode)
    model.load_weights(payload)
    finish_reload(model, None)


@pytest.mark.parametrize("mode", MODES)
def test_model_level_derived_state_is_rebuilt(mode):
    """Without this the model serves state derived from the previous
    checkpoint: the fused copy would still hold the cold-start weights."""
    model = _Fused()
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()
    fused_storage = model.fused.data_ptr()

    _update(model, 7.0, mode)

    assert torch.equal(model.fused, torch.full_like(model.fused, 7.0))
    assert model.fused.data_ptr() == fused_storage, "rebuilt out of place"


@pytest.mark.parametrize("mode", MODES)
def test_the_hook_runs_once_and_sees_the_new_weights(mode):
    """It derives from the weights, so it cannot run before they have landed."""
    model = _Fused()
    record_metadata_for_reloading(model)

    _update(model, 5.0, mode)

    assert len(model.seen) == 1
    assert torch.equal(model.seen[0], torch.full_like(model.seen[0], 5.0))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cls", [_RebindsStorage, _ChangesLayout])
def test_a_hook_that_does_not_write_in_place_is_refused(mode, cls):
    """Dropping the storage a CUDA graph captured cannot be allowed to report
    success, so the outcome is checked and the model named."""
    model = cls()
    record_metadata_for_reloading(model)

    with pytest.raises(RuntimeError, match="relocated fused"):
        _update(model, 3.0, mode)


@pytest.mark.parametrize("mode", MODES)
def test_a_model_that_finalizes_during_load_is_refreshed_anyway(mode):
    """`finalizes_weights_during_load` says the model runs its own hook at the
    end of `load_weights`. It is not evidence the hook ran during this update:
    sharded-RDT replays baked scatters instead of calling `load_weights`, and
    skipping on the flag would leave that path serving the previous derived
    state."""
    model = _FinalizesDuringLoad()
    record_metadata_for_reloading(model)

    _update(model, 2.0, mode)

    assert torch.equal(model.fused, torch.full_like(model.fused, 2.0))


@pytest.mark.parametrize("mode", MODES)
def test_an_undeclared_model_level_hook_is_refused(mode):
    """Re-running a cold-start hook is only correct for a hook written to allow
    it, and that cannot be decided by inspection: an in-tree hook may skip on a
    "have I ever built this" guard, consume an input the reload cannot restore,
    or recompile a submodule. Refuse until the model says it was reviewed --
    and refuse before a single weight is written, because nothing on the
    inference path consults the reload lifecycle, so a model left holding new
    weights beside old derived state would go on serving that mixture."""
    model = _NotDeclaredSafe()
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()
    before = model.a.weight.clone()

    with pytest.raises(RuntimeError, match="reload_safe_post_load"):
        _update(model, 6.0, mode)

    assert torch.equal(model.a.weight, before), "refused after mutating the model"


@pytest.mark.parametrize("mode", MODES)
def test_a_hook_that_rebinds_a_plain_attribute_is_refused(mode):
    """A bare attribute is read by forward exactly as a buffer is, so
    reallocating one leaves a captured graph on freed storage."""
    model = _RebindsPlainAttribute()
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()

    with pytest.raises(RuntimeError, match="relocated stacked"):
        _update(model, 3.0, mode)


def test_a_hook_that_consumed_its_input_refuses_the_reload():
    """The MegaMoE shape: a model-level hook fuses a parameter and drops it.

    Layerwise restores the parameter so the checkpoint can load into it, but
    the layer keeps only the tensors it held before the reload, and the fusion
    that would carry the new value into one of those lives on the model, not on
    the layer, so it does not run in the layer's window. The loaded value would
    be discarded and the model would keep serving the previous checkpoint.
    Refreshing the model-level state afterwards cannot fix this -- its input is
    already gone -- so the reload has to be refused instead of reported clean.
    """
    model = _ConsumesInput()
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()
    assert model.raw is None

    start_reload(model, "layerwise")
    model.load_weights([("raw", torch.full((4, 4), 5.0, dtype=torch.bfloat16))])
    with pytest.raises(RuntimeError, match="does not keep after post-processing"):
        finish_reload(model, None)


def test_a_dropped_weight_is_reported_at_finish_not_during_the_load():
    """The sharded-RDT bake drives `load_weights` over fake tensors only to
    record a plan, and never finishes. It discards every value by design, so
    the refusal has to wait for finish_reload rather than fire mid-load."""
    model = _ConsumesInput()
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()

    start_reload(model, "layerwise")
    model.load_weights([("raw", torch.full((4, 4), 5.0, dtype=torch.bfloat16))])


@pytest.mark.parametrize("mode", MODES)
def test_a_model_without_the_hook_is_untouched(mode):
    model = _NoHook()
    record_metadata_for_reloading(model)

    _update(model, 4.0, mode)

    assert torch.equal(model.a.weight, torch.full_like(model.a.weight, 4.0))


def test_every_no_op_model_level_hook_is_declared():
    """A root hook that does nothing costs nothing to re-run, so refusing its
    model's reload is a pure regression. Two were missed by reading alone (one
    body is `pass`, the other a bare `return`), so enumerate them instead."""
    import ast
    import pathlib

    import vllm

    def is_no_op(fn: ast.FunctionDef) -> bool:
        body = [
            s
            for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        return all(
            isinstance(s, ast.Pass) or (isinstance(s, ast.Return) and s.value is None)
            for s in body
        )

    root = pathlib.Path(vllm.__file__).parent
    undeclared = []
    for path in (
        *root.glob("models/**/*.py"),
        *root.glob("model_executor/models/*.py"),
    ):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            hooks = [
                f
                for f in cls.body
                if isinstance(f, ast.FunctionDef)
                and f.name == "process_weights_after_loading"
                and is_no_op(f)
            ]
            declared = any(
                isinstance(s, ast.Assign)
                and any(
                    getattr(t, "id", "") == "reload_safe_post_load" for t in s.targets
                )
                for s in cls.body
            )
            if hooks and not declared:
                undeclared.append(f"{path.name}:{hooks[0].lineno} {cls.name}")

    assert not undeclared, (
        "these root hooks do nothing, so declare reload_safe_post_load rather "
        f"than refusing their reloads: {undeclared}"
    )


class _GuardsOnExistence(_Fused):
    """Kimi K3's shape: the fusion returns early once it has ever run, which at
    cold start is the same question as whether the state is current."""

    def __init__(self) -> None:
        super().__init__()
        self.built = False

    def process_weights_after_loading(self) -> None:
        from vllm.model_executor.model_loader.reload.derived import (
            rebuilding_derived_state,
        )

        if self.built and not rebuilding_derived_state():
            return
        self.built = True
        self.fused.copy_(torch.stack([self.a.weight, self.b.weight]))


@pytest.mark.parametrize("mode", MODES)
def test_a_hook_that_guards_on_existence_is_told_it_is_a_rebuild(mode):
    """Without the scope the guard fires and the model keeps serving the
    previous checkpoint's fused weights."""
    model = _GuardsOnExistence()
    record_metadata_for_reloading(model)
    model.process_weights_after_loading()  # cold start
    fused = model.fused
    address = fused.data_ptr()

    _update(model, 7.0, mode)

    assert model.fused is fused
    assert model.fused.data_ptr() == address
    assert torch.equal(model.fused, torch.full_like(model.fused, 7.0))


def test_the_scope_is_closed_outside_a_reload():
    """A cold-start call must see the same answer as before this existed."""
    from vllm.model_executor.model_loader.reload.derived import (
        rebuilding_derived_state,
    )

    assert not rebuilding_derived_state()
    model = _GuardsOnExistence()
    model.process_weights_after_loading()
    assert not rebuilding_derived_state()
