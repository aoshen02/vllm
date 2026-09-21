# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A reload keeps the storage of tensors a layer reaches outside the module tree.

Shapes mirror MXFP4: `quant_method.moe_kernel.fused_experts` holds the SwiGLU
constants a graph reads, and the quant method holds permutation indices in a
dict. `process_weights_after_loading` rebuilds the kernel, which allocates both
afresh.
"""

import pytest
import torch

from vllm.model_executor.model_loader.reload.owned import (
    restore_owned_tensors,
    snapshot_owned_tensors,
)

# The traversal follows vLLM's own classes; these stand in for them.
VLLM = "vllm.model_executor.layers.fused_moe.test_double"


class _FusedExperts:
    __module__ = VLLM

    def __init__(self, alpha: float) -> None:
        self.gemm1_alpha = torch.full((1,), alpha)
        self.gemm1_clamp_limit = torch.full((1,), 7.0)


class _Kernel:
    __module__ = VLLM

    def __init__(self, alpha: float) -> None:
        self.fused_experts = _FusedExperts(alpha)
        self.back_reference: object | None = None


class _QuantMethod:
    __module__ = VLLM

    def __init__(self) -> None:
        self.moe_kernel = _Kernel(1.702)
        self._cache_permute_indices: dict[str, torch.Tensor] = {
            "a": torch.zeros(4, dtype=torch.int32)
        }
        self.helper: torch.nn.Module | None = None

    def rebuild(self) -> None:
        """What `_setup_kernel` does: a fresh kernel, hence fresh constants."""
        self.moe_kernel = _Kernel(2.5)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2, 2))
        self.quant_method = _QuantMethod()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Layer()


def _names(model: torch.nn.Module) -> set[str]:
    return set(snapshot_owned_tensors(model))


def test_tensors_behind_owned_objects_are_found():
    """`named_buffers` stops at the module tree, so nothing else sees these."""
    assert _names(_Model()) == {
        "experts.quant_method.moe_kernel.fused_experts.gemm1_alpha",
        "experts.quant_method.moe_kernel.fused_experts.gemm1_clamp_limit",
        "experts.quant_method._cache_permute_indices['a']",
    }


def test_a_rebuilt_kernel_keeps_the_storage_a_graph_captured():
    """The constant's value must change and its address must not."""
    model = _Model()
    before = snapshot_owned_tensors(model)
    alpha = model.experts.quant_method.moe_kernel.fused_experts.gemm1_alpha
    address = alpha.data_ptr()

    model.experts.quant_method.rebuild()
    assert restore_owned_tensors(before, snapshot_owned_tensors(model)) == []

    restored = model.experts.quant_method.moe_kernel.fused_experts.gemm1_alpha
    assert restored is alpha
    assert restored.data_ptr() == address
    assert restored.item() == pytest.approx(2.5)


def test_a_cleared_cache_is_reported_rather_than_silently_dropped():
    """Clearing frees storage the graph still reads, and nothing can put it back."""
    model = _Model()
    before = snapshot_owned_tensors(model)
    model.experts.quant_method._cache_permute_indices.clear()

    assert restore_owned_tensors(before, snapshot_owned_tensors(model)) == [
        "experts.quant_method._cache_permute_indices['a']"
    ]


def test_a_differently_shaped_replacement_is_reported():
    model = _Model()
    before = snapshot_owned_tensors(model)
    experts = model.experts.quant_method.moe_kernel.fused_experts
    experts.gemm1_alpha = torch.zeros(9)

    assert restore_owned_tensors(before, snapshot_owned_tensors(model)) == [
        "experts.quant_method.moe_kernel.fused_experts.gemm1_alpha"
    ]


def test_a_cache_entry_replaced_in_place_is_restored():
    model = _Model()
    before = snapshot_owned_tensors(model)
    original = model.experts.quant_method._cache_permute_indices["a"]
    model.experts.quant_method._cache_permute_indices["a"] = torch.full(
        (4,), 3, dtype=torch.int32
    )

    assert restore_owned_tensors(before, snapshot_owned_tensors(model)) == []
    assert model.experts.quant_method._cache_permute_indices["a"] is original
    assert original.tolist() == [3, 3, 3, 3]


class _Foreign:
    """Stands in for a third-party object a layer happens to hold."""

    def __init__(self) -> None:
        self.tensor = torch.zeros(2)


def test_a_foreign_object_is_not_followed():
    """Its identity is not ours to police, and its graph may be anything."""
    model = _Model()
    model.experts.foreign = _Foreign()
    assert not any("foreign" in name for name in _names(model))


def test_a_cycle_terminates():
    model = _Model()
    method = model.experts.quant_method
    method.moe_kernel.back_reference = method
    assert "experts.quant_method.moe_kernel.fused_experts.gemm1_alpha" in _names(model)


def test_a_submodule_reached_through_an_owned_object_is_left_to_the_module_tree():
    """It is already named by `named_parameters`; naming it twice would make a
    parameter look relocated when only the alias went away."""
    model = _Model()
    model.experts.quant_method.helper = torch.nn.Linear(2, 2)
    assert not any("helper" in name for name in _names(model))
