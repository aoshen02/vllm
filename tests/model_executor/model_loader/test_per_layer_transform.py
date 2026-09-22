# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A direct reload is refused for a model whose layers transform weights.

`direct` does not run per-layer post-load processing, so a model that
requantizes at load would serve kernel weights derived from the previous
checkpoint. Whether a layer transforms is measured while the cold start runs
that pass anyway.
"""

import pytest
import torch

from vllm.model_executor.model_loader.reload.per_layer import (
    layers_that_transform,
    observe,
    refuse_direct_reload_if_layers_transform,
    signature,
)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))


def _run_post_load(model, name, layer, transform):
    """What the cold-start pass does around one layer's post-load."""
    before = signature(layer)
    transform()
    observe(model, name, layer, before)


def test_a_layer_that_rewrites_its_weight_is_recorded():
    model, layer = torch.nn.Module(), _Layer()
    _run_post_load(model, "mlp", layer, lambda: layer.weight.data.fill_(3.0))

    assert layers_that_transform(model) == ["mlp"]
    with pytest.raises(RuntimeError, match="transforms weights"):
        refuse_direct_reload_if_layers_transform(model)


def test_a_layer_that_replaces_its_weight_is_recorded():
    """Requantization swaps in a fresh Parameter rather than writing in place."""
    model, layer = torch.nn.Module(), _Layer()

    def requantize():
        layer.weight = torch.nn.Parameter(torch.full((2, 2), 5.0))

    _run_post_load(model, "mlp", layer, requantize)
    assert layers_that_transform(model) == ["mlp"]


def test_a_layer_that_changes_nothing_is_not_recorded():
    """The unquantized path is what `direct` exists for; it must stay usable."""
    model, layer = torch.nn.Module(), _Layer()
    _run_post_load(model, "mlp", layer, lambda: None)

    assert layers_that_transform(model) == []
    refuse_direct_reload_if_layers_transform(model)  # must not raise


def test_a_layer_that_only_adds_state_is_not_recorded():
    """Caching a kernel choice is not a weight transform."""
    model, layer = torch.nn.Module(), _Layer()
    _run_post_load(model, "mlp", layer, lambda: setattr(layer, "cpu_linear", len))

    assert layers_that_transform(model) == []


def test_the_message_names_the_layers_and_the_way_out():
    model = torch.nn.Module()
    for index in range(7):
        layer = _Layer()
        _run_post_load(
            model, f"l{index}", layer, lambda each=layer: each.weight.data.fill_(2)
        )

    with pytest.raises(RuntimeError) as error:
        refuse_direct_reload_if_layers_transform(model)
    message = str(error.value)
    assert "l0" in message and "..." in message
    assert "layerwise" in message
