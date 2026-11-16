# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Widen the dummy weight init for the MoRIIO dummy-weight P/D test.

The dummy loader fills float weights in +-1e-3, which keeps every activation so
small that outputs barely depend on the input, so a wrong KV prefix would go
unnoticed. The P/D test names `WideDummyWeights` as each server's
`--worker-extension-cls`. Workers import it before they load the model, which
re-registers the `dummy` load format in every worker process with a loader
that fills 16/32-bit float weights in +-`MORIIO_DUMMY_WEIGHT_RANGE`; 8-bit
weights and scales keep the default init. The format keeps the name `dummy`
so every config path that special-cases dummy weights still applies.

The test process itself never imports this module.
"""

import os

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.online_processing import (
    finish_online_layer,
    is_online_processing,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)

_RANGE = float(os.environ["MORIIO_DUMMY_WEIGHT_RANGE"])


@register_model_loader("dummy")
class WideDummyModelLoader(DummyModelLoader):
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # DummyModelLoader.load_weights with a wider init for 16/32-bit floats.
        for layer in model.modules():
            if is_online_processing(layer):
                finish_online_layer(layer, fill=initialize_single_dummy_weight)
                continue
            for name, tensor in (*layer._parameters.items(), *layer._buffers.items()):
                if tensor is None or name in layer._non_persistent_buffers_set:
                    continue
                if tensor.is_floating_point() and torch.finfo(tensor.dtype).bits >= 16:
                    initialize_single_dummy_weight(
                        tensor, low=-_RANGE, high=_RANGE, name=name
                    )
                else:
                    initialize_single_dummy_weight(tensor, name=name)


class WideDummyWeights:
    """Worker extension that only exists to import this module in workers."""
