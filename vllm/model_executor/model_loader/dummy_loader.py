# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.online_processing import (
    finish_online_layer,
    is_online_processing,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        for layer in model.modules():
            if is_online_processing(layer):
                # materialize, apply dummy weights, run quantization processing
                finish_online_layer(layer, fill=initialize_single_dummy_weight)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                # Initialize only local tensors: state_dict() would recursively
                # initialize descendants that model.modules() also visits.
                # Exclude nonpersistent buffers, matching state_dict().
                tensors = (*layer._parameters.items(), *layer._buffers.items())
                for name, tensor in tensors:
                    if (
                        tensor is not None
                        and name not in layer._non_persistent_buffers_set
                    ):
                        initialize_single_dummy_weight(tensor, name=name)
