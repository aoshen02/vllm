# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
from torch.utils._python_dispatch import _disable_current_modes

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform


def _rename(arg: Any, old: str, new: str) -> Any:
    return arg.replace(old, new) if isinstance(arg, str) else arg


def quantize_on_load(
    layer: torch.nn.Module,
    name: str,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    *scale_names: str,
) -> None:
    """Make `layer.<name>`, created in its quantized checkpoint layout, load
    high-precision checkpoint tensors: each arriving tensor is quantized by
    `quantize` into `(values, *scales)`, loaded into `layer.<name>` and
    `layer.<scale_names>` by their own loaders, as a quantized checkpoint is."""
    load = getattr(getattr(layer, name), "weight_loader", default_weight_loader)
    layer.quantizes_on_load = True

    def loader(param, loaded_weight, *args, **kwargs):
        # A layerwise reload counts the elements each loader copies, into meta
        # parameters: count only the values, the scales' loaders count theirs.
        with _disable_current_modes():
            device = current_platform.device_type if param.is_meta else param.device
            values, *scales = quantize(loaded_weight.to(device, layer.orig_dtype))
            for scale_name, scale in zip(scale_names, scales):
                scale_param = getattr(layer, scale_name)
                # the MoE loader picks how to load a scale by its parameter name
                getattr(scale_param, "weight_loader", default_weight_loader)(
                    scale_param,
                    scale,
                    *(_rename(a, name, scale_name) for a in args),
                    **{k: _rename(v, name, scale_name) for k, v in kwargs.items()},
                )
        return load(param, values, *args, **kwargs)

    getattr(layer, name).weight_loader = loader
