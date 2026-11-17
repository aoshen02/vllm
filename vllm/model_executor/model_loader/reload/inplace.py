# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_listeners import weights_updating
from vllm.model_executor.model_loader.weight_utils import finalize_model_weights
from vllm.model_executor.utils import weights_already_processed

from .lifecycle import finish_reload, start_reload

logger = init_logger(__name__)

__all__ = ["get_parameter_for_reload", "reload_weights"]


def get_parameter_for_reload(model: nn.Module, name: str) -> nn.Parameter:
    """Resolve checkpoint names without changing the model's module tree."""
    from vllm.lora.layers import BaseLayerWithLoRA

    module_name, _, parameter_name = name.rpartition(".")
    module = model.get_submodule(module_name)
    if isinstance(module, BaseLayerWithLoRA):
        module = module.base_layer
    return module.get_parameter(parameter_name)


def reload_weights(
    vllm_config: VllmConfig,
    model: nn.Module,
    weights_iterator: Iterable[tuple[str, torch.Tensor]] | None,
    weights_path: str | None,
    is_checkpoint_format: bool,
    model_loader: BaseModelLoader | None = None,
) -> None:
    """Reload a model's weights in place from a weights iterator or from disk.

    Callers are responsible for invalidating any state derived from the old
    weights (e.g. LoRA, encoder and multimodal caches).

    Args:
        vllm_config: config of the model being reloaded. If `weights_path` is
            given, `vllm_config.model_config` is updated to point at it.
        model: model to load weights into
        weights_iterator: weights to load into model
        weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        is_checkpoint_format: set to False if weights have already been
            processed into kernel format (repacking, renaming, etc.)
        model_loader: loader to reload from if weights_iterator is not
            provided, by default a new one for `vllm_config.load_config`

    """
    if weights_iterator is None and not is_checkpoint_format:
        logger.warning(
            "`is_checkpoint_format=False` is ignored when reloading from the "
            "model loader"
        )

    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    weights_to_load = {
        name.replace(".base_layer.", ".") if vllm_config.lora_config else name
        for name, _ in model.named_parameters()
    }
    counter_before_reloading = time.perf_counter()

    logger.info_once("Reloading weights inplace...")
    if weights_iterator is None:
        if weights_path is not None:
            # The revision and any object-storage `model_weights` source
            # belong to the model we are reloading away from, so they must
            # not be carried over to the new path.
            model_config.model = weights_path
            model_config.model_weights = ""
            model_config.revision = None
        if model_loader is None:
            model_loader = get_model_loader(load_config)
        loaded_weights = model_loader.reload_weights(vllm_config, model)

    elif is_checkpoint_format:
        # load weights from checkpoint/ original model format
        device = torch.device(load_config.device or vllm_config.device_config.device)
        start_reload(model)
        loaded_weights = model.load_weights(weights_iterator)
        finish_reload(model, model_config, device)

    else:
        # load weights from kernel format
        logger.warning_once(
            "Reloading with `is_checkpoint_format=True` requires that "
            "weights be in kernel format and already sharded",
        )
        loaded_weights = set()
        with weights_updating():
            for name, loaded_weight in weights_iterator:
                param = get_parameter_for_reload(model, name)  # TODO: buffers?
                param.copy_(loaded_weight)
                loaded_weights.add(name)
            # kernel-format weights: hooks rebuild derived state, transform nothing
            with weights_already_processed():
                finalize_model_weights(model)

    # logging and validation
    counter_after_reloading = time.perf_counter()
    diff_seconds = counter_after_reloading - counter_before_reloading
    logger.info_once(
        "Reloading and processing weights took %.2f seconds",
        diff_seconds,
    )
    if model_config.quantization is None and loaded_weights is not None:
        weights_not_loaded = weights_to_load - loaded_weights
        if weights_not_loaded:
            logger.warning(
                "Following weights were not loaded from checkpoint: %s",
                weights_not_loaded,
            )
