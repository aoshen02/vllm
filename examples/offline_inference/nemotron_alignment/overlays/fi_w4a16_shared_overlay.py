"""Opt-in construction-time candidate; no public kernel-selector mutation."""

import inspect
from contextlib import contextmanager
from functools import wraps


class HummingScaleCompatibility:
    """Preserve the existing Humming FP32 scale roundtrip during fresh packing."""

    def process_weights_after_loading(self, layer):
        import torch

        raw_scale = layer.weight_global_scale.detach().clone()
        if raw_scale.dtype != torch.float32 or raw_scale.numel() != 1:
            raise ValueError("Shared W4A16 compatibility requires scalar FP32 scale")
        effective_scale = 1.0 / (1.0 / raw_scale)
        super().process_weights_after_loading(layer)
        with torch.no_grad():
            layer.weight_global_scale.copy_(
                effective_scale.reshape_as(layer.weight_global_scale)
            )


def select_instances(mlp, prefix, *, batch_invariant, tp_size, eligible, factory):
    """Replace exactly two unprocessed shared W4A16 kernel instances atomically."""
    if not batch_invariant or tp_size != 1 or not prefix.endswith(".shared_experts"):
        return []
    layers = [mlp.up_proj, mlp.down_proj]
    if not all(eligible(layer) for layer in layers):
        return []
    for layer in layers:
        if type(layer.quant_method.kernel).__name__ != "HummingNvFp4LinearKernel":
            raise ValueError(
                "Shared candidate expected the unchanged BI Humming selection"
            )
        if not hasattr(layer, "weight_scale_2") or hasattr(
            layer, "weight_global_scale"
        ):
            raise ValueError(
                "Shared candidate must select before weight postprocessing"
            )
    replacements = [factory(), factory()]
    records = []
    for name, layer, kernel in zip(("up_proj", "down_proj"), layers, replacements):
        previous = type(layer.quant_method.kernel).__name__
        layer.quant_method.kernel = kernel
        record = {
            "prefix": prefix + "." + name,
            "previous": previous,
            "selected": type(kernel).__name__,
            "before_packing": True,
        }
        layer._fi_shared_candidate = record
        records.append(record)
    return records


@contextmanager
def construction_overlay():
    import vllm.envs as envs
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferCuteDslNvFp4W4A16LinearKernel,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    class NemotronSharedHummingScaleFI(
        HummingScaleCompatibility, FlashInferCuteDslNvFp4W4A16LinearKernel
    ):
        pass

    original = NemotronHMLP.__init__
    signature = inspect.signature(original)
    records = []

    def eligible(layer):
        method = layer.quant_method
        return (
            isinstance(method, ModelOptLinearMethod)
            and method.spec.weight == kNvfp4Static
            and method.spec.activation is None
        )

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if type(self) is not NemotronHMLP:
            return
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        records.extend(
            select_instances(
                self,
                bound.arguments["prefix"],
                batch_invariant=envs.VLLM_BATCH_INVARIANT,
                tp_size=get_tensor_model_parallel_world_size(),
                eligible=eligible,
                factory=lambda: NemotronSharedHummingScaleFI(NvFp4LinearLayerConfig()),
            )
        )

    NemotronHMLP.__init__ = initialize
    try:
        yield records
    finally:
        NemotronHMLP.__init__ = original
