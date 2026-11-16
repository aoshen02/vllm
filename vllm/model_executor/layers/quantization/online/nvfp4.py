# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.loading import quantize_on_load
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    trtllm_nvfp4_hidden_alignment,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    kNvfp4DynamicToken,
    kNvfp4Static,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_moe_weight_to_nvfp4(
    weight: torch.Tensor,
    moe_tp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE expert weights `(E, N, K)` to NVFP4.

    One FP32 global scale per expert plus per-block (group-16) FP8 scales,
    matching the ModelOpt NVFP4 checkpoint layout. Returns packed FP4 weights
    `(E, N, K // 2)`, block scales `(E, N, K // 16)`, and the per-expert
    global scale `(E,)` stored as `amax / (fp4_max * fp8_max)`.
    """
    assert weight.dim() == 3, f"expected 3D expert weights, got {weight.shape}"
    k = weight.shape[-1]
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight_amax(weight.flatten(1), dim=-1).to(torch.float32)
    amax = amax_for_moe_weight_quant(amax, moe_tp_size).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # Keep the original BF16/FP16 values as the quantizer input. Folding each
    # expert's FP32 global scale into the weight would add a BF16/FP16 rounding
    # before the group-16 scale and E2M1 values are selected.
    weight = weight.contiguous()
    quantized_experts = [
        scaled_fp4_quant(
            expert_weight,
            expert_scale,
            is_sf_swizzled_layout=False,
        )
        for expert_weight, expert_scale in zip(
            weight,
            global_scale,
            strict=True,
        )
    ]
    qweight = torch.stack([quantized for quantized, _ in quantized_experts])
    block_scale = torch.stack([block_scale for _, block_scale in quantized_experts])
    return (
        qweight,
        block_scale,
        weight_scale_2,
    )


def _quantize_nvfp4(
    weight: torch.Tensor, amax: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`weight` to NVFP4 (as `_quantize_moe_weight_to_nvfp4` does one expert),
    its global scale from `amax`, by default its own."""
    if amax is None:
        amax = weight_amax(weight)
    amax = amax.to(weight.device, torch.float32)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax.clamp_min(1e-8)
    qweight, block_scale = scaled_fp4_quant(
        weight.contiguous(), global_scale, is_sf_swizzled_layout=False
    )
    return qweight, block_scale, (1.0 / global_scale).reshape(1)


class Nvfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """Online NVFP4 MoE quantization with per-token activation scales.

    Quantizes fp16/bf16 expert weights to NVFP4 at load time; the FlashInfer
    TRTLLM kernel computes per-token activation scales at runtime. Blackwell
    (SM100) only.
    """

    activation_quant_key = kNvfp4DynamicToken
    uses_meta_device = False

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        if not current_platform.is_device_capability_family(100):
            raise ValueError(
                "nvfp4_per_token online quantization requires a Blackwell (SM100) GPU."
            )
        super().__init__(moe)
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=self.activation_quant_key,
        )

    @property
    def per_token_activation(self) -> bool:
        return self.activation_quant_key == kNvfp4DynamicToken

    def create_quantized_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        **extra_weight_attrs,
    ):
        """As a ModelOpt NVFP4 checkpoint."""
        inter = intermediate_size_per_partition
        w13_rows = self.moe.w13_num_shards * inter
        fp8 = torch.float8_e4m3fn
        self.register_weights(
            layer,
            FusedMoeWeightScaleSupported.GROUP.value,
            {
                "w13_weight": torch.zeros(
                    num_experts, w13_rows, hidden_size // 2, dtype=torch.uint8
                ),
                "w2_weight": torch.zeros(
                    num_experts, hidden_size, inter // 2, dtype=torch.uint8
                ),
                "w13_weight_scale": torch.zeros(
                    num_experts, w13_rows, hidden_size // 16, dtype=fp8
                ),
                "w2_weight_scale": torch.zeros(
                    num_experts, hidden_size, inter // 16, dtype=fp8
                ),
            },
            **extra_weight_attrs,
        )
        self.register_weights(
            layer,
            FusedMoeWeightScaleSupported.TENSOR.value,
            {
                "w13_weight_scale_2": torch.ones(
                    num_experts, self.moe.w13_num_shards, dtype=torch.float32
                ),
                "w2_weight_scale_2": torch.ones(num_experts, dtype=torch.float32),
            },
            **extra_weight_attrs,
        )
        quantize_on_load(
            layer, "w2_weight", _quantize_nvfp4, "w2_weight_scale", "w2_weight_scale_2"
        )
        # w1 and w3 of an expert share one global scale: the first to arrive
        # waits for the other
        amax = None
        quantize_on_load(
            layer,
            "w13_weight",
            lambda weight: _quantize_nvfp4(weight, amax),
            "w13_weight_scale",
            "w13_weight_scale_2",
        )
        load = layer.w13_weight.weight_loader
        pending: dict[int, tuple[str, torch.Tensor]] = {}

        def w13_loader(
            param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
        ):
            nonlocal amax
            if not self.moe.is_act_and_mul:
                pair = [(shard_id, loaded_weight)]
            elif (other := pending.pop(expert_id, None)) is None:
                pending[expert_id] = (shard_id, loaded_weight.clone())
                return True if return_success else None
            else:
                pair = [other, (shard_id, loaded_weight)]
            amax = max(weight_amax(w) for _, w in pair).to(layer.orig_dtype)
            for shard, weight in pair:
                ret = load(param, weight, weight_name, shard, expert_id, return_success)
            return ret

        layer.w13_weight.weight_loader = w13_loader

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if self.requantization_source is None:
            replace_parameter(
                layer,
                "w13_weight_scale_2",
                layer.w13_weight_scale_2[:, 0].contiguous(),
            )
            ones = torch.ones(
                layer.num_experts, device=layer.w13_weight.device, dtype=torch.float32
            )
            replace_parameter(layer, "w13_input_scale", ones)
            replace_parameter(layer, "w2_input_scale", ones.clone())
            self._setup_kernel(layer)
            layer._already_called_process_weights_after_loading = True
            return

        w13_weight, w2_weight = self.get_weights_for_quantization(layer)
        self._quantize_weights(layer, w13_weight, w2_weight)
        self._setup_kernel(layer)
        self.release_requantization_source_weights(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(
        self,
        layer: Module,
        w13_weight: torch.Tensor | None = None,
        w2_weight: torch.Tensor | None = None,
    ) -> None:
        w13_weight = layer.w13_weight if w13_weight is None else w13_weight
        w2_weight = layer.w2_weight if w2_weight is None else w2_weight
        moe_tp_size = self.moe.tp_size
        w13, w13_scale, w13_scale_2 = _quantize_moe_weight_to_nvfp4(
            w13_weight, moe_tp_size
        )
        w2, w2_scale, w2_scale_2 = _quantize_moe_weight_to_nvfp4(w2_weight, moe_tp_size)

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)

        # Neutral (1.0) activation global scales: the kernel derives per-token
        # scales at runtime, so the output scalars reduce to the weight scales.
        ones = torch.ones(layer.num_experts, device=w13.device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
            trtllm_hidden_alignment=trtllm_nvfp4_hidden_alignment(
                per_token_activation=self.per_token_activation,
                is_act_and_mul=self.moe.is_act_and_mul,
            ),
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale", a13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale", a2_scale)

        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=self.per_token_activation,
            )
            self.moe_kernel.fused_experts.keep_constants(layer)
        else:
            # Reload creates new scale tensors; derived kernel scales must use
            # their new values before layerwise reload restores captured storage.
            assert self.moe_quant_config is not None
            assert self.moe_quant_config.g1_alphas is not None
            assert self.moe_quant_config.g2_alphas is not None
            self.moe_quant_config.g1_alphas.copy_(w13_scale_2)
            self.moe_quant_config.g2_alphas.copy_(w2_scale_2)

        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )
