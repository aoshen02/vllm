# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit Nemotron W4A16 routed-expert backend; not batch invariant."""

import weakref

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static


def nemotron_w4a16_support(config, model_type, batch_invariant):
    """Limit explicit backend selection to the supported Nemotron recipe."""
    parallel = config.moe_parallel_config
    if batch_invariant:
        return False, "W4A16 routed backend does not support batch invariance"
    if model_type != "nemotron_h" or config.moe_backend != "flashinfer_cutedsl":
        return False, "requires explicit Nemotron flashinfer_cutedsl selection"
    if config.in_dtype != torch.bfloat16:
        return False, "requires BF16 activations"
    if config.activation != MoEActivation.RELU2_NO_MUL:
        return False, "requires non-gated ReLU2"
    if (
        config.hidden_dim,
        config.intermediate_size_per_partition,
        config.num_experts,
        config.experts_per_token,
    ) != (2688, 1856, 128, 6):
        return False, "requires Lightning H2688/I1856/E128/top6"
    if parallel.tp_size != 1 or parallel.ep_size not in (1, 4):
        return False, "requires TP1 and EP1 or EP4"
    if parallel.enable_eplb or parallel.use_batched_activation_format:
        return False, "EPLB and batched-expert layouts are unsupported"
    if config.num_local_experts != 128 // parallel.ep_size:
        return False, "requires contiguous uniform expert ownership"
    return True, None


def prepare_w4a16_scales(scale):
    """Keep padded SF storage registered; the runner consumes its raw pointer."""
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    if scale.dtype != torch.float8_e4m3fn or scale.ndim != 3:
        raise ValueError("Expected expert-major E4M3 block scales")
    experts, rows, cols = scale.shape
    swizzled = block_scale_interleave(scale.contiguous().view(torch.uint8))
    return swizzled.view(torch.float8_e4m3fn).reshape(
        experts, (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    )


class FlashInferCuteDSLW4A16Experts(FlashInferCuteDSLExperts):
    @staticmethod
    def is_supported_config(cls, config, weight_key, activation_key, activation_format):
        current = get_current_vllm_config_or_none()
        model = current.model_config if current is not None else None
        model_type = getattr(getattr(model, "hf_config", None), "model_type", None)
        supported, reason = nemotron_w4a16_support(
            config, model_type, envs.VLLM_BATCH_INVARIANT
        )
        if not supported:
            return supported, reason
        try:
            from flashinfer.fused_moe.cute_dsl.tuner import (
                CuteDslFusedMoEW4A16Runner,  # noqa: F401
            )
        except ImportError:
            return False, "installed FlashInfer lacks the W4A16 runner"
        return mk.FusedMoEExperts.is_supported_config(
            cls, config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key == kNvfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.RELU2_NO_MUL

    @staticmethod
    def _supports_batch_invariance():
        return False

    def __init__(self, moe_config, quant_config):
        from flashinfer.fused_moe.cute_dsl.tuner import CuteDslFusedMoEW4A16Runner
        from flashinfer.tllm_enums import ActivationType

        mk.FusedMoEExpertsModular.__init__(self, moe_config, quant_config)
        if quant_config.quant_dtype is not None:
            raise ValueError("W4A16 must not quantize activations")
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.local_expert_offset = (
            moe_config.moe_parallel_config.ep_rank * self.local_num_experts
        )
        self.runner = CuteDslFusedMoEW4A16Runner(
            num_experts=self.global_num_experts,
            top_k=moe_config.experts_per_token,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            use_fused_finalize=False,
            output_dtype=torch.bfloat16,
            activation_type=ActivationType.Relu2.value,
        )

    def process_weights_after_loading(self, layer):
        if layer.activation != MoEActivation.RELU2_NO_MUL:
            raise ValueError("Only non-gated ReLU2 is covered")
        if getattr(layer, "w13_input_scale", None) is not None:
            raise ValueError("Unexpected activation scale after W4A16 conversion")
        if layer.expert_map is not None:
            expected = torch.full_like(layer.expert_map, -1)
            start = self.local_expert_offset
            expected[start : start + self.local_num_experts] = torch.arange(
                self.local_num_experts, device=expected.device, dtype=expected.dtype
            )
            if not torch.equal(layer.expert_map, expected):
                raise ValueError("Non-contiguous expert maps are unsupported")
        self._layer_ref = weakref.ref(layer)

    def _registered_scale(self, name, initial):
        ref = getattr(self, "_layer_ref", None)
        if ref is None:
            return initial
        layer = ref()
        if layer is None:
            raise RuntimeError("W4A16 expert layer was released")
        # Layerwise reload restores registered storage after building the kernel.
        return getattr(layer, name)

    @property
    def w1_scale(self):
        return self._registered_scale("w13_weight_scale", super().w1_scale)

    @property
    def w2_scale(self):
        return self._registered_scale("w2_weight_scale", super().w2_scale)

    @property
    def g1_alphas(self):
        return self._registered_scale("w13_weight_scale_2", super().g1_alphas)

    @property
    def g2_alphas(self):
        return self._registered_scale("w2_weight_scale_2", super().g2_alphas)

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        # The modular caller derives K from unquantized a1, not packed weights.
        if self.hidden_dim != K:
            raise ValueError("W4A16 workspace requires the full BF16 hidden dimension")
        return (0,), (0,), (M, self.hidden_dim)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if hidden_states.dtype != torch.bfloat16 or a1q_scale is not None:
            raise ValueError("Expected unquantized BF16 activations")
        if apply_router_weight_on_input or activation != MoEActivation.RELU2_NO_MUL:
            raise ValueError("Unsupported W4A16 routing/activation contract")
        if global_num_experts != self.global_num_experts:
            raise ValueError("Global expert count changed")
        inputs = [
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.float(),
            w1,
            self.w1_scale,
            self.g1_alphas.reshape(-1),
            w2,
            self.w2_scale,
            self.g2_alphas.reshape(-1),
            output,
        ]
        # Explicit screened candidates, chosen from host-known shapes before
        # launch. No autotuning or device synchronization inside Graph replay.
        tile = (
            ((128, 8, 256), (1, 1), True)
            if hidden_states.shape[0] <= 64
            else ((256, 128, 256), (2, 1), True)
        )
        self.runner.forward(inputs, tactic=(tile, tile))
