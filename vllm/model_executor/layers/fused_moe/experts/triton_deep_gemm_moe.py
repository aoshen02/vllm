# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmExperts,
    _valid_deep_gemm,
    _valid_deep_gemm_shape,
)
from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
)
from vllm.utils.import_utils import has_deep_gemm


def _bi_use_deep_gemm(N: int, K: int) -> bool:
    """Batch-invariant implementation pick from deployment constants only.

    The default selection gates on ``align <= M``, which switches between
    Triton and DeepGemm as the batch grows; both are invariant individually
    but not mutually bitwise identical. Grouped padding covers small M, so
    under BI only the weight-shape constants may decide.
    """
    align = get_mk_alignment_for_contiguous_layout()[0]
    return has_deep_gemm() and N % align == 0 and K % align == 0 and N > 512


class TritonOrDeepGemmExperts(FallbackExperts):
    """DeepGemm with fallback to Triton for low latency shapes."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(
            experts=DeepGemmExperts(moe_config, quant_config),
            fallback_experts=TritonExperts(moe_config, quant_config),
        )

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular],
        type[mk.FusedMoEExpertsModular],
    ]:
        return (DeepGemmExperts, TritonExperts)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        #
        # Under batch invariance the implementation is picked from N and K
        # alone, so ask the same question here instead of always assuming
        # DeepGemm: a deployment whose weight shapes always route to Triton
        # would otherwise pay for the larger workspace it never uses.
        if envs.VLLM_BATCH_INVARIANT:
            use_deep_gemm = _bi_use_deep_gemm(N, K)
        else:
            use_deep_gemm = is_deep_gemm_e8m0_used() or _valid_deep_gemm_shape(M, N, K)
        if use_deep_gemm:
            return self.experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        else:
            return self.fallback_experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )

    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEExpertsModular:
        if envs.VLLM_BATCH_INVARIANT:
            _, K, N = w2.size()
            return self.experts if _bi_use_deep_gemm(N, K) else self.fallback_experts
        if is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2):
            return self.experts
        else:
            return self.fallback_experts
