# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_bf16_grouped_supported,
    is_deep_gemm_bf16_masked_supported,
    is_deep_gemm_supported,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    mk_alignment_scope,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def _fp8_workspace_shape(
    num_rows: int, num_columns: int, workspace_dtype: torch.dtype
) -> tuple[int, int]:
    """Size an FP8 byte buffer stored in a workspace of another dtype."""
    bytes_per_workspace_element = workspace_dtype.itemsize
    fp8_columns_per_workspace_element = (
        bytes_per_workspace_element // torch.float8_e4m3fn.itemsize
    )
    assert bytes_per_workspace_element % torch.float8_e4m3fn.itemsize == 0
    return (
        num_rows,
        -(-num_columns // fp8_columns_per_workspace_element),
    )


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align = get_mk_alignment_for_contiguous_layout()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()

    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug_once(
            "DeepGemm disabled due to unaligned problem size. "
            "M: %s, N: %s, K: %s. M should >= %s "
            "and N and K must be multiples of %s. "
            "This is not an error and we will fall back to triton.",
            M,
            N,
            K,
            align,
            align,
        )
        return False
    elif N <= 512:
        logger.debug_once(
            "DeepGemm disabled for N <= 512. M: %s, N: %s, K: %s. "
            "This means we will fallback to triton "
            "for this specific shape for further speed up.",
            M,
            N,
            K,
        )
        return False

    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


class DeepGemmExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # MXFP8: FP8 e4m3 values + UE8M0 1x32 block scales (Blackwell). Reuses
        # the same grouped GEMM (aliased to fp8_fp4) with recipe (1, 32).
        self.mxfp8 = quant_config.block_shape == [1, 32]
        if self.mxfp8:
            assert quant_config.quant_dtype == "mxfp8"
        else:
            assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()
            assert quant_config.quant_dtype == torch.float8_e4m3fn
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        # Gated-activation params: silu == swigluoai with alpha=1, beta=0.
        # FP8 (silu) configs leave these None, reproducing plain silu.
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if (weight_key, activation_key) == (kFp8Static128BlockSym, kFp8Dynamic128Sym):
            return True
        # MXFP8 1x32 uses the fp8_fp4 grouped GEMM with recipe (1, 32) — only
        # available on Blackwell (SM100).
        if (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic):
            return current_platform.is_device_capability_family(100)
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # silu/swigluoai go through the fused alpha/beta kernel; swiglustep
        # uses the unfused activation path. The fused kernel reads packed w13
        # (gate = first half, up = second half), so it implements the
        # *uninterleaved* SwiGLU-OAI variant.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        assert self.block_shape is not None
        # Use the contiguous-layout M alignment (matches apply()); block_shape[0]
        # is the quant block (1 for MXFP8) and would under-size the workspace.
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace1 is allocated in the activation dtype by the workspace
        # manager, but is only ever viewed and used as FP8 in apply(). Size it
        # by bytes instead of reserving one BF16/FP16 element per FP8 element.
        workspace1 = _fp8_workspace_shape(
            M_sum,
            max(activation_out_dim, K),
            self.workspace_dtype(self.moe_config.in_dtype),
        )
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        activation: MoEActivation,
        expert_ends: torch.Tensor | None = None,
        expert_alignment: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.block_shape is not None
        block_k = self.block_shape[1]
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        # silu and swigluoai are both expressible by the fused gated kernel via
        # (alpha, beta): silu uses alpha=1, beta=0; swigluoai uses config values.
        # The fused kernel reads packed w13, hence SWIGLUOAI_UNINTERLEAVE.
        fused_gated = activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

        # 1. DeepGemm UE8M0: fused gate+mul+clamp+quant+pack
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if fused_gated:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                    alpha=self.gemm1_alpha,
                    beta=self.gemm1_beta,
                    expert_ends=expert_ends,
                    expert_alignment=expert_alignment,
                )
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        # 2. Hopper / non‑E8M0: prefer the fused gate+mul+quant kernel
        if fused_gated:
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
                group_size=block_k,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
                expert_ends=expert_ends,
                expert_alignment=expert_alignment,
            )

        # 3. fallback path for non-SiLU activations in non‑UE8M0 cases.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        return per_token_group_quant_fp8(
            act_out, block_k, column_major_scales=True, out_q=output
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        # Both GEMMs and quantization must agree on the live expert ranges.
        use_psum_layout = (
            not self.mxfp8
            and (
                current_platform.is_device_capability_family(90)
                or (
                    current_platform.is_device_capability_family(100)
                    and DeepGemmQuantScaleFMT.from_oracle()
                    == DeepGemmQuantScaleFMT.UE8M0
                )
            )
            and activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE)
        )
        a1q, a1q_scale, grouped_layout, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            # MXFP8 uses a 32-element activation-scale group (block_shape[1]);
            # FP8-block keeps the default (128) alignment.
            block_size=self.block_shape[1] if self.mxfp8 else None,
            use_psum_layout=use_psum_layout,
        )
        assert a1q.size(0) == M_sum

        # MXFP8 (1x32) drives the fp8_fp4-aliased grouped GEMM with recipe
        # (1, 32); the FP8 block path keeps the default (128) recipe.
        gemm_kwargs: dict = (
            {"recipe_a": (1, self.block_shape[1]), "recipe_b": (1, self.block_shape[1])}
            if self.mxfp8
            else {}
        )

        if use_psum_layout:
            gemm_kwargs["use_psum_layout"] = True

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # otherwise the scheduler can pick the wrong expert id from m_indices
        # under cudagraph replay.
        with mk_alignment_scope(align_used):
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                grouped_layout,
                **gemm_kwargs,
            )

            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N),
                output=quant_out,
                activation=activation,
                expert_ends=grouped_layout if use_psum_layout else None,
                expert_alignment=align_used if use_psum_layout else 0,
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                grouped_layout,
                **gemm_kwargs,
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepGemmFP4Experts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation for FP4 weights.

    Uses m_grouped_fp8_fp4_gemm_nt_contiguous with FP8 activations and
    MXFP4 (FP4 E2M1 packed as uint8) weights. Requires Blackwell-family
    GPUs (SM100 datacenter or SM120 consumer).
    """

    # FP8 activation block size (hardcoded since mxfp4_w4a8 quant config
    # does not set a block_shape on the activation descriptor).
    _ACT_BLOCK_K = 128
    # FP4 weight block size
    _WEIGHT_BLOCK_K = 32

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.weight_quant_dtype == "mxfp4"
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        return is_deep_gemm_supported() and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # SILU has fused gate+mul+quant kernels; SWIGLUSTEP/SITU take the
        # general path (activation applied via self.activation, which forwards
        # the situ betas, then FP8 requant).
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SITU,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace1 holds the permuted and requantized FP8 activations. The
        # workspace manager allocates it in the model activation dtype, so
        # account for the dtype sizes rather than overallocating BF16/FP16.
        workspace1 = _fp8_workspace_shape(
            M_sum,
            max(activation_out_dim, K),
            self.workspace_dtype(self.moe_config.in_dtype),
        )
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_k = self._ACT_BLOCK_K
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        if activation == MoEActivation.SILU:
            # Fused gate+mul+quant kernels for the common SILU case.
            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                    clamp_limit=self.gemm1_clamp_limit,
                )
            use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            return silu_mul_per_token_group_quant_fp8_colmajor(
                input=input,
                output=output,
                use_ue8m0=use_ue8m0,
                clamp_limit=self.gemm1_clamp_limit,
            )

        # General gated activations (SWIGLUSTEP, SITU): apply the activation
        # (self.activation forwards the situ betas from moe_config) then
        # FP8-requant into the layout DeepGEMM expects for this scale format.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            return per_token_group_quant_fp8_packed_for_deepgemm(
                act_out, block_k, use_ue8m0=True, out_q=output
            )
        use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        return per_token_group_quant_fp8(
            act_out,
            block_k,
            column_major_scales=True,
            out_q=output,
            use_ue8m0=use_ue8m0,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states
        _, N, _ = w1.size()
        # K comes from activations (full hidden dim), not from w1 which is
        # packed FP4 (E, N, K//2).
        K = a1q.size(1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, K)
        )
        a1q, a1q_scale, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )
        assert a1q.size(0) == M_sum

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # see DeepGemmExperts.apply for rationale.
        with mk_alignment_scope(align_used):
            # FC1: FP8 activations x FP4 weights
            # DeepGEMM 2.4.2 requires FP4-packed weights as int8 (kPackedFP4).
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1.view(torch.int8), self.w1_scale),
                mm1_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

            # SwiGLU activation + FP8 requant
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn), (M_sum, activation_out_dim)
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            # FC2: FP8 activations x FP4 weights
            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2.view(torch.int8), self.w2_scale),
                mm2_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


def _valid_deep_gemm_bf16(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """Whether the bf16 DeepGEMM grouped contiguous kernel supports this problem.

    All of M, N, K must be aligned to
    ``get_mk_alignment_for_contiguous_layout()`` (128) and every tensor must be
    contiguous bfloat16.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm(bf16) disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()
    align = get_mk_alignment_for_contiguous_layout()[0]

    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug_once(
            "DeepGemm(bf16) disabled due to unaligned problem size. "
            "M=%s N=%s K=%s (need M>=%s and N,K multiples of %s). "
            "Falling back to the default backend.",
            M,
            N,
            K,
            align,
            align,
        )
        return False

    if (
        hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        logger.debug_once(
            "DeepGemm(bf16) disabled: expected bfloat16 activations and weights. "
            "hidden_states=%s w1=%s w2=%s",
            hidden_states.dtype,
            w1.dtype,
            w2.dtype,
        )
        return False

    if not (
        hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm(bf16) disabled: activations or weights not contiguous."
        )
        return False

    return True


class DeepGemmBf16Experts(mk.FusedMoEExpertsModular):
    """DeepGEMM-based fused MoE experts for unquantized bf16 weights.

    Uses ``m_grouped_bf16_gemm_nt_contiguous`` for both grouped GEMMs, with a
    plain silu-and-mul in between. This is the ``Standard`` (contiguous)
    activation-format analogue of the FP8 :class:`DeepGemmExperts`.
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # Unquantized only: no weight/activation scales, no block quant.
        assert quant_config.quant_dtype is None, (
            "DeepGemmBf16Experts only supports unquantized (bf16) weights."
        )
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_bf16_grouped_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Unquantized bf16 only: the oracle passes (None, None) for this path.
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Phase 1: plain gated SiLU only (silu_and_mul). Other gated variants
        # (e.g. swigluoai) can be added later.
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Match DeepGemmExperts: exclude the FlashInfer NVL fused kernels.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        block_m = get_mk_alignment_for_contiguous_layout()[0]
        M_sum, align_used = compute_aligned_M_and_alignment(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % align_used == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        # workspace13 holds the permuted bf16 activations (M_sum, K) and later
        # the post-activation input (M_sum, activation_out_dim); workspace2
        # holds either grouped-GEMM output (M_sum, N) or (M_sum, K).
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Unquantized bf16: no activation scales are produced by prepare/finalize.
        assert a1q_scale is None
        assert a2_scale is None

        a1 = hidden_states
        _, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum, _ = compute_aligned_M_and_alignment(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        # Permute bf16 activations into the per-expert contiguous layout and
        # build m_indices (expert_ids). No scale scatter for bf16.
        a1_perm = _resize_cache(workspace13, (M_sum, K))
        a1, _, expert_ids, inv_perm, align_used = deepgemm_moe_permute(
            aq=a1,
            aq_scale=None,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1_perm,
        )
        assert a1.size(0) == M_sum

        # Cap DG's BLOCK_M heuristic at the workspace's per-expert alignment;
        # see DeepGemmExperts.apply for the IMA-under-cudagraph rationale.
        with mk_alignment_scope(align_used):
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_bf16_gemm_nt_contiguous(a1, w1, mm1_out, expert_ids)

            activation_out_dim = self.adjust_N_for_activation(N, activation)
            act_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
            # Plain (bf16) silu-and-mul; no requantization.
            self.activation(activation, act_out, mm1_out.view(-1, N))

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_bf16_gemm_nt_contiguous(act_out, w2, mm2_out, expert_ids)

        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


class DeepGemmBf16BatchedExperts(mk.FusedMoEExpertsModular):
    """DeepGEMM bf16 masked grouped-GEMM experts (``BatchedExperts`` / EP format).

    Unquantized bf16 analogue of ``BatchedDeepGemmExperts``: uses
    ``m_grouped_bf16_gemm_nt_masked`` for both grouped GEMMs with a plain
    silu-and-mul in between (no fp8 quantization). Selected via the opt-in
    ``deep_gemm`` MoE backend when the layer runs in the batched activation
    format (expert/data parallel with an all2all dispatcher).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        assert quant_config.quant_dtype is None, (
            "DeepGemmBf16BatchedExperts only supports unquantized (bf16) weights."
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_bf16_masked_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the reduce impl.
        return TopKWeightAndReduceDelegate()

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
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int:
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            logger.warning_once(
                "DPMetadata unavailable. Defaulting expected_m to "
                f"{max_tokens_per_expert}.",
            )
            return max_tokens_per_expert

        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        # Assume even load balancing across experts.
        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # Unquantized bf16: no activation scales.
        assert a1q_scale is None
        assert a2_scale is None
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        a1 = hidden_states

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        assert w2.size(1) == K

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )

        # GroupGemm-0: (E, T, K) x (E, N, K) -> (E, T, N)
        m_grouped_bf16_gemm_nt_masked(a1, w1, workspace1, expert_num_tokens, expected_m)

        # Plain (bf16) silu-and-mul over the full padded batch. Rows past
        # expert_num_tokens are junk but are never read by the masked
        # down-GEMM below (it is gated by expert_num_tokens), so no masked
        # activation kernel is required.
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        act_out = _resize_cache(workspace2, (E, max_num_tokens, activation_out_dim))
        self.activation(
            activation,
            act_out.view(-1, activation_out_dim),
            workspace1.view(-1, N),
        )

        # GroupGemm-1: (E, T, N//2) x (E, K, N//2) -> (E, T, K)
        m_grouped_bf16_gemm_nt_masked(
            act_out, w2, output, expert_num_tokens, expected_m
        )
