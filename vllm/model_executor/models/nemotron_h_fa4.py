# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-schedule FP8 attention for Nemotron-H batch invariance."""

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.platforms import current_platform
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd


class NemotronHFixedFA4Impl(FlashAttentionImpl):
    supports_quant_query_input = True
    supports_dcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        config = get_current_vllm_config_or_none()
        capability = current_platform.get_device_capability()
        if (
            capability is None
            or capability.major != 10
            or (num_heads, num_kv_heads, head_size) != (32, 2, 128)
            or kv_cache_dtype not in ("fp8", "fp8_e4m3")
            or alibi_slopes is not None
            or sliding_window is not None
            or logits_soft_cap is not None
            or sinks is not None
            or attn_type != AttentionType.DECODER
            or kv_sharing_target_layer_name is not None
            or config is None
            or config.parallel_config.decode_context_parallel_size != 1
        ):
            raise ValueError("Unsupported Nemotron-H fixed FA4 attention contract")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.sliding_window = (-1, -1)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Nemotron-H fixed FA4 does not fuse output quantization")
        if attn_metadata is None:
            return output.fill_(0)
        if (
            attn_metadata.use_cascade
            or attn_metadata.causal is not True
            or attn_metadata.max_query_len > 9216
            or attn_metadata.max_seq_len > 9216
        ):
            raise ValueError("Unsupported Nemotron-H fixed FA4 metadata")

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache).view(
            current_platform.fp8_dtype()
        )
        value_cache = canonicalize_singleton_dim_strides(value_cache).view(
            current_platform.fp8_dtype()
        )
        num_tokens = attn_metadata.num_actual_tokens
        scale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        _flash_attn_fwd(
            query[:num_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_q=9216,
            max_seqlen_k=9216,
            page_table=attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=True,
            q_descale=layer._q_scale.expand(scale_shape),
            k_descale=layer._k_scale.expand(scale_shape),
            v_descale=layer._v_scale.expand(scale_shape),
            tile_mn=(128, 128),
            pack_gqa=True,
            num_splits=16,
            seqlen_k_per_split=640,
            disable_scheduler_metadata=True,
            out=output[:num_tokens],
        )
        return output


class NemotronHFixedFA4Backend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[NemotronHFixedFA4Impl]:
        return NemotronHFixedFA4Impl
