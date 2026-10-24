# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts rows stored as a full-attention KV cache layer."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.extract_hidden_states import (
    CacheOnlyAttentionBackend,
    CacheOnlyAttentionMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, KVCacheSpec


class RoutedExpertsCacheMetadataBuilder(CacheOnlyAttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class RoutedExpertsCacheBackend(CacheOnlyAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.uint8, torch.uint16]

    @staticmethod
    def get_name() -> str:
        return "ROUTED_EXPERTS_CACHE"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[RoutedExpertsCacheMetadataBuilder]:
        return RoutedExpertsCacheMetadataBuilder


class RoutedExpertsCacheLayer(nn.Module, AttentionLayerBase):
    """Keep each token's routed experts in a full-attention KV cache group.

    Prefix caching and KV connectors then hit, evict, and transfer routed
    experts together with the KV of the same blocks.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        shape_per_token: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        num_layers = vllm_config.model_config.get_total_num_hidden_layers()
        self.layer_name = f"routed_experts_cache.{num_layers}"
        self.shape_per_token = shape_per_token
        self.head_size = math.prod(shape_per_token)
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        forward_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return RoutedExpertsCacheBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return HiddenStateCacheSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
        )

    def write(self, rows: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Scatter one step's rows; padding slots land in the null block."""
        block_size = self.kv_cache.shape[2]
        slot_mapping = slot_mapping[: len(rows)].clamp_min(0)
        self.kv_cache[slot_mapping // block_size, 0, slot_mapping % block_size] = (
            rows.reshape(len(rows), -1)
        )

    def read(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        block_size = self.kv_cache.shape[2]
        rows = self.kv_cache[slot_mapping // block_size, 0, slot_mapping % block_size]
        return rows.view(len(slot_mapping), *self.shape_per_token)
