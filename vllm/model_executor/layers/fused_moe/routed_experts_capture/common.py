# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, torch-free helpers for routed-experts capture."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_num_experts_per_token(hf_config: Any) -> int:
    """Resolve the number of experts selected per token."""
    num_experts_per_token = getattr(hf_config, "num_experts_per_tok", None)
    if num_experts_per_token is None:
        num_experts_per_token = getattr(hf_config, "top_k_experts", None)
    if num_experts_per_token is None:
        raise ValueError(
            "Cannot determine the number of experts selected per token: "
            "HF config has neither "
            "'num_experts_per_tok' nor 'top_k_experts'"
        )
    return num_experts_per_token


def get_num_experts(hf_config: Any) -> int:
    """Resolve the global logical expert count from the HF config."""
    for attribute_name in (
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
    ):
        num_experts = getattr(hf_config, attribute_name, None)
        if num_experts is not None:
            return num_experts
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


def get_routing_shape_and_dtype(
    vllm_config: "VllmConfig",
) -> tuple[tuple[int, int], str]:
    """Return the logical per-token routing shape and dtype."""
    hf_config = vllm_config.model_config.hf_text_config
    num_layers = hf_config.num_hidden_layers
    moe_top_k = get_num_experts_per_token(hf_config)
    num_experts = get_num_experts(hf_config)
    dtype = "uint8" if num_experts <= 256 else "uint16"
    return (num_layers, moe_top_k), dtype
