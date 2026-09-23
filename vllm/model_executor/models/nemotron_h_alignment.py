# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit normalization forwards shared by single-rank Nemotron execution."""

import torch


def rms_forward(x, weight, eps, residual=None):
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

    if x.dtype != weight.dtype:
        raise ValueError("Shared RMS requires matching activation and weight dtype")
    if residual is None:
        return rms_norm_batch_invariant(x, weight, eps)
    return rms_norm_batch_invariant(x.clone(), weight, eps, residual.clone())


@torch.library.custom_op("nemotron_alignment::gated_rms", mutates_args=())
def gated_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    # Keep compiler-generated and eager callers on the same kernel launch.
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated

    return rms_norm_gated(
        x,
        weight,
        None,
        z=gate,
        eps=eps,
        group_size=group_size,
        norm_before_gate=False,
    )


@gated_forward.register_fake
def _gated_forward_fake(x, gate, weight, group_size, eps):
    return torch.empty_like(x)
