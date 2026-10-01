# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit normalization forwards shared by single-rank Nemotron execution."""

import torch


def rms_forward(x, weight, eps, residual=None):
    import vllm._custom_ops as ops

    if x.dtype != weight.dtype:
        raise ValueError("Shared RMS requires matching activation and weight dtype")
    if residual is None:
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        ops.rms_norm(output, x, weight, eps)
        return output
    output = x.clone()
    residual_out = residual.clone()
    ops.fused_add_rms_norm(output, residual_out, weight, eps)
    return output, residual_out


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
