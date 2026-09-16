# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit normalization forwards shared by single-rank Nemotron execution."""

import torch
from torch import nn


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
        x, weight, None, z=gate, eps=eps,
        group_size=group_size, norm_before_gate=False,
    )


@gated_forward.register_fake
def _gated_forward_fake(x, gate, weight, group_size, eps):
    return torch.empty_like(x)


class SharedRMSNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.eps = original.variance_epsilon

    def forward(self, x, residual=None):
        return rms_forward(x, self.weight, self.eps, residual)


class SharedGatedRMSNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not original.use_rms_norm:
            raise ValueError("Shared gated normalization requires RMS normalization")
        self.weight = original.weight
        self.eps = original.variance_epsilon
        self.group_size = original.group_size

    def forward(self, x, gate):
        return gated_forward(x, gate, self.weight, self.group_size, self.eps)


def install_inference_norms(model):
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("Shared Nemotron normalization currently requires TP=1")
    for name, module in list(model.named_modules()):
        if isinstance(module, RMSNorm):
            replacement = SharedRMSNorm(module)
        elif isinstance(module, Mixer2RMSNormGated):
            replacement = SharedGatedRMSNorm(module)
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, replacement)
