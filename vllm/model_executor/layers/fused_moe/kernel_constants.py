# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep per-expert kernel constants at fixed addresses across weight reloads."""

import torch

_ATTRS = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")


def keep_kernel_constants(layer: torch.nn.Module, experts: object) -> None:
    """Pin the experts' SwiGLU constants to buffers on ``layer``.

    The experts object is rebuilt on every ``process_weights_after_loading``
    and allocates these tensors in its constructor, but captured CUDA graphs
    keep reading them at their first address. The first call registers them
    as non-persistent buffers; later calls copy the fresh values in place and
    hand the original tensors back to the new experts object.
    """
    for attr in _ATTRS:
        fresh = getattr(experts, attr, None)
        if not isinstance(fresh, torch.Tensor):
            continue
        key = f"_kept_{attr}"
        kept = layer._buffers.get(key)
        if (
            kept is not None
            and kept.shape == fresh.shape
            and kept.dtype == fresh.dtype
            and kept.device == fresh.device
        ):
            kept.copy_(fresh)
        else:
            layer.register_buffer(key, fresh, persistent=False)
            kept = fresh
        setattr(experts, attr, kept)
