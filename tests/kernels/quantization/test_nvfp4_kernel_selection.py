# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 linear kernel selection order (CPU-only)."""

import pytest

import vllm.model_executor.kernels.linear as linear_kernels
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    CutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.platforms.interface import PlatformEnum

# W4A4 kernels that run on SM120/121, where the head of the list is gated to
# sm_10x and selection falls through to whatever follows.
W4A4_KERNELS_ON_SM12X = (
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferB12xNvFp4LinearKernel,
    CutlassNvFp4LinearKernel,
)


@pytest.mark.parametrize("w4a4_kernel", W4A4_KERNELS_ON_SM12X)
def test_w4a16_kernel_does_not_precede_w4a4_kernels(w4a4_kernel):
    candidates = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
    w4a16_index = candidates.index(FlashInferCuteDslNvFp4W4A16LinearKernel)
    assert candidates.index(w4a4_kernel) < w4a16_index, (
        f"{w4a4_kernel.__name__} must be preferred over "
        f"{FlashInferCuteDslNvFp4W4A16LinearKernel.__name__}"
    )


@pytest.mark.parametrize("supported", [True, False])
def test_bi_w4a16_keeps_a16_or_fails_closed(monkeypatch, supported):
    """BI must not select a W4A4 kernel for weight-only NVFP4."""
    monkeypatch.setattr(linear_kernels.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda **_: "auto")
    monkeypatch.setattr(
        linear_kernels.HummingNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls: (supported, None if supported else "unavailable")),
    )
    if supported:
        kernel = linear_kernels.init_nvfp4_linear_kernel(use_a16=True)
        assert isinstance(kernel, linear_kernels.HummingNvFp4LinearKernel)
    else:
        with pytest.raises(ValueError, match="unavailable"):
            linear_kernels.init_nvfp4_linear_kernel(use_a16=True)
