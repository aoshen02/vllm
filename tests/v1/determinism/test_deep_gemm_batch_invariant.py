# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance wiring of the DeepGEMM MoE path (no GPU required).

The kernel-level guarantees live in the fork (set_batch_invariant) and are
probed on hardware; these tests pin the vLLM-side wiring: fail-closed
capability declaration, M-independent implementation selection, and the
alignment ladder staying pinned under VLLM_BATCH_INVARIANT.
"""

import pytest

import vllm.utils.deep_gemm as dg_wrapper
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    _bi_use_deep_gemm,
)


def test_supports_batch_invariance_fails_closed(monkeypatch):
    """The declaration must be exactly the wrapper's capability probe: no
    set_batch_invariant API (or enabling failed) means no support."""
    monkeypatch.setattr(dg_wrapper, "deep_gemm_batch_invariant_enabled", lambda: False)
    assert DeepGemmExperts._supports_batch_invariance() is False
    monkeypatch.setattr(dg_wrapper, "deep_gemm_batch_invariant_enabled", lambda: True)
    assert DeepGemmExperts._supports_batch_invariance() is True


def test_bi_selection_ignores_batch(monkeypatch):
    """Under BI the Triton-or-DeepGemm pick comes from weight-shape constants
    only — the helper takes no M at all, unlike the default align <= M gate."""
    import vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe as m

    monkeypatch.setattr(m, "get_mk_alignment_for_contiguous_layout", lambda: [128, 128])
    monkeypatch.setattr(m, "has_deep_gemm", lambda: True)
    assert _bi_use_deep_gemm(N=1024, K=4096) is True
    assert _bi_use_deep_gemm(N=1000, K=4096) is False  # N unaligned
    assert _bi_use_deep_gemm(N=512, K=4096) is False  # small-N carve-out
    monkeypatch.setattr(m, "has_deep_gemm", lambda: False)
    assert _bi_use_deep_gemm(N=1024, K=4096) is False


@pytest.mark.parametrize("m_tokens", [1, 4, 64])
def test_alignment_ladder_pinned_under_bi(monkeypatch, m_tokens):
    """The theoretical 224->32 shrink makes grouped block_m a function of the
    batch; under BI the caller's alignment must come back unchanged."""
    shrink_called = []
    monkeypatch.setattr(
        dg_wrapper,
        "get_theoretical_mk_alignment_for_contiguous_layout",
        lambda **kw: shrink_called.append(kw) or 32,
    )
    m_sum, align = compute_aligned_M_and_alignment(
        M=m_tokens,
        num_topk=6,
        local_num_experts=8,
        alignment=128,
        expert_tokens_meta=None,
    )
    assert align == 128, "alignment must not shrink with the batch under BI"
    assert not shrink_called
    assert m_sum % 128 == 0
