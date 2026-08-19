# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance wiring of the DeepGEMM MoE path (no GPU required).

The kernel-level guarantees live in the fork (set_batch_invariant) and are
probed on hardware; these tests pin the vLLM-side wiring: fail-closed
capability declaration, M-independent implementation selection, and the
alignment ladder staying pinned under VLLM_BATCH_INVARIANT.
"""

import inspect
import types

import pytest

import vllm.envs as envs
import vllm.utils.deep_gemm as dg_wrapper
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
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


def test_probe_fails_closed_on_every_way_the_api_can_be_missing():
    """The probe decides whether DeepGEMM is offered under batch invariance, so
    every way it can fail has to leave it False.

    Structural, not behavioural: the enabling block cannot be driven without a
    vendored deep_gemm build. What is asserted is that the flag is only ever set
    on the success path -- looked up with getattr, set inside a try, and never
    assigned in the except or the missing-API branch.
    """
    import vllm.utils.deep_gemm as dg

    body = inspect.getsource(dg._lazy_init)
    block = body[body.index("if envs.VLLM_BATCH_INVARIANT") :]
    assert 'getattr(_dg, "set_batch_invariant", None)' in block
    assignments = [
        line for line in block.splitlines() if "_batch_invariant_enabled = " in line
    ]
    assert assignments == ["                _batch_invariant_enabled = True"], (
        f"the flag must be set only on the success path, found: {assignments}"
    )
    assert dg.deep_gemm_batch_invariant_enabled() is dg._batch_invariant_enabled


def test_flag_is_set_before_any_impl_global():
    """A thread taking the fast path must never read the flag as False.

    ``_lazy_init``'s fast path returns as soon as *any* impl global is not None,
    so a second thread can leave with the batch-invariance flag already read.
    That is safe only while the flag is assigned before every impl global -- and
    since the flag decides whether DeepGEMM is offered under batch invariance,
    two ranks reading it differently would pick different MoE backends for the
    same configuration.

    Asserted on the source order rather than by racing threads: the enabling
    block cannot be driven without a vendored deep_gemm build, and a race test
    that cannot fail is worse than none.
    """
    import vllm.utils.deep_gemm as dg

    lines = inspect.getsource(dg._lazy_init).splitlines()
    flag_at = next(
        i for i, ln in enumerate(lines) if "_batch_invariant_enabled = True" in ln
    )
    impl_assignments = [
        i
        for i, ln in enumerate(lines)
        if ln.startswith("    _") and "_impl = getattr(" in ln
    ]
    assert impl_assignments, "no impl-global assignments found; test is vacuous"
    assert flag_at < min(impl_assignments), (
        "_batch_invariant_enabled must be assigned before the first impl global"
    )


def test_bi_workspace_matches_the_implementation_it_will_use(monkeypatch):
    """Workspace sizing must ask the same question the dispatch does.

    The two entry points do not speak the same dimension: workspace_shapes gets
    w1's output dim, _select_experts_impl reads w2's, which for a gated
    activation is half of it. Comparing them for consistent w1/w2 is the point
    -- checking either alone is what let them disagree.
    """
    import torch

    import vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe as m

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(m, "get_mk_alignment_for_contiguous_layout", lambda: [128, 128])
    monkeypatch.setattr(m, "has_deep_gemm", lambda: True)

    cls = m.TritonOrDeepGemmExperts
    K, E = 4096, 8
    for w2_n in (512, 1024):
        sized: list[str] = []

        def _record(tag, _sized=sized):
            return lambda *a, **k: (_sized.append(tag), ((), (), ()))[1]

        experts = types.SimpleNamespace(workspace_shapes=_record("deep_gemm"))
        fallback = types.SimpleNamespace(workspace_shapes=_record("triton"))
        obj = types.SimpleNamespace(
            experts=experts,
            fallback_experts=fallback,
            adjust_N_for_activation=cls.adjust_N_for_activation,
        )
        # Gated activation: w1's output dim is twice w2's.
        w1 = torch.empty((E, 2 * w2_n, K))
        w2 = torch.empty((E, K, w2_n))
        cls.workspace_shapes(
            obj,
            M=128,
            N=w1.shape[1],
            K=K,
            topk=6,
            global_num_experts=E,
            local_num_experts=E,
            expert_tokens_meta=None,
            activation=MoEActivation.SILU,
        )
        chosen = cls._select_experts_impl(obj, torch.empty((128, K)), w1, w2)
        dispatched = "deep_gemm" if chosen is experts else "triton"
        assert sized == [dispatched], (
            f"w2_n={w2_n}: sized for {sized}, dispatch takes {dispatched}"
        )


def test_o_proj_refuses_a_deep_gemm_that_cannot_be_pinned(monkeypatch):
    """The DSv4 output projection reaches DeepGEMM without asking the probe.

    `deep_gemm_fp8_o_proj` calls `fp8_einsum` directly -- it never goes through
    MoE backend selection, so "DeepGEMM MoE disabled" leaves it running with its
    config chosen from the batch, and it has no other implementation to fall
    back to.
    """
    import vllm.models.deepseek_v4.nvidia.ops.o_proj as op

    guard = op._require_batch_invariant_deep_gemm
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    # o_proj binds the probe at import, so patch the name it actually reads.
    monkeypatch.setattr(op, "deep_gemm_batch_invariant_enabled", lambda: True)
    guard.cache_clear()
    guard()  # pinned: no complaint

    monkeypatch.setattr(op, "deep_gemm_batch_invariant_enabled", lambda: False)
    guard.cache_clear()
    with pytest.raises(RuntimeError, match="set_batch_invariant"):
        guard()

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    guard.cache_clear()
    guard()
    guard.cache_clear()
