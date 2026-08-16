# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatchers that pick a cudagraph per step, outside cudagraph_mode's reach.

Pinning cudagraph_mode removes the main runner's per-step choice, but several
components schedule graphs of their own and never consult it. Each one is a
numeric path selected from whatever else the step happens to carry, so batch
invariance has to switch them off too. These are unit tests over the decision
points -- no weights, no GPU.
"""

import types

import pytest

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode


@pytest.mark.parametrize("supports_capability", [False, True])
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_encoder_cudagraph_manager_disabled_only_where_it_would_exist(
    monkeypatch, supports_capability, batch_invariant
):
    """The encoder manager is disabled under batch invariance -- but only for
    models that would actually have got one.

    The capability is not knowable at config time, so the decision lives here.
    Ordering it before the capability check would warn at deployments where the
    flag was inert.
    """
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        gmr, "supports_encoder_cudagraph", lambda _m: supports_capability, raising=False
    )
    import vllm.model_executor.models.interfaces as interfaces

    monkeypatch.setattr(
        interfaces, "supports_encoder_cudagraph", lambda _m: supports_capability
    )

    import vllm.v1.worker.encoder_cudagraph as enc

    sentinel_manager = object()
    monkeypatch.setattr(enc, "EncoderCudaGraphManager", lambda **_kw: sentinel_manager)

    warned = []
    monkeypatch.setattr(
        gmr.logger, "warning_once", lambda msg, *a, **k: warned.append(msg)
    )

    runner = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        get_model=lambda: object(),
        vllm_config=None,
        device=None,
        dtype=None,
    )
    manager = gmr.GPUModelRunner._create_encoder_cudagraph_manager(runner)

    if not supports_capability:
        assert manager is None
        assert not warned, "a model without the capability must not be told anything"
    elif batch_invariant:
        assert manager is None
        assert any("VLLM_BATCH_INVARIANT" in m for m in warned)
    else:
        assert manager is sentinel_manager
        assert not warned


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_gemma4_centroid_graphs_skipped_under_batch_invariance(
    monkeypatch, batch_invariant
):
    """Gemma4's proposer captures centroid graphs at fixed sizes and replays
    whichever fits the step's token count, eager above the largest -- its own
    per-step choice, reached through neither cudagraph_mode nor the proposer
    dispatcher.

    Skipping the capture must not change *what* is computed: with no graphs, the
    same centroid top-k has to run eagerly rather than falling through to the
    base class's full-vocab argmax.
    """
    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    captured = []
    sentinel = object()

    proposer = types.SimpleNamespace(
        model=types.SimpleNamespace(
            masked_embedding=object(),
            get_top_tokens=lambda _h: sentinel,
        ),
        _centroids_sizes=[],
        _uses_centroids=getattr,  # replaced below
        _setup_centroids_cuda_graphs=lambda: captured.append(True),
    )
    proposer._uses_centroids = (
        getattr(proposer.model, "masked_embedding", None) is not None
    )
    if proposer._uses_centroids and not envs.VLLM_BATCH_INVARIANT:
        proposer._setup_centroids_cuda_graphs()

    assert proposer._uses_centroids
    assert bool(captured) is (not batch_invariant)

    # With no graphs captured, the eager centroid path must still be the one
    # taken -- falling through to the base class would swap in a full-vocab
    # argmax over a different code path.
    if batch_invariant:
        assert Gemma4Proposer._greedy_sample(proposer, hidden_states=None) is sentinel


def test_microbatching_is_refused_regardless_of_cudagraph_mode(monkeypatch):
    """Microbatching decides per step whether to split and which graph to
    replay, and never consults cudagraph_mode -- so the refusal must not be
    gated on it either."""
    from vllm.config import (
        CompilationConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    for mode in (CUDAGraphMode.FULL, CUDAGraphMode.NONE):
        with pytest.raises(ValueError, match="microbatching"):
            VllmConfig(
                model_config=ModelConfig("Qwen/Qwen3-0.6B", max_model_len=2048),
                scheduler_config=SchedulerConfig(
                    max_model_len=2048, is_encoder_decoder=False, max_num_seqs=64
                ),
                parallel_config=ParallelConfig(enable_dbo=True),
                compilation_config=CompilationConfig(cudagraph_mode=mode),
            )
