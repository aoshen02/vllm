# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatchers that pick a cudagraph per step, outside cudagraph_mode's reach.

Microbatching schedules graphs of its own and never consults cudagraph_mode, so
neither a mode choice nor --enforce-eager switches it off. That makes it a
numeric path selected from whatever else the step happens to carry, which batch
invariance has to refuse outright. This is a unit test over the decision point
itself -- no weights, no GPU.
"""

import pytest

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode


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
