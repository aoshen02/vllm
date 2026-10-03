# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming launch schedule for Nemotron-H Lightning routed experts.

With VLLM_BATCH_INVARIANT=1, Humming's indexed W4A16 heuristics leave the
Lightning EP4 expert GEMMs (N tile 256) at 3 stages with 64-wide warps. Four
stages, and 32-wide warps below 448 rows, cut BI=1 TPOT by 5-12% at
concurrency 16/64 on GB200 (8K in / 1K out, TP1/DP4/EP4). Only launch geometry
changes: K tiles stay 32 and accumulation stays FP32, so outputs are bitwise
unchanged (tests/kernels/moe/test_moe.py). This belongs in Humming's heuristics
eventually.
"""

import copy

from vllm import envs

SCHEDULE_KEY = "nemotron_h_humming_schedule"

# The configuration the schedule was measured on. Local experts follow from
# n_routed_experts / EP size.
_MEASURED = {
    "hidden_size": 2688,
    "moe_intermediate_size": 1856,
    "num_local_experts": 32,
    "num_experts_per_tok": 6,
}


def schedule_unsupported_reason(vllm_config) -> str | None:
    """Why the measured schedule does not apply to this engine configuration."""
    hf_config = vllm_config.model_config.hf_config
    parallel = vllm_config.parallel_config
    if not envs.VLLM_BATCH_INVARIANT:
        return "batch invariance is off"
    if vllm_config.kernel_config.moe_backend != "humming":
        return "the MoE backend is not Humming"
    if envs.VLLM_HUMMING_MOE_GEMM_TYPE != "indexed":
        return "the Humming GEMM type is not indexed"
    if parallel.tensor_parallel_size != 1 or not parallel.enable_expert_parallel:
        return "it needs TP1 with expert parallelism"
    ep_size = parallel.data_parallel_size
    actual = {
        "hidden_size": getattr(hf_config, "hidden_size", None),
        "moe_intermediate_size": getattr(hf_config, "moe_intermediate_size", None),
        "num_local_experts": getattr(hf_config, "n_routed_experts", 0) // ep_size,
        "num_experts_per_tok": getattr(hf_config, "num_experts_per_tok", None),
    }
    if actual != _MEASURED:
        return f"it was measured for {_MEASURED}, not {actual}"
    return None


def nemotron_humming_schedule(entries: list) -> list:
    """Change launch geometry without changing K32 or FP32 accumulation."""
    if not entries:
        raise ValueError("Missing Humming indexed tuning intervals")
    candidate = copy.deepcopy(entries)
    changed = False
    for lower, upper, config in candidate:
        block, warp = config["block_shape"], config["warp_shape"]
        if (
            not lower < upper
            or len(block) != 3
            or len(warp) != 3
            or block[2] != 32
            or warp[2] != 32
            or config.get("use_stream_k") is not False
            or config.get("mma_type") != "mma"
            or config.get("use_f16_accum") is not False
        ):
            raise ValueError("Unsupported Nemotron Humming reduction recipe")
        if block[1] == 256:
            if config.get("num_stages") != 3 or warp[1] != 64:
                raise ValueError("Unsupported Nemotron Humming N256 schedule")
            config["num_stages"] = 4
            if lower < 448:
                updated = list(warp)
                updated[1] = 32
                config["warp_shape"] = type(warp)(updated)
            changed = True
        elif block[1] not in (64, 128):
            raise ValueError("Unsupported Nemotron Humming tile")
    if not changed:
        raise ValueError("Missing Nemotron Humming N256 intervals")
    return candidate
