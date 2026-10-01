# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed indexed W4A16 schedules for Nemotron Lightning on GB200."""

import copy
import json


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


def configure_nemotron_humming(experts) -> None:
    """Set both serialized schedules once, before graph capture or execution."""
    geometry = {"w13": (2048, 2688, 192, 0), "w2": (2816, 1920, 128, 64)}
    for name, expected in geometry.items():
        config = experts.humming_configs[name]
        actual = tuple(
            getattr(config, key)
            for key in ("shape_n", "shape_k", "pad_shape_n", "pad_shape_k")
        )
        if actual != expected:
            raise ValueError("Unsupported Nemotron packed W4A16 geometry")
    if (
        experts.num_experts != 32
        or experts.global_num_experts != 128
        or experts.compute_config["use_batch_invariant"] is not True
        or experts.compute_config["use_f16_accum"] is not False
    ):
        raise ValueError("Nemotron indexed schedule requires BI1/EP4/FP32")
    tables = {
        name: nemotron_humming_schedule(getattr(experts, name + "_tuning_config"))
        for name in geometry
    }
    for name, table in tables.items():
        setattr(experts, name + "_tuning_config", table)
        setattr(experts, name + "_tuning_config_str", json.dumps(table))
