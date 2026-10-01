"""Strict private-instance N-only tuning transform and synthetic EP4 geometry."""

import copy
import dataclasses
import enum
import json


def config_record(config):
    """Capture every dataclass field, not just shape or MMA labels."""
    fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    return json.loads(
        json.dumps(
            fields,
            default=lambda value: (
                value.value if isinstance(value, enum.Enum) else str(value)
            ),
        )
    )


def resolved_analysis_layer(layer_config, tuning):
    """Mirror the execution merge on an analysis-only copy, never packed state."""
    chosen = tuning.get("mma_type")
    if chosen != "mma":
        raise ValueError("Require explicit actual chosen MMA in both arms")
    before = config_record(layer_config)
    resolved = dataclasses.replace(
        layer_config, mma_type=type(layer_config.mma_type)(chosen)
    )
    after = config_record(resolved)
    changed = [name for name in before if before[name] != after[name]]
    if set(changed) - {"mma_type"}:
        raise ValueError(f"Analysis resolution altered packed fields: {changed}")
    if config_record(layer_config) != before:
        raise ValueError("Analysis mutated the actual packed configuration")
    return resolved, {
        "base": before,
        "resolved": after,
        "changed_fields": changed,
        "chosen_mma": chosen,
    }


def separate_launch_metadata(tuning, max_sms):
    """Mirror HummingKernel's num_sms pop and ComputeConfig's FP32 contract."""
    values = copy.deepcopy(tuning)
    accumulation = values.pop("use_f16_accum", False)
    if accumulation is not False:
        raise ValueError("Require unchanged FP32 accumulator")
    num_sms = values.pop("num_sms", max_sms)
    if type(num_sms) is not int or not 0 < num_sms <= max_sms:
        raise ValueError("Invalid launch SM budget")
    return values, {"use_f16_accum": accumulation, "num_sms": num_sms}


def analysis_shapes(values):
    """Normalize the official analysis API's tuple inputs without editing launch."""
    copied = copy.deepcopy(values)
    for name in ("block_shape", "warp_shape"):
        copied[name] = tuple(copied[name])
    return copied


def analyze_schedule(layer_config, tuning, shape_m, device):
    """Use installed Humming's full admission; never mutate the launch config."""
    from humming.config import GemmType, TuningConfig
    from humming.tune.candidate import (
        ScheduleCandidate,
        TuningProblem,
        analyze_candidate,
    )

    values, metadata = separate_launch_metadata(tuning, device.num_sms)
    resolved_layer, layer_record = resolved_analysis_layer(layer_config, tuning)
    values.pop("mma_type")
    # Same TuningConfig default, resolved from recorded hardware for no-CUDA tests.
    if values.get("use_cp_async") is None:
        values["use_cp_async"] = device.sm_version >= 80
    resolved = TuningConfig(**values)
    # The official estimator assumes these values; reject other execution modes.
    if (
        resolved.reduce_overlap_last_stage_only
        or resolved.num_write_splits != 1
        or resolved.multi_cast_size_b != 1
    ):
        raise ValueError("Unsupported resource-estimator execution mode")
    fields = {
        f.name
        for f in dataclasses.fields(ScheduleCandidate)
        if not f.name.startswith("_") and f.name != "candidate_id"
    }
    schedule = ScheduleCandidate.from_config(
        "literal-n-only",
        analysis_shapes(
            {
                key: False if key == "use_f16_accum" else getattr(resolved, key)
                for key in fields
            }
        ),
    )
    problem = TuningProblem(
        layer_config=resolved_layer,
        shape_m=shape_m,
        gemm_type=GemmType.INDEXED,
        device=device,
        use_f16_accum=False,
        use_batch_invariant=True,
    )
    analysis = analyze_candidate(problem, schedule)
    result = dataclasses.asdict(analysis)
    result["candidate"].pop("_explicit_fields", None)
    result.update(
        launchable=analysis.launchable,
        meets_resource_target=analysis.meets_resource_target,
        actual_resolved_threads=resolved.num_threads,
        launch_metadata=metadata,
        layer_config=layer_record,
        resolved_tuning=dataclasses.asdict(resolved),
    )
    if resolved.num_threads != analysis.num_threads:
        raise ValueError("Thread calculation differs from actual TuningConfig")
    return result


def n128_config(entries):
    candidate = copy.deepcopy(entries)
    if not isinstance(candidate, list) or not candidate:
        raise ValueError("Require actual runtime tuning interval list")
    changed = 0
    for original, entry in zip(entries, candidate, strict=True):
        lower, upper, cfg = entry
        if not lower < upper or cfg.get("use_stream_k") is not False:
            raise ValueError("Require ordered intervals and explicit stream-K off")
        block = cfg["block_shape"]
        if len(block) != 3 or block[2] != 32 or block[1] not in (64, 128, 256):
            raise ValueError("Only admitted K32/N64-or128-or256 configurations")
        if cfg.get("mma_type", "mma") != "mma":
            raise ValueError("Do not change MMA family")
        updated = list(block)
        updated[1] = 128 if block[1] == 256 else block[1]
        cfg["block_shape"] = type(block)(updated)
        changed += block[1] == 256
        restored = copy.deepcopy(entry)
        restored[2]["block_shape"] = copy.deepcopy(original[2]["block_shape"])
        if restored != original:
            raise ValueError("Non-N configuration drift")
    if not changed:
        raise ValueError("No N change; not a candidate comparison")
    return candidate


def selected_schedule(entries, shape_m):
    """Match the actual frozen indexed expert's lower-exclusive selection."""
    selected = [entry for entry in entries if entry[0] < shape_m <= entry[1]]
    if len(selected) != 1:
        raise ValueError("Missing or ambiguous measured-shape tuning interval")
    return copy.deepcopy(selected[0])


def route_fixture(capacity, valid, owner):
    """CPU synthetic capacity metadata; never labeled real post-dispatch data."""
    import torch

    if not 0 <= valid <= capacity or owner not in range(4):
        raise ValueError("Invalid local capacity/ownership")
    mapping = torch.full((128,), -1, dtype=torch.int32)
    mapping[owner * 32 : (owner + 1) * 32] = torch.arange(32, dtype=torch.int32)
    pattern = torch.tensor([0, 32, 64, 96, 1, 33], dtype=torch.int32)
    ids = pattern.repeat(capacity, 1)
    # Retain the same row0 while other rows change local expert loads.
    if valid > 1:
        ids[1:valid] = (ids[1:valid] + torch.arange(1, valid)[:, None] * 7) % 128
    ids[valid:] = -1
    scores = torch.full((capacity, 6), 1 / 6, dtype=torch.float32)
    scores[valid:] = float("nan")
    return ids, scores, mapping
