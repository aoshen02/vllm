"""Private stage-only transform, preserving K32 and every original warp tile."""

import copy


def stage4_config(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Require actual runtime interval table")
    candidate = copy.deepcopy(entries)
    changed = 0
    for old, new in zip(entries, candidate, strict=True):
        lower, upper, config = new
        block, warp = config["block_shape"], config["warp_shape"]
        if not lower < upper or len(block) != 3 or len(warp) != 3:
            raise ValueError("Invalid interval/geometry")
        if block[2] != 32 or warp[2] != 32 or config.get("use_stream_k") is not False:
            raise ValueError("Require original block/warp K32 and stream-K off")
        if config.get("mma_type") != "mma" or config.get("use_f16_accum") is not False:
            raise ValueError("Require original chosen MMA/FP32 accumulation")
        if block[1] == 256:
            if config.get("num_stages") != 3:
                raise ValueError("Expected original stage3")
            config["num_stages"] = 4
            changed += 1
        elif block[1] not in (64, 128):
            raise ValueError("Unknown original block N")
        restored = copy.deepcopy(new)
        restored[2]["num_stages"] = old[2]["num_stages"]
        if restored != old:
            raise ValueError("Non-stage configuration drift")
    if not changed:
        raise ValueError("No original N256 intervals to compare")
    return candidate
