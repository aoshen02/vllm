"""Humming stage4 with warp-N32 only for small-token intervals."""

import copy


def hybrid_config(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Require actual runtime interval table")
    candidate = copy.deepcopy(entries)
    changed = 0
    for old, new in zip(entries, candidate, strict=True):
        lower, upper, config = new
        block, warp = config["block_shape"], config["warp_shape"]
        if not lower < upper or len(block) != 3 or len(warp) != 3:
            raise ValueError("Invalid interval geometry")
        if block[2] != 32 or warp[2] != 32:
            raise ValueError("Require original K32 reduction")
        if config.get("use_stream_k") is not False:
            raise ValueError("Require stream-K off")
        if config.get("mma_type") != "mma":
            raise ValueError("Require original MMA path")
        if config.get("use_f16_accum") is not False:
            raise ValueError("Require FP32 accumulation")
        if block[1] == 256:
            if config.get("num_stages") != 3 or warp[1] != 64:
                raise ValueError("Unexpected original N256 schedule")
            config["num_stages"] = 4
            if lower < 448:
                updated = list(warp)
                updated[1] = 32
                config["warp_shape"] = type(warp)(updated)
            changed += 1
        elif block[1] not in (64, 128):
            raise ValueError("Unknown original block N")
        restored = copy.deepcopy(new)
        restored[2]["num_stages"] = old[2]["num_stages"]
        restored[2]["warp_shape"] = copy.deepcopy(old[2]["warp_shape"])
        if restored != old:
            raise ValueError("Non-stage/warp configuration drift")
    if not changed:
        raise ValueError("No N256 intervals to optimize")
    return candidate
