# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57j: the warmup session is released only after its requests' teardown wrote their
manifests (engine stepped, manifests read); a bounded leak probe lists surviving keys of
known sessions before the job-end sweep. Run: python -I t57j.py [driver.py]  (uses
t57f's fakes)"""

import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t57f as base  # noqa: E402
from t57f import driver_ns, oc, run_with_timeout  # noqa: E402

SRC = base.SRC


def t_warmup_released_after_teardown():
    blk = SRC[
        SRC.index("# ---------------- warmup (untimed) ----------------") : SRC.index(
            'say("warmup done")'
        )
    ]
    i_gen = blk.index("_warm = llm.generate(")
    i_step = blk.index("SamplingParams(max_tokens=1)")
    i_man = blk.index("get_manifest(WARM, o.request_id")
    i_rel = blk.index("release(WARM)")
    assert i_gen < i_step < i_man < i_rel, (i_gen, i_step, i_man, i_rel)
    print(
        "warmup: engine stepped and every warmup manifest read before the warmup "
        "release"
    )


def t_leak_probe():
    live = set()

    class Reader:
        def is_exist(self, k):
            return int(k in live)

    ns = driver_ns(["leak_probe", "SESSION_RIDS", "KEY_HINTS"], reader=Reader())
    ns["SESSION_RIDS"].update(
        {
            "ps-n-p0-L1-full-0": ["r1", "r2"],
            "ps-n-warm": ["w0", "w1"],
            "probe-n-p0-v0a": ["q"],
        }
    )
    ns["KEY_HINTS"]["probe-n-p0-v0a"] = [oc.block_key("probe-n-p0-v0a", "b0")]
    live.update(
        {
            oc.manifest_key("ps-n-warm", "w1"),
            oc.tail_key("ps-n-warm", "w0"),
            oc.block_key("probe-n-p0-v0a", "b0"),
            oc.cone_key("ps-n-p0-L1-full-0", "r2", "idx"),
        }
    )
    r = ns["leak_probe"]()
    assert r["complete"] and set(r["sample"]) == live, r
    assert (
        r["sample"][0].startswith(f"{oc.PREFIX}/ps-n-warm/")
        or "probe-" in r["sample"][0]
    ), r["sample"][0]
    # Bounded: a stuck store does not hold the job end.
    block = threading.Event()
    ns2 = driver_ns(
        ["leak_probe", "SESSION_RIDS", "KEY_HINTS"],
        reader=types.SimpleNamespace(is_exist=lambda k: block.wait() or 0),
    )
    ns2["SESSION_RIDS"]["x"] = ["r"]
    os.environ["OC_LEAK_PROBE_S"] = "0.2"
    try:
        hung, r = run_with_timeout(ns2["leak_probe"], 3)
    finally:
        os.environ.pop("OC_LEAK_PROBE_S")
        block.set()
    assert not hung and r["complete"] is False, r
    fin = SRC[
        SRC.index("def finish(deadlock=False):") : SRC.index("def _rank_dump_compare")
    ]
    assert fin.index('"leaked_sample"') < fin.index('_step("sweep", sweep)')
    print(
        "leak probe: surviving keys of known sessions sampled before the job-end "
        "sweep (bounded)"
    )


if __name__ == "__main__":
    t_warmup_released_after_teardown()
    t_leak_probe()
    print("all t57j tests passed")
