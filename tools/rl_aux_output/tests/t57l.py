# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57k compaction workload (OC_COMPACT=1), pure logic from the driver's own functions:
continuation prompt construction, trajectory ids per segment, the segment/summary state
machine, the byte estimate, block sharing count and the shifted-row check. Run: python
-I t57l.py [driver.py]"""

import ast
import os
import pathlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "x10_driver.py")
SRC = pathlib.Path(DRIVER).read_text(encoding="utf8")
NAMES = (
    "track_of",
    "compact_prompt",
    "compact_sess_bytes",
    "compact_init",
    "compact_advance",
    "compact_archive",
    "compact_new_segment",
    "compact_shared_blocks",
    "compact_prev_served",
    "gen_len",
)
US = chr(0x1F)


def load(**env):
    ns = dict(
        np=np,
        TRAJ_TAG=True,
        GROUP=1,
        COMPACT_AT=8192,
        COMPACT_N=2,
        COMPACT_MAX_TURNS=64,
        SUMMARY_TOK=256,
        GEN_DIST="",
        MAXTOK=512,
    )
    ns.update(env)
    tree = ast.parse(SRC)
    got = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            exec(compile(ast.Module([node], []), DRIVER, "exec"), ns)
            got.add(node.name)
    missing = set(NAMES) - got
    assert not missing, f"driver lacks {sorted(missing)}"
    return ns


def t_prompt_and_tracks():
    ns = load()
    sys_ids, summ = [1] * 4, [9, 9]
    chunks = [[2, 2], [3, 3, 3], [4], [5, 5]]
    assert ns["compact_prompt"](sys_ids, summ, chunks) == [1, 1, 1, 1, 9, 9, 4, 5, 5]
    assert ns["compact_prompt"](sys_ids, summ, [[7]]) == [1, 1, 1, 1, 9, 9, 7]
    seg = dict(name="ps-n-p0-L8192-full-3", traj=2, compact=True)
    assert ns["track_of"](seg) == "ps-n-p0-L8192-full-3" + US + "2"
    assert (
        ns["track_of"](dict(name="x", traj=1)) == "x"
    )  # ungrouped, not compaction: the session itself
    print(
        "continuation prompt = system prompt + summary + last 2 turns; segment k is "
        "trajectory k"
    )


def t_state_machine():
    ns = load(COMPACT_AT=100, COMPACT_N=2)
    sys_ids, task, summarize = list(range(1000, 1020)), [7] * 10, [5, 5]
    s = dict(name="sess", i=0, g=0, v0=0, reqs=[], rows=None, turn=0)
    ns["compact_init"](s, sys_ids, task)
    log, summaries = [], 0
    for step in range(200):
        assert ns["gen_len"](s) == (256 if s["mode"] == "summary" else 512)
        s["reqs"].append((f"r{step}", len(s["ids"])))  # the engine ran one request
        gen = [3] * (8 if s["mode"] == "summary" else 6)
        obs = [4] * 14 if s["mode"] == "turns" else []
        act = ns["compact_advance"](s, gen, obs, summarize)
        s["turn"] += 1
        log.append(act)
        if act == "new_segment":
            prev_chunks = s["chunks"]
            summary = s["summary"]
            ns["compact_new_segment"](s, sys_ids)
            summaries += 1
            assert s["ids"] == sys_ids + summary + [
                t for c in prev_chunks[-2:] for t in c
            ]
            assert (
                s["traj"] == summaries
                and s["reqs"] == []
                and s["rows"] is None
                and s["mode"] == "turns"
            )
        elif act == "done":
            ns["compact_archive"](s)
            break
    segs = s["segments"]
    assert (
        act == "done" and summaries == 2 and [g["traj"] for g in segs] == [0, 1, 2]
    ), (log, summaries)
    # Every compacted segment ends with its summary request (its prompt ends in the
    # summarize instruction; the generated summary opens the next segment).
    for g in segs[:-1]:
        assert g["ids"][-2:] == summarize and len(g["reqs"]) >= 2
    assert log.count("summary") == 2 and log.count("new_segment") == 2
    print(
        f"state machine: {len(segs)} segments (trajectories 0..2), 2 summaries, "
        f"{len(log)} requests"
    )


def t_byte_estimate():
    ns = load()
    est = ns["compact_sess_bytes"](8192, 2, 480, False, 512, 1024)
    assert est == 3 * 8192 * 480 * 1.1
    est_x = ns["compact_sess_bytes"](8192, 2, 480 + 5240, True, 512, 1024)
    assert est_x == 3 * 8192 * 5720 * 1.1 + 3 * (8192 // 1536 + 2) * 512 * 5124
    print(
        "byte estimate: segments x threshold x bytes/position (+ hidden rows), per "
        "rollout"
    )


def t_sharing_and_shift():
    ns = load()
    m = [
        dict(blocks=["s0", "s1", "a2", "a3"]),
        dict(blocks=["s0", "s1", "b2"]),
        dict(blocks=["s0", "s1", "c2", "c3"]),
    ]
    assert ns["compact_shared_blocks"](m) == (
        11,
        7,
    )  # 4 shared = 2 system blocks x 2 later segments
    rng = np.random.default_rng(0)
    prev = rng.integers(0, 100, (20, 3, 2)).astype(np.uint16)
    own = rng.integers(0, 100, (20, 3, 2)).astype(np.uint16)
    own[:8] = prev[:8]  # the shared system prompt (prefix hit)
    got = own.copy()
    assert ns["compact_prev_served"](got, own, prev, 8) == 0
    got[10:13] = prev[10:13]  # three continuation rows served from the previous segment
    assert ns["compact_prev_served"](got, own, prev, 8) == 3
    print(
        "sharing: listed vs unique block objects; continuation rows served from the "
        "previous segment counted"
    )


def _has(frag, src):
    return "".join(frag.split()) in "".join(src.split())


def t_driver_wiring():
    assert _has(
        '**({"oc_traj": str(s["traj"])} if (TRAJ_TAG and GROUP > 1) or COMPACT else '
        "{}),",
        SRC,
    )
    assert "            track = track_of(s)" in SRC
    loop = SRC[SRC.index("        for s in done:") :]
    loop = loop[: loop.index('ph["read_s"]')]
    assert _has('segs = s["segments"] if COMPACT else [s]', loop) and _has(
        "check_request(seg, rec, ph", loop
    )
    assert _has("compact_check(s, ph)", loop)
    adm = SRC[SRC.index("    if COMPACT:\n        # One rollout") :]
    assert _has("compact_sess_bytes(COMPACT_AT, COMPACT_N", adm[:300])
    print(
        "driver: compaction segments tagged per trajectory, each checked, bytes "
        "estimated per rollout"
    )


if __name__ == "__main__":
    t_prompt_and_tracks()
    t_state_machine()
    t_byte_estimate()
    t_sharing_and_shift()
    t_driver_wiring()
    print("all t57l tests passed")
