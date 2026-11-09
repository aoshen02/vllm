# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57f (R22): writer drops a released session's put on every attempt; fences wait for
every expected writer and only for the released session; bounded store calls and a
finish deadline; one finish owner and a summary before every exit; an admission
reservation ledger; the SSD dir handed to the holder by token; rejections through
cleanup; more failure counters; no fences without SSD or release dir. Run: python -I
t57f.py [driver.py]"""

import ast
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pylib"))
for k in (
    "OC_MC_SSD_PATH",
    "OC_MC_SSD_WIPE",
    "OC_HARD_PIN",
    "SLURM_JOB_ID",
    "OC_RELEASE_DIR",
    "OC_SSD_RESERVE_BYTES",
    "OC_WRITER_FAIR",
    "OC_STATS_FILE",
    "OC_MC_SSD_TOKEN",
):
    os.environ.pop(k, None)
os.environ["OC_MC_MASTER"] = "fake:1"
os.environ["OC_HARD_PIN"] = "0"


class Store:
    """Fake Mooncake: `full` makes every put fail (pool at capacity)."""

    def __init__(self):
        self.objs, self.full = {}, False

    def setup(self, *a):
        return 0

    def put(self, key, value, cfg=None):
        if self.full:
            return 1
        self.objs[key] = bytes(value)
        return 0

    def get(self, key):
        return self.objs.get(key)

    def is_exist(self, key):
        return int(key in self.objs)


STORES = []
sys.modules["mooncake"] = types.ModuleType("mooncake")
sys.modules["mooncake.store"] = types.SimpleNamespace(
    MooncakeDistributedStore=lambda: STORES.append(Store()) or STORES[-1],
    ReplicateConfig=object,
)

import numpy as np  # noqa: E402
import oc_common as oc  # noqa: E402

DRIVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "x10_driver.py")
SRC = pathlib.Path(DRIVER).read_text(encoding="utf8")
TMP = tempfile.mkdtemp(prefix="t57f_")
GiB = 1 << 30


# Helpers later driver versions add around the requested names (loaded when present).
OPTIONAL = (
    "FIN",
    "FINISH_WAIT_S",
    "FINISH_DEADLINE_S",
    "STORE_CALL_S",
    "write_min_summary",
    "write_summary",
    "SUMMARY_LOCK",
    "hard_exit",
    "refresh_writers",
    "WRITERS_EXPECTED",
    "read_reports",
    "REPORT_READ_S",
    "HARD_LOCK_S",
    "HARD_WRITE_S",
    "_replace_out",
    "_hard_marked",
    "check_writer_fatal",
    "RELEASE_OWNER",
    "leak_probe",
    "SESSION_RIDS",
    "KEY_HINTS",
)


def driver_ns(names, **env):
    tree = ast.parse(SRC)
    ns = dict(
        os=os,
        sys=sys,
        json=json,
        contextlib=contextlib,
        pathlib=pathlib,
        time=time,
        threading=threading,
        subprocess=subprocess,
        np=np,
        oc=oc,
        say=lambda *a: None,
        MEM={},
        READ_STATS={},
        REP="",
        GEN_TIMEOUT=1800,
        last_writer_stats=lambda: {},
        _REF_PIDS=[],
        OUT=os.path.join(TMP, "out.json"),
        GET_CHUNK=32,
        procs=[],
        reader=None,
        STATS={},
        FENCE_ON=True,
        WRITERS=[],
        SSD=False,
        DRIVER_T0=0.0,
    )
    ns.update(env)
    want = set(names) | set(OPTIONAL)
    got = set()
    for node in tree.body:
        tgt = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
            tgt = [node.name]
        elif isinstance(node, ast.Assign):
            tgt = [
                t.id for t in node.targets if isinstance(t, ast.Name) and t.id in want
            ]
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in want
        ):
            tgt = [node.target.id]
        if tgt:
            exec(compile(ast.Module([node], []), DRIVER, "exec"), ns)
            got.update(tgt)
    missing = set(names) - got
    assert not missing, f"driver lacks {sorted(missing)}"
    ns.update({k: v for k, v in env.items() if k in want and not callable(v)})
    return ns


def exits_into(ns, log):
    ns["os"] = types.SimpleNamespace(
        environ=os.environ,
        path=os.path,
        replace=os.replace,
        getpid=os.getpid,
        _exit=lambda c: log.append(c) or (_ for _ in ()).throw(SystemExit(c)),
    )


def run_with_timeout(fn, s):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn()), daemon=True)
    th.start()
    th.join(s)
    return th.is_alive(), box.get("r")


# ------------------------------------------------------------------ 1. writer drops
# after release


def t_writer_drops_after_release():
    rel = os.path.join(TMP, "released")
    os.makedirs(rel, exist_ok=True)
    oc.RELEASE_DIR = rel
    try:
        for fair in ("0", "1"):
            os.environ["OC_WRITER_FAIR"] = fair
            w = oc.Writer()
            st = STORES[-1]
            a, b = f"{fair}A", f"{fair}B"
            # Released while it was encoded: never written.

            def enc(a=a):
                open(os.path.join(rel, a), "w").close()
                return b"late"

            w.put(oc.block_key(a, "h0"), enc)
            # Released while it retried against a full pool: dropped, the other session
            # proceeds.
            st.full = True
            w.put(oc.block_key(b, "h1"), b"x" * 8)
            time.sleep(0.3)
            assert w.pending_sessions().get(b) == 1, w.pending_sessions()
            open(os.path.join(rel, b), "w").close()
            st.full = False
            c = f"{fair}C"
            w.put(oc.block_key(c, "h2"), b"y")
            w.flush()
            time.sleep(0.2)
            assert (
                oc.block_key(a, "h0") not in st.objs
                and oc.block_key(b, "h1") not in st.objs
            ), st.objs
            assert st.objs.get(oc.block_key(c, "h2")) == b"y"
            assert (
                w.stats.get("dropped_after_release") == 2 and w.pending_sessions() == {}
            ), (w.stats, w.pending_sessions())
    finally:
        oc.RELEASE_DIR = ""
        os.environ.pop("OC_WRITER_FAIR", None)
    print(
        "writer: a released session's put is dropped at every attempt (after "
        "encode, while retrying), both queues"
    )


# ------------------------------------------------------------------ 2. fence membership


def t_fence_membership():
    t0 = 1000.0
    reports = {
        1: dict(t=t0 + 5, pending_sessions={"B": 3}),
        2: dict(t=t0 + 6, pending_sessions={}),
    }
    assert (
        oc.fence_waiting(reports, [1, 2], "A", t0) == []
    )  # B's backlog does not hold A
    assert oc.fence_waiting(reports, [1, 2], "B", t0) == [1]
    assert oc.fence_waiting(reports, [1, 2, 3], "A", t0) == [
        3
    ]  # rank 3 never registered
    assert oc.fence_waiting(reports, [1, 2], "A", t0 + 5.5) == [
        1
    ]  # rank 1's report predates the release
    removed = []
    rd = types.SimpleNamespace(remove_by_regex=lambda rx, f: removed.append(rx) or 1)
    stats = os.path.join(TMP, "ws")
    for pid, rec in ((1, dict(t=time.time() + 5, pending_sessions={})),):
        pathlib.Path(f"{stats}.{pid}").write_text(json.dumps(rec) + "\n")
    os.environ["OC_STATS_FILE"] = stats
    ns = driver_ns(
        ["settle_fences", "FENCES", "STORE_CALL_S"],
        reader=rd,
        WRITERS=[1, 2],
        WRITERS_EXPECTED=[2],
    )
    ns["FENCES"]["A"] = time.time()
    acc = {}
    ns["settle_fences"](acc)
    assert (
        removed == [] and acc["fences_open"] == 1 and acc["fence_waiting"] == {"A": [2]}
    ), acc
    pathlib.Path(f"{stats}.2").write_text(
        json.dumps(dict(t=time.time() + 5, pending_sessions={"Z": 1})) + "\n"
    )
    ns["settle_fences"](acc)
    assert (
        removed == ["^" + oc.PREFIX + "/A/"]
        and acc["fences_open"] == 0
        and acc["late_recreated"] == 1
    ), acc
    print(
        "fence: clears only when every expected writer reported nothing pending for "
        "that session after release"
    )


# ------------------------------------------------------------------ 3. bounded store
# calls, finish deadline


def t_bounded_calls():
    block = threading.Event()
    rd = types.SimpleNamespace(remove_by_regex=lambda rx, f: block.wait())
    stats = os.path.join(TMP, "wb")
    pathlib.Path(f"{stats}.1").write_text(
        json.dumps(dict(t=time.time() + 9, pending_sessions={})) + "\n"
    )
    os.environ["OC_STATS_FILE"] = stats
    ns = driver_ns(
        ["settle_fences", "FENCES", "STORE_CALL_S"],
        reader=rd,
        WRITERS=[1],
        WRITERS_EXPECTED=[1],
    )
    ns["STORE_CALL_S"] = 0.2
    ns["FENCES"]["A"] = time.time()
    acc = {}
    hung, _ = run_with_timeout(lambda: ns["settle_fences"](acc), 5)
    assert (
        not hung and acc.get("fence_sweep_failed") == 1 and acc["fences_open"] == 1
    ), acc
    out = os.path.join(TMP, "bounded.json")
    names = [
        "finish",
        "_step",
        "_fail",
        "_rank_dump_compare",
        "settle_fences",
        "FAIL_KEYS",
        "FINISHING",
        "FENCES",
        "FIN",
        "FINISH_WAIT_S",
        "STORE_CALL_S",
        "write_min_summary",
    ]
    ns = driver_ns(names, OUT=out, STATS=dict(phases=[]), reader=rd, FENCE_ON=False)
    ns["STORE_CALL_S"] = 0.2
    hung, _ = run_with_timeout(ns["finish"], 5)
    st = json.loads(pathlib.Path(out).read_text())
    assert not hung and st["ok"] is False and "sweep" in str(st["cleanup_errors"]), st
    # The watchdog keeps watching a finish that overruns its deadline.
    log = []
    ns = driver_ns(
        [
            "Guard",
            "FINISHING",
            "FIN",
            "FINISH_DEADLINE_S",
            "FINISH_WAIT_S",
            "hard_exit",
            "write_min_summary",
        ],
        OUT=os.path.join(TMP, "deadline.json"),
        STATS=dict(x=1),
    )
    exits_into(ns, log)
    ns["FINISHING"].set()
    ns["FIN"]["t"] = time.time() - 10_000
    ns["FINISH_WAIT_S"] = 0.1
    ns["time"] = types.SimpleNamespace(time=time.time, sleep=lambda s: None)
    g = ns["Guard"].__new__(ns["Guard"])
    g.t, g.tag = None, ""
    with contextlib.suppress(SystemExit):
        g._loop()
    assert (
        log == [11]
        and json.loads(pathlib.Path(os.path.join(TMP, "deadline.json")).read_text())[
            "exit_code"
        ]
        == 11
    ), log
    print(
        "bounded: a blocked second sweep or final sweep is abandoned; a finish past "
        "its deadline is ended"
    )


# ------------------------------------------------------------------ 4. one finish
# owner, summary before exit


def t_single_owner():
    out = os.path.join(TMP, "owner.json")
    names = [
        "finish",
        "_step",
        "_fail",
        "_rank_dump_compare",
        "settle_fences",
        "FAIL_KEYS",
        "FINISHING",
        "FENCES",
        "FIN",
        "FINISH_WAIT_S",
        "STORE_CALL_S",
        "write_min_summary",
        "hard_exit",
        "check_helpers",
    ]
    gate = threading.Event()
    log = []

    class Dead:
        args = ["holder"]

        def poll(self):
            return -9

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return -9

        def kill(self):
            pass

    ns = driver_ns(names, OUT=out, STATS=dict(phases=[]), procs=[], FENCE_ON=False)
    exits_into(ns, log)
    ns["FINISH_WAIT_S"] = 0.5
    ns["_rank_dump_compare"] = lambda: gate.wait(
        10
    )  # the main thread's cleanup is slow
    main = threading.Thread(target=ns["finish"], daemon=True)
    main.start()
    time.sleep(0.2)
    # The watchdog saw a dead holder just before the main thread started finishing.
    ns["procs"] = [Dead()]
    ns["FINISHING"].clear()
    seen = {}

    def watchdog():
        try:
            ns["check_helpers"]()
        except SystemExit:
            seen["summary_at_exit"] = os.path.exists(out)

    th = threading.Thread(target=watchdog, daemon=True)
    th.start()
    th.join(5)
    assert log == [10] and seen["summary_at_exit"], (log, seen)
    assert json.loads(pathlib.Path(out).read_text())["exit_code"] == 10
    gate.set()
    main.join(5)
    # The deadlock path: a failing stack dump still finishes and exits with a summary.
    out2 = os.path.join(TMP, "dead.json")
    log2, calls = [], []
    ns = driver_ns(
        [
            "Guard",
            "FINISHING",
            "FIN",
            "FINISH_WAIT_S",
            "hard_exit",
            "write_min_summary",
        ],
        OUT=out2,
        STATS={},
        dump_stacks=lambda tag: (_ for _ in ()).throw(OSError("py-spy missing")),
        finish=lambda deadlock=False: calls.append(deadlock),
    )
    exits_into(ns, log2)
    ns["GEN_TIMEOUT"] = 0
    ns["FINISH_WAIT_S"] = 0.1
    ns["time"] = types.SimpleNamespace(time=time.time, sleep=lambda s: None)
    g = ns["Guard"].__new__(ns["Guard"])
    g.t, g.tag = time.time() - 5, "gen"
    with contextlib.suppress(SystemExit):
        g._loop()
    assert (
        calls == [True]
        and log2 == [9]
        and json.loads(pathlib.Path(out2).read_text())["exit_code"] == 9
    ), (calls, log2)
    print(
        "finish: one owner; a second caller waits for its summary (bounded) and "
        "every exit writes one"
    )


# ------------------------------------------------------------------ 5. admission ledger


def t_admission_ledger():
    loop = SRC[SRC.index("        while pending and len(inflight) < C * GROUP:") :]
    loop = loop[: loop.index("        prompts, params = [], []")]
    ns = dict(
        pending=[0, 1, 2],
        inflight=[],
        C=3,
        GROUP=1,
        SSD=True,
        POOL=8 * GiB,
        sess_bytes=2 * GiB,
        RESERVED={},
        L=1000,
        P0=10,
        ph={},
        text_ids=lambda seed, n: [1] * n,
        group_left={},
        REP="",
        DESIGN="ps",
        ARM="full",
        FORK=False,
        os=os,
        RUN_NONCE="n0",
        STATS=dict(phases=[]),
        COMPACT=False,
        oc=types.SimpleNamespace(ssd_headroom=lambda path, seg: 2 * GiB),
    )
    os.environ["OC_MC_SSD_PATH"] = "/ssd"
    try:
        # Pretend one group is already in flight so the pause does not raise.
        ns["inflight"].append(dict(g=-1))
        exec(textwrap.dedent(loop), ns)
    finally:
        os.environ.pop("OC_MC_SSD_PATH")
    admitted = sorted({s["g"] for s in ns["inflight"] if s["g"] >= 0})
    assert admitted == [0] and ns["ph"].get("admission_paused") == 1, (
        admitted,
        ns["ph"],
    )
    assert 'RESERVED.pop((L, s["g"]), None)' in SRC
    print(
        "admission: a reservation ledger keeps 2 GiB of headroom from admitting "
        "three 2 GiB groups"
    )


# ------------------------------------------------------------------ 6. SSD dir handoff
# by token


def t_ssd_token():
    os.environ["SLURM_JOB_ID"] = "9"
    os.environ["OC_MC_SSD_PATH"] = path = os.path.join(TMP, "ssd")
    try:
        token = oc.ssd_prepare_dir(path)  # the driver
        os.environ["OC_MC_SSD_TOKEN"] = token
        oc.make_store(segment_bytes=1 << 20)  # its holder
        for bad in ("other-driver", None):
            if bad is None:
                os.environ.pop("OC_MC_SSD_TOKEN")
            else:
                os.environ["OC_MC_SSD_TOKEN"] = bad
            try:
                oc.make_store(segment_bytes=1 << 20)
                raise AssertionError(
                    f"holder accepted a dir prepared by another driver (token {bad})"
                )
            except RuntimeError as e:
                assert "another driver" in str(e), e
    finally:
        for k in ("OC_MC_SSD_PATH", "OC_MC_SSD_TOKEN", "SLURM_JOB_ID"):
            os.environ.pop(k, None)
    print(
        "SSD dir: a holder only takes the marker-only dir of the driver that spawned it"
    )


# ------------------------------------------------------------------ 7. rejections
# through cleanup


def t_rejections():
    prep = SRC.index("oc.ssd_prepare_dir(")
    assert prep < SRC.index('margs = ["mooncake_master"'), (
        "SSD checks must run before any helper starts"
    )
    blk = SRC[SRC.rfind("try:", 0, prep) : SRC.index("port = int(", prep)]
    assert "write_min_summary(" in blk and "raise" in blk
    calls = []
    ns = driver_ns(["_uncaught"], STATS={}, finish=lambda: calls.append(1))
    ns["sys"] = types.SimpleNamespace(__excepthook__=lambda *a: calls.append("hook"))
    ns["_uncaught"](RuntimeError, RuntimeError("headroom"), None)
    assert calls == [1, "hook"] and ns["STATS"]["ok"] is False
    assert "sys.excepthook = _uncaught" in SRC
    print(
        "rejections: SSD checks before helpers start (min summary); uncaught errors "
        "go through finish"
    )


# ------------------------------------------------------------------ 8./9. ok counters,
# fences off without SSD


def t_ok_and_fence_off():
    out = os.path.join(TMP, "ok.json")
    names = [
        "finish",
        "_step",
        "_fail",
        "_rank_dump_compare",
        "settle_fences",
        "release",
        "FAIL_KEYS",
        "FINISHING",
        "FENCES",
        "FIN",
        "FINISH_WAIT_S",
        "STORE_CALL_S",
        "write_min_summary",
    ]
    rd = types.SimpleNamespace(remove_by_regex=lambda rx, f: 0)
    for k in ("idx_out_of_range", "hid_meta_bad", "ref_wrong"):
        ns = driver_ns(
            names, OUT=out, STATS=dict(phases=[{k: 1}]), reader=rd, FENCE_ON=False
        )
        ns["finish"]()
        assert json.loads(pathlib.Path(out).read_text())["ok"] is False, k
    # An open fence at the end fails the run.
    os.environ["OC_STATS_FILE"] = os.path.join(TMP, "nofiles")
    os.environ["OC_FENCE_WAIT_S"] = "0"
    ns = driver_ns(
        names,
        OUT=out,
        STATS=dict(phases=[{}]),
        reader=rd,
        FENCE_ON=True,
        WRITERS=[1],
        WRITERS_EXPECTED=[1],
    )
    ns["FENCES"]["A"] = time.time()
    ns["finish"]()
    st = json.loads(pathlib.Path(out).read_text())
    assert st["ok"] is False and st["fences_open"] == 1, st
    # No SSD and no release dir: release is a plain removal, nothing fenced, no fence
    # wait at the end.
    os.environ["OC_FENCE_WAIT_S"] = "3"
    ns = driver_ns(names, OUT=out, STATS=dict(phases=[{}]), reader=rd, FENCE_ON=False)
    ns["release"]("S")
    t = time.time()
    ns["finish"]()
    assert (
        not ns["FENCES"]
        and time.time() - t < 1
        and json.loads(pathlib.Path(out).read_text())["ok"] is True
    )
    assert 'FENCE_ON = SSD or bool(os.environ.get("OC_RELEASE_DIR"))' in SRC
    os.environ.pop("OC_FENCE_WAIT_S")
    print(
        "ok: False on idx_out_of_range / hid_meta_bad / ref_wrong / an open fence; "
        "no fences without SSD or release dir"
    )


if __name__ == "__main__":
    t_writer_drops_after_release()
    t_fence_membership()
    t_bounded_calls()
    t_single_owner()
    t_admission_ledger()
    t_ssd_token()
    t_rejections()
    t_ok_and_fence_off()
    print("all t57f tests passed")
