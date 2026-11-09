# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57e (R21): SSD dir removed only by the process that created it, finish/watchdog
robustness, helper liveness from the watchdog, SSD admission with segment + reserve,
release fence, run ok flag, shared get backoff (and v57c behaviour without SSD),
per-phase probe names. Driver functions are executed from x10_driver.py's source with
fakes (it imports vllm at module level).
Run: python -I t57e.py [driver.py] [v57c driver.py for the SSD-unset comparison]"""

import ast
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
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
):
    os.environ.pop(k, None)
os.environ["OC_MC_MASTER"] = "fake:1"
sys.modules["mooncake"] = types.ModuleType("mooncake")
sys.modules["mooncake.store"] = types.SimpleNamespace(
    MooncakeDistributedStore=lambda: types.SimpleNamespace(
        setup=lambda *a: 0, put=lambda *a: 0, get=lambda k: None
    ),
    ReplicateConfig=object,
)

import numpy as np  # noqa: E402
import oc_common as oc  # noqa: E402

DRIVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "x10_driver.py")
SRC = pathlib.Path(DRIVER).read_text(encoding="utf8")
TMP = tempfile.mkdtemp(prefix="t57e_")
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


def driver_ns(names, src=SRC, **env):
    """Exec the named top-level functions/classes of a driver source into a namespace
    of fakes."""
    tree = ast.parse(src)
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
        PREFIX=oc.PREFIX,
        REP="",
        GEN_TIMEOUT=1800,
        last_writer_stats=lambda: {},
        _REF_PIDS=[],
        OUT=os.path.join(TMP, "out.json"),
        GET_CHUNK=32,
        FENCE_ON=True,
        WRITERS=[],
        SSD=False,
        DRIVER_T0=0.0,
    )
    ns.update(env)
    want = set(names) | set(OPTIONAL)
    got = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
            exec(compile(ast.Module([node], []), DRIVER, "exec"), ns)
            got.add(node.name)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in want for t in node.targets
        ):
            exec(compile(ast.Module([node], []), DRIVER, "exec"), ns)
            got.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in want
        ):
            exec(compile(ast.Module([node], []), DRIVER, "exec"), ns)
            got.add(node.target.id)
    missing = set(names) - got
    assert not missing, f"driver lacks {sorted(missing)}"
    ns.update({k: v for k, v in env.items() if k in want and not callable(v)})
    return ns


def env(**kw):
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def t_ssd_owner_is_creator():
    env(OC_HARD_PIN="0", SLURM_JOB_ID="7")
    mine = os.path.join(TMP, "mine")
    oc.ssd_prepare_dir(mine)
    # Driver B, same job and path, did not create the dir: never removes it.
    other = os.path.join(TMP, "other")
    os.makedirs(other)
    pathlib.Path(os.path.join(other, oc.SSD_OWNER)).write_text(
        json.dumps(dict(job="7", pid=os.getpid() + 1, host="x"))
    )
    pathlib.Path(os.path.join(other, "bucket")).write_text("live")
    assert not oc.ssd_remove_owned(other) and os.path.exists(
        os.path.join(other, "bucket")
    )
    # Same marker contents as ours, but this process did not create it.
    pathlib.Path(os.path.join(other, oc.SSD_OWNER)).write_text(
        json.dumps(
            json.loads(pathlib.Path(os.path.join(mine, oc.SSD_OWNER)).read_text())
        )
    )
    assert not oc.ssd_remove_owned(other)
    # Its holder may use a dir holding only the creator's marker.
    reuse = os.path.join(TMP, "reuse")
    os.makedirs(reuse)
    pathlib.Path(os.path.join(reuse, oc.SSD_OWNER)).write_text(
        json.dumps(dict(job="7", pid=1, host="h", token="T"))
    )
    oc.ssd_prepare_dir(reuse, "T")
    # A missing job id never matches, even for the creator.
    env(SLURM_JOB_ID=None)
    nojob = os.path.join(TMP, "nojob")
    oc.ssd_prepare_dir(nojob)
    assert not oc.ssd_remove_owned(nojob) and os.path.isdir(nojob)
    env(SLURM_JOB_ID="7")
    assert oc.ssd_remove_owned(mine) and not os.path.exists(mine)
    print("SSD dir: removed only by the process that created it, for a set job id")


def t_finish_always_writes_summary():
    out = os.path.join(TMP, "sum.json")

    class Stuck:
        args = ["python", "holder"]

        def terminate(self):
            raise OSError("terminate failed")

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("holder", timeout)

        def kill(self):
            pass

        def poll(self):
            return None

    class Reader:
        def remove_by_regex(self, rx, force):
            raise RuntimeError("store gone")

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
    ]
    ns = driver_ns(
        names,
        OUT=out,
        STATS=dict(phases=[]),
        procs=[Stuck()],
        reader=Reader(),
        FENCE_ON=False,
    )
    os.environ["OC_STATS_FILE"] = os.path.join(TMP, "w")
    ns["finish"]()
    st = json.loads(pathlib.Path(out).read_text())
    assert st["ok"] is False and st["cleanup_errors"], st
    # The watchdog path: finish raising still ends in os._exit.
    exits = []
    ns2 = driver_ns(
        [
            "Guard",
            "FINISHING",
            "FIN",
            "FINISH_WAIT_S",
            "STORE_CALL_S",
            "write_min_summary",
            "hard_exit",
        ],
        STATS={},
        dump_stacks=lambda tag: "d",
        finish=lambda deadlock=False: (_ for _ in ()).throw(
            RuntimeError("finish broke")
        ),
    )
    ns2["os"] = types.SimpleNamespace(
        **{k: getattr(os, k) for k in ("environ", "path")},
        _exit=lambda c: exits.append(c) or (_ for _ in ()).throw(SystemExit(c)),
    )
    ns2["GEN_TIMEOUT"] = 0
    ns2["time"] = types.SimpleNamespace(time=time.time, sleep=lambda s: None)
    g = ns2["Guard"].__new__(ns2["Guard"])
    g.t, g.tag = time.time() - 5, "x"
    with contextlib.suppress(SystemExit):
        g._loop()
    assert exits == [9], exits
    print("finish: every cleanup step guarded, summary written, watchdog always exits")


def t_watchdog_polls_helpers():
    exits, polls = [], [0]

    class Dead:
        args = ["python", "holder"]

        def poll(self):
            polls[0] += 1
            return -9

    names = [
        "Guard",
        "check_helpers",
        "FINISHING",
        "FIN",
        "FINISH_WAIT_S",
        "STORE_CALL_S",
        "write_min_summary",
        "hard_exit",
    ]
    ns = driver_ns(
        names,
        STATS={},
        procs=[types.SimpleNamespace(poll=lambda: None, args=["m"]), Dead()],
        finish=lambda deadlock=False: None,
        dump_stacks=lambda t: "d",
    )
    ns["os"] = types.SimpleNamespace(
        **{k: getattr(os, k) for k in ("environ", "path")},
        _exit=lambda c: exits.append(c) or (_ for _ in ()).throw(SystemExit(c)),
    )
    sleeps = [0]

    def sleep(s):
        sleeps[0] += 1
        if sleeps[0] > 3:
            raise SystemExit("no exit")

    ns["time"] = types.SimpleNamespace(time=time.time, sleep=sleep)
    g = ns["Guard"].__new__(ns["Guard"])
    g.t, g.tag = None, ""  # no generate() armed: generation is stuck elsewhere
    with contextlib.suppress(SystemExit):
        g._loop()
    assert (
        exits == [10] and sleeps[0] == 1 and ns["STATS"]["helper_died"][0]["rc"] == -9
    ), (exits, sleeps)
    print(
        "liveness: the watchdog polls helpers every 10 s and aborts on the first "
        "poll after a death"
    )


def t_ssd_admission():
    real = oc.shutil.disk_usage
    try:
        oc.shutil.disk_usage = lambda p: types.SimpleNamespace(
            total=0, used=0, free=10 * GiB
        )
        env(OC_SSD_RESERVE_BYTES="0")
        assert (
            oc.admission_c(100, 1 * GiB, 2, 8 * GiB, 1.0, "/ssd") == 1
        )  # (10 - 8) // 2
        env(OC_SSD_RESERVE_BYTES=str(GiB))
        try:
            oc.admission_c(100, 1 * GiB, 2, 8 * GiB, 1.0, "/ssd")
            raise AssertionError("admitted a group that does not fit")
        except RuntimeError as e:
            assert "headroom" in str(e)
        oc.shutil.disk_usage = lambda p: types.SimpleNamespace(total=0, used=0, free=0)
        env(OC_SSD_RESERVE_BYTES=None)
        try:
            oc.admission_c(100, 1e6, 1, 0, 1.0, "/ssd")
            raise AssertionError("admitted with no free space")
        except RuntimeError:
            pass
        oc.shutil.disk_usage = lambda p: types.SimpleNamespace(
            total=0, used=0, free=100 * GiB
        )
        assert oc.ssd_headroom("/ssd", 8 * GiB) == 100 * GiB - 8 * GiB - 16 * GiB
    finally:
        oc.shutil.disk_usage = real
        env(OC_SSD_RESERVE_BYTES=None)
    loop = SRC[SRC.index("    while pending or inflight:") :]
    loop = loop[: loop.index("prompts, params = [], []")]
    assert "oc.ssd_headroom(" in loop and "admission_paused" in loop and "break" in loop
    print(
        "SSD admission: segment and reserve subtracted, error when one group does "
        "not fit, re-checked per step"
    )


def t_release_fence():
    # The fence itself (expected writers, per-session pending) is tested in t57f.
    src = pathlib.Path(os.path.join(HERE, "oc_common.py")).read_text(encoding="utf8")
    assert "pending=self._pending if self._fair else self._q.unfinished_tasks" in src
    assert "pending_sessions=self.pending_sessions()" in src
    rel = SRC[SRC.index("def release(name):") :]
    rel = rel[: rel.index("\ndef ")]
    assert "oc.release_session(" in rel and "FENCES[name]" in rel
    print(
        "release fence: writers report what is pending per session; release records "
        "the fence"
    )


def t_ok_flag_and_warmup():
    out = os.path.join(TMP, "ok.json")
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
    ]

    class Reader:
        def remove_by_regex(self, rx, force):
            return 0

    os.environ["OC_STATS_FILE"] = os.path.join(TMP, "none")
    for phase, ok in (
        (dict(missing_manifest=0, r3_missing_rows=0, hid_bad=0), True),
        (dict(missing_manifest=1), False),
        (dict(r3_missing_rows=128), False),
        (dict(hid_bad=1), False),
    ):
        ns = driver_ns(
            names,
            OUT=out,
            STATS=dict(phases=[phase]),
            procs=[],
            reader=Reader(),
            READ_STATS={},
            FENCE_ON=False,
        )
        ns["finish"]()
        assert json.loads(pathlib.Path(out).read_text())["ok"] is ok, (
            phase,
            json.loads(pathlib.Path(out).read_text()),
        )
    ns = driver_ns(
        names,
        OUT=out,
        STATS=dict(phases=[{}]),
        procs=[],
        reader=Reader(),
        READ_STATS=dict(read_fail_keys=1),
        FENCE_ON=False,
    )
    ns["finish"]()
    assert json.loads(pathlib.Path(out).read_text())["ok"] is False
    warm = SRC[SRC.index('say("warmup done")') - 200 : SRC.index('say("warmup done")')]
    assert any(
        f in warm
        for f in (
            'release(f"{DESIGN}-warm")',
            'release(f"{DESIGN}-{RUN_NONCE}-warm")',
            "release(WARM)",
        )
    )
    print(
        "ok flag: False on missing manifests/rows, hid_bad, read failures; warmup "
        "session released"
    )


def t_get_many(v57c_driver):
    # With SSD: one shared round for every missing key, ~1 s whatever their number.
    slept = []
    keys = [f"k{i}" for i in range(4096)]
    vals = [None] * len(keys)
    oc.refetch_missing(lambda k: None, keys, vals, sleep=slept.append)
    assert sum(slept) <= 1.05 and len(slept) == 5, (sum(slept), len(slept))
    vals = [None, b"a", None]
    left = {"x": [None, b"X"], "z": [b"Z"]}
    n = oc.refetch_missing(
        lambda k: left[k].pop(0), ["x", "y", "z"], vals, sleep=slept.append
    )
    assert vals == [b"X", b"a", b"Z"] and n == 3
    # Without SSD: the same gets, retries and counters as v57c.

    def run(src, ssd):
        log = []

        class R:
            def get_batch(self, part):
                log.append(("batch", len(part)))
                return [None if int(k[1:]) % 3 == 0 else b"v" for k in part]

            def get(self, k):
                log.append(("get", k))
                return None

        ns = driver_ns(
            ["get_many"],
            src=src,
            reader=R(),
            READ_STATS=dict(batch_calls=0, single_retries=0, read_fail_keys=0),
            SSD=ssd,
        )
        ns["oc"] = types.SimpleNamespace(refetch_missing=lambda get, ks, vs: 0)
        t = time.time()
        out = ns["get_many"]([f"k{i}" for i in range(70)])
        return out, ns["READ_STATS"], log, time.time() - t

    old = run(pathlib.Path(v57c_driver).read_text(encoding="utf8"), False)
    new = run(SRC, False)
    assert old[:3] == new[:3] and new[3] < 0.5, "SSD-unset get_many differs from v57c"
    print(
        "get_many: shared 1 s backoff with SSD (4096 keys: 1.0 s); identical to "
        "v57c without SSD"
    )


def t_probe_names():
    probe = SRC[SRC.index("def probe_session(tag):") :]
    probe = probe[: probe.index("\n    ids = ")]
    ns = {"REP": "", "STATS": dict(phases=[{}, {}]), "RUN_NONCE": "n0"}
    line = [ln for ln in probe.splitlines() if ln.strip().startswith("name = ")][0]
    exec("tag='v0a'\n" + line.strip(), ns)
    assert ns["name"] in ("probe-p2-v0a", "probe-n0-p2-v0a"), ns["name"]
    print("probes: one session per phase and probe (a released name is never reused)")


if __name__ == "__main__":
    v57c = sys.argv[2] if len(sys.argv) > 2 else None
    t_ssd_owner_is_creator()
    t_finish_always_writes_summary()
    t_watchdog_polls_helpers()
    t_ssd_admission()
    t_release_fence()
    t_ok_flag_and_warmup()
    if v57c:
        t_get_many(v57c)
    t_probe_names()
    print("all t57e tests passed")
