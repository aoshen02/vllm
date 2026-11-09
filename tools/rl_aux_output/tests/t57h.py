# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57h (R24): bounded report reads during registration (guard armed); a hard exit
always leaves an ok=False summary with its exit code and never waits forever on the
summary lock; exclusive SSD claim; per-run session names and an own release dir; stale
writer reports ignored; SSD wipe by persisted ownership; writer put-stall deadline
surfaced as an abort; release_failed fails the run; purge marker checks outside the
queue mutex. Run: python -I t57h.py [driver.py]  (uses t57f's fakes)"""

import json
import os
import pathlib
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t57f as base  # noqa: E402
from t57f import TMP, driver_ns, oc, run_with_timeout  # noqa: E402

SRC = base.SRC
FIN_NAMES = [
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
    "FINISH_DEADLINE_S",
    "STORE_CALL_S",
    "write_min_summary",
    "write_summary",
    "SUMMARY_LOCK",
    "hard_exit",
]


def fake_os(log):
    return types.SimpleNamespace(
        environ=os.environ,
        path=os.path,
        replace=os.replace,
        getpid=os.getpid,
        _exit=lambda c: log.append(c),
    )


def t_registration_bounded():
    real = oc.writer_reports
    block = threading.Event()
    oc.writer_reports = lambda *a, **k: block.wait() or {}
    try:
        ns = driver_ns(
            [
                "register_writers",
                "refresh_writers",
                "read_reports",
                "WRITERS_EXPECTED",
                "REPORT_READ_S",
            ],
            WRITERS=[],
        )
        hung, got = run_with_timeout(lambda: ns["register_writers"](2, wait_s=0.3), 5)
        assert not hung and got == [] and ns["STATS"].get("report_read_failed"), (
            hung,
            ns["STATS"],
        )
    finally:
        oc.writer_reports = real
        block.set()
    reg = SRC[SRC.index('guard.arm("writer registration")') :]
    assert reg.index("register_writers(") < reg.index("guard.disarm()")
    print("registration: report reads are bounded and the guard is armed around it")


def t_hard_exit_marks_summary():
    out = os.path.join(TMP, "hard_b.json")
    log = []
    ns = driver_ns(
        FIN_NAMES, OUT=out, STATS=dict(phases=[]), procs=[], reader=None, FENCE_ON=False
    )
    ns["os"] = fake_os(log)
    gate = threading.Event()
    ns["_rank_dump_compare"] = lambda: gate.wait(5)
    main = threading.Thread(target=ns["finish"], daemon=True)
    main.start()
    time.sleep(0.1)
    ns["FINISH_WAIT_S"] = 3
    timer = threading.Thread(
        target=ns["hard_exit"], args=(11, "finish exceeded"), daemon=True
    )
    timer.start()
    time.sleep(0.1)
    gate.set()  # the normal finish completes while the timer waits for it
    timer.join(5)
    main.join(5)
    st = json.loads(pathlib.Path(out).read_text())
    assert log == [11] and st["ok"] is False and st["exit_code"] == 11, (log, st)
    print(
        "hard exit: a finish completing during its wait still leaves ok=False with "
        "the exit code"
    )


def t_hard_exit_lock_bounded():
    out = os.path.join(TMP, "hard_c.json")
    log = []
    ns = driver_ns(
        FIN_NAMES,
        OUT=out,
        STATS=dict(phases=[], x=1),
        procs=[],
        reader=None,
        FENCE_ON=False,
    )
    ns["os"] = fake_os(log)
    ns["HARD_LOCK_S"] = 0.2
    ns["SUMMARY_LOCK"].acquire()  # a normal summary write stuck in os.replace
    try:
        hung, _ = run_with_timeout(lambda: ns["hard_exit"](11, "deadline"), 5)
    finally:
        ns["SUMMARY_LOCK"].release()
    st = json.loads(pathlib.Path(out + ".hard.json").read_text())
    assert not hung and log == [11] and st["exit_code"] == 11 and st["ok"] is False, (
        hung,
        log,
    )
    print("hard exit: a held summary lock is waited on for a bound, then OUT.hard.json")


def t_ssd_claim_exclusive():
    os.environ["SLURM_JOB_ID"] = "4"
    path = os.path.join(TMP, "claim")
    os.makedirs(path)
    real = os.listdir
    try:
        oc.ssd_prepare_dir(path)  # driver A claims the empty dir
        # Driver B had seen the dir empty just before A's marker landed.
        oc.os.listdir = (
            lambda p: [] if os.path.realpath(p) == os.path.realpath(path) else real(p)
        )
        try:
            oc.ssd_prepare_dir(path)
            raise AssertionError("second claim of the same dir succeeded")
        except RuntimeError as e:
            assert "claimed by another driver" in str(e), e
    finally:
        oc.os.listdir = real
        os.environ.pop("SLURM_JOB_ID")
    print(
        "SSD claim: the marker is created exclusively; a concurrent second claim fails"
    )


def run_startup(out, release_dir):
    """Exec the driver's startup block (stale reports, release dir) with a fake
    environment."""
    blk = SRC[SRC.index("DRIVER_T0 = time.time()") : SRC.index("from vllm import LLM")]
    import contextlib
    import glob
    import uuid

    ns = dict(os=os, time=time, OUT=out, _glob=glob, _uuid=uuid, contextlib=contextlib)
    os.environ["OC_RELEASE_DIR"] = release_dir
    try:
        exec(blk, ns)
    finally:
        os.environ.pop("OC_RELEASE_DIR")
    return ns


def t_run_nonce_and_release_dir():
    run = os.path.join(TMP, "run1")
    inside = os.path.join(run, "released")
    outside = os.path.join(TMP, "shared_released")
    for d in (inside, outside):
        os.makedirs(d)
        open(os.path.join(d, "ps-L131072-full-0"), "w").close()
    out = os.path.join(run, "x.json")
    pathlib.Path(out + ".writer.111").write_text("{}\n")
    ns = run_startup(out, inside)
    # v57i: a dir is wiped only with a driver's creation marker (t57i); here only stale
    # reports go.
    assert os.listdir(inside) == ["ps-L131072-full-0"] and not os.path.exists(
        out + ".writer.111"
    )
    run_startup(out, outside)
    assert os.listdir(outside) == [
        "ps-L131072-full-0"
    ]  # not ours: kept; the nonce keeps it from matching
    assert len(ns["RUN_NONCE"]) == 8
    for frag in (
        "name=f\"{REP}{DESIGN}-{RUN_NONCE}-p{len(STATS['phases'])}-L{L}-{ARM}-{g}\"",
        'f"{DESIGN}-{RUN_NONCE}-warm"',
        "name = f\"{REP}probe-{RUN_NONCE}-p{len(STATS['phases'])}-{tag}\"",
    ):
        assert frag in SRC, frag
    print(
        "release dir: ours emptied at start, a shared one left; session names carry "
        "a run nonce and phase"
    )


def t_stale_reports():
    pre = os.path.join(TMP, "stale.writer")
    t0 = time.time()
    pathlib.Path(pre + ".11").write_text(
        json.dumps(dict(t=t0 - 3600, pending_sessions={})) + "\n"
    )
    os.utime(pre + ".11", (t0 - 3600, t0 - 3600))
    pathlib.Path(pre + ".12").write_text(
        json.dumps(dict(t=t0 + 1, pending_sessions={})) + "\n"
    )
    assert sorted(oc.writer_reports(pre, t0 - 10)) == [12]
    print(
        "writer reports: files of an earlier run (older than the driver start) are "
        "ignored"
    )


def t_ssd_wipe_by_ownership():
    path = os.path.join(TMP, "crashed")
    os.makedirs(path)
    real_alive = oc._pid_alive
    try:
        for case, marker, alive, ok in (
            (
                "same job, live holder elsewhere",
                dict(job="77", pid=1, host="elsewhere", token="t", created=True),
                True,
                False,
            ),
            (
                "dead pid here",
                dict(job="", pid=999999, host=oc._host(), token="t", created=False),
                False,
                True,
            ),
            (
                "live pid here",
                dict(job="", pid=999999, host=oc._host(), token="t", created=False),
                True,
                False,
            ),
            (
                "other job",
                dict(job="78", pid=1, host="elsewhere", token="t", created=True),
                False,
                False,
            ),
        ):
            for n in os.listdir(path):
                os.remove(os.path.join(path, n))
            pathlib.Path(os.path.join(path, oc.SSD_OWNER)).write_text(
                json.dumps(marker)
            )
            pathlib.Path(os.path.join(path, "bucket")).write_text("old")
            os.environ.update(SLURM_JOB_ID="77", OC_MC_SSD_WIPE="1")
            oc._pid_alive = lambda pid, a=alive: a
            try:
                oc.ssd_prepare_dir(path)
                assert ok, case
                rec = json.loads(
                    pathlib.Path(os.path.join(path, oc.SSD_OWNER)).read_text()
                )
                assert (
                    os.listdir(path) == [oc.SSD_OWNER]
                    and rec["created"] == marker["created"]
                ), (case, rec)
            except RuntimeError:
                assert not ok, case
                assert "bucket" in os.listdir(path), case
    finally:
        oc._pid_alive = real_alive
        for k in ("SLURM_JOB_ID", "OC_MC_SSD_WIPE"):
            os.environ.pop(k, None)
    print(
        "SSD wipe: allowed for a dead process's dir on this host (not for a live "
        "same-job holder); the dir and its created flag stay"
    )


def t_put_stall():
    oc.PUT_STALL_S = 0.3
    try:
        for fair in ("0", "1"):
            os.environ["OC_WRITER_FAIR"] = fair
            w = oc.Writer()
            st = base.STORES[-1]
            st.full = True  # every put fails (disk write failure, pool full)
            w.put(oc.block_key(f"s{fair}", "h"), b"x")
            hung, _ = run_with_timeout(w.flush, 5)
            assert (
                not hung
                and w.stats["fatal"].startswith("put_stall")
                and w.pending_sessions() == {}
            ), w.stats
    finally:
        oc.PUT_STALL_S = 600.0
        os.environ.pop("OC_WRITER_FAIR", None)
    # The driver's watchdog turns a writer's fatal into an abort with that reason.
    stats = os.path.join(TMP, "fatal")
    pathlib.Path(stats + ".5").write_text(
        json.dumps(dict(t=time.time(), fatal="put_stall: 600 s")) + "\n"
    )
    os.environ["OC_STATS_FILE"] = stats
    out = os.path.join(TMP, "fatal.json")
    log = []
    ns = driver_ns(
        FIN_NAMES + ["check_writer_fatal", "read_reports", "REPORT_READ_S"],
        OUT=out,
        STATS=dict(phases=[]),
        procs=[],
        reader=None,
        FENCE_ON=False,
    )
    ns["os"] = fake_os(log)
    ns["check_writer_fatal"]()
    st = json.loads(pathlib.Path(out).read_text())
    assert (
        log == [12]
        and st["ok"] is False
        and "put_stall" in st["fail_reasons"]
        and st["exit_code"] == 12
    ), st
    print(
        "put stall: a writer with no successful put for OC_PUT_STALL_S stops and "
        "the driver aborts (put_stall)"
    )


def t_release_failed_fails():
    out = os.path.join(TMP, "relfail.json")
    ns = driver_ns(
        FIN_NAMES,
        OUT=out,
        STATS=dict(phases=[], release_failed={"A": "timeout"}),
        procs=[],
        reader=None,
        FENCE_ON=False,
    )
    ns["finish"]()
    st = json.loads(pathlib.Path(out).read_text())
    assert st["ok"] is False and any(
        r.startswith("release_failed") for r in st["fail_reasons"]
    ), st
    print("release_failed fails the run")


def t_purge_outside_mutex():
    rel = os.path.join(TMP, "rel_h")
    os.makedirs(rel, exist_ok=True)
    oc.RELEASE_DIR = rel
    try:
        os.environ["OC_WRITER_FAIR"] = "0"
        w = oc.Writer()
        st = base.STORES[-1]
        st.full = True
        held = []
        real = w._aborted

        def aborted(key):
            held.append(w._q.mutex.locked() and threading.current_thread() is w._thread)
            return real(key)

        w._aborted = aborted
        w.put(oc.block_key("hB", "1"), b"b")
        time.sleep(0.15)
        w.put(oc.block_key("hA", "2"), b"a")
        open(os.path.join(rel, "hA"), "w").close()
        time.sleep(0.3)
        st.full = False
        w.flush()
        assert held and not any(held) and w.stats.get("dropped_after_release") == 1, (
            held[:5],
            w.stats,
        )
    finally:
        oc.RELEASE_DIR = ""
        os.environ.pop("OC_WRITER_FAIR", None)
    print("purge: release-marker checks run outside the queue mutex")


if __name__ == "__main__":
    t_registration_bounded()
    t_hard_exit_marks_summary()
    t_hard_exit_lock_bounded()
    t_ssd_claim_exclusive()
    t_run_nonce_and_release_dir()
    t_stale_reports()
    t_ssd_wipe_by_ownership()
    t_put_stall()
    t_release_failed_fails()
    t_purge_outside_mutex()
    print("all t57h tests passed")
