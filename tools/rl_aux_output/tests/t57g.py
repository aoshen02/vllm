# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57g (R23): FIFO writer purges released puts behind a retrying head; exactly the
expected writers before a fence clears, late reporters join; bounded first removal
(warmup with the guard armed) and a finish deadline timer; atomic, locked summary
writes; a pre-existing empty SSD dir is taken over and emptied at cleanup; the abort
check never looks at the working directory.
Run: python -I t57g.py [driver.py]   (uses t57f's fakes)"""

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


def t_fifo_purge():
    rel = os.path.join(TMP, "rel_g")
    os.makedirs(rel, exist_ok=True)
    oc.RELEASE_DIR = rel
    try:
        os.environ["OC_WRITER_FAIR"] = "0"
        w = oc.Writer()
        st = base.STORES[-1]
        st.full = True
        w.put(oc.block_key("gB", "h1"), b"b")  # head: retries against the full pool
        time.sleep(0.15)
        w.put(oc.block_key("gA", "h2"), b"a")  # queued behind it
        w.put(oc.block_key("gA", "h3"), b"a")
        time.sleep(0.1)
        assert w.pending_sessions() == {"gB": 1, "gA": 2}, w.pending_sessions()
        open(
            os.path.join(rel, "gA"), "w"
        ).close()  # A released while B still blocks the head
        time.sleep(0.3)
        assert w.pending_sessions() == {"gB": 1}, w.pending_sessions()
        assert w.stats.get("dropped_after_release") == 2, w.stats
        st.full = False
        w.flush()
        assert st.objs == {oc.block_key("gB", "h1"): b"b"}, st.objs
    finally:
        oc.RELEASE_DIR = ""
        os.environ.pop("OC_WRITER_FAIR", None)
    print(
        "FIFO writer: queued puts of a released session are purged while the head "
        "retries"
    )


def t_writer_registration():
    stats = os.path.join(TMP, "reg")
    os.environ["OC_STATS_FILE"] = stats
    now = time.time()
    pathlib.Path(f"{stats}.101").write_text(
        json.dumps(dict(t=now + 5, pending_sessions={})) + "\n"
    )
    removed = []
    rd = types.SimpleNamespace(remove_by_regex=lambda rx, f: removed.append(rx) or 0)
    ns = driver_ns(
        [
            "settle_fences",
            "register_writers",
            "refresh_writers",
            "FENCES",
            "STORE_CALL_S",
            "WRITERS_EXPECTED",
        ],
        reader=rd,
        WRITERS=[],
    )
    os.environ["OC_TP"] = "2"
    got = ns["register_writers"](2, wait_s=0)
    assert got == [101], got
    ns["FENCES"]["A"] = now
    acc = {}
    ns["settle_fences"](acc)
    assert (
        removed == [] and acc["fences_open"] == 1 and "1/2" in str(acc["fence_waiting"])
    ), acc
    # Rank 202 reports later (with A in flight, then idle): it joins and the fence
    # follows it.
    pathlib.Path(f"{stats}.202").write_text(
        json.dumps(dict(t=now + 5, pending_sessions={"A": 1})) + "\n"
    )
    ns["settle_fences"](acc)
    assert removed == [] and ns["WRITERS"] == [101, 202], (removed, ns["WRITERS"])
    with open(f"{stats}.202", "a") as fh:
        fh.write(json.dumps(dict(t=now + 6, pending_sessions={})) + "\n")
    ns["settle_fences"](acc)
    assert removed == ["^" + oc.PREFIX + "/A/"] and acc["fences_open"] == 0, acc
    # Zero writers at the timeout is recoverable the same way.
    ns2 = driver_ns(
        [
            "settle_fences",
            "register_writers",
            "refresh_writers",
            "FENCES",
            "STORE_CALL_S",
            "WRITERS_EXPECTED",
        ],
        reader=rd,
        WRITERS=[],
    )
    os.environ["OC_STATS_FILE"] = os.path.join(TMP, "reg_none")
    assert ns2["register_writers"](1, wait_s=0) == []
    pathlib.Path(os.path.join(TMP, "reg_none.7")).write_text(
        json.dumps(dict(t=now + 5, pending_sessions={})) + "\n"
    )
    ns2["FENCES"]["Z"] = now
    ns2["settle_fences"](acc)
    assert acc["fences_open"] == 0 and ns2["WRITERS"] == [7], acc
    # A set still incomplete at finish fails the run.
    out = os.path.join(TMP, "reg.json")
    names = [
        "finish",
        "_step",
        "_fail",
        "_rank_dump_compare",
        "settle_fences",
        "refresh_writers",
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
        "WRITERS_EXPECTED",
    ]
    os.environ["OC_STATS_FILE"] = stats
    os.environ["OC_FENCE_WAIT_S"] = "0"
    ns3 = driver_ns(names, OUT=out, STATS=dict(phases=[]), reader=rd, WRITERS=[])
    ns3["WRITERS_EXPECTED"][0] = 5
    ns3["finish"]()
    st = json.loads(pathlib.Path(out).read_text())
    assert st["ok"] is False and any(
        "writers_registered" in r for r in st["fail_reasons"]
    ), st
    os.environ.pop("OC_FENCE_WAIT_S")
    print(
        "writers: fences wait for exactly OC_TP writers; late reporters join; "
        "incomplete fails the run"
    )


def t_bounded_release_and_deadline():
    block = threading.Event()
    rd = types.SimpleNamespace(remove_by_regex=lambda rx, f: block.wait())
    ns = driver_ns(
        ["release", "FENCES", "STORE_CALL_S"], reader=rd, STATS={}, FENCE_ON=True
    )
    ns["STORE_CALL_S"] = 0.2
    hung, n = run_with_timeout(lambda: ns["release"]("ps-warm"), 5)
    assert (
        not hung and n is None and "timeout" in ns["STATS"]["release_failed"]["ps-warm"]
    ), ns["STATS"]
    assert "ps-warm" in ns["FENCES"]
    warm = SRC[SRC.index('guard.arm("warmup release")') :]
    i_rel = (
        warm.index("release(WARM)")
        if "release(WARM)" in warm
        else warm.index('-warm")')
    )
    assert i_rel < warm.index("guard.disarm()")
    # The watchdog owns finish and its SSD cleanup blocks: the deadline timer still
    # exits.
    out = os.path.join(TMP, "deadline_g.json")
    log = []
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
        "FINISH_DEADLINE_S",
        "STORE_CALL_S",
        "write_min_summary",
        "write_summary",
        "SUMMARY_LOCK",
        "hard_exit",
    ]
    dead = types.SimpleNamespace(
        poll=lambda: 0,
        terminate=lambda: None,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
        args=["h"],
    )
    ns = driver_ns(
        names, OUT=out, STATS=dict(phases=[]), procs=[dead], reader=None, FENCE_ON=False
    )
    ns["os"] = types.SimpleNamespace(
        environ=os.environ,
        path=os.path,
        replace=os.replace,
        getpid=os.getpid,
        _exit=lambda c: log.append(c),
    )
    ns["oc"] = types.SimpleNamespace(
        ssd_remove_owned=lambda p: block.wait(), call_bounded=oc.call_bounded
    )
    ns["FINISH_DEADLINE_S"], ns["FINISH_WAIT_S"] = 0.3, 0.2
    th = threading.Thread(target=ns["finish"], daemon=True)
    th.start()
    t = time.time()
    while not log and time.time() - t < 5:
        time.sleep(0.05)
    assert (
        log == [11] and json.loads(pathlib.Path(out).read_text())["exit_code"] == 11
    ), log
    block.set()
    print(
        "bounded: a blocked first removal is abandoned (warmup under the guard); "
        "the finish deadline is a timer"
    )


def t_summary_atomic():
    out = os.path.join(TMP, "atomic.json")
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
        "FINISH_DEADLINE_S",
        "STORE_CALL_S",
        "write_min_summary",
        "write_summary",
        "SUMMARY_LOCK",
        "hard_exit",
    ]
    ns = driver_ns(
        names, OUT=out, STATS=dict(phases=[]), procs=[], reader=None, FENCE_ON=False
    )
    ns["write_min_summary"](10, "helper died")  # the watchdog's fallback came first
    gate, entered = threading.Event(), threading.Event()
    real = json

    class SlowJson:
        def __getattr__(self, k):
            return getattr(real, k)

        def dump(self, obj, fp, **kw):
            if kw.get("indent") == 1:
                entered.set()
                gate.wait(5)
            return real.dump(obj, fp, **kw)

        def dumps(self, obj, **kw):
            if kw.get("indent") == 1 and not entered.is_set():
                entered.set()
                gate.wait(5)
            return real.dumps(obj, **kw)

    ns["json"] = SlowJson()
    th = threading.Thread(target=ns["finish"], daemon=True)
    th.start()
    assert entered.wait(5)
    # The process exits here, mid-write of the normal summary: what is on disk must be
    # whole.
    data = pathlib.Path(out).read_text()
    assert json.loads(data)["exit_code"] == 10, data[:80]
    gate.set()
    th.join(5)
    st = json.loads(pathlib.Path(out).read_text())
    assert "exit_code" not in st and st["ok"] is True, st
    ns["write_min_summary"](9, "late")  # never replaces a complete summary
    assert "exit_code" not in json.loads(pathlib.Path(out).read_text())
    print(
        "summary: written atomically under one lock; a minimal summary never "
        "replaces a complete one"
    )


def t_empty_dir_taken_over():
    os.environ["SLURM_JOB_ID"] = "5"
    try:
        path = os.path.join(TMP, "mnt_ssd")
        os.makedirs(path)
        token = oc.ssd_prepare_dir(path)
        assert (
            token
            and json.loads(pathlib.Path(os.path.join(path, oc.SSD_OWNER)).read_text())[
                "token"
            ]
            == token
        )
        os.environ["OC_MC_SSD_TOKEN"] = token
        os.environ["OC_MC_SSD_PATH"] = path
        oc.make_store(segment_bytes=1 << 20)
        try:
            os.environ["OC_MC_SSD_TOKEN"] = "other"
            oc.make_store(segment_bytes=1 << 20)
            raise AssertionError("another driver's holder took the dir")
        except RuntimeError:
            pass
        pathlib.Path(os.path.join(path, "bucket0")).write_text("x")
        os.makedirs(os.path.join(path, "sub"))
        assert oc.ssd_remove_owned(path)
        assert os.path.isdir(path) and os.listdir(path) == [], os.listdir(
            path
        )  # emptied, kept
    finally:
        for k in ("SLURM_JOB_ID", "OC_MC_SSD_TOKEN", "OC_MC_SSD_PATH"):
            os.environ.pop(k, None)
    print(
        "SSD dir: a pre-existing empty dir is marked and handed over; cleanup "
        "empties it and keeps it"
    )


def t_abort_check_no_cwd():
    cwd = os.getcwd()
    rel = os.path.join(TMP, "rel_cwd")
    os.makedirs(rel, exist_ok=True)
    work = os.path.join(TMP, "work")
    os.makedirs(work, exist_ok=True)
    oc.RELEASE_DIR, oc.ABORT_DIR = rel, ""
    try:
        os.chdir(work)
        open("sessX", "w").close()  # an unrelated file named like the session
        assert not oc.Writer._aborted(oc.block_key("sessX", "h"))
        open(os.path.join(rel, "sessX"), "w").close()
        assert oc.Writer._aborted(oc.block_key("sessX", "h"))
    finally:
        os.chdir(cwd)
        oc.RELEASE_DIR = ""
    print("abort check: only the configured abort dir, never the working directory")


if __name__ == "__main__":
    t_fifo_purge()
    t_writer_registration()
    t_bounded_release_and_deadline()
    t_summary_atomic()
    t_empty_dir_taken_over()
    t_abort_check_no_cwd()
    print("all t57g tests passed")
