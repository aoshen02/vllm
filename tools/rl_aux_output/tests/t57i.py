# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57i (R25 P1s): put-stall timer of the oldest pending put, never a first attempt;
release dir wiped only with our creation marker; SSD wipe of a same-job dir only with
our token or a dead holder; one registration deadline for all reads; after a hard exit's
flag a finish never writes OUT; the hard exit's write is bounded. Run: python -I t57i.py
[driver.py]  (uses t57f's fakes)"""

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


class FlakyStore(base.Store):
    """Fails the next `fail_next` puts, then works; `full` fails every put."""

    fail_next = 0

    def put(self, key, value, cfg=None):
        if self.fail_next:
            self.fail_next -= 1
            return 1
        return super().put(key, value, cfg)


def t_put_stall_fresh_session():
    oc.PUT_STALL_S = 0.2
    real = sys.modules["mooncake.store"].MooncakeDistributedStore
    sys.modules["mooncake.store"].MooncakeDistributedStore = (
        lambda: base.STORES.append(FlakyStore()) or base.STORES[-1]
    )
    try:
        for fair in ("0", "1"):
            os.environ["OC_WRITER_FAIR"] = fair
            w = oc.Writer()
            st = base.STORES[-1]
            st.full = True
            w.put(
                oc.block_key(f"old{fair}", "h"), b"x"
            )  # stalls past the deadline, is dropped
            hung, _ = run_with_timeout(w.flush, 5)
            assert not hung and w.stats.get("dropped_put_stall") == 1, w.stats
            time.sleep(0.4)  # idle longer than the deadline
            st.full, st.fail_next = False, 1  # a transient failure
            w.put(oc.block_key(f"new{fair}", "h"), b"y")
            hung, _ = run_with_timeout(w.flush, 5)
            assert not hung and st.objs.get(oc.block_key(f"new{fair}", "h")) == b"y", (
                fair,
                w.stats,
            )
            assert w.stats.get("dropped_put_stall") == 1, w.stats
    finally:
        oc.PUT_STALL_S = 600.0
        sys.modules["mooncake.store"].MooncakeDistributedStore = real
        os.environ.pop("OC_WRITER_FAIR", None)
    print(
        "put stall: the timer restarts when nothing is pending; a fresh put's first "
        "failure is retried"
    )


def run_startup(out, release_dir):
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


def t_release_dir_ownership():
    run = os.path.join(TMP, "run_i")
    child = os.path.join(run, "results")
    os.makedirs(child)
    pathlib.Path(os.path.join(child, "valuable.csv")).write_text("keep")
    out = os.path.join(run, "x.json")
    run_startup(
        out, child
    )  # an existing dir under the run dir, not created by a driver
    assert os.listdir(child) == ["valuable.csv"], os.listdir(child)
    fresh = os.path.join(run, "released")
    ns = run_startup(out, fresh)  # created by this driver: marked
    owner = ns["RELEASE_OWNER"]
    open(os.path.join(fresh, "ps-old-session"), "w").close()
    run_startup(out, fresh)  # the next run of a driver-created dir: stale markers go
    assert sorted(os.listdir(fresh)) == [owner], os.listdir(fresh)
    print(
        "release dir: wiped only when a driver created it (its marker); location "
        "alone is not ownership"
    )


def t_ssd_wipe_needs_token_or_dead_holder():
    path = os.path.join(TMP, "samejob")
    os.makedirs(path)
    real_alive = oc._pid_alive
    os.environ.update(SLURM_JOB_ID="77", OC_MC_SSD_WIPE="1")
    try:
        for case, token, alive, ok in (
            ("same job, live holder, other token", "mine", True, False),
            ("same job, live holder, our token", "theirs", True, True),
            ("same job, dead holder", None, False, True),
        ):
            for n in os.listdir(path):
                os.remove(os.path.join(path, n))
            pathlib.Path(os.path.join(path, oc.SSD_OWNER)).write_text(
                json.dumps(
                    dict(
                        job="77",
                        pid=4242,
                        host=oc._host(),
                        token="theirs",
                        created=True,
                    )
                )
            )
            pathlib.Path(os.path.join(path, "bucket")).write_text("live")
            oc._pid_alive = lambda pid, a=alive: a
            try:
                oc.ssd_prepare_dir(path, token)
                assert ok, case
            except RuntimeError:
                assert not ok and "bucket" in os.listdir(path), case
    finally:
        oc._pid_alive = real_alive
        for k in ("SLURM_JOB_ID", "OC_MC_SSD_WIPE"):
            os.environ.pop(k, None)
    print("SSD wipe: a same-job dir needs our token or a dead holder")


def t_registration_one_deadline():
    calls = []
    block = threading.Event()
    real = oc.writer_reports

    def stuck(*a, **k):
        calls.append(time.time())
        block.wait()
        return {}

    oc.writer_reports = stuck
    os.environ["OC_STATS_FILE"] = os.path.join(TMP, "reg_i")
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
        t = time.time()
        hung, got = run_with_timeout(lambda: ns["register_writers"](2, wait_s=0.2), 3)
        assert not hung and got == [] and time.time() - t < 1.0, (
            hung,
            time.time() - t,
            len(calls),
        )
    finally:
        oc.writer_reports = real
        block.set()
    print("registration: one deadline for every report read")


def t_finish_never_overwrites_hard_summary():
    out = os.path.join(TMP, "hard_f.json")
    log = []
    ns = driver_ns(
        FIN_NAMES, OUT=out, STATS=dict(phases=[]), procs=[], reader=None, FENCE_ON=False
    )
    ns["os"] = fake_os(log)
    ns["FINISH_WAIT_S"] = 0.1
    gate, serialized = threading.Event(), threading.Event()
    real = json

    class PausingJson:
        def __getattr__(self, k):
            return getattr(real, k)

        def dumps(self, obj, **kw):
            data = real.dumps(obj, **kw)
            if kw.get("indent") == 1 and threading.current_thread() is main:
                serialized.set()
                gate.wait(5)  # finish paused after serializing its (successful) summary
            return data

    ns["json"] = PausingJson()
    main = threading.Thread(target=ns["finish"], daemon=True)
    main.start()
    assert serialized.wait(5)
    ns["hard_exit"](11, "finish exceeded")  # the timer: writes the failure summary
    gate.set()
    main.join(5)
    st = json.loads(pathlib.Path(out).read_text())
    assert log == [11] and st["ok"] is False and st["exit_code"] == 11, st
    assert json.loads(pathlib.Path(out + ".finish.json").read_text())["ok"] is True
    print(
        "hard exit: once its flag is up a normal finish writes OUT.finish.json, "
        "never OUT"
    )


def t_hard_write_bounded():
    out = os.path.join(TMP, "hard_w.json")
    log = []
    ns = driver_ns(
        FIN_NAMES + ["_replace_out", "HARD_WRITE_S"],
        OUT=out,
        STATS=dict(phases=[]),
        procs=[],
        reader=None,
        FENCE_ON=False,
    )
    ns["os"] = fake_os(log)
    block = threading.Event()
    ns["_replace_out"] = lambda data, path: block.wait()
    ns["HARD_WRITE_S"] = 0.2
    hung, _ = run_with_timeout(lambda: ns["hard_exit"](11, "deadline"), 3)
    block.set()
    assert not hung and log == [11], (hung, log)
    print(
        "hard exit: a blocked summary write is abandoned after OC_HARD_WRITE_S; the "
        "process exits"
    )


if __name__ == "__main__":
    t_put_stall_fresh_session()
    t_release_dir_ownership()
    t_ssd_wipe_needs_token_or_dead_holder()
    t_registration_one_deadline()
    t_finish_never_overwrites_hard_summary()
    t_hard_write_bounded()
    print("all t57i tests passed")
