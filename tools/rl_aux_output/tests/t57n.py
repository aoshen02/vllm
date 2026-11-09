# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57n (R27): the guarantee is at read time. A manifest listing an absent object fails
the request in the reader and in both checkers (failed_requests, never missing rows),
whichever writer failed it and whether or not its manifest was written first. The
writer's error marker is a fast path: a put that raised but whose object exists is no
failure; a request fails only for objects its manifest lists; without a marker the
manifest is written anyway; no substitution of failed names; injected failures counted
apart; x9 sees the marker at once; failed names leave with their session.
Run: python -I t57n.py [x10_driver.py] [x9_driver.py]  (uses t57f/t57m fakes)"""

import ast
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import oc_reader as rd  # noqa: E402
import t57f as base  # noqa: E402
import t57m  # noqa: E402
from t57f import TMP, driver_ns, oc, run_with_timeout  # noqa: E402

SRC = base.SRC
X9 = (
    sys.argv[2]
    if len(sys.argv) > 2
    else os.path.join(os.path.dirname(base.DRIVER), "x9_driver.py")
)


def ns_from(src, names, path, **env):
    tree = ast.parse(src)
    ns = dict(
        os=os,
        sys=sys,
        json=json,
        contextlib=contextlib,
        pathlib=pathlib,
        time=time,
        np=np,
        oc=oc,
        threading=threading,
        subprocess=subprocess,
    )
    ns.update(env)
    for node in tree.body:
        tgt = (
            [node.name]
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            else [t.id for t in node.targets if isinstance(t, ast.Name)]
            if isinstance(node, ast.Assign)
            else [node.target.id]
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            else []
        )
        if any(t in names for t in tgt):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    missing = [n for n in names if n not in ns]
    assert not missing, f"driver lacks {missing}"
    ns.update({k: v for k, v in env.items() if k in names})
    return ns


class Mem:
    def __init__(self):
        self.objs = {}

    def get(self, k):
        return self.objs.get(k)


def t_absent_listed_object_fails_request():
    import t_reader as tr

    tr.set_mode("pack20-int32")
    w = tr.W()
    T, Ix = tr.r3_truth(31, 8000), tr.idx_truth(31, 8000)
    mk, total = tr.request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
    m = json.loads(w.objs[mk])
    del w.objs[
        oc.block_key("s", m["blocks"][2])
    ]  # listed, absent (whoever failed it, whenever)
    smp = rd.read_sample(w.get, mk, total)
    assert (
        not smp.ok
        and (smp.error or "").startswith("listed object absent")
        and smp.missing.rows.all()
    ), smp.error
    for path, src in (
        (base.DRIVER, SRC),
        (X9, pathlib.Path(X9).read_text(encoding="utf8")),
    ):
        ph = dict(
            missing_manifest=0,
            failed_requests=0,
            r3_missing_objects=0,
            r3_missing_rows=0,
            r3_wrong_rows=0,
            r3_rows_checked=0,
            wrong_detail=[],
            idx_missing_objects=0,
            hid_missing_objects=0,
        )
        reader = types.SimpleNamespace(
            get=w.get, is_exist=lambda k: 1 if k in w.objs else 0
        )
        names = ["check_request", "request_failed", "FAILED_REQ", "INJECT_EVERY"]
        names += ["absent_listed"] if "def absent_listed" in src else []
        ns = ns_from(
            src,
            names,
            path,
            reader=reader,
            get_manifest=lambda sess, r, timeout=0: json.loads(
                w.objs[oc.manifest_key(sess, r)]
            ),
            get_many=lambda keys: [w.get(k) for k in keys],
            PAUSE_PERTURB=False,
            WCHANGE_AFTER=0,
            WCHANGE_EVERY=False,
            FORK_CUT=0,
            EXTRA=False,
            TRAJ_TAG=True,
            GROUP=1,
            READ_LAST_ONLY=False,
            _writer_snap=[None],
        )
        rec = ("r0", total, T[:total], [1], [0.0], 3000, 0, 0)
        ns["check_request"](dict(name="s", reqs=[rec], i=0, traj=0), rec, ph, full=True)
        assert (
            ph["failed_requests"] == 1
            and ph["r3_missing_rows"] == 0
            and ph["r3_missing_objects"] == 0
        ), (path, ph)
    print(
        "a manifest listing an absent block: read_sample fails; x10 and x9 count "
        "one failed request, no missing rows"
    )


def writer_with(store_cls, fair="0"):
    real = sys.modules["mooncake.store"].MooncakeDistributedStore
    sys.modules["mooncake.store"].MooncakeDistributedStore = (
        lambda: base.STORES.append(store_cls()) or base.STORES[-1]
    )
    os.environ["OC_WRITER_FAIR"] = fair
    try:
        return oc.Writer(), base.STORES[-1]
    finally:
        sys.modules["mooncake.store"].MooncakeDistributedStore = real
        os.environ.pop("OC_WRITER_FAIR", None)


class CommitThenRaise(base.Store):
    raise_on: set = set()

    def put(self, key, value, cfg=None):
        super().put(key, value, cfg)
        if key in self.raise_on:
            raise RuntimeError("TRANSFER_FAIL after commit")
        return 0


class NoMarkers(t57m.FailingStore):
    def put(self, key, value, cfg=None):
        if key.endswith("!err"):
            return 1
        return super().put(key, value, cfg)


def t_writer_fast_path():
    for fair in ("0", "1"):
        S = f"wf{fair}"
        # (3) the put raised but the object is there: no failure, the manifest stays.
        w, st = writer_with(CommitThenRaise, fair)
        st.raise_on = {oc.block_key(S, "b1")}
        w.put(oc.block_key(S, "b1"), b"x")
        w.put(oc.manifest_key(S, "A"), t57m.manifest(S, "A", ["b1"]))
        assert not run_with_timeout(w.flush, 5)[0]
        assert (
            oc.manifest_key(S, "A") in st.objs
            and oc.err_key(oc.manifest_key(S, "A")) not in st.objs
        ), fair
        assert w.stats.get("exist_after_error") == 1 and not w.stats.get(
            "failed_objects"
        ), w.stats
        # (4) a failed hidden chunk that the manifest does not list (field withheld)
        #     fails nothing.
        w, st = writer_with(t57m.FailingStore, fair)
        st.fail = {oc.hid_key(S, "B", 0)}
        w.put(oc.hid_key(S, "B", 0), b"h")
        w.put(oc.block_key(S, "b2"), b"x")
        w.put(oc.manifest_key(S, "B"), t57m.manifest(S, "B", ["b2"]))  # no hidden field
        assert not run_with_timeout(w.flush, 5)[0]
        assert (
            oc.manifest_key(S, "B") in st.objs
            and oc.err_key(oc.manifest_key(S, "B")) not in st.objs
        ), fair
        # (5) no marker can be written: the manifest is written after all (the reader
        #     fails it).
        w, st = writer_with(NoMarkers, fair)
        st.fail = {oc.block_key(S, "b3")}
        w.put(oc.block_key(S, "b3"), b"x")
        w.put(oc.manifest_key(S, "C"), t57m.manifest(S, "C", ["b3"]))
        assert not run_with_timeout(w.flush, 15)[0]
        assert (
            oc.manifest_key(S, "C") in st.objs and w.stats.get("err_marker_lost") == 1
        ), (fair, w.stats)
    print(
        "writer: an existing object is no failure; only listed objects fail; no "
        "marker -> manifest anyway"
    )


def t_no_substitution():
    os.environ["OC_REPUBLISH_LOST"] = "1"
    try:
        w, st = writer_with(base.Store)
        w._poisoned.setdefault("sn", set()).add("b9")
        name, new = w.put_block_checked("sn", "b9", b"x", "tag")
        w.flush()
        assert (name, new) == ("b9", False) and not any("~r" in k for k in st.objs), (
            name,
            new,
            list(st.objs),
        )
        assert not hasattr(w, "retry_name")
    finally:
        os.environ.pop("OC_REPUBLISH_LOST", None)
    assert "retry_name" not in pathlib.Path(
        os.path.join(os.path.dirname(base.DRIVER), "worker_attr.py")
    ).read_text(encoding="utf8")
    print("no substitution: a failed name stays missing even with OC_REPUBLISH_LOST=1")


def t_inject_accounting():
    out = os.path.join(TMP, "inj.json")
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
        "write_summary",
        "SUMMARY_LOCK",
        "hard_exit",
    ]
    for inject, phase, ok in (
        ("0", dict(failed_requests=2), False),
        ("5", dict(failed_requests=1), False),
        ("5", dict(injected_failed_requests=3), True),
        (None, dict(failed_requests=0), True),
    ):
        if inject is not None:
            os.environ["OC_INJECT_ERR_EVERY"] = inject
        try:
            ns = driver_ns(
                names,
                OUT=out,
                STATS=dict(phases=[phase]),
                procs=[],
                reader=None,
                FENCE_ON=False,
            )
            ns["finish"]()
        finally:
            os.environ.pop("OC_INJECT_ERR_EVERY", None)
        assert json.loads(pathlib.Path(out).read_text())["ok"] is ok, (inject, phase)
    # Classification by the marker's failed names (an injected key), only when injection
    # is on.
    store = Mem()
    oc.INJECTED_KEYS.add("k")
    store.objs[oc.err_key(oc.manifest_key("si", "r"))] = json.dumps(
        dict(failed=["k"])
    ).encode()
    for every, key in ((5, "injected_failed_requests"), (0, "failed_requests")):
        ns = ns_from(
            SRC,
            ["request_failed", "FAILED_REQ"],
            base.DRIVER,
            reader=types.SimpleNamespace(get=store.get),
            INJECT_EVERY=every,
        )
        ph = {}
        ns["request_failed"](ph, dict(name="si"), "r", "error marker")
        assert ph == {key: 1, "failed_detail": ph["failed_detail"]}, (every, ph)
    print(
        "injection: OC_INJECT_ERR_EVERY parsed as int; injected failures counted "
        "apart; genuine ones fail the run"
    )


def t_x9_marker_no_wait():
    store = Mem()
    store.objs[oc.err_key(oc.manifest_key("sx", "r"))] = b"failed"
    ns = ns_from(
        pathlib.Path(X9).read_text(encoding="utf8"),
        ["get_manifest"],
        X9,
        reader=types.SimpleNamespace(get=store.get),
        READ_STATS={},
    )
    t = time.time()
    assert ns["get_manifest"]("sx", "r", timeout=5) is None and time.time() - t < 1
    store.objs[oc.manifest_key("sx", "r")] = (
        b"{}"  # with a manifest the marker is stale
    )
    assert ns["get_manifest"]("sx", "r", timeout=5) == {}
    print("x9: an error marker without a manifest fails the request at once")


def t_prune_failed_names():
    rel = os.path.join(TMP, "rel_n")
    os.makedirs(rel, exist_ok=True)
    w, st = writer_with(base.Store)
    w._fail_object(oc.block_key("gone", "b"), RuntimeError("x"))
    w._fail_object(oc.block_key("live", "b"), RuntimeError("x"))
    open(os.path.join(rel, "gone"), "w").close()
    oc.RELEASE_DIR = rel
    try:
        w._prune_released()
    finally:
        oc.RELEASE_DIR = ""
    assert "gone" not in w._failed and "live" in w._failed, w._failed
    print("failed names leave with their released session")


if __name__ == "__main__":
    t_absent_listed_object_fails_request()
    t_writer_fast_path()
    t_no_substitution()
    t_inject_accounting()
    t_x9_marker_no_wait()
    t_prune_failed_names()
    print("all t57n tests passed")
