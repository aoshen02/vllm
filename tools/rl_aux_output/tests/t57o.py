# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57o (R28): a request is delivered only if every object its manifest lists exists,
whatever rows or outputs are read (reader and both checkers, one batched existence
call); x9 checks indexer cone patches; a historically failed name that exists now does
not suppress the manifest; the error marker is the request's own manifest key + '!err'
(request ids may contain '/'); a failed request counts as injected only when one of its
failed objects is an injected key.
Run: python -I t57o.py [x10_driver.py] [x9_driver.py]  (uses t57f/t57m/t57n fakes)"""

import json
import os
import pathlib
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc_reader as rd  # noqa: E402
import t57f as base  # noqa: E402
import t57m  # noqa: E402
import t_reader as tr  # noqa: E402
from t57f import TMP, oc, run_with_timeout  # noqa: E402
from t57n import X9, ns_from, writer_with  # noqa: E402

SRC = base.SRC


def build(P=3000, On=73):
    tr.set_mode("pack20-int32")
    w = tr.W()
    T, Ix = tr.r3_truth(P + On, 8000), tr.idx_truth(P + On, 8000)
    mk, total = tr.request(w, "a", "r0", T, Ix, 0, 0, P, On, 0)
    return w, mk, total, json.loads(w.objs[mk]), T


def t_reader_checks_every_listed_object():
    w, mk, total, m, _ = build()
    assert rd.read_sample(w.get, mk, total).ok
    fh = next(f for f in m["fields"] if f["name"] == "hidden")
    fi = next(f for f in m["fields"] if f["name"] == "indexer_topk")
    idx_patch = next(p["idx_key"] for p in m["r3_cone"]["patches"] if p.get("idx_key"))
    assert fi["tail_rows"] == 0, fi
    cases = [  # (absent key, n_rows, want)
        (
            oc.block_key("s", m["blocks"][-1]),
            10,
            ("r3", "indexer", "hidden"),
        ),  # last block, rows truncated
        (f"{fh['key_prefix']}0", total, ("r3",)),  # hidden not wanted
        (fi["tail_key"], total, ("r3", "indexer", "hidden")),  # empty tail placeholder
        (idx_patch, total, ("r3",)),  # indexer cone patch
        (m["tail_key"], 10, ("indexer",)),  # r3 tail not read
    ]
    for key, n, want in cases:
        v = w.objs.pop(key)
        try:
            smp = rd.read_sample(w.get, mk, n, want=want)
            assert not smp.ok and smp.error == f"listed object absent: {key}", (
                key,
                smp.error,
            )
        finally:
            w.objs[key] = v

    class Batched:
        calls = 0

        def get(self, k):
            return w.get(k)

        def batch_is_exist(self, keys):
            Batched.calls += 1
            return [1 if k in w.objs else 0 for k in keys]

        def is_exist(self, k):
            raise AssertionError("per-key call with a batch call available")

    st = Batched()
    assert rd.read_sample(st.get, mk, 10, want=("r3",)).ok and Batched.calls == 1
    print(
        "reader: every listed object checked (truncated rows, unwanted fields, "
        "empty placeholders, "
        "cone patches) in one batched call"
    )


def checker_ph(path, w, total, T, rid="r0"):
    src = pathlib.Path(path).read_text(encoding="utf8")
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
        check_cones=lambda s, rec, m, ph, got: got,
    )
    rec = (rid, total, T[:total], [1], [0.0], 3000, 0, 0)
    ns["check_request"](dict(name="s", reqs=[rec], i=0, traj=0), rec, ph, full=True)
    return ph


def t_checkers_check_every_listed_object():
    w, mk, total, m, T = build()
    fh = next(f for f in m["fields"] if f["name"] == "hidden")
    fi = next(f for f in m["fields"] if f["name"] == "indexer_topk")
    idx_patch = next(p["idx_key"] for p in m["r3_cone"]["patches"] if p.get("idx_key"))
    for path in (base.DRIVER, X9):
        for key in (f"{fh['key_prefix']}0", fi["tail_key"], idx_patch):
            v = w.objs.pop(key)
            try:
                ph = checker_ph(path, w, total, T)
                assert ph["failed_requests"] == 1 and ph["r3_missing_rows"] == 0, (
                    path,
                    key,
                    ph,
                )
            finally:
                w.objs[key] = v
    print(
        "x10 and x9: a full check fails the request on any absent listed object "
        "(hidden, indexer "
        "placeholder, indexer cone patch), even with EXTRA off"
    )


def t_present_name_not_suppressed():
    for fair in ("0", "1"):
        S = f"pn{fair}"
        w, st = writer_with(t57m.FailingStore, fair)
        bk = oc.block_key(S, "b1")
        st.fail = {bk}
        w.put(bk, b"x")
        assert not run_with_timeout(w.flush, 5)[0]
        st.fail = set()
        st.put(bk, b"x")  # another writer stored the same name
        w.put(oc.manifest_key(S, "A"), t57m.manifest(S, "A", ["b1"]))
        assert not run_with_timeout(w.flush, 5)[0]
        assert (
            oc.manifest_key(S, "A") in st.objs
            and oc.err_key(oc.manifest_key(S, "A")) not in st.objs
        ), fair
    print("writer: a failed name that exists now does not suppress the manifest")


def t_marker_is_own_manifest_key():
    for fair in ("0", "1"):
        S = f"mk{fair}"
        own, other = oc.manifest_key(S, "job/r0"), oc.manifest_key(S, "job")
        own2 = oc.manifest_key(S, "job/r1")
        w, st = writer_with(t57m.FailingStore, fair)
        st.fail = {own, oc.block_key(S, "b2")}
        w.put(own, t57m.manifest(S, "job/r0", []))  # the manifest write fails
        w.put(oc.block_key(S, "b2"), b"x")
        w.put(own2, t57m.manifest(S, "job/r1", ["b2"]))  # names a failed block
        assert not run_with_timeout(w.flush, 5)[0]
        assert oc.err_key(own) in st.objs and oc.err_key(own2) in st.objs, (
            fair,
            sorted(st.objs),
        )
        assert oc.err_key(other) not in st.objs, (fair, sorted(st.objs))
        assert oc.marker_failed_keys(st.objs[oc.err_key(own2)]) == [
            oc.block_key(S, "b2")
        ]
        smp = rd.read_sample(st.get, own2, 1, store=st)
        assert not smp.ok and smp.error.startswith("error marker")
    print(
        "error marker: the request's own manifest key + '!err', ids with '/' mark "
        "no other request"
    )


def t_inject_by_cause():
    log = os.path.join(TMP, "inj.log")
    if os.path.exists(log):
        os.remove(log)
    S = "ij"
    os.environ.update(OC_INJECT_ERR_EVERY="1", OC_INJECT_LOG=log)
    try:
        w, st = writer_with(base.Store, "1")
        w.put(oc.block_key(S, "bi"), b"x")
        assert not run_with_timeout(w.flush, 5)[0]
    finally:
        os.environ.pop("OC_INJECT_ERR_EVERY", None)
    # the checker reads the shared log, not the writer's process state
    oc.INJECTED_KEYS.clear()
    try:
        assert oc.injected_keys() == {oc.block_key(S, "bi")}, oc.injected_keys()
        store = t57n_mem = types.SimpleNamespace(objs={})
        store.get = lambda k: t57n_mem.objs.get(k)
        inj, gen = oc.block_key(S, "bi"), oc.block_key(S, "bg")
        store.objs[oc.err_key(oc.manifest_key(S, "m1"))] = json.dumps(
            dict(failed=[inj])
        ).encode()
        store.objs[oc.err_key(oc.manifest_key(S, "m2"))] = json.dumps(
            dict(failed=[gen], error="RuntimeError('injected write error')")
        ).encode()
        for path in (base.DRIVER, X9):
            ns = ns_from(
                pathlib.Path(path).read_text(encoding="utf8"),
                ["request_failed", "FAILED_REQ"],
                path,
                reader=types.SimpleNamespace(get=store.get),
                INJECT_EVERY=1,
            )
            ph = {}
            ns["request_failed"](
                ph, dict(name=S), "a1", "listed object absent", [inj]
            )  # no marker
            ns["request_failed"](ph, dict(name=S), "a2", "listed object absent", [gen])
            ns["request_failed"](ph, dict(name=S), "m1", "error marker")
            ns["request_failed"](ph, dict(name=S), "m2", "error marker")
            assert ph["injected_failed_requests"] == 2 and ph["failed_requests"] == 2, (
                path,
                ph,
            )
            assert [d["injected"] for d in ph["failed_detail"]] == [
                True,
                False,
                True,
                False,
            ], (path, ph)
    finally:
        os.environ.pop("OC_INJECT_LOG", None)
    print(
        "injection: classified by cause (an injected key among the failed objects, "
        "marker or not); "
        "genuine failures in the same run stay genuine"
    )


if __name__ == "__main__":
    t_reader_checks_every_listed_object()
    t_checkers_check_every_listed_object()
    t_present_name_not_suppressed()
    t_marker_is_own_manifest_key()
    t_inject_by_cause()
    print("all t57o tests passed")
