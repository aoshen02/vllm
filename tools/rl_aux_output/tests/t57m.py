# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57m: a store write that fails fails the request. Every request whose manifest names
a failed object (its own or an earlier request's block) gets its error marker instead of
a manifest; siblings that do not name it are unaffected; failed names are never
rewritten. The reader fails those requests; the x10 checker counts failed_requests (a
run failure unless OC_INJECT_ERR_EVERY).
Run: python -I t57m.py [driver.py]  (uses t57f's fakes)"""

import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc_reader as rd  # noqa: E402
import t57f as base  # noqa: E402
from t57f import TMP, driver_ns, oc, run_with_timeout  # noqa: E402

SRC = base.SRC


class FailingStore(base.Store):
    """Raises on the keys in `fail` (a write error the writer does not retry)."""

    fail: set = set()

    def put(self, key, value, cfg=None):
        if key in self.fail:
            raise RuntimeError(f"FILE_WRITE_FAIL {key}")
        return super().put(key, value, cfg)


def manifest(sess, rid, blocks, idx=False, hid=0, cone=False):
    pre = f"{oc.PREFIX}/{sess}"
    fields = [
        dict(
            name="routed_experts",
            dtype="uint16",
            shape=[1],
            codec="raw",
            layout="blocks+tail",
            rows=1,
        )
    ]
    if idx:
        fields.append(
            dict(
                name="indexer_topk",
                block_suffix=oc.IDX_SUFFIX,
                tail_key=oc.idx_tail_key(sess, rid),
            )
        )
    if hid:
        fields.append(
            dict(name="hidden", key_prefix=oc.hid_key(sess, rid, ""), chunks=hid)
        )
    m = dict(
        key_prefix=pre, tail_key=oc.tail_key(sess, rid), blocks=blocks, fields=fields
    )
    if cone:
        m["r3_cone"] = dict(patches=[dict(key=oc.cone_key(sess, rid), lo=0, end=1)])
    return json.dumps(m).encode()


def t_failed_write_fails_requests():
    real = sys.modules["mooncake.store"].MooncakeDistributedStore
    sys.modules["mooncake.store"].MooncakeDistributedStore = (
        lambda: base.STORES.append(FailingStore()) or base.STORES[-1]
    )
    try:
        for fair in ("0", "1"):
            os.environ["OC_WRITER_FAIR"] = fair
            w = oc.Writer()
            st = base.STORES[-1]
            S = f"fs{fair}"
            st.fail = {
                oc.block_key(S, "b1"),
                oc.block_key(S, "b4") + oc.IDX_SUFFIX,
                oc.hid_key(S, "E", 0),
                oc.cone_key(S, "F"),
            }
            for h in ("b1", "b2", "b3", "b4"):
                w.put(oc.block_key(S, h), b"x")
            w.put(oc.block_key(S, "b4") + oc.IDX_SUFFIX, b"i")
            for rid in "ABCDEF":
                w.put(oc.tail_key(S, rid), b"t")
            w.put(oc.hid_key(S, "E", 0), b"h")
            w.put(oc.cone_key(S, "F"), b"c")
            w.put(
                oc.manifest_key(S, "A"), manifest(S, "A", ["b1", "b2"])
            )  # names its failed block
            w.put(
                oc.manifest_key(S, "B"), manifest(S, "B", ["b1", "b3"])
            )  # a later request naming it
            w.put(
                oc.manifest_key(S, "C"), manifest(S, "C", ["b2", "b3"])
            )  # sibling: unaffected
            w.put(
                oc.manifest_key(S, "D"), manifest(S, "D", ["b4"], idx=True)
            )  # its indexer block failed
            w.put(
                oc.manifest_key(S, "E"), manifest(S, "E", ["b2"], hid=1)
            )  # its hidden chunk failed
            w.put(
                oc.manifest_key(S, "F"), manifest(S, "F", ["b3"], cone=True)
            )  # its cone patch failed
            hung, _ = run_with_timeout(w.flush, 5)
            assert not hung
            for rid, failed in (
                ("A", True),
                ("B", True),
                ("C", False),
                ("D", True),
                ("E", True),
                ("F", True),
            ):
                mk = oc.manifest_key(S, rid)
                has_m, has_err = mk in st.objs, oc.err_key(mk) in st.objs
                assert (has_m, has_err) == (not failed, failed), (
                    fair,
                    rid,
                    has_m,
                    has_err,
                )
                smp = rd.read_sample(st.get, mk, 1, store=st)
                # (The sibling's manifest here is minimal: the reader must just not
                # report an error marker.)
                assert ("error marker" in (smp.error or "")) == failed and (
                    not failed or smp.ok is False
                ), (rid, smp.error)
            assert oc.block_key(S, "b1") not in st.objs, (
                "a failed name is never rewritten"
            )
            assert w.stats["failed_requests"] == 5 and w.stats["failed_objects"] == 4, (
                w.stats
            )
    finally:
        sys.modules["mooncake.store"].MooncakeDistributedStore = real
        os.environ.pop("OC_WRITER_FAIR", None)
    print(
        "failed write: every request naming the failed object gets its error "
        "marker; the sibling reads"
    )


def t_checker_counts_failed_requests():
    loop = SRC[SRC.index("def check_request(s, rec, ph, full=True):") :]
    i_fail = loop.index('request_failed(ph, s, rid, "error marker")')
    assert i_fail < loop.index('ph["missing_manifest"] += 1')
    out = os.path.join(TMP, "failed_req.json")
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
    # v57n: injected failures are counted apart (injected_failed_requests, t57n);
    # genuine ones fail the run.
    for inject, phase, ok in (
        (None, dict(failed_requests=2), False),
        ("5", dict(injected_failed_requests=2), True),
    ):
        if inject:
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
        st = json.loads(pathlib.Path(out).read_text())
        assert st["ok"] is ok, (inject, st.get("fail_reasons"))
    print(
        "checker: failed requests counted (not missing rows); a run failure unless "
        "errors are injected"
    )


if __name__ == "__main__":
    t_failed_write_fails_requests()
    t_checker_counts_failed_requests()
    print("all t57m tests passed")
