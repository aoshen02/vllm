# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57q: top-k logprobs through the store, modelled on hidden. The engine snapshot (ids
by descending logit + the sampler kernel's logprobs) is packed per row as k int32 + k
float32; the worker connector chunks the sampled rows like hidden (same positions,
preemption, generation reset, forks, oc-apply lag); the manifest lists the chunks
(failure rule); the reader returns ids and logprobs [rows, k]; the x10 checker verifies
them against the sampled tokens and inline logprobs.
Run: python -I t57q.py  (driver dir on sys.path via t57f when given: python -I t57q.py
x10_driver.py)"""

import json
import os
import pathlib
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault(
    "T55_WORKER",
    os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[1])) if len(sys.argv) > 1 else HERE,
        "worker_attr.py",
    ),
)
import numpy as np  # noqa: E402
import t55 as h  # noqa: E402

oc = h.oc
import oc_reader as rd  # noqa: E402

KK = 4
EXPECT: dict = {}
WITHHOLD: list = [None]  # (sim step number) at which the snapshot is missing


def tk_row(rid, pos):
    ids = ((pos * 7 + len(rid) * 11 + np.arange(KK) * 3) % 1000).astype(np.int32)
    lps = (-(np.arange(KK) * 0.5) - pos * 1e-3).astype(np.float32)
    return np.concatenate([ids.view(np.uint8), lps.view(np.uint8)])


_init = h.Sim.__init__


def _sim_init(self, *a, **kw):
    _init(self, *a, **kw)
    orig, self.nstep = self.conn.prepare_output, 0

    def prep(ib):
        self.nstep += 1
        qsl, ts = ib.query_start_loc_np, ib.num_computed_tokens_np
        rows = []
        for i, rid in enumerate(ib.req_ids):
            pos = int(ts[i]) + int(qsl[i + 1] - qsl[i]) - 1
            EXPECT[(rid, pos)] = r = tk_row(rid, pos)
            rows.append(r)
        oc.TOPK[0] = (
            None
            if WITHHOLD[0] == self.nstep
            else (np.stack(rows), np.arange(len(rows) + 1))
        )
        p = orig(ib)
        p.oc_tk = getattr(p, "oc_tk_gpu", None)
        return p

    self.conn.prepare_output = prep


h.Sim.__init__ = _sim_init


def fields_of(mc):
    out = {}
    for k, v in mc.objs.items():
        if k.endswith("/m"):
            out[k] = {f["name"]: f for f in json.loads(v)["fields"]}
    return out


def check_topk_fields(mc, rids):
    n = 0
    for mk, fs in fields_of(mc).items():
        ft, fh = fs.get("topk"), fs.get("hidden")
        assert ft is not None, (mk, list(fs))
        # Sampling-following output: exactly hidden's positions and chunking.
        assert (ft["start"], ft["rows"], ft["chunk_lens"]) == (
            fh["start"],
            fh["rows"],
            fh["chunk_lens"],
        ), (ft, fh)
        assert (
            ft["codec"] == "raw"
            and ft["dtype"] == "int32+float32"
            and ft["shape"] == [KK]
        ), ft
        assert ft["nbytes"] == ft["rows"] * 8 * KK, ft
        kid = mk.rsplit("/", 2)[-2]
        rid = next(r for r in rids if oc.key_id(r) == kid)
        smp = rd.read_sample(
            mc.get, mk, ft["start"] + ft["rows"], want=("topk",), store=mc
        )
        assert smp.ok, (mk, smp.error, smp.missing.counts)
        assert smp.topk_start == ft["start"] and smp.topk_ids.shape == (
            ft["rows"],
            KK,
        ), smp.topk_ids.shape
        want = np.stack(
            [EXPECT[(rid, p)] for p in range(ft["start"], ft["start"] + ft["rows"])]
        )
        assert (smp.topk_ids == want[:, : 4 * KK].copy().view(np.int32)).all(), mk
        assert (smp.topk_logprobs == want[:, 4 * KK :].copy().view(np.float32)).all(), (
            mk
        )
        n += ft["rows"]
    assert n > 0
    return n


def t_worker_chunks_follow_hidden():
    oc.TOPK_ON, oc.TOPK_K = True, KK
    try:
        for case, rids in (
            (h.case_preempt_resume, "ab"),
            (h.case_generation, "abc"),
            (h.case_fork_parent_queued, ["p", "n2", "c"]),
            (h.case_parked_empty_stash, "pq"),
        ):
            EXPECT.clear()
            h.check(case)
            print("  topk rows checked:", check_topk_fields(h.MC[-1], list(rids)))
    finally:
        oc.TOPK_ON = False
    print(
        "worker: top-k chunks at hidden's positions (preemption, reset, fork, "
        "parked), identical at every lag"
    )


def t_withheld_without_snapshot():
    oc.TOPK_ON, oc.TOPK_K = True, KK
    WITHHOLD[0] = 3
    try:
        ref = h.check(h.case_generation)
    finally:
        oc.TOPK_ON, WITHHOLD[0] = False, None
    fs = fields_of(h.MC[-1])
    missing = [mk for mk, f in fs.items() if "topk" not in f]
    assert missing and ref["stats"].get("topk_withheld", 0) >= 1, (
        missing,
        ref["stats"],
    )
    mk = missing[0]
    smp = rd.read_sample(h.MC[-1].get, mk, 10, want=("topk",), store=h.MC[-1])
    assert not smp.ok and smp.missing.counts.get("topk_field"), smp.missing.counts
    print(
        "a sampled row without its snapshot withholds the field; the reader reports "
        "topk_field"
    )


class Mem:
    def __init__(self):
        self.objs = {}

    def get(self, k):
        return self.objs.get(k)

    def batch_is_exist(self, keys):
        return [int(k in self.objs) for k in keys]


def build(start=9, rows=7, lens=(4, 3)):
    st, pre = Mem(), f"{oc.PREFIX}/s"
    data = np.stack([tk_row("r", p) for p in range(start, start + rows)])
    keys, at = [], 0
    for i, r in enumerate(lens):
        keys.append(oc.topk_key("s", "r", i))
        st.objs[keys[-1]] = data[at : at + r].tobytes()
        at += r
    st.objs[f"{pre}/r/t"] = b"\0"
    m = dict(
        key_prefix=pre,
        tail_key=f"{pre}/r/t",
        rows=start + rows,
        block_size=16,
        blocks=[],
        tail_start=0,
        tail_nbytes=0,
        dtype="uint16",
        shape=[1],
        fields=[
            dict(
                name="topk",
                dtype="int32+float32",
                shape=[KK],
                codec="raw",
                layout="chunks",
                key_prefix=oc.topk_key("s", "r", ""),
                start=start,
                rows=rows,
                chunk_rows=4,
                chunks=len(lens),
                chunk_lens=list(lens),
                nbytes=rows * 8 * KK,
            )
        ],
    )
    mk = oc.manifest_key("s", "r")
    st.objs[mk] = json.dumps(m).encode()
    return st, mk, m, data, keys


def t_reader_topk():
    st, mk, m, data, keys = build()
    smp = rd.read_sample(st.get, mk, 16, want=("topk",))
    assert smp.ok, (smp.error, smp.missing.counts)
    assert (
        smp.topk_ids == data[:, : 4 * KK].copy().view(np.int32)
    ).all() and smp.topk_start == 9
    assert smp.topk_ids.dtype == np.int32 and smp.topk_logprobs.dtype == np.float32
    smp = rd.read_sample(st.get, mk, 12, want=("topk",))  # truncated: rows below n only
    assert smp.ok and smp.topk_ids.shape == (3, KK), smp.topk_ids.shape
    smp = rd.read_sample(st.get, mk, 20, want=("topk",))  # fewer rows than asked
    assert not smp.ok and smp.missing.counts.get("topk_rows_short") == 4, (
        smp.missing.counts
    )
    assert oc.manifest_object_keys(m) >= set(keys)
    v = st.objs.pop(keys[1])
    smp = rd.read_sample(
        st.get, mk, 12, want=("r3",)
    )  # listed, not read, absent: request fails
    assert not smp.ok and smp.error == f"listed object absent: {keys[1]}", smp.error
    st.objs[keys[1]] = v[:-8]
    smp = rd.read_sample(st.get, mk, 16, want=("topk",))
    assert (
        not smp.ok
        and smp.missing.counts.get("topk_chunks")
        and (smp.topk_ids[4:] == -1).all()
    )
    assert np.isnan(smp.topk_logprobs[4:]).all()
    print(
        "reader: want=topk returns ids/logprobs [rows, k]; truncation, short rows, "
        "bad chunk, listed-absent"
    )


def t_topk_check():
    st, mk, m, data, keys = build(start=9, rows=7)
    ft = m["fields"][0]
    bufs = [st.objs[k] for k in keys]
    ids = data[:, : 4 * KK].copy().view(np.int32)
    lps = data[:, 4 * KK :].copy().view(np.float32)
    gen = [int(ids[j, j % KK]) for j in range(7)]
    inl = [float(lps[j, j % KK]) for j in range(7)]
    r = oc.topk_check(ft, bufs, gen, inl, 9)
    assert r == dict(
        meta_bad=0,
        rows=7,
        bad_rows=0,
        in_topk=7,
        lp_checked=7,
        lp_mismatch=0,
        lp_err_max=0.0,
    ), r
    assert oc.topk_check(ft, bufs, gen, inl, 10)["meta_bad"] == 1
    r = oc.topk_check(ft, bufs, gen, [x + 0.01 for x in inl], 9)
    assert r["lp_mismatch"] == 7 and abs(r["lp_err_max"] - 0.01) < 1e-6, r
    out_ = list(gen)
    out_[2] = 999999
    r = oc.topk_check(
        ft, bufs, out_, inl[:2] + [0.0] + inl[3:], 9
    )  # better than the k-th, not listed
    assert r["bad_rows"] == 1 and r["in_topk"] == 6, r
    r = oc.topk_check(
        ft, bufs, out_, inl[:2] + [-50.0] + inl[3:], 9
    )  # outside the top-k, legitimately
    assert r["bad_rows"] == 0 and r["in_topk"] == 6, r
    bad = data.copy()
    bad[0, 4:8] = bad[0, 0:4]  # repeated id
    bad[1, 4 * KK + 4 : 4 * KK + 8] = np.frombuffer(
        np.float32(5.0).tobytes(), np.uint8
    )  # increasing logprob
    r = oc.topk_check(ft, [bad[:4].tobytes(), bad[4:].tobytes()], gen, inl, 9)
    assert r["bad_rows"] == 2, r
    print(
        "topk_check: distinct ids, non-increasing logprobs, sampled logprob vs "
        "inline, missed sampled token"
    )


def t_snapshot_layout():
    import torch

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 300, generator=g) * 4
    buf = oc.topk_snapshot(logits, KK).numpy()
    assert buf.dtype == np.uint8 and buf.shape == (5, 8 * KK), buf.shape
    ids = buf[:, : 4 * KK].copy().view(np.int32)
    lps = buf[:, 4 * KK :].copy().view(np.float32)
    ref = torch.log_softmax(logits.float(), -1)
    want = torch.topk(logits, KK, dim=-1).indices.numpy()
    assert (ids == want).all() and np.allclose(
        lps, ref.gather(-1, torch.from_numpy(want)).numpy(), atol=1e-6
    )
    assert (np.diff(lps, axis=1) <= 0).all()
    print(
        "snapshot: [n, 8k] = k int32 ids by descending logit, then their logprobs "
        "(temperature 1)"
    )


def t_x10_check_topk():
    import t57f as base  # noqa: F401
    from t57n import ns_from

    st, mk, m, data, keys = build(start=9, rows=7)
    ft = m["fields"][0]
    ids = data[:, : 4 * KK].copy().view(np.int32)
    lps = data[:, 4 * KK :].copy().view(np.float32)
    gen, inl = [int(x) for x in ids[:, 0]], [float(x) for x in lps[:, 0]]
    failed = []
    ns = ns_from(
        pathlib.Path(base.DRIVER).read_text(encoding="utf8"),
        ["check_topk", "FAIL_KEYS"],
        base.DRIVER,
        get_many=lambda ks: [st.get(k) for k in ks],
        request_failed=lambda ph, s, rid, why, keys=None: failed.append(why),
    )
    ph = {}
    ns["check_topk"](ph, dict(name="s"), "r", ft, gen, inl, 10)
    assert (
        ph["topk_rows"] == 7
        and ph["topk_in_topk"] == 7
        and ph["topk_lp_mismatch"] == 0
        and ph["topk_bad_rows"] == 0
    ), ph
    ns["check_topk"](ph, dict(name="s"), "r", None, gen, inl, 10)
    assert ph["topk_manifest_missing"] == 1
    assert {
        "topk_manifest_missing",
        "topk_meta_bad",
        "topk_bad_rows",
        "topk_lp_mismatch",
    } <= set(ns["FAIL_KEYS"])
    del st.objs[keys[0]]
    ns["check_topk"](ph, dict(name="s"), "r", ft, gen, inl, 10)
    assert failed == ["absent topk chunk"], failed
    print(
        "x10: check_topk counts rows / in-top-k / mismatches; missing field and "
        "absent chunk fail"
    )


if __name__ == "__main__":
    t_snapshot_layout()
    t_reader_topk()
    t_topk_check()
    t_worker_chunks_follow_hidden()
    t_withheld_without_snapshot()
    t_x10_check_topk()
    print("all t57q tests passed")
