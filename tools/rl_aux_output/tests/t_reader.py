# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure tests of oc_reader.read_sample against objects built with the writer's own
encoders and the writer's real _oc_cone_finish (in-memory store, as in t52f). Run:
python -I t_reader.py"""

import hashlib
import json
import os
import pathlib
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pylib"))
os.environ["OC_R3_PARTIAL"] = "1"

import numpy as np  # noqa: E402
import oc_common as oc  # noqa: E402
import oc_reader as rd  # noqa: E402

src = pathlib.Path(os.path.join(HERE, "worker_attr.py")).read_text(encoding="utf8")
a = src.index("    def _oc_cone_finish(self, state, key_id, total: int, w):")
b = src.index(
    "    def _oc_idx_complete(self, state, start: int, n: int, w, want_cov=False):"
)
ns = {"_oc": oc, "np": np}
exec("class F:\n" + src[a:b], ns)
cone_finish = ns["F"]._oc_cone_finish

BS = 256
NO = 65535
R3_LAYERS = list(range(20, 40))
IDX_LAYERS = [0, 10, 20, 30]
K = 8
DEEP_IDX = [lyr for lyr in IDX_LAYERS if lyr >= 20]
D = 16  # hidden width
oc.IDX_CAP = types.SimpleNamespace(
    layer_ids=IDX_LAYERS, shape=[len(IDX_LAYERS), K], topk=K
)
GARBAGE = 60000  # outside the truth range [1, 59006)

MODES = {  # name: (IDX_MODE, IDX_ZSTD, manifest codec)
    "pack20-int32": ("full", "1", "pack20+zstd3"),
    "raw-int32": ("raw", "1", "pack20+zstd3"),
    "gpupack": ("gpupack", "1", "pack20+zstd3"),
    "gpupack-nozstd": ("gpupack", "0", "pack20+zstd3"),
    "gpudelta": ("gpudelta", "1", "dplane20+zstd3"),
    "gpudelta-nozstd": ("gpudelta", "0", "dplane20+zstd3"),
}


def set_mode(mode):
    oc.IDX_MODE, oc.IDX_ZSTD, oc.IDX_CODEC = MODES[mode]


def packed_mode():
    return oc.IDX_MODE in ("gpupack", "gpudelta")


# ------------------------------------------------------------ truth


def r3_truth(seed, n):
    r = np.random.default_rng(seed)
    base = r.integers(1, 59000, (n, 40, 1))
    return (base + np.arange(6)).astype(np.uint16)


def idx_truth(seed, n):
    r = np.random.default_rng(seed + 1000)
    v = r.integers(-1, 300000, (n, len(IDX_LAYERS), K))
    v[r.random(v.shape) < 0.2] = -1
    s = np.sort(np.where(v < 0, np.iinfo(np.int64).max, v), -1)
    return np.where(s == np.iinfo(np.int64).max, -1, s).astype(
        np.int32
    )  # engine order: ascending, pads last


def enc_rows(Ix):
    """Indexer rows as the GPU capture holds them for the current mode."""
    n, L = len(Ix), len(IDX_LAYERS)
    v = Ix.reshape(n, -1).astype(np.int64) + 1
    if oc.IDX_MODE == "gpudelta":
        s = np.sort(v.reshape(n, L, K), axis=-1)
        d = np.diff(s, axis=-1, prepend=0)
        hi = (d >> 16) & 15
        planes = (d & 255, (d >> 8) & 255, hi[..., 0::2] | (hi[..., 1::2] << 4))
        return np.concatenate([p.astype(np.uint8) for p in planes], -1).reshape(n, -1)
    if oc.IDX_MODE == "gpupack":
        u = v.astype(np.uint64)
        p = (u[:, 0::2] | (u[:, 1::2] << np.uint64(20))).astype("<u8")
        return p.view(np.uint8).reshape(n, -1, 8)[:, :, :5].reshape(n, -1).copy()
    return Ix.copy()


def deep_parts(E):
    """Deep indexer slots of capture rows, [n, deep layers, cell]."""
    return E.reshape(len(E), len(IDX_LAYERS), -1)[:, len(IDX_LAYERS) - len(DEEP_IDX) :]


def step(deep, layers, start, end):
    """A prefill step's deep cells: exact only in each layer's cone, garbage before
    it."""
    rows = np.array(deep[start:end], copy=True)
    flat = rows.view(np.uint8).reshape(end - start, len(layers), -1)
    for j, layer in enumerate(layers):
        lo = max(start, end - oc.cone_size(layer))
        flat[: lo - start, j] = 0x5A
    return rows


# ------------------------------------------------------------ in-memory writer


class W:
    def __init__(self):
        self.objs, self.stats = {}, {}
        self.rowver, self.ends, self.hid = {}, {}, {}

    def put(self, k, v):
        self.objs[k] = v() if callable(v) else v

    def get(self, k):
        return self.objs.get(k)

    def batch_is_exist(self, keys):
        return [1 if k in self.objs else 0 for k in keys]


def segs(rv):
    out = []
    for p, v in enumerate(rv):
        if not out or out[-1][1] != int(v):
            out.append([p, int(v)])
    return out


def request(w, track, rid, T, Ix, aux, start, P, On, version, parent=None):
    """One request: prefill [start, P), decode to total = P + O - 1 rows. Mirrors
    _oc_finish."""
    total = P + On - 1
    if parent is not None:
        rv0 = w.rowver[parent][:aux]
        w.ends[track] = [e for e in w.ends[parent] if e[0] <= aux]
        # As the writer's _oc_inherit (v54): the parent's patches and ends up to the
        # fork point.
        w.cone_list = getattr(w, "cone_list", {})
        w.cone_list[track] = [
            dict(p, end=min(p["end"], aux + 1))
            for p in w.cone_list.get(parent, [])
            if p["lo"] <= aux
        ]
        w.cone_ends = getattr(w, "cone_ends", {})
        w.cone_ends[track] = [
            [p_, min(e_, aux + 1)]
            for p_, e_ in w.cone_ends.get(parent, [])
            if p_ <= aux + 1
        ]
    else:
        rv0 = w.rowver.get(track, np.zeros(0, np.int64))[:aux]
    rv = np.concatenate([rv0, np.full(total - len(rv0), version, np.int64)])
    w.rowver[track] = rv
    w.ends[track] = w.ends.get(track, []) + [(P, total)]
    E = enc_rows(Ix[:total])
    nfull = total // BS
    names = []
    for i in range(nfull):
        h = hashlib.blake2b(T[: (i + 1) * BS].tobytes(), digest_size=8).hexdigest()
        vs = rv[i * BS : (i + 1) * BS]
        name = (
            h
            if (vs == 0).all()
            else h + "~" + hashlib.blake2b(vs.tobytes(), digest_size=3).hexdigest()
        )
        names.append(name)
        key = oc.block_key("s", name)
        if key not in w.objs:
            blk = np.array(T[i * BS : (i + 1) * BS], copy=True)
            blk[:, 20:] = GARBAGE  # trimmed layers of blocks are not exact
            w.put(key, blk.tobytes())
            w.put(key + oc.IDX_SUFFIX, oc.idx_encode(E[i * BS : (i + 1) * BS]))
    tail_start = nfull * BS
    tail = np.array(T[tail_start:total], copy=True)
    tail[:, 20:] = GARBAGE
    w.put(oc.tail_key("s", rid), tail.tobytes() or b"\0")
    w.put(
        oc.idx_tail_key("s", rid),
        oc.idx_encode(E[tail_start:total]) if total > tail_start else b"\0",
    )
    st = types.SimpleNamespace(
        oc_track=track,
        session="s",
        oc_aux_start=aux,
        oc_cone_prompt=P,
        oc_cone=oc.ConeParts(R3_LAYERS),
        oc_cone_idx=oc.ConeParts(DEEP_IDX),
    )
    st.oc_cone.add(start, P, step(T[:, 20:], R3_LAYERS, start, P), version)
    st.oc_cone_idx.add(start, P, step(deep_parts(E), DEEP_IDX, start, P), version)
    for p in range(P, total):
        st.oc_cone.add(p, p + 1, T[p : p + 1, 20:], version)
        st.oc_cone_idx.add(p, p + 1, deep_parts(E)[p : p + 1], version)
    cone = cone_finish(None, st, rid, total, w)
    rng = np.random.default_rng(abs(hash(rid)) % 1000)
    q = rng.standard_normal((On, D)).astype(np.float32)
    lens = []
    for c, i in enumerate(range(0, On, oc.HID_CHUNK)):
        k = min(oc.HID_CHUNK, On - i)
        w.put(oc.hid_key("s", rid, c), q[i : i + k].tobytes())
        lens.append(k)
    w.hid[rid] = (q, P - 1)
    fields = [
        dict(
            name="routed_experts",
            dtype="uint16",
            shape=[40, 6],
            codec="raw",
            layout="blocks+tail",
            rows=total,
            nbytes=0,
        ),
        dict(
            name="indexer_topk",
            dtype="int32",
            shape=[len(IDX_LAYERS), K],
            codec=oc.IDX_CODEC,
            layout="blocks+tail",
            block_suffix=oc.IDX_SUFFIX,
            tail_key=oc.idx_tail_key("s", rid),
            layers=IDX_LAYERS,
            rows=total,
            tail_start=tail_start,
            tail_rows=total - tail_start,
            missing=0,
            holes=0,
            nbytes=0,
        ),
        dict(
            name="hidden",
            dtype="float32",
            shape=[D],
            codec="raw",
            layout="chunks",
            key_prefix=oc.hid_key("s", rid, ""),
            start=P - 1,
            rows=On,
            chunk_rows=oc.HID_CHUNK,
            chunks=len(lens),
            chunk_lens=lens,
            nbytes=0,
        ),
    ]
    m = dict(
        key_prefix=f"{oc.PREFIX}/s",
        tail_key=oc.tail_key("s", rid),
        blocks=names,
        tail_start=tail_start,
        rows=total,
        tail_nbytes=len(tail.tobytes()),
        dtype="uint16",
        shape=[40, 6],
        block_size=BS,
        missing_from_session=0,
        invalid_from=None,
        weight_versions=segs(rv),
        r3_cone=cone,
        fields=fields,
    )
    mk = oc.manifest_key("s", rid)
    w.put(mk, json.dumps(m).encode())
    return mk, total


def check_full(w, mk, n, T, Ix, track, name, versions_deep=None, ends=None):
    ends = w.ends[track] if ends is None else ends
    s = rd.read_sample(w.get, mk, n)
    assert s.ok, (name, s.error, s.missing.counts)
    assert s.r3.shape == (n, 40, 6) and s.indexer.shape == (n, 4, K), name
    # Shallow cells: every row delivered, equal to the truth, versions from the
    # segments.
    assert s.r3_replay_mask[:, :20].all() and (s.r3[:, :20] == T[:n, :20]).all(), name
    assert (s.weight_version == w.rowver[track][:n]).all(), name
    assert (s.r3_version[:, :20] == w.rowver[track][:n, None]).all(), name
    # Deep cells: every needed cell replayed and right; the block garbage never shows.
    need = oc.cone_mask(n, ends, R3_LAYERS)
    rm = s.r3_replay_mask[:, 20:]
    assert rm[need].all(), (
        name,
        "r3 needed cells not replayed",
        int((need & ~rm).sum()),
    )
    assert (s.r3[:, 20:][rm] == T[:n, 20:][rm]).all(), (name, "r3 deep wrong")
    assert (s.r3[:, 20:][~rm] == NO).all() and not (s.r3 == GARBAGE).any(), name
    assert (s.r3_version[:, 20:][~rm] == -1).all() and (
        s.r3_version[:, 20:][rm] >= 0
    ).all(), name
    # Indexer: shallow slots exact, deep slots from the patches, -2 where not replayed.
    im = s.indexer_replay_mask
    assert im[:, :2].all() and (s.indexer[:, :2] == Ix[:n, :2]).all(), (
        name,
        "indexer shallow",
    )
    ineed = oc.cone_mask(n, ends, DEEP_IDX)
    assert im[:, 2:][ineed].all(), (
        name,
        "indexer needed cells",
        int((ineed & ~im[:, 2:]).sum()),
    )
    assert (s.indexer[:, 2:][im[:, 2:]] == Ix[:n, 2:][im[:, 2:]]).all(), (
        name,
        "indexer deep wrong",
    )
    assert (s.indexer[:, 2:][~im[:, 2:]] == rd.IDX_NO_REPLAY).all(), name
    assert s.uncovered == {"r3": 0, "indexer": 0}, (name, s.uncovered)
    assert [tuple(e) for e in s.manifest["r3_cone"]["ends"]] == [
        tuple(e) for e in ends
    ], (name, "ends")
    assert not s.r3_needed_missing.any() and not s.indexer_needed_missing.any(), name
    # Hidden: this request's generated positions only.
    q, start = w.hid[mk.split("/")[-2]]
    k = n - start
    assert s.hidden_start == start and len(s.hidden) == k, (
        name,
        s.hidden_start,
        len(s.hidden),
    )
    assert (s.hidden == q[:k]).all(), (name, "hidden")
    if versions_deep is not None:
        versions_deep(s)
    return s


# ------------------------------------------------------------ tests


def t_plain(mode):
    set_mode(mode)
    w = W()
    T, Ix = r3_truth(1, 8000), idx_truth(1, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    check_full(w, mk, total, T, Ix, "a", "plain/" + mode)
    # Manifest covering more rows than L - 1 (truncated drafts): truncated, still
    # complete.
    s2 = rd.read_sample(w.get, mk, total - 70)
    assert (
        s2.ok
        and len(s2.r3) == total - 70
        and (s2.r3[:, :20] == T[: total - 70, :20]).all()
    )
    assert len(s2.hidden) == total - 70 - (3000 - 1)
    # Asking for more rows than the manifest covers: reported, never padded silently.
    s3 = rd.read_sample(w.get, mk, total + 5)
    assert (
        not s3.ok
        and s3.missing.counts["rows_short"] == 5
        and s3.missing.rows[total:].all()
    )
    assert not s3.missing.rows[:total].any() and not s3.r3_replay_mask[total:].any()
    assert (s3.indexer[total:] == -2).all() and (s3.r3[total:] == NO).all()
    for want in (("r3",), ("indexer",)):
        s4 = rd.read_sample(w.get, mk, total + 5, want=want)
        assert (
            not s4.ok
            and s4.missing.rows[total:].all()
            and not s4.missing.rows[:total].any()
        ), want
    print(f"plain request ok [{mode}]")


def t_multiturn():
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(2, 20 * 1000), idx_truth(2, 20 * 1000)
    turns = [
        (0, 0, 3000, 100, 0),
        (3099, 3072, 6200, 80, 1),
        (6279, 6144, 6500, 300, 2),
        (6799, 6656, 9800, 60, 3),
    ]
    mks = []
    for i, (aux, start, P, On, v) in enumerate(turns):
        mk, total = request(w, "a", f"t{i}", T, Ix, aux, start, P, On, v)
        mks.append(mk)
        assert total == (turns[i + 1][0] if i + 1 < len(turns) else total)
    m = json.loads(w.objs[mk])
    assert len([p for p in m["r3_cone"]["patches"] if "key" in p]) >= 3, m["r3_cone"][
        "patches"
    ]

    def vers(s):
        # First computation wins: a cell any earlier turn needed keeps that turn's
        # version.
        rm = s.r3_replay_mask[:, 20:]
        for j, (aux, start, P, On, v) in enumerate(turns):
            ends = w.ends["a"][: j + 1]
            mine = oc.cone_mask(total, ends, R3_LAYERS)
            if j:
                mine &= ~oc.cone_mask(total, w.ends["a"][:j], R3_LAYERS)
            got = s.r3_version[:, 20:][mine & rm]
            assert (got == v).all(), (j, np.unique(got))
        assert len(np.unique(s.r3_version[:, 20:][rm])) == 4

    check_full(w, mk, total, T, Ix, "a", "multi-turn", vers)
    # An earlier turn's manifest is a complete sample of its own.
    check_full(
        w, mks[1], 6200 + 80 - 1, T, Ix, "a", "multi-turn turn 1", ends=w.ends["a"][:2]
    )
    print("multi-turn trajectory with patches from 4 requests ok")
    return w, mk, total, T, Ix


def t_fork():
    set_mode("gpudelta")
    w = W()
    T, Ix = r3_truth(3, 20 * 1000), idx_truth(3, 20 * 1000)
    request(w, "p", "p0", T, Ix, 0, 0, 5000, 200, 0)
    request(w, "p", "p1", T, Ix, 5199, 5120, 9000, 100, 1)
    C, J = T.copy(), Ix.copy()
    C[7000:], J[7000:] = r3_truth(9, 20 * 1000)[7000:], idx_truth(9, 20 * 1000)[7000:]
    mk, total = request(w, "c", "c0", C, J, 7000, 4864, 7400, 50, 2, parent="p")
    s = check_full(w, mk, total, C, J, "c", "fork")
    # Rows below the fork point carry the parent's versions; the fork's own rows its
    # own.
    assert (s.weight_version[:5199] == 0).all() and (
        s.weight_version[5199:7000] == 1
    ).all()
    assert (s.weight_version[7000:] == 2).all()
    # The parent's later turn is not visible to the fork: its deep cells past 7000 are
    # the fork's.
    rm = s.r3_replay_mask[7000:, 20:]
    assert (s.r3_version[7000:, 20:][rm] == 2).all()
    # Below the fork point the parent's patches come first: cells the parent's first
    # turn needed keep version 0 although the fork recomputed (and stored) them at
    # version 2.
    pneed = oc.cone_mask(5199, w.ends["p"][:1], R3_LAYERS)
    assert (s.r3_version[:5199, 20:][pneed] == 0).all(), np.unique(
        s.r3_version[:5199, 20:][pneed]
    )
    assert (
        s.indexer_version[:5199, 2:][oc.cone_mask(5199, w.ends["p"][:1], DEEP_IDX)] == 0
    ).all()
    assert (s.r3_version[7000:, 20:][rm] == 2).all()
    print("fork ok")


def t_missing_block():
    set_mode("gpupack")
    w = W()
    T, Ix = r3_truth(4, 8000), idx_truth(4, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    m = json.loads(w.objs[mk])
    del w.objs[oc.block_key("s", m["blocks"][3])]
    s = rd.read_sample(w.get, mk, total)
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
        and s.missing.counts["r3_blocks"] == 1
    ), s.missing.counts
    assert not s.r3_replay_mask.any() and not s.indexer_replay_mask.any()
    # Cells on failed rows are not double-reported as needed-but-missing.
    assert not s.r3_needed_missing.any() and "r3_needed_missing" not in s.missing.counts
    # Indexer block alone missing: the same rows, counted under the indexer.
    w2 = W()
    mk, total = request(w2, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    del w2.objs[oc.block_key("s", m["blocks"][5]) + "/idx"]
    s = rd.read_sample(w2.get, mk, total)
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
        and s.missing.counts["indexer_blocks"] == 1
    ), s.missing.counts
    # A missing tail.
    w3 = W()
    mk, total = request(w3, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    del w3.objs[oc.tail_key("s", "r0")]
    s = rd.read_sample(w3.get, mk, total)
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
        and s.missing.counts["r3_tail"] == 1
    ), s.missing.counts
    try:
        rd.read_sample(w3.get, mk, total, strict=True)
        raise AssertionError("strict read returned")
    except rd.ReadError:
        pass
    print("missing block / indexer block / tail: the request fails")


def t_missing_patch():
    w, mk, total, T, Ix = t_multiturn_quiet()
    m = json.loads(w.objs[mk])
    p = [p for p in m["r3_cone"]["patches"] if "key" in p][1]
    del w.objs[p["key"]]
    s = rd.read_sample(w.get, mk, total)
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
        and s.missing.counts["r3_cone_patches"] == 1
    ), s.missing.counts
    assert not s.r3_replay_mask.any()
    # Indexer patch missing.
    w, mk, total, T, Ix = t_multiturn_quiet()
    m = json.loads(w.objs[mk])
    p = [p for p in m["r3_cone"]["patches"] if "idx_key" in p][2]
    del w.objs[p["idx_key"]]
    s = rd.read_sample(w.get, mk, total)
    assert (
        s.missing.counts["indexer_cone_patches"] == 1
        and s.missing.counts.get("r3_cone_patches", 0) == 0
    )
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
    )
    print("missing cone patches: the request fails")


def t_multiturn_quiet():
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(2, 20 * 1000), idx_truth(2, 20 * 1000)
    for i, (aux, start, P, On, v) in enumerate(
        [
            (0, 0, 3000, 100, 0),
            (3099, 3072, 6200, 80, 1),
            (6279, 6144, 6500, 300, 2),
            (6799, 6656, 9800, 60, 3),
        ]
    ):
        mk, total = request(w, "a", f"t{i}", T, Ix, aux, start, P, On, v)
    return w, mk, total, T, Ix


def t_error_marker():
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(5, 8000), idx_truth(5, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    w.objs[oc.err_key(mk)] = b"tail write failed"
    s = rd.read_sample(w.get, mk, total)
    assert s.ok and s.stale_marker, s.error  # a complete manifest wins over the marker
    del w.objs[mk]
    s = rd.read_sample(w.get, mk, total)
    assert (
        not s.ok
        and s.error.startswith("error marker")
        and s.missing.rows.all()
        and s.r3 is None
        and s.indexer is None
    )
    try:
        rd.read_sample(w.get, mk, total, strict=True)
        raise AssertionError("strict read returned")
    except rd.ReadError:
        pass
    s = rd.read_sample(w.get, oc.manifest_key("s", "nope"), 10)
    assert not s.ok and "manifest" in s.error
    print(
        "error marker without a manifest and absent manifest fail the sample; "
        "beside a manifest it is stale"
    )


def t_codecs():
    for mode in MODES:
        t_plain(mode)
    # Value past 20 bits in a shallow slot: int32 fallback object (tag 2).
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(6, 8000), idx_truth(6, 8000)
    Ix[100, 0, -1] = 1 << 21
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    assert (
        w.objs[oc.block_key("s", json.loads(w.objs[mk])["blocks"][0]) + "/idx"][:1]
        == b"\x02"
    )
    check_full(w, mk, total, T, Ix, "a", "int32 fallback")
    # An object whose tag disagrees with the manifest's codec is not decoded as the
    # wrong codec.
    set_mode("gpupack")
    w = W()
    mk, total = request(w, "a", "r0", T, idx_truth(6, 8000), 0, 0, 3000, 300, 0)
    m = json.loads(w.objs[mk])
    m["fields"][1]["codec"] = "dplane20+zstd3"
    s = rd.read_sample(w.get, mk, total, manifest=m, want=("indexer",))
    assert (
        s.missing.counts["indexer_blocks"] == total // BS
        and s.missing.counts["indexer_tail"] == 1
    )
    assert not s.indexer_replay_mask[:, :2].any()
    print("all indexer codecs ok; codec mismatch reported")


def t_reported_and_invalid():
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(7, 8000), idx_truth(7, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    m = json.loads(w.objs[mk])
    m["fields"][1]["missing"] = 1
    s = rd.read_sample(w.get, mk, total, manifest=m)
    assert not s.ok and s.missing.rows.all() and not s.indexer_replay_mask.any()
    m = json.loads(w.objs[mk])
    m["invalid_from"] = 2000
    s = rd.read_sample(w.get, mk, total, manifest=m)
    assert (
        s.missing.counts["invalid_rows"] == total - 2000 and s.missing.rows[2000:].all()
    )
    assert not s.missing.rows[:2000].any() and s.r3_replay_mask[:2000, :20].all()
    # A delivered row that fails the R3 sanity check (repeated expert id) is dropped.
    blk = bytearray(w.objs[oc.block_key("s", json.loads(w.objs[mk])["blocks"][1])])
    arr = np.frombuffer(blk, np.uint16).reshape(BS, 40, 6).copy()
    arr[10, 3, 1] = arr[10, 3, 0]
    w.objs[oc.block_key("s", json.loads(w.objs[mk])["blocks"][1])] = arr.tobytes()
    s = rd.read_sample(w.get, mk, total)
    assert (
        s.missing.counts["r3_invalid_rows"] == 1
        and s.missing.rows[BS + 10]
        and s.missing.rows.sum() == 1
    )
    # Missing hidden chunk: the request fails.
    w = W()
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    del w.objs[oc.hid_key("s", "r0", 2)]
    s = rd.read_sample(w.get, mk, total)
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
        and s.missing.counts["hidden_chunks"] == 1
    ), s.missing.counts
    print("reported missing, invalid_from, uncovered, R3 sanity, hidden chunk ok")


def t_needed_missing():
    set_mode("pack20-int32")
    T, Ix = r3_truth(11, 8000), idx_truth(11, 8000)
    # The replay started at 2944: needed cells of the window before it were never
    # computed.
    w = W()
    mk, total = request(w, "a", "r0", T, Ix, 0, 2944, 3000, 50, 0)
    m = json.loads(w.objs[mk])
    assert m["r3_cone"]["uncovered"] > 0 and m["r3_cone"]["uncovered_idx"] > 0
    s = rd.read_sample(w.get, mk, total)
    need = oc.cone_mask(total, w.ends["a"], R3_LAYERS)
    want_nm = need & ~s.r3_replay_mask[:, 20:]
    assert (s.r3_needed_missing[:, 20:] == want_nm).all() and not s.r3_needed_missing[
        :, :20
    ].any()
    assert (
        s.missing.counts["r3_needed_missing"]
        == m["r3_cone"]["uncovered"]
        == int(want_nm.sum())
    )
    assert s.missing.counts["indexer_needed_missing"] == m["r3_cone"]["uncovered_idx"]
    assert not s.ok and not s.missing.rows.any()
    assert not s.r3_needed_missing[
        :500
    ].any()  # no-replay cells nobody needs are not missing
    # A later turn: its own manifest counts 0 uncovered, the per-position mask still has
    # the holes.
    mk2, total2 = request(w, "a", "r1", T, Ix, total, 3072, 6000, 40, 1)
    m2 = json.loads(w.objs[mk2])
    assert m2["r3_cone"]["uncovered"] == 0
    s2 = rd.read_sample(w.get, mk2, total2)
    assert s2.missing.counts["r3_needed_missing"] == m["r3_cone"]["uncovered"], (
        s2.missing.counts
    )
    assert (s2.r3_needed_missing[:total] == s.r3_needed_missing).all()
    # Manifests before v54 (no key_prefix, tail_key, ends, idx_codec): keys derived, the
    # request's own uncovered count reported, no per-position mask.
    old = json.loads(w.objs[mk])
    for k in ("key_prefix", "tail_key"):
        del old[k]
    del old["r3_cone"]["ends"]
    for p in old["r3_cone"]["patches"]:
        p.pop("idx_codec", None)
    s3 = rd.read_sample(w.get, mk, total, manifest=old)
    assert (
        s3.r3_needed_missing is None
        and s3.missing.counts["r3_uncovered"] == m["r3_cone"]["uncovered"]
    )
    assert (s3.r3 == s.r3).all() and (s3.indexer == s.indexer).all()
    print("needed-but-missing cells reported per position; pre-v54 fallback ok")


def t_patch_codec():
    # Patches decode by their own idx_codec: blocks stored as int32 (valid under any
    # codec), the field naming pack20, the patches written as dplane20.
    set_mode("gpudelta")
    w = W()
    T, Ix = r3_truth(12, 8000), idx_truth(12, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 300, 0)
    m = json.loads(w.objs[mk])
    assert all(
        p.get("idx_codec") == "dplane20+zstd3"
        for p in m["r3_cone"]["patches"]
        if "idx_key" in p
    )
    oc.IDX_MODE = "raw"
    for i, b in enumerate(m["blocks"]):
        w.objs[oc.block_key("s", b) + "/idx"] = oc.idx_encode(Ix[i * BS : (i + 1) * BS])
    w.objs[oc.idx_tail_key("s", "r0")] = oc.idx_encode(Ix[m["tail_start"] : total])
    m["fields"][1]["codec"] = "pack20+zstd3"
    w.objs[mk] = json.dumps(m).encode()
    check_full(w, mk, total, T, Ix, "a", "patch codec")
    print("indexer patches decode by idx_codec")


def manual(dtype, shape, tail_rows, idx=None, idx_shape=None):
    """A one-request manifest with only a tail (no full blocks), for edge cases."""
    w = W()
    n = len(tail_rows)
    w.put(oc.tail_key("s", "e"), tail_rows.tobytes())
    fields = [
        dict(
            name="routed_experts",
            dtype=dtype,
            shape=shape,
            codec="raw",
            layout="blocks+tail",
            rows=n,
        )
    ]
    if idx is not None:
        w.put(oc.idx_tail_key("s", "e"), idx)
        fields.append(
            dict(
                name="indexer_topk",
                dtype="int32",
                shape=idx_shape,
                codec="pack20+zstd3",
                layout="blocks+tail",
                block_suffix="/idx",
                tail_key=oc.idx_tail_key("s", "e"),
                layers=list(range(idx_shape[0])),
                rows=n,
                tail_start=0,
                tail_rows=n,
                missing=0,
                holes=0,
            )
        )
    m = dict(
        key_prefix=f"{oc.PREFIX}/s",
        tail_key=oc.tail_key("s", "e"),
        blocks=[],
        tail_start=0,
        rows=n,
        tail_nbytes=len(tail_rows.tobytes()),
        dtype=dtype,
        shape=shape,
        block_size=BS,
        invalid_from=None,
        weight_versions=[[0, 0]],
        fields=fields,
    )
    mk = oc.manifest_key("s", "e")
    w.put(mk, json.dumps(m).encode())
    return w, mk


def t_r17_deep():
    set_mode("pack20-int32")
    # P1: deep cells delivered by a patch get the same sanity check as shallow ones.
    T, Ix = r3_truth(13, 8000), idx_truth(13, 8000)
    T[2990, 25] = NO
    T[2995, 30, 1] = T[2995, 30, 0]
    w = W()
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
    s = rd.read_sample(w.get, mk, total)
    assert s.missing.counts.get("r3_invalid_rows") == 2, s.missing.counts
    assert list(np.flatnonzero(s.missing.rows)) == [2990, 2995]
    assert (
        not s.r3_replay_mask[[2990, 2995]].any()
        and not s.indexer_replay_mask[[2990, 2995]].any()
    )


def t_r17_uint8():
    # P2: one-byte expert ids: 255 is an id, not a sentinel.
    rows = np.array(
        [[[250, 251, 252, 253, 254, 255], [0, 1, 2, 3, 4, 5]]] * 3, np.uint8
    )
    w, mk = manual("uint8", [2, 6], rows)
    s = rd.read_sample(w.get, mk, 3, want=("r3",))
    assert s.ok and s.r3_replay_mask.all() and (s.r3 == rows).all(), s.missing.counts
    rows[1, 1, 2] = 1
    w, mk = manual("uint8", [2, 6], rows)
    s = rd.read_sample(w.get, mk, 3, want=("r3",))
    assert list(np.flatnonzero(s.missing.rows)) == [1]


def t_r17_odd_pack20():
    set_mode("pack20-int32")
    # P2: pack20 with an odd value count (the writer pads the last pair).
    saved = oc.IDX_CAP
    oc.IDX_CAP = types.SimpleNamespace(layer_ids=[0], shape=[1, 3], topk=3)
    for idx in (
        np.array([[[3, 5, 7]]], np.int32),
        np.array([[[3, 5, 7]], [[-1, 0, 9]], [[1, 2, 4]]], np.int32),
    ):
        buf = oc.idx_encode(idx)
        assert buf[:1] == b"\x01" and (oc.idx_decode(buf, (1, 3)) == idx).all()
        r3 = np.tile(np.arange(6, dtype=np.uint16), (len(idx), 2, 1))
        w, mk = manual("uint16", [2, 6], r3, idx=buf, idx_shape=[1, 3])
        s = rd.read_sample(w.get, mk, len(idx), want=("r3", "indexer"))
        assert s.ok and (s.indexer == idx).all() and s.indexer_replay_mask.all(), (
            s.missing.counts
        )
    oc.IDX_CAP = saved


def t_r17_absent_field():
    set_mode("pack20-int32")
    # P2: a wanted output whose field is absent is missing, routed_experts included.
    w = W()
    mk, total = request(
        w, "a", "r0", r3_truth(14, 8000), idx_truth(14, 8000), 0, 0, 3000, 50, 0
    )
    m = json.loads(w.objs[mk])
    m["fields"] = [f for f in m["fields"] if f["name"] != "routed_experts"]
    s = rd.read_sample(w.get, mk, total, manifest=m)
    assert (
        not s.ok
        and s.missing.counts["r3_field"] == 1
        and s.missing.rows.all()
        and s.r3 is None
    )
    assert not s.indexer_replay_mask.any()
    s = rd.read_sample(w.get, mk, total, manifest=m, want=("indexer", "hidden"))
    assert s.ok


def t_r17_zero_rows():
    set_mode("pack20-int32")
    w = W()
    mk, total = request(
        w, "a", "r0", r3_truth(14, 8000), idx_truth(14, 8000), 0, 0, 3000, 50, 0
    )
    # P2: n_rows = 0.
    s = rd.read_sample(w.get, mk, 0)
    assert (
        s.ok
        and s.r3.shape == (0, 40, 6)
        and s.indexer.shape == (0, 4, K)
        and len(s.missing.rows) == 0
    )
    print(
        "R17 findings fixed: deep sanity, uint8 ids, odd pack20, absent field, n_rows=0"
    )


def _raw_hidden(w, mk, rid, dtype, bits_dtype):
    """Replace a request's hidden chunks and field with raw rows of `dtype` (bit
    patterns drawn as `bits_dtype`)."""
    m = json.loads(w.objs[mk])
    fh = [f for f in m["fields"] if f["name"] == "hidden"][0]
    rng = np.random.default_rng(len(rid))
    bits = rng.integers(
        0, np.iinfo(bits_dtype).max, (fh["rows"], D), dtype=np.uint64
    ).astype(bits_dtype)
    at = 0
    for i, k in enumerate(fh["chunk_lens"]):
        w.objs[f"{fh['key_prefix']}{i}"] = bits[at : at + k].tobytes()
        at += k
    fh.update(dtype=dtype, codec="raw")
    w.objs[mk] = json.dumps(m).encode()
    return bits, fh


def t_hidden_raw():
    # v56: raw hidden rows read back bit-exactly in their stored dtype.
    set_mode("pack20-int32")
    T, Ix = r3_truth(19, 8000), idx_truth(19, 8000)
    for dtype, bits_dtype, np_dtype in (
        ("float32", np.uint32, np.float32),
        ("float16", np.uint16, np.float16),
        ("bfloat16", np.uint16, None),
    ):
        w = W()
        mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 150, 0)
        bits, fh = _raw_hidden(w, mk, "r0", dtype, bits_dtype)
        s = rd.read_sample(w.get, mk, total, want=("hidden",))
        assert s.ok, (dtype, s.missing.counts)
        assert s.hidden_dtype == dtype, dtype
        assert s.hidden_start == fh["start"] and s.hidden.shape == (
            total - fh["start"],
            D,
        ), dtype
        if np_dtype is not None:
            assert s.hidden.dtype == np_dtype, (dtype, s.hidden.dtype)
        got = s.hidden.view(bits_dtype)
        assert (got == bits[: len(got)]).all(), dtype
        # A lost chunk: the request fails (its positions still hold NaN; bfloat16 bits:
        # 0x7FC0).
        del w.objs[oc.hid_key("s", "r0", 1)]
        s = rd.read_sample(w.get, mk, total, want=("hidden",))
        lo, k = fh["chunk_lens"][0], fh["chunk_lens"][1]
        assert (
            not s.ok
            and (s.error or "").startswith("listed object absent")
            and s.missing.rows.all()
            and s.missing.counts["hidden_chunks"] == 1
        ), (dtype, s.missing.counts)
        gap = s.hidden[lo : lo + k]
        assert (
            (gap.view(np.uint16) == 0x7FC0).all()
            if np_dtype is None
            else np.isnan(gap).all()
        ), dtype
        # Unknown dtype: reported, never misread.
        m = json.loads(w.objs[mk])
        [f for f in m["fields"] if f["name"] == "hidden"][0]["dtype"] = "float8_e5m2"
        s = rd.read_sample(w.get, mk, total, manifest=m, want=("hidden",))
        assert s.missing.counts.get("hidden_meta") == 1, s.missing.counts
    print("hidden: raw float32/float16/bfloat16 bit-exact, lost chunk reported")


def t_hidden_codec_strict():
    # Hidden is raw only: a field declaring any other codec is rejected (reader error),
    # never decoded.
    set_mode("pack20-int32")
    T, Ix = r3_truth(20, 8000), idx_truth(20, 8000)
    w = W()
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 70, 0)
    m = json.loads(w.objs[mk])
    fh = [f for f in m["fields"] if f["name"] == "hidden"][0]
    rows = np.tile(np.array([1, 2, 3, 4], np.float16), (fh["rows"], 1))
    at = 0
    for i, k in enumerate(fh["chunk_lens"]):
        w.objs[f"{fh['key_prefix']}{i}"] = rows[at : at + k].tobytes()
        at += k
    fh.update(dtype="float16", shape=[4], codec="raw")
    s = rd.read_sample(w.get, mk, total, manifest=m, want=("hidden",))
    assert s.ok and (s.hidden == rows[: len(s.hidden)]).all()
    for bad in (
        dict(codec="typo"),
        dict(codec=None),
        dict(codec="fp8e4m3+rowscale-f32"),
        dict(codec="fp8e4m3+rowscale-f32", dtype="float8_e4m3fn"),
    ):
        f2 = dict(fh, **bad)
        m2 = dict(m, fields=[f2 if f["name"] == "hidden" else f for f in m["fields"]])
        s = rd.read_sample(w.get, mk, total, manifest=m2, want=("hidden",))
        assert not s.ok and "only raw" in (s.error or "") and s.hidden is None, (
            bad,
            s.error,
        )
    print("hidden: only raw is decoded; any other codec is rejected")


def t_key_prefix():
    # A caller request id with '/' : only the manifest's key_prefix/tail_key find the
    # objects.
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(15, 8000), idx_truth(15, 8000)
    mk, total = request(w, "a", "job7/r0", T, Ix, 0, 0, 3000, 50, 0)
    s = rd.read_sample(w.get, mk, total)
    assert s.ok and (s.r3[:, :20] == T[:total, :20]).all(), s.missing.counts
    print("keys from key_prefix / tail_key")


def _rewrite_patch(w, key, dtype, cell, fn):
    """Re-encode a stored patch with another dtype / cell shape (values through fn)."""
    import zstandard

    buf = w.objs[key]
    k = int.from_bytes(buf[:4], "little")
    h = json.loads(buf[4 : 4 + k])
    body = buf[4 + k :]
    if h.get("zstd"):
        body = zstandard.ZstdDecompressor().decompress(body)
    nc = int(np.prod(h["cov"]))
    packed = (nc + 7) // 8
    cells = int(np.unpackbits(np.frombuffer(body, np.uint8, packed), count=nc).sum())
    nv = cells * int(np.prod(h["shape"][2:]))
    vals = np.frombuffer(body, np.dtype(h["dtype"]), nv, packed)
    newvals = fn(vals)
    h2 = dict(h, dtype=dtype, shape=[*h["shape"][:2], *cell])
    h2.pop("zstd", None)
    hb = json.dumps(h2).encode()
    w.objs[key] = (
        len(hb).to_bytes(4, "little")
        + hb
        + body[:packed]
        + newvals.tobytes()
        + body[packed + vals.nbytes :]
    )


def t_r18_version_unknown():
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(16, 8000), idx_truth(16, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
    # The writer's own segment builder: a prefix-hit gap recorded as -1 ("as recorded by
    # the object") and no block record for it.
    ns2 = {"np": np}
    wa = pathlib.Path(os.path.join(HERE, "worker_attr.py")).read_text(encoding="utf8")
    exec(wa[wa.index("def _clip_segs") : wa.index("def _oc_keep_tail")], ns2)
    segs = ns2["_object_segs"](
        [[0, 3], [2800, -1], [3000, 4]],
        [("s", f"h{i}") for i in range(total // BS)],
        BS,
        {},
        total,
    )
    m = json.loads(w.objs[mk])
    m["weight_versions"] = segs
    s = rd.read_sample(w.get, mk, total, manifest=m)
    unknown = np.zeros(total, bool)
    unknown[2800:3000] = True
    assert (
        s.ok and s.version_unknown is not None and (s.version_unknown == unknown).all()
    ), segs
    assert s.version_unknown_count == 200
    assert (
        s.r3_replay_mask[:, :20].all() and s.indexer_replay_mask[:, :2].all()
    )  # still replayable
    assert (s.weight_version[unknown] == -1).all() and (
        s.weight_version[3000:] == 4
    ).all()
    s = rd.read_sample(w.get, mk, total)
    assert s.version_unknown_count == 0 and not s.version_unknown.any()
    # A lost object fails the request: nothing is reported as version-unknown.
    del w.objs[oc.block_key("s", m["blocks"][10])]
    s = rd.read_sample(w.get, mk, total, manifest=m)
    assert (
        not s.ok
        and (s.error or "").startswith("listed object absent")
        and s.missing.rows.all()
        and s.version_unknown_count == 0
    )
    print("unknown-version rows reported, still replayable")


def t_r18_patch_type():
    set_mode("pack20-int32")
    T, Ix = r3_truth(17, 8000), idx_truth(17, 8000)
    for name, dtype, cell, fn in (
        ("uint8 ids", "uint8", [6], lambda v: (v % 251).astype(np.uint8)),
        ("uint32 x3", "uint32", [3], lambda v: v.view(np.uint32).copy()),
    ):
        w = W()
        mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
        p = [p for p in json.loads(w.objs[mk])["r3_cone"]["patches"] if "key" in p][0]
        _rewrite_patch(w, p["key"], dtype, cell, fn)
        s = rd.read_sample(w.get, mk, total, want=("r3",))
        assert not s.ok and s.missing.counts["r3_cone_patches"] == 1, (
            name,
            s.missing.counts,
        )
        rm = s.r3_replay_mask[:, 20:]
        assert (s.r3[:, 20:][rm] == T[:total, 20:][rm]).all(), name
        assert s.missing.rows[p["lo"] : min(p["end"], total)].all(), name
    # Header that places the cells elsewhere (shifted rows, other layers): reported,
    # never applied.
    for change in (
        lambda h: dict(h, lo=h["lo"] + 1, hi=h["hi"] + 1),
        lambda h: dict(h, layers=h["layers"][::-1]),
    ):
        w = W()
        mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
        p = [p for p in json.loads(w.objs[mk])["r3_cone"]["patches"] if "key" in p][0]
        buf = w.objs[p["key"]]
        k = int.from_bytes(buf[:4], "little")
        hb = json.dumps(change(json.loads(buf[4 : 4 + k]))).encode()
        w.objs[p["key"]] = len(hb).to_bytes(4, "little") + hb + buf[4 + k :]
        s = rd.read_sample(w.get, mk, total, want=("r3",))
        assert s.missing.counts.get("r3_cone_patches") == 1, s.missing.counts
        rm = s.r3_replay_mask[:, 20:]
        assert (s.r3[:, 20:][rm] == T[:total, 20:][rm]).all()
    # An indexer patch with int32 cells of the wrong width.
    w = W()
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
    p = [p for p in json.loads(w.objs[mk])["r3_cone"]["patches"] if "idx_key" in p][0]
    _rewrite_patch(w, p["idx_key"], "int32", [K // 2], lambda v: v.copy())
    s = rd.read_sample(w.get, mk, total, want=("indexer",))
    assert s.missing.counts["indexer_cone_patches"] == 1, s.missing.counts
    print("patch dtype / cell shape checked against the field")


def t_r18_nonstrict():
    set_mode("pack20-int32")
    w = W()
    T, Ix = r3_truth(18, 8000), idx_truth(18, 8000)
    mk, total = request(w, "a", "r0", T, Ix, 0, 0, 3000, 50, 0)
    m = json.loads(w.objs[mk])
    m["fields"][1]["layers"] = [0, 30, 10, 20]
    s = rd.read_sample(w.get, mk, total, manifest=m)
    assert not s.ok and s.missing.counts["indexer_layout"] == 1 and s.missing.rows.all()
    m = json.loads(w.objs[mk])
    del m["block_size"]
    s = rd.read_sample(w.get, mk, total, manifest=m)
    assert not s.ok and "malformed" in s.error and s.missing.rows.all()
    try:
        rd.read_sample(w.get, mk, total, manifest=m, strict=True)
        raise AssertionError("strict read returned")
    except rd.ReadError:
        pass
    w.objs[mk] = b"{not json"
    assert not rd.read_sample(w.get, mk, total).ok
    print("strict=False reports malformed input instead of raising")


if __name__ == "__main__":
    t_codecs()
    t_multiturn()
    t_fork()
    t_missing_block()
    t_missing_patch()
    t_error_marker()
    t_reported_and_invalid()
    t_needed_missing()
    t_patch_codec()
    t_key_prefix()
    t_r17_deep()
    t_r17_uint8()
    t_r17_odd_pack20()
    t_r17_absent_field()
    t_r17_zero_rows()
    t_r18_version_unknown()
    t_r18_patch_type()
    t_r18_nonstrict()
    t_hidden_raw()
    t_hidden_codec_strict()
    print("all reader tests passed")
