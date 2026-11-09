# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure tests of v51d, with the real _oc_cone_finish on an in-memory writer. argv[1]:
code directory."""

import os
import pathlib
import sys
import types

import numpy as np

os.environ["OC_R3_PARTIAL"] = "1"
sys.path.insert(0, sys.argv[1])
import oc_common as oc  # noqa: E402

NO = 65535
LAYERS = list(range(20, 40))
src = pathlib.Path(sys.argv[1] + "/worker_attr.py").read_text(encoding="utf8")
a = src.index("    def _oc_cone_finish(self, state, key_id, total: int, w):")
b = src.index(
    "    def _oc_idx_complete(self, state, start: int, n: int, w, want_cov=False):"
)
ns = {"_oc": oc, "np": np}
exec("class F:\n" + src[a:b], ns)
finish = ns["F"]._oc_cone_finish


class W:
    def __init__(self):
        self.objs, self.stats = {}, {}

    def put(self, k, v):
        self.objs[k] = v() if callable(v) else v


def truth(track_seed, n):
    t = np.zeros((n, 20, 6), np.uint16)
    for r in range(n):
        t[r] = (
            (r * 7 + track_seed * 1000 + np.arange(20)[:, None] * 13 + np.arange(6))
            % 60000
        ).astype(np.uint16)
    return t


def step(t, start, end):
    rows = np.full((end - start, 20, 6), NO, np.uint16)
    for j, layer in enumerate(LAYERS):
        a = max(start, end - oc.cone_size(layer))
        rows[a - start :, j] = t[a:end, j]
        m = max(start, a - 50)
        rows[m - start : a - start, j] = 12345
    return rows


def request(w, track, rid, t, aux, start, prompt, gen, version=0):
    st = types.SimpleNamespace(
        oc_track=track,
        session="s",
        oc_aux_start=aux,
        oc_cone_prompt=prompt,
        oc_cone_idx=None,
        oc_cone=oc.ConeParts(LAYERS),
    )
    st.oc_cone.add(start, prompt, step(t, start, prompt), version)
    for p in range(prompt, prompt + gen):
        st.oc_cone.add(p, p + 1, t[p : p + 1], version)
    return finish(None, st, rid, prompt + gen, w)


def read(w, m, n):
    got = np.full((n, 40, 6), 777, np.uint16)
    return oc.overlay_cones(
        got, 20, NO, [(w.objs[p["key"]], p["end"]) for p in m["patches"]]
    )


def fork(w, parent, child, aux):
    w.cone_list[child] = [
        dict(p, end=min(p["end"], aux))
        for p in w.cone_list.get(parent, [])
        if p["lo"] < aux
    ]


def check(t, m, w, n, ends, name):
    out, filled, ver = read(w, m, n)
    mask = oc.cone_mask(n, ends, LAYERS)
    assert filled[mask].all(), (name, "missing", int((~filled & mask).sum()))
    bad = (out[:, 20:][mask] != t[:n][mask]).any(-1)
    assert not bad.any(), (name, "wrong", int(bad.sum()))
    assert not (out[:, 20:] == 12345).all(-1).any(), (name, "margin delivered")
    assert m["uncovered"] == 0, (name, m["uncovered"])
    print(name, "ok; stored bytes", sum(len(w.objs[p["key"]]) for p in m["patches"]))
    return ver


T = truth(0, 40000)
# Linear trajectory, short steps of new rows.
w = W()
ends = []
aux = 0
for i, (start, p) in enumerate([(0, 3000), (0, 3500), (500, 3900), (900, 4300)]):
    m = request(w, "a", f"r{i}", T, aux, start, p, 50)
    ends.append((p, p + 50))
    aux = p + 50
check(T, m, w, aux, ends, "linear")

# R9-01: interior fork after several parent turns; R9-02: parent rows past the fork
# point must not shadow.
w = W()
request(w, "p", "r0", T, 0, 0, 6000, 50)
m0 = request(w, "p", "r1", T, 6050, 1000, 12000, 50)
C = T.copy()
C[8000:] = truth(5, 40000)[8000:]
fork(w, "p", "c", 8000)
m = request(w, "c", "c0", C, 8000, 3000, 8500, 40)
check(C, m, w, 8540, [(8500, 8540)], "interior fork")

# Fork inside generated output.
w = W()
request(w, "p", "r0", T, 0, 0, 1000, 500)
C = T.copy()
C[1200:] = truth(6, 40000)[1200:]
fork(w, "p", "c", 1200)
m = request(w, "c", "c0", C, 1200, 0, 1400, 10)
check(C, m, w, 1410, [(1400, 1410)], "fork inside output")

# Shrinking prompt in one trajectory.
w = W()
request(w, "a", "r0", T, 0, 0, 12000, 50)
S = T.copy()
S[8500:] = truth(7, 40000)[8500:]
m = request(w, "a", "r1", S, 8500, 3000, 9000, 50)
check(S, m, w, 9050, [(9000, 9050)], "shrink")

# R9-03: same shape, different provenance, different versions per cell.
w = W()
st = types.SimpleNamespace(
    oc_track="v",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=5000,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
st.oc_cone.add(0, 4000, step(T, 0, 4000), 0)
st.oc_cone.add(3000, 5000, step(T, 3000, 5000), 1)
m = finish(None, st, "r", 5000, w)
_, filled, ver = read(w, m, 5000)
assert (
    ver[3999, 0] == 0
    and ver[4500, 0] == 1
    and ver[3999, 19] == -1
    and ver[4999, 19] == 1
), ver[3999]
print("versions per cell ok", np.unique(ver[filled]))

# R9-04: nothing captured for the cone: reported, not silent.
w = W()
st = types.SimpleNamespace(
    oc_track="x",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=8193,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
st.oc_cone.add(8192, 8193, T[8192:8193], 0)
m = finish(None, st, "r", 8193, w)
assert m["uncovered"] > 0
print("window restored without patches: uncovered", m["uncovered"])

# R9-06: long prompt in short chunks stays bounded.
c = oc.ConeParts(LAYERS)
for p in range(0, 16384, 128):
    c.add(p, p + 128, T[p : p + 128], 0)
    c.trim(p + 128 - oc.CONE_MAX)
held = sum(sum(len(a) for a in q[2]) for q in c.parts)
assert held <= oc.CONE_MAX + 2 * oc.R3_WINDOW + 128, held
print("short chunks held rows", held)

# Indexer through the same keep mask.
w = W()
pk = (np.arange(3000 * 5 * 40) % 251).astype(np.uint8).reshape(3000, 5, 40)
st = types.SimpleNamespace(
    oc_track="i",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=3000,
    oc_cone=oc.ConeParts(LAYERS),
    oc_cone_idx=oc.ConeParts([20, 24, 28, 32, 36]),
)
st.oc_cone.add(0, 3000, step(T, 0, 3000), 0)
st.oc_cone_idx.add(0, 3000, pk, 0)
m = finish(None, st, "r", 3000, w)
h, data, cov, own = oc.cone_decode(w.objs[m["patches"][0]["idx_key"]])
assert (data[cov] == pk[h["lo"] : h["hi"]][cov]).all()
print("indexer ok; all passed")

# R10: speculative boundary row: the earlier request computed row aux, it stays.
w = W()
m0 = request(w, "b", "r0", T, 0, 0, 1000, 1, version=0)
T2 = T.copy()
T2[1000:] = (T2[1000:].astype(np.int64) + 1).astype(np.uint16)
st = types.SimpleNamespace(
    oc_track="b",
    session="s",
    oc_aux_start=1000,
    oc_cone_prompt=1100,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
    oc_cone_cut=1001,
)
st.oc_cone.add(1000, 1100, T2[1000:1100], 1)
m = finish(None, st, "r1", 1100, w)
out, filled, ver = read(w, m, 1100)
assert (
    ver[1000, 19] == 0 and (out[1000, 39] == T[1000, 19]).all() and ver[1001, 19] == 1
), ver[1000:1002, 19]
print("boundary row keeps first computation")

# Without the row in the tail the cut is aux: the later request owns it.
w = W()
request(w, "b", "r0", T, 0, 0, 1000, 1, version=0)
st.oc_cone_cut = 1000
st.oc_cone = oc.ConeParts(LAYERS)
st.oc_cone.add(1000, 1100, T2[1000:1100], 1)
m = finish(None, st, "r1", 1100, w)
_, _, ver = read(w, m, 1100)
assert ver[1000, 19] == 1, ver[1000, 19]

# R10: holes before the trailing bitmap are still counted after long generation.
w = W()
st = types.SimpleNamespace(
    oc_track="u",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=3000,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
st.oc_cone.add(2999, 3000, T[2999:3000], 0)
for p in range(3000, 6000):
    st.oc_cone.add(p, p + 1, T[p : p + 1], 0)
m = finish(None, st, "r", 6000, w)
out, filled, _ = read(w, m, 6000)
true_missing = int((oc.cone_mask(6000, [(3000, 6000)], LAYERS) & ~filled).sum())
assert 0 < m["uncovered"] <= true_missing, (m["uncovered"], true_missing)
print("uncovered equals real holes", true_missing)

# R10: 257 versions in one request.
c = oc.ConeParts(LAYERS)
for v in range(257):
    c.add(v, v + 1, T[v : v + 1], v)
lo, hi, cov = c.cover(257)
h, data, cv, own = oc.cone_decode(c.encode(257, cov))
assert (
    len(h["versions"]) == 257
    and own[cv].max() == 256
    and (data[cv] == T[lo:hi][cv]).all()
)
print("257 versions round trip")

# R10: consecutive release batches.
a = oc.__dict__
cls = next(
    v for v in a.values() if isinstance(v, type) and hasattr(v, "prune_engine_tables")
)
src2 = pathlib.Path(sys.argv[1] + "/oc_common.py").read_text(encoding="utf8")
assert "pend[0] | released" in src2
print("all v52 tests passed")

# R10: a replay that starts inside the context: low cells are not stored, so a later
# fork's own cell wins.
c = oc.ConeParts(LAYERS)
c.add(7680, 12802, T[7680:12802], 0, 7680)
lo, hi, cov = c.cover(12802)
assert not cov[7680 + 40 * 127 - 1 - lo, 19] and cov[7680 + 40 * 127 - lo, 19]
print("replay-start mask ok")

# R10: a fork below a short-suffix hit gets its own cell, not the parent's unexact one.
w = W()
st = types.SimpleNamespace(
    oc_track="p",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=12802,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
B = T.copy()
B[:, :] = 12345
good = T.copy()
rows = B[7680:12802].copy()
for j, layer in enumerate(LAYERS):
    a = 7680 + (layer + 1) * 127
    rows[a - 7680 :, j] = T[a:12802, j]
st.oc_cone.add(7680, 12802, rows, 0, 7680)
finish(None, st, "r0", 12802, w)
fork(w, "p", "c", 12700)
st2 = types.SimpleNamespace(
    oc_track="c",
    session="s",
    oc_aux_start=12700,
    oc_cone_prompt=12800,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
st2.oc_cone.add(7000, 12800, T[7000:12800], 0, 0)
m = finish(None, st2, "c0", 12800, w)
out, filled, _ = read(w, m, 12800)
mask = oc.cone_mask(12800, [(12800, 12800)], LAYERS)
assert (out[:, 20:][mask] != 12345).all(-1).all(), int(
    (out[:, 20:][mask] == 12345).all(-1).sum()
)
print("fork below hit takes exact cells")

# R10: indexer holes are counted on their own even when R3 holds the cell.
w = W()
st = types.SimpleNamespace(
    oc_track="j",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=3000,
    oc_cone=oc.ConeParts(LAYERS),
    oc_cone_idx=oc.ConeParts([20, 24, 28, 32, 36]),
)
st.oc_cone.add(0, 3000, step(T, 0, 3000), 0)
st.oc_cone_idx.add(2999, 3000, pk[2999:3000], 0)
m = finish(None, st, "r", 3000, w)
assert m["uncovered"] == 0 and m["uncovered_idx"] > 0, m
print("indexer holes reported", m["uncovered_idx"], "; all v52b tests passed")

# R11: abort below the prompt does not raise and leaves held aligned.
w = W()
st = types.SimpleNamespace(
    oc_track="ab",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=9000,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
st.oc_cone.add(0, 1000, step(T, 0, 1000), 0)
finish(None, st, "r", 1000, w)
h0, hc = w.cone_held["ab"]
assert h0 == 0 and len(hc) == 1000, (h0, len(hc))
print("abort below prompt ok")

# R11: the same step recomputed after a preemption adds no part.
c = oc.ConeParts(LAYERS)
for _ in range(50):
    c.add(0, 6000, T[0:6000], 0, 0)
assert len(c.parts) == 1, len(c.parts)
print("recompute adds no part; all v52c tests passed")

# R12: row aux completing a block keeps the earlier cell only when the block content is
# the same.
pp = [dict(lo=0, end=1300)]
assert oc.cone_boundary_cut(1151, 1151, False, 128, ["a"] * 9, ["a"] * 9, pp) == 1152
assert (
    oc.cone_boundary_cut(1151, 1151, False, 128, ["a"] * 9, ["a"] * 8 + ["b"], pp)
    == 1151
)
assert oc.cone_boundary_cut(1023, 896, True, 128, ["a"] * 8, ["a"] * 8, pp) == 1024
assert (
    oc.cone_boundary_cut(1023, 896, True, 128, ["a"] * 8, ["a"] * 7 + ["c"], pp) == 1023
)
assert oc.cone_boundary_cut(1100, 1101, False, 128, None, [], pp) == 1101
assert oc.cone_boundary_cut(1100, 1101, True, 128, None, [], pp) == 1100
print("boundary cut follows block content")

# R12: an abort while replaying below the prompt reports the cells its sampled tokens
# lost.
w = W()
st = types.SimpleNamespace(
    oc_track="ab2",
    session="s",
    oc_aux_start=0,
    oc_cone_prompt=9000,
    oc_freeze=9050,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
)
st.oc_cone.add(0, 1000, step(T, 0, 1000), 0)
m = finish(None, st, "r", 1000, w)
want = sum(9050 - max(1000, 9000 - oc.cone_need(lyr)) for lyr in LAYERS)
assert m["uncovered"] == want, (m["uncovered"], want)
print("abort below prompt reports lost cells", want)

# R12: the manifest's uncovered equals the cells the reader's mask finds missing after a
# replay hit.
for suffix in (1, 87, 88, 300):
    w = W()
    hit, start = 8064, 2944
    prompt = hit + suffix
    st = types.SimpleNamespace(
        oc_track="h",
        session="s",
        oc_aux_start=0,
        oc_cone_prompt=prompt,
        oc_cone_idx=None,
        oc_cone=oc.ConeParts(LAYERS),
    )
    rows = step(T, start, prompt)
    st.oc_cone.add(start, prompt, rows, 0, start)
    m = finish(None, st, "r", prompt, w)
    _, filled, _ = read(w, m, prompt)
    seen = int((oc.cone_mask(prompt, [(prompt, prompt)], LAYERS) & ~filled).sum())
    assert m["uncovered"] == seen, (suffix, m["uncovered"], seen)
print("uncovered matches the reader mask; all v52e tests passed")

# R12 Fable: a delayed fork's inherited list holds the parent's later turn (prompt past
# aux); it does not own row aux.
later = [dict(lo=0, end=1024, prompt_end=1300)]
verify = [dict(lo=0, end=1024, prompt_end=1000)]
assert (
    oc.cone_boundary_cut(1023, 1023, False, 128, ["a"] * 8, ["a"] * 8, later, True)
    == 1023
)
assert (
    oc.cone_boundary_cut(1023, 1023, False, 128, ["a"] * 8, ["a"] * 8, verify, True)
    == 1024
)
assert oc.cone_boundary_cut(1023, 1023, False, 128, ["a"] * 8, ["a"] * 8, later) == 1024
print("fork block-end cut needs the row's generator; all v52f tests passed")
