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


T = truth(1, 4000)
w = W()
m0 = request(w, "a", "r0", T, 0, 0, 1000, 10)
m1 = request(w, "a", "r1", T, 1010, 1010, 1500, 20)
assert m0["ends"] == [[1000, 1010]], m0["ends"]
assert m1["ends"] == [[1000, 1010], [1500, 1520]], m1["ends"]
_, filled, _ = read(w, m1, 1520)
need = oc.cone_mask(1520, [tuple(e) for e in m1["ends"]], LAYERS)
assert need.any() and filled[need].all(), int((need & ~filled).sum())
print("ends recorded per request; needed cells all covered")
# An edit at 800 cuts the trajectory: the first request's end is clipped, nothing past
# the cut stays.
st = types.SimpleNamespace(
    oc_track="a",
    session="s",
    oc_aux_start=800,
    oc_cone_prompt=900,
    oc_cone_idx=None,
    oc_cone=oc.ConeParts(LAYERS),
    oc_cone_cut=800,
)
st.oc_cone.add(0, 900, step(T, 0, 900), 1)
m2 = finish(None, st, "r2", 900, w)
assert m2["ends"] == [[900, 900]], m2["ends"]
print("ends clipped at the cut; all v54 tests passed")
