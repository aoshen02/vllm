# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np
import oc_common as oc

assert oc.CONE_ZSTD
c = oc.ConeParts(list(range(20, 40)))
T = (
    np.arange(6000)[:, None, None] * 7
    + np.arange(20)[None, :, None] * 13
    + np.arange(6)
) % 60000
c.add(0, 6000, T[:6000].astype(np.uint16), 0)
lo, hi, cov = c.cover(6000)
buf = c.encode(6000, cov)
h, data, cv, own = oc.cone_decode(buf)
assert (
    h.get("zstd") == 1
    and (data[cv] == T[lo:hi].astype(np.uint16)[cv]).all()
    and (cv == cov).all()
)
raw = oc.ConeParts.encode.__globals__
os.environ["OC_CONE_ZSTD"] = "0"
oc.CONE_ZSTD = False
buf2 = c.encode(6000, cov)
h2, d2, c2, _ = oc.cone_decode(buf2)
assert "zstd" not in h2 and (d2[c2] == data[cv]).all()
print("cone zstd round trip; bytes", len(buf), "vs", len(buf2))
