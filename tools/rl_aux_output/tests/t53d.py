# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v53d: partial capture writes deep indexer slots of blocks as pad. argv[1]: code
dir."""

import os
import sys
import types

os.environ["OC_IDX_MODE"] = "gpudelta"
os.environ["OC_R3_PARTIAL"] = "1"
sys.path.insert(0, sys.argv[1])
import numpy as np
import oc_common as oc

oc.IDX_CAP = types.SimpleNamespace(layer_ids=[2, 8, 14, 20, 24, 28, 32, 36], topk=512)
rng = np.random.default_rng(1)
a = rng.integers(0, 5000, (128, 8, 512)).astype(np.int64)
s = np.sort(a, -1)
d = np.diff(s, axis=-1, prepend=0)
hi = (d >> 16) & 15
b = np.concatenate(
    [
        x.astype(np.uint8)
        for x in (d & 255, (d >> 8) & 255, hi[..., 0::2] | (hi[..., 1::2] << 4))
    ],
    -1,
).reshape(128, -1)
b = (
    b
    if False
    else np.concatenate(
        [
            x.astype(np.uint8)
            for x in (
                np.diff(np.sort(a + 1, -1), axis=-1, prepend=0) & 255,
                (np.diff(np.sort(a + 1, -1), axis=-1, prepend=0) >> 8) & 255,
                ((np.diff(np.sort(a + 1, -1), axis=-1, prepend=0) >> 16) & 15)[
                    ..., 0::2
                ]
                | (
                    ((np.diff(np.sort(a + 1, -1), axis=-1, prepend=0) >> 16) & 15)[
                        ..., 1::2
                    ]
                    << 4
                ),
            )
        ],
        -1,
    ).reshape(128, -1)
)
buf = oc.idx_encode(b)
out = oc.idx_decode(buf, (8, 512))
assert (out[:, :3] == np.sort(a[:, :3], -1)).all() and (out[:, 3:] == -1).all()
assert b.sum() > 0 and (oc.idx_encode(b) == buf)
full = len(oc.idx_encode.__globals__["_ZC_LOCAL"].c.compress(b.tobytes()))
print("deep slots dropped; shallow intact; bytes", len(buf), "vs", full)
i32 = a.astype(np.int32)
r = oc.idx_drop_deep(i32)
assert (
    (r[:, 3:] == -1).all()
    and (r[:, :3] == i32[:, :3]).all()
    and (i32[:, 3:] != -1).any()
)
print("int32 path ok; input untouched")
