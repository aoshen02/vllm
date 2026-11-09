# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""gpudelta decode restores the engine order: ascending positions, pads last.
argv[1]: code dir."""

import os
import sys

os.environ["OC_IDX_MODE"] = "gpudelta"
sys.path.insert(0, sys.argv[1])
import numpy as np
import oc_common as oc

rng = np.random.default_rng(5)
L, K = 8, 512
rows = []
for r in range(40):
    n = int(rng.integers(0, K + 1))
    for lyr in range(L):
        v = np.sort(rng.choice(20000, n, replace=False)) if n else np.zeros(0, int)
        rows.append(np.concatenate([v, -np.ones(K - n, int)]))
a = np.array(rows).reshape(40, L, K)
s = np.sort(a + 1, -1)
d = np.diff(s, axis=-1, prepend=0)
hi = (d >> 16) & 15
b = np.concatenate(
    [
        x.astype(np.uint8)
        for x in (d & 255, (d >> 8) & 255, hi[..., 0::2] | (hi[..., 1::2] << 4))
    ],
    -1,
).reshape(40, -1)
assert (oc.dplane20(b.tobytes(), (L, K)) == a).all()
assert (oc.idx_decode(oc.idx_encode(b), (L, K)) == a).all()
import torch  # noqa: E402

assert (oc.dplane20_torch(torch.from_numpy(b.reshape(-1)), (L, K)).numpy() == a).all()
import oc_reader as R  # noqa: E402

assert (R._dplane20(b.tobytes(), L, K) == a).all()
print("engine order restored (numpy, torch, reader)")
