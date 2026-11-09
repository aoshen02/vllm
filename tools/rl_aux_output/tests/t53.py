# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure tests of the gpudelta codec (numpy encoder mirrors the torch one). argv[1]:
code dir."""

import os
import sys

os.environ["OC_IDX_MODE"] = "gpudelta"
sys.path.insert(0, sys.argv[1])
import numpy as np
import oc_common as oc


def enc_np(v, L, K):
    n = v.shape[0]
    s = np.sort(v.reshape(n, L, K), axis=-1)
    d = np.diff(s, axis=-1, prepend=0)
    hi = (d >> 16) & 15
    planes = (d & 255, (d >> 8) & 255, hi[..., 0::2] | (hi[..., 1::2] << 4))
    return np.concatenate([p.astype(np.uint8) for p in planes], -1).reshape(n, -1)


def canon(x):
    s_ = np.sort(np.where(x < 0, np.iinfo(np.int64).max, x), -1)
    return np.where(s_ == np.iinfo(np.int64).max, -1, s_)


rng = np.random.default_rng(0)
L, K = 8, 512
for maxv in (100, 8191, 131071, (1 << 20) - 2):
    a = rng.integers(0, maxv + 1, (37, L, K)).astype(np.int64)
    a[rng.random(a.shape) < 0.2] = -1
    a[3] = -1
    b = enc_np(a.reshape(37, -1) + 1, L, K)
    assert b.shape == (37, L * K * 5 // 2)
    out = oc.dplane20(b.tobytes(), (L, K))
    assert (out == canon(a)).all(), maxv
    buf = oc.idx_encode(b)
    assert buf[:1] in (b"\x05", b"\x06")
    assert (oc.idx_decode(buf, (L, K)) == canon(a)).all()
    assert (oc.unpack_idx_rows(b.tobytes(), (L, K)) == out).all()
    one = b.reshape(37, L, -1)[:, 2]
    assert (oc.dplane20(one.tobytes(), (1, K))[:, 0] == canon(a[:, 2])).all()
z = np.zeros((4, L * K * 5 // 2), np.uint8)
assert (oc.dplane20(z.tobytes(), (L, K)) == -1).all()
print("dplane20 round trip ok; layer slices decode alone; zero rows are pad")
try:
    import torch

    v = torch.from_numpy(rng.integers(0, 9000, (5, L * K)))
    tb = oc.dplane20_encode_torch(v, L, K).numpy()
    assert (tb == enc_np(v.numpy(), L, K)).all()
    assert (
        oc.dplane20_torch(torch.from_numpy(tb.reshape(-1)), (L, K)).numpy()
        == oc.dplane20(tb.tobytes(), (L, K))
    ).all()
    print("torch matches numpy")
except ImportError:
    print("no torch here")

# R13: cone rows decode by the named codec, whatever this process runs.
a = rng.integers(0, 9000, (6, L, K)).astype(np.int64)
b = enc_np(a.reshape(6, -1) + 1, L, K)
assert (oc.unpack_idx_rows(b.tobytes(), (L, K), "dplane20+zstd3") == canon(a)).all()
v = (a.reshape(6, -1) + 1).astype(np.uint64)
p20 = (
    (v[:, 0::2] | (v[:, 1::2] << np.uint64(20)))
    .astype("<u8")
    .view(np.uint8)
    .reshape(6, -1, 8)[:, :, :5]
    .reshape(6, -1)
)
assert (oc.unpack_idx_rows(p20.tobytes(), (L, K), "pack20+zstd3") == a).all()
print("codec named by the manifest wins")

import torch  # noqa: E402

a = rng.integers(-1, (1 << 20) - 1, (9, L, K)).astype(np.int64)
a[rng.random(a.shape) < 0.3] = -1
v = torch.from_numpy(a.reshape(9, -1) + 1).to(torch.int32)
tb = oc.dplane20_encode_torch(v, L, K)
assert (tb.numpy() == enc_np(a.reshape(9, -1) + 1, L, K)).all()
out = oc.dplane20_torch(tb.reshape(-1), (L, K))
assert out.dtype == torch.int32 and (out.numpy() == canon(a)).all()
print("int32 torch codec ok up to 2**20 - 2")
