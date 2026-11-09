# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v56: raw hidden rows round-trip bit-exactly from the lm_head input to the readers'
rows, for float32 and bfloat16 (and float16) inputs; hidden is raw only. Run: python -I
t56.py"""

import os
import pathlib
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pylib"))

import numpy as np  # noqa: E402
import oc_common as oc  # noqa: E402
import oc_reader as rd  # noqa: E402
import torch  # noqa: E402

D = 4096


def lm_head_input(dtype, n=37):
    g = torch.Generator().manual_seed(7)
    h = torch.randn(n, D, generator=g) * torch.logspace(-30, 30, D)
    h[0, :6] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 1e-45, 3.4e38]
    )
    h = h.to(dtype)
    # As in the model runner: the rows of the logits indices, a non-contiguous gather
    # source.
    full = torch.zeros(2 * n, D, dtype=dtype)
    full[1::2] = h
    return full[1::2], h


def bits(t):
    return t.contiguous().view({4: torch.int32, 2: torch.int16}[t.element_size()])


def stored_chunks(q_host, lens):
    """The worker's _oc_put_hid: one object per chunk, its rows' bytes back to back."""
    out, at = [], 0
    for k in lens:
        out.append(np.stack([q_host[at + j] for j in range(k)]).tobytes())
        at += k
    return out


def main():
    for dtype, name in (
        (torch.float32, "float32"),
        (torch.bfloat16, "bfloat16"),
        (torch.float16, "float16"),
    ):
        src, h = lm_head_input(dtype)
        q, dt, d = oc.hid_snapshot(src)
        assert dt == name and d == D and q.dtype == torch.uint8
        # the snapshot owns its rows: later writes to the source do not reach it
        src.zero_()
        q_host = q.numpy()
        lens = [16, 16, 5]
        bufs = stored_chunks(q_host, lens)
        field = dict(
            name="hidden",
            dtype=dt,
            shape=[d],
            codec="raw",
            layout="chunks",
            chunk_lens=lens,
            chunks=len(lens),
            rows=sum(lens),
        )
        assert oc.hid_row_nbytes(field) == D * h.element_size()
        assert [len(b) for b in bufs] == [k * oc.hid_row_nbytes(field) for k in lens]
        got = oc.hid_torch(field, bufs, device="cpu")
        assert got.dtype == dtype and torch.equal(bits(got), bits(h)), name
        # The reader (no torch): same bits in the stored dtype (bfloat16 as uint16
        # without ml_dtypes).
        dt_np, _ = rd._hid_raw_dtype(name)
        rows = np.frombuffer(b"".join(bufs), dt_np).reshape(-1, D)
        assert (
            rows.view(f"u{rows.itemsize}") == bits(h).numpy().view(f"u{rows.itemsize}")
        ).all(), name
        print(
            f"{name}: {len(h)} rows bit-exact; {oc.hid_row_nbytes(field)} bytes per "
            "generated token"
        )
    # Only raw with a known dtype is decodable; any other codec (the former fp8 one
    # included) is reported ("encoding") before any chunk arithmetic, by the helpers and
    # the checkers.
    bufs = [b"\0" * 8 * 3]
    raw_e5 = dict(
        dtype="float8_e5m2", shape=[4], codec="raw", chunk_lens=[3], chunks=1, rows=3
    )
    assert (
        oc.hid_row_nbytes(raw_e5) is None and oc.hid_check(raw_e5, bufs) == "encoding"
    )
    for bad in (
        dict(codec="typo"),
        dict(codec=None),
        dict(codec="fp8e4m3+rowscale-f32", dtype="float16"),
        dict(
            codec="fp8e4m3+rowscale-f32", dtype="float8_e4m3fn", scale_dtype="float32"
        ),
    ):
        f = dict(
            dict(dtype="float16", shape=[4], chunk_lens=[3], chunks=1, rows=3), **bad
        )
        assert oc.hid_row_nbytes(f) is None and oc.hid_check(f, bufs) == "encoding", bad
        try:
            oc.hid_torch(f, bufs, device="cpu")
            raise AssertionError(f"decoded {bad}")
        except ValueError:
            pass
    ok = dict(dtype="float16", shape=[4], codec="raw", chunk_lens=[3], chunks=1, rows=3)
    assert (
        oc.hid_check(ok, bufs) is None
        and oc.hid_check(dict(ok, chunk_lens=[2], rows=2), bufs) == "meta"
    )
    for name in ("x10_driver.py", "x9_driver.py"):
        src = pathlib.Path(os.path.join(HERE, name)).read_text(encoding="utf8")
        sec = src[src.index('fh = fields.get("hidden")') :]
        sec = sec[: sec.index("oc.hid_torch(")]
        assert "oc.hid_check(fh, vals)" in sec and "hid_row_nbytes" not in sec, name
    print(
        "unknown hidden encodings reported by hid_check / hid_torch and both checkers"
    )
    # R20b #1: an lm_head input dtype raw readers do not know is stored as exact
    # float32, counted.
    import types

    saved = oc.ENABLED, oc._writer
    oc.ENABLED, oc._writer = True, types.SimpleNamespace(stats={})
    try:
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            src = (torch.randn(5, D) * 4).to(dtype)
            q, dt, d = oc.hid_snapshot(src)
            assert dt == "float32" and d == D, (dtype, dt)
            field = dict(
                dtype=dt, shape=[d], codec="raw", chunk_lens=[5], chunks=1, rows=5
            )
            buf = [q.numpy().tobytes()]
            assert oc.hid_check(field, buf) is None
            assert torch.equal(oc.hid_torch(field, buf, device="cpu"), src.float()), (
                dtype
            )
        assert oc._writer.stats["hid_upcast"] == 2, oc._writer.stats
    finally:
        oc.ENABLED, oc._writer = saved
    print("fp8 lm_head input stored as exact float32 rows, hid_upcast counted")
    # R20b #2: a field without chunk_lens, or chunks it does not list, is "meta" for the
    # checkers.
    ok = dict(dtype="float16", shape=[4], codec="raw", chunk_lens=[3], chunks=1, rows=3)
    assert (
        oc.hid_check({k: v for k, v in ok.items() if k != "chunk_lens"}, bufs) == "meta"
    )
    assert oc.hid_check(dict(ok, chunk_lens=[3, 0], chunks=2), bufs) == "meta"
    sec = pathlib.Path(os.path.join(HERE, "x10_driver.py")).read_text(encoding="utf8")
    sec = sec[sec.index('fh = fields.get("hidden")') :]
    sec = sec[: sec.index("oc.hid_torch(")]
    assert 'if why == "meta":' in sec and 'fh.get("chunk_lens") is not None' not in sec
    print("hidden fields without matching chunk_lens rejected before decoding")
    print("all t56 tests passed")


if __name__ == "__main__":
    main()
