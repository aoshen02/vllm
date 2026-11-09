# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Consumer side of the aux-output delivery protocol: one call reads one sample.

    sample = read_sample(store.get, manifest_key, n_rows)

The store dependencies are `get(key) -> bytes | None` and an existence check:
`exists(keys)`, the `batch_is_exist` of get's store, or `store=` (batch_is_exist or
is_exist). Nothing here imports the writer. Every row or cell that was not delivered
shows up in `sample.missing` or in a replay mask that is False; data under a False mask
holds the no-replay sentinel, never a guess.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

IDX_NO_REPLAY = -2
IDX_PAD = -1
_PACK20_TAGS = {1: True, 4: False}  # tag -> zstd compressed
_DPLANE20_TAGS = {5: True, 6: False}
_INT32_TAGS = {2: True, 3: False}


class ReadError(Exception):
    pass


@dataclass
class Missing:
    counts: dict[str, int] = field(default_factory=dict)
    rows: np.ndarray | None = None  # bool [n]: rows the trainer must drop or mask

    def add(self, what: str, n: int = 1) -> None:
        self.counts[what] = self.counts.get(what, 0) + int(n)

    @property
    def any(self) -> bool:
        return any(v for v in self.counts.values())


@dataclass
class Sample:
    n_rows: int
    manifest: dict | None = None
    error: str | None = None
    weight_version: np.ndarray | None = (
        None  # int64 [n], -1 unknown (see version_unknown)
    )
    first_deep_layer: int | None = None  # set under partial capture
    r3: np.ndarray | None = None  # [n, layers, k], sentinel = dtype max
    r3_no_replay: int | None = None
    r3_replay_mask: np.ndarray | None = None  # bool [n, layers]
    r3_version: np.ndarray | None = (
        None  # int64 [n, layers], -1 unknown or not replayed
    )
    indexer: np.ndarray | None = None  # int32 [n, idx_layers, topk], -2 no replay
    indexer_layers: list[int] | None = None
    indexer_replay_mask: np.ndarray | None = None  # bool [n, idx_layers]
    indexer_version: np.ndarray | None = None  # int64 [n, idx_layers]
    # [m, d], positions hidden_start + i: the lm_head input as stored, in hidden_dtype
    # (bfloat16 without ml_dtypes: its uint16 bit patterns); NaN (bf16: 0x7FC0) where
    # missing.
    hidden: np.ndarray | None = None
    hidden_dtype: str | None = None
    hidden_start: int | None = None
    # Top-k of the raw distribution (temperature 1) at positions topk_start + i, as
    # hidden: int32 ids by descending logit (-1 where missing) and float32 logprobs (NaN
    # where missing).
    topk_ids: np.ndarray | None = None
    topk_logprobs: np.ndarray | None = None
    topk_start: int | None = None
    r3_needed_missing: np.ndarray | None = (
        None  # bool [n, layers]: needed deep cell not replayed
    )
    indexer_needed_missing: np.ndarray | None = None  # bool [n, idx_layers]
    version_unknown: np.ndarray | None = (
        None  # bool [n]: delivered row, weight version unknown
    )
    version_unknown_count: int = 0
    stale_marker: bool = False  # error marker beside a complete manifest: ignored
    missing: Missing = field(default_factory=Missing)
    uncovered: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.missing.any


# ---------------------------------------------------------------- decoders


def _zstd(b: bytes) -> bytes:
    import zstandard

    return zstandard.ZstdDecompressor().decompress(b)


def _unpack20(raw: bytes, per_row: int) -> np.ndarray:
    b = np.frombuffer(raw, np.uint8)
    if len(b) % 5:
        raise ValueError("pack20 length")
    b = b.reshape(-1, 5).astype(np.int32)
    out = np.empty((len(b), 2), np.int32)
    out[:, 0] = b[:, 0] | (b[:, 1] << 8) | ((b[:, 2] & 15) << 16)
    out[:, 1] = (b[:, 2] >> 4) | (b[:, 3] << 4) | (b[:, 4] << 12)
    flat = out.reshape(-1) - 1
    # The writer pads an odd value count with one zero value (decodes to -1).
    rows = len(flat) // per_row
    if len(flat) != rows * per_row + (rows * per_row) % 2 or (
        len(flat) > rows * per_row and flat[-1] != -1
    ):
        raise ValueError("pack20 rows")
    return flat[: rows * per_row]


def _dplane20(raw: bytes, layers: int, topk: int) -> np.ndarray:
    per_layer = topk * 5 // 2
    b = np.frombuffer(raw, np.uint8)
    if len(b) % (layers * per_layer):
        raise ValueError("dplane20 length")
    b = b.reshape(-1, layers, per_layer).astype(np.int64)
    lo, mid, hi = b[..., :topk], b[..., topk : 2 * topk], b[..., 2 * topk :]
    h = np.empty(lo.shape, np.int64)
    h[..., 0::2], h[..., 1::2] = hi & 15, hi >> 4
    out = (np.cumsum(lo | (mid << 8) | (h << 16), axis=-1) - 1).astype(np.int32)
    # Sorting put the pads first; the engine emits ascending positions with pads last.
    npad = (out < 0).sum(-1, keepdims=True)
    k = out.shape[-1]
    return np.take_along_axis(out, (np.arange(k) + npad) % k, axis=-1)


def _unpack_packed(raw: bytes, layers: int, topk: int, codec: str) -> np.ndarray:
    if codec.startswith("dplane20"):
        return _dplane20(raw, layers, topk)
    return _unpack20(raw, layers * topk).reshape(-1, layers, topk)


def decode_indexer_object(buf: bytes, layers: int, topk: int, codec: str) -> np.ndarray:
    """One stored indexer block or tail -> int32 [rows, layers, topk]. The tag byte must
    agree with the codec the manifest names."""
    tag, body = buf[0], bytes(buf[1:])
    dplane = codec.startswith("dplane20")
    if tag in _INT32_TAGS:
        raw = _zstd(body) if _INT32_TAGS[tag] else body
        if len(raw) % (4 * layers * topk):
            raise ValueError("int32 rows")
        return np.frombuffer(raw, np.int32).reshape(-1, layers, topk).copy()
    if tag in _PACK20_TAGS and not dplane:
        return _unpack_packed(
            _zstd(body) if _PACK20_TAGS[tag] else body, layers, topk, "pack20"
        )
    if tag in _DPLANE20_TAGS and dplane:
        return _dplane20(_zstd(body) if _DPLANE20_TAGS[tag] else body, layers, topk)
    raise ValueError(f"indexer tag {tag} does not match codec {codec!r}")


def decode_patch(buf: bytes):
    """Cone patch -> (header, data [hi-lo, layers, ...], covered [hi-lo, layers],
    version)."""
    k = int.from_bytes(buf[:4], "little")
    h = json.loads(bytes(buf[4 : 4 + k]))
    body = bytes(buf[4 + k :])
    if h.get("zstd"):
        body = _zstd(body)
    shape, dt = list(h["shape"]), np.dtype(h["dtype"])
    nc = int(np.prod(h["cov"]))
    packed = (nc + 7) // 8
    cov = (
        np.unpackbits(np.frombuffer(body, np.uint8, packed), count=nc)
        .astype(bool)
        .reshape(h["cov"])
    )
    cells = int(cov.sum())
    per = int(np.prod(shape[2:]))
    nvals = cells * per
    need = packed + nvals * dt.itemsize
    versions = list(h["versions"])
    vt = np.uint8 if len(versions) <= 256 else np.uint16
    if len(versions) > 1:
        need += cells * np.dtype(vt).itemsize
    if len(body) != need:
        raise ValueError("patch length")
    data = np.zeros(shape, dt)
    data[cov] = np.frombuffer(body, dt, nvals, packed).reshape(-1, *shape[2:])
    ver = np.full(h["cov"], -1, np.int64)
    if len(versions) > 1:
        ver[cov] = np.asarray(versions, np.int64)[
            np.frombuffer(body, vt, cells, packed + nvals * dt.itemsize)
        ]
    elif cells:
        ver[cov] = versions[0]
    return h, data, cov, ver


def _row_versions(segs, n: int) -> np.ndarray:
    out = np.full(n, -1, np.int64)
    for i, (start, v) in enumerate(segs or []):
        end = segs[i + 1][0] if i + 1 < len(segs) else n
        out[max(0, int(start)) : max(0, min(int(end), n))] = int(v)
    return out


def cone_mask(n: int, ends, layers, last_layer: int, window: int) -> np.ndarray:
    """Deep cells [n, layers] some sampled token of the trajectory depends on. ends:
    [prompt rows, rows at its end] per request; a token at t depends on rows [t -
    need(L), t) of layer L."""
    mask = np.zeros((n, len(layers)), bool)
    for p, e in ends:
        for j, layer in enumerate(layers):
            need = 1 + (last_layer - layer) * (window - 1)
            mask[max(0, int(p) - need) : max(0, min(int(e), n)), j] = True
    return mask


# ---------------------------------------------------------------- reader


def _get(get, key):
    v = get(key)
    return bytes(v) if v else None


def _overlay(
    out, mask, ver, deep0, sentinel, patches, n, deep_layers, missing, what, row_bad
):
    """Deep slots start at `sentinel`; patches (key, lo, end, bytes, unpack) in list
    order, each up to its end, first written wins. A patch that is absent or whose cells
    do not fit the field (layers, start, coverage, dtype, cell shape) is reported and
    fills nothing."""
    out[:, deep0:] = sentinel
    ver[:, deep0:] = -1
    nd = out.shape[1] - deep0
    filled = np.zeros((n, nd), bool)
    for key, lo, end, buf, unpack in patches:
        hi_m = min(int(end), n)
        try:
            if buf is None:
                raise KeyError(key)
            h, data, cov, own = decode_patch(buf)
            plo, phi = int(h["lo"]), int(h["hi"])
            if (
                [int(x) for x in h["layers"]] != deep_layers
                or plo != int(lo)
                or data.shape[:2] != (phi - plo, nd)
                or cov.shape != (phi - plo, nd)
            ):
                raise ValueError("patch header")
            if unpack is not None:
                data = unpack(data)
            if data.shape != (phi - plo, nd, *out.shape[2:]) or data.dtype != out.dtype:
                raise ValueError("patch cells")
            hi = min(phi, hi_m)
            if plo < hi:
                take = cov[: hi - plo] & ~filled[plo:hi]
                out[plo:hi, deep0:][take] = data[: hi - plo][take]
                ver[plo:hi, deep0:][take] = own[: hi - plo][take]
                filled[plo:hi] |= take
        except Exception:
            missing.add(what)
            if lo < hi_m:
                row_bad[int(lo) : hi_m] = True
    mask[:, deep0:] = filled


def _r3_bad_cells(cells, sent, has_sent: bool) -> np.ndarray:
    """[n, layers, k] -> bool [n, layers]: a repeated expert id, or the never-computed
    sentinel (only for ids wider than one byte: one-byte ids have no spare value)."""
    srt = np.sort(cells, axis=-1)
    bad = np.zeros(cells.shape[:-1], bool)
    if cells.shape[-1] > 1:
        bad |= (srt[..., 1:] == srt[..., :-1]).any(-1)
    if has_sent:
        bad |= (srt == sent).any(-1)
    return bad


def _listed_keys(m, sess, r3_tail_key) -> list[str]:
    """Every object the manifest lists, for every field it declares, whatever rows
    are read."""
    keys = [f"{sess}/{b}" for b in m["blocks"]] + [r3_tail_key]
    for f in m.get("fields", []):
        if f.get("name") == "indexer_topk":
            keys += [f"{sess}/{b}{f.get('block_suffix', '/idx')}" for b in m["blocks"]]
            if f.get("tail_key"):
                keys.append(str(f["tail_key"]))
        elif f.get("name") in ("hidden", "topk"):
            keys += [f"{f['key_prefix']}{i}" for i in range(int(f.get("chunks", 0)))]
    for p in (m.get("r3_cone") or {}).get("patches", []):
        keys += [str(p[k]) for k in ("key", "idx_key") if p.get(k)]
    return list(dict.fromkeys(keys))


def _exist_fn(get, exists, store) -> Callable[[list[str]], list[bool]]:
    """The existence check: exists=, else batch_is_exist of get's store, else store=
    (batched or per key). Never by downloading the objects."""
    if exists is None:
        exists = getattr(getattr(get, "__self__", None), "batch_is_exist", None)
    if exists is None and store is not None:
        exists = getattr(store, "batch_is_exist", None)
        if exists is None and hasattr(store, "is_exist"):
            exists = lambda keys: [store.is_exist(k) for k in keys]
    if exists is None:
        raise ValueError(
            "read_sample needs an existence check: exists=, a get bound to a store "
            "with "
            "batch_is_exist, or store= with batch_is_exist or is_exist"
        )
    return lambda keys: [r is True or r == 1 for r in exists(keys)] if keys else []


def read_sample(
    get: Callable[[str], bytes | None],
    manifest_key: str,
    n_rows: int,
    want: Iterable[str] = ("r3", "indexer", "hidden"),
    manifest: dict | bytes | None = None,
    strict: bool = False,
    exists: Callable[[list[str]], list] | None = None,
    store=None,
) -> Sample:
    """Reads rows [0, n_rows) of one request's manifest. n_rows = L - 1 for a sample of
    L tokens (prompt + generated). With strict=True any missing data raises ReadError;
    otherwise nothing raises on stored data, however malformed: it is reported in the
    returned Sample."""
    want = tuple(want)
    bad = set(want) - {"r3", "indexer", "hidden", "topk"}
    if bad:
        raise ValueError(f"unknown outputs {bad}")
    n = int(n_rows)
    if n < 0:
        raise ValueError("n_rows < 0")
    exists = _exist_fn(get, exists, store)
    try:
        return _read_sample(get, manifest_key, n, want, manifest, strict, exists)
    except ReadError:
        raise
    except Exception as e:
        s = Sample(
            n_rows=n, error=f"malformed manifest or object: {type(e).__name__}: {e}"
        )
        s.missing.rows = np.ones(n, bool)
        s.missing.add("error")
        if strict:
            raise ReadError(s.error) from e
        return s


def _read_sample(get, manifest_key, n, want, manifest, strict, exists) -> Sample:
    s = Sample(n_rows=n)
    s.missing.rows = np.zeros(n, bool)
    rows_bad = s.missing.rows

    def fail(msg):
        s.error = msg
        s.missing.add("error")
        rows_bad[:] = True
        if strict:
            raise ReadError(msg)
        return s

    # With a manifest, the verdict is whether every object it lists exists: an error
    # marker beside it is stale (another writer stored the objects). Without one, the
    # marker fails it at once.
    if manifest is None:
        manifest = _get(get, manifest_key)
        if manifest is None:
            marker = exists([manifest_key + "!err"])[0]
            return fail(
                "error marker: the writer failed to store this request"
                if marker
                else "manifest not found"
            )
    if isinstance(manifest, (bytes, bytearray, memoryview)):
        manifest = json.loads(bytes(manifest))
    m = s.manifest = manifest
    fields = {f["name"]: f for f in m.get("fields", [])}
    fr, fi, fh = (
        fields.get("routed_experts"),
        fields.get("indexer_topk"),
        fields.get("hidden"),
    )
    ft = fields.get("topk")
    if "key_prefix" in m and "tail_key" in m:
        sess, r3_tail_key = str(m["key_prefix"]), str(m["tail_key"])
    else:
        # Manifests before v54: derive the keys from the manifest key and check the
        # derivation against the keys the manifest does carry.
        if not manifest_key.endswith("/m"):
            return fail("manifest key must end with /m")
        req_base = manifest_key[:-2]
        sess, r3_tail_key = req_base.rsplit("/", 1)[0], req_base + "/t"
        if fi is not None and fi.get("tail_key") != req_base + "/t/idx":
            return fail("indexer tail key does not match the manifest key layout")
        if fh is not None and fh.get("key_prefix") != req_base + "/h/":
            return fail("hidden key prefix does not match the manifest key layout")

    rows = int(m["rows"])
    bs = int(m["block_size"])
    blocks = list(m["blocks"])
    tail_start = int(m["tail_start"])
    if tail_start != len(blocks) * bs:
        return fail("tail_start does not follow the blocks")
    if rows < n:
        s.missing.add("rows_short", n - rows)
        rows_bad[rows:] = True
    inv = m.get("invalid_from")
    if inv is not None and int(inv) < n:
        s.missing.add("invalid_rows", n - int(inv))
        rows_bad[int(inv) :] = True
    s.weight_version = _row_versions(m.get("weight_versions"), n)
    cone = m.get("r3_cone") if (m.get("r3_cone") or {}).get("partial") else None
    s.first_deep_layer = None if cone is None else int(cone["first_deep_layer"])
    if cone is not None:
        s.uncovered = {
            "r3": int(cone.get("uncovered", 0)),
            "indexer": int(cone.get("uncovered_idx", 0)),
        }
    nb = min(len(blocks), (n + bs - 1) // bs)  # blocks that hold a row below n
    absent, seen = [], set()

    def get_listed(key):
        # Every object read below is one the manifest lists: if it is not there, the
        # request failed (its write failed, or it was never written); the request is not
        # partially delivered.
        seen.add(key)
        v = get(key)
        if not v:
            absent.append(key)
        return v

    ctx = dict(
        get=get_listed,
        m=m,
        cone=cone,
        sess=sess,
        n=n,
        nb=nb,
        bs=bs,
        rows=rows,
        inv=inv,
        tail_start=tail_start,
    )

    for name, f, reader in (
        ("r3", fr, _read_r3),
        ("indexer", fi, _read_indexer),
        ("hidden", fh, _read_hidden),
        ("topk", ft, _read_topk),
    ):
        if name not in want:
            continue
        if f is None:
            s.missing.add(name + "_field")
            rows_bad[:] = True
            continue
        reader(s, f, ctx, r3_tail_key)
    rest = [manifest_key + "!err"] + [
        k for k in _listed_keys(m, sess, r3_tail_key) if k not in seen
    ]
    got = exists(rest)
    s.stale_marker = got[0]
    absent += [k for k, ok in zip(rest[1:], got[1:]) if not ok]
    if absent:
        s.missing.add("absent_objects", len(absent))
        fail(f"listed object absent: {absent[0]}")

    # A row any wanted output lost is dropped from every output: no mask stays True on
    # it.
    for data, mask, ver, sent in (
        (s.r3, s.r3_replay_mask, s.r3_version, s.r3_no_replay),
        (s.indexer, s.indexer_replay_mask, s.indexer_version, IDX_NO_REPLAY),
    ):
        if data is not None:
            mask &= ~rows_bad[:, None]
            data[~mask] = sent
            ver[~mask] = -1
    # Needed deep cells that nothing replays, per position (rows already missing
    # excluded).
    for name, data, mask, layer_ids in (
        (
            "r3",
            s.r3,
            s.r3_replay_mask,
            None if s.r3 is None else list(range(s.r3.shape[1])),
        ),
        ("indexer", s.indexer, s.indexer_replay_mask, s.indexer_layers),
    ):
        if data is None or cone is None:
            continue
        if cone.get("ends") is None:
            # Manifests before v54 only count this request's own holes.
            if s.uncovered.get(name):
                s.missing.add(name + "_uncovered", s.uncovered[name])
            continue
        deep = [j for j, lyr in enumerate(layer_ids) if lyr >= s.first_deep_layer]
        nm = np.zeros(mask.shape, bool)
        if deep:
            need = cone_mask(
                n,
                cone["ends"],
                [layer_ids[j] for j in deep],
                int(cone["last_layer"]),
                int(cone["window"]),
            )
            nm[:, deep] = need & ~mask[:, deep] & ~rows_bad[:, None]
        if nm.any():
            s.missing.add(name + "_needed_missing", int(nm.sum()))
        setattr(s, name + "_needed_missing", nm)
    # Rows delivered with an unknown weight version (manifest segment -1): replayable,
    # but the version cannot be used for staleness or importance weighting.
    vu = s.weight_version < 0
    for mask, ver in (
        (s.r3_replay_mask, s.r3_version),
        (s.indexer_replay_mask, s.indexer_version),
    ):
        if mask is not None:
            vu |= (mask & (ver < 0)).any(-1)
    s.version_unknown = vu & ~rows_bad
    s.version_unknown_count = int(s.version_unknown.sum())
    if strict and not s.ok:
        raise ReadError(f"incomplete sample: {s.missing.counts}")
    return s


def _read_r3(s, fr, c, tail_key):
    get, m, n, bs, rows, inv, tail_start = (
        c["get"],
        c["m"],
        c["n"],
        c["bs"],
        c["rows"],
        c["inv"],
        c["tail_start"],
    )
    rows_bad = s.missing.rows
    dt = np.dtype(fr.get("dtype", m["dtype"]))
    shape = [int(x) for x in fr.get("shape", m["shape"])]
    layers = shape[0]
    sent = np.iinfo(dt).max
    has_sent = dt.itemsize >= 2
    r3 = np.full((n, *shape), sent, dt)
    ok = np.zeros(n, bool)
    row_bytes = int(np.prod(shape)) * dt.itemsize
    for i, b in enumerate(m["blocks"][: c["nb"]]):
        buf = _get(get, f"{c['sess']}/{b}")
        if buf is None or len(buf) != bs * row_bytes:
            s.missing.add("r3_blocks")
            continue
        a, e = i * bs, min((i + 1) * bs, n)
        r3[a:e] = np.frombuffer(buf, dt).reshape(bs, *shape)[: e - a]
        ok[a:e] = True
    t_hi = min(rows, n)
    if t_hi > tail_start:
        tn = int(m["tail_nbytes"])
        buf = _get(get, tail_key)
        if tn == 0 or buf is None or len(buf) < tn or tn % row_bytes:
            s.missing.add("r3_tail")
        else:
            tail = np.frombuffer(buf[:tn], dt).reshape(-1, *shape)
            got = min(len(tail), t_hi - tail_start)
            r3[tail_start : tail_start + got] = tail[:got]
            ok[tail_start : tail_start + got] = True
            if got < t_hi - tail_start:
                s.missing.add("r3_tail_rows", t_hi - tail_start - got)
    lost = ~ok
    lost[rows:] = False  # counted as rows_short
    if inv is not None:
        lost[int(inv) :] = False
    rows_bad |= lost
    cone = c["cone"]
    d0 = layers if cone is None else int(cone["first_deep_layer"])
    # Sanity (§7) on the shallow cells: ids distinct, no sentinel, not an all-zero row.
    sh = r3[:, :d0]
    insane = ok & (
        _r3_bad_cells(sh, sent, has_sent).any(-1)
        | (sh.reshape(n, d0 * shape[1]) == 0).all(-1)
    )
    mask = np.repeat(ok[:, None], layers, 1)
    ver = np.where(mask, s.weight_version[:, None], -1)
    if cone is not None:
        plist = [
            (p["key"], p["lo"], p["end"], _get(get, p["key"]), None)
            for p in cone["patches"]
            if "key" in p and p["lo"] < n
        ]
        _overlay(
            r3,
            mask,
            ver,
            d0,
            sent,
            plist,
            n,
            list(range(d0, layers)),
            s.missing,
            "r3_cone_patches",
            rows_bad,
        )
        # Deep cells of a row whose shallow cells failed are not trusted either.
        mask[:, d0:] &= ok[:, None]
        # The same sanity on the replayed deep cells.
        insane |= (_r3_bad_cells(r3[:, d0:], sent, has_sent) & mask[:, d0:]).any(-1)
    if insane.any():
        s.missing.add("r3_invalid_rows", int(insane.sum()))
        rows_bad |= insane
    s.r3, s.r3_replay_mask, s.r3_version, s.r3_no_replay = r3, mask, ver, int(sent)


def _read_indexer(s, fi, c, _tail_key):
    get, m, n, bs, rows, inv, cone = (
        c["get"],
        c["m"],
        c["n"],
        c["bs"],
        c["rows"],
        c["inv"],
        c["cone"],
    )
    rows_bad = s.missing.rows
    layer_ids = [int(x) for x in fi["layers"]]
    L, K = int(fi["shape"][0]), int(fi["shape"][1])
    codec = str(fi["codec"])
    out = np.full((n, L, K), IDX_NO_REPLAY, np.int32)
    ok = np.zeros(n, bool)
    suffix = fi.get("block_suffix", "/idx")
    for i, b in enumerate(m["blocks"][: c["nb"]]):
        buf = _get(get, f"{c['sess']}/{b}{suffix}")
        try:
            if buf is None or len(buf) <= 1:
                raise KeyError(b)
            r = decode_indexer_object(buf, L, K, codec)
            if len(r) != bs:
                raise ValueError("block rows")
        except Exception:
            s.missing.add("indexer_blocks")
            continue
        a, e = i * bs, min((i + 1) * bs, n)
        out[a:e] = r[: e - a]
        ok[a:e] = True
    tail_start = int(fi.get("tail_start", m["tail_start"]))
    t_hi = min(int(fi.get("rows", rows)), n)
    if t_hi > tail_start:
        buf = _get(get, fi["tail_key"])
        try:
            if buf is None or len(buf) <= 1:
                raise KeyError("tail")
            r = decode_indexer_object(buf, L, K, codec)
            if len(r) != int(fi["tail_rows"]):
                raise ValueError("tail rows")
            got = min(len(r), t_hi - tail_start)
            out[tail_start : tail_start + got] = r[:got]
            ok[tail_start : tail_start + got] = True
            if got < t_hi - tail_start:
                s.missing.add("indexer_tail_rows", t_hi - tail_start - got)
        except Exception:
            s.missing.add("indexer_tail")
    # The writer's own counts name no block: the whole field is untrusted.
    for what in ("missing", "holes"):
        if int(fi.get(what, 0) or 0):
            s.missing.add("indexer_reported_" + what, int(fi[what]))
            ok[:] = False
    lost = ~ok
    lost[rows:] = False
    if inv is not None:
        lost[int(inv) :] = False
    rows_bad |= lost
    d0 = L
    if cone is not None:
        dslots = [
            j for j, lyr in enumerate(layer_ids) if lyr >= int(cone["first_deep_layer"])
        ]
        d0 = dslots[0] if dslots else L
        if dslots != list(range(d0, L)):
            # Deep slots must be the trailing slots; anything else cannot be overlaid.
            s.missing.add("indexer_layout")
            ok[:] = False
            rows_bad[:] = True
            d0 = L
    insane = ok & (out[:, :d0] < IDX_PAD).any((-1, -2))
    if insane.any():
        s.missing.add("indexer_invalid_rows", int(insane.sum()))
        rows_bad |= insane
    mask = np.repeat(ok[:, None], L, 1)
    ver = np.where(mask, s.weight_version[:, None], -1)
    if d0 < L:
        deep_layers = layer_ids[d0:]

        def unpacker(pc):
            return (
                lambda d: d
                if d.dtype != np.uint8
                else _unpack_packed(d.tobytes(), len(deep_layers), K, pc)
            )

        plist = [
            (
                p["idx_key"],
                p["lo"],
                p["end"],
                _get(get, p["idx_key"]),
                unpacker(str(p.get("idx_codec", codec))),
            )
            for p in cone["patches"]
            if "idx_key" in p and p["lo"] < n
        ]
        _overlay(
            out,
            mask,
            ver,
            d0,
            IDX_NO_REPLAY,
            plist,
            n,
            deep_layers,
            s.missing,
            "indexer_cone_patches",
            rows_bad,
        )
        mask[:, d0:] &= ok[:, None]
        bad_deep = mask[:, d0:] & (out[:, d0:] < IDX_PAD).any(-1)
        if bad_deep.any():
            s.missing.add("indexer_invalid_cells", int(bad_deep.sum()))
            rows_bad |= bad_deep.any(-1)
    out[~mask] = IDX_NO_REPLAY
    ver[~mask] = -1
    s.indexer, s.indexer_layers, s.indexer_replay_mask, s.indexer_version = (
        out,
        layer_ids,
        mask,
        ver,
    )


_HID_RAW = {"float64": np.float64, "float32": np.float32, "float16": np.float16}


def _hid_raw_dtype(name):
    """Numpy dtype of a raw hidden row element; bfloat16 is ml_dtypes' or its uint16
    bits."""
    if name in _HID_RAW:
        return np.dtype(_HID_RAW[name]), np.nan
    if name == "bfloat16":
        try:
            import ml_dtypes

            return np.dtype(ml_dtypes.bfloat16), np.nan
        except ImportError:
            return np.dtype(np.uint16), 0x7FC0
    return None, None


def _read_hidden(s, fh, c, _tail_key):
    get, n = c["get"], c["n"]
    rows_bad = s.missing.rows
    d = int(fh["shape"][0])
    start, total = int(fh["start"]), int(fh["rows"])
    lens = fh.get("chunk_lens")
    nch = int(fh["chunks"])
    if fh.get("codec") != "raw":
        raise ValueError(f"hidden codec {fh.get('codec')!r}: only raw is supported")
    dt, fill = _hid_raw_dtype(fh.get("dtype"))
    row = d * dt.itemsize if dt is not None else 0
    if dt is None or lens is None or len(lens) != nch or sum(lens) != total:
        s.missing.add("hidden_meta")
        rows_bad[min(start, n) :] = True
        lens = None
        if dt is None:
            dt, fill = np.dtype(np.uint8), 0
    q = np.zeros((max(0, min(start + total, n) - start), d), dt)
    got = np.zeros(len(q), bool)
    at = start
    for i in range(nch if lens is not None else 0):
        k = int(lens[i])
        if at >= n:
            break
        buf = _get(get, f"{fh['key_prefix']}{i}")
        a, b = at - start, min(at + k, n) - start
        if buf is None or len(buf) != k * row:
            s.missing.add("hidden_chunks")
        else:
            q[a:b] = np.frombuffer(buf, dt).reshape(k, d)[: b - a]
            got[a:b] = True
        at += k
    if start + total < n and start < n:
        s.missing.add("hidden_rows_short", n - (start + total))
        rows_bad[start + total :] = True
    rows_bad[start : start + len(got)] |= ~got
    s.hidden_start = start
    q[~got] = fill
    s.hidden, s.hidden_dtype = q, fh.get("dtype")


def _read_topk(s, ft, c, _tail_key):
    """Chunks of rows of k int32 ids then k float32 logprobs, positions
    start..start+rows-1."""
    get, n = c["get"], c["n"]
    rows_bad = s.missing.rows
    k = int(ft["shape"][0])
    start, total = int(ft["start"]), int(ft["rows"])
    lens = ft.get("chunk_lens")
    nch = int(ft["chunks"])
    m = max(0, min(start + total, n) - start)
    ids = np.full((m, k), -1, np.int32)
    lps = np.full((m, k), np.nan, np.float32)
    got = np.zeros(m, bool)
    if (
        ft.get("codec") != "raw"
        or ft.get("dtype") != "int32+float32"
        or lens is None
        or len(lens) != nch
        or sum(lens) != total
    ):
        s.missing.add("topk_meta")
        rows_bad[min(start, n) :] = True
        lens = None
    at = start
    for i in range(nch if lens is not None else 0):
        r = int(lens[i])
        if at >= n:
            break
        buf = _get(get, f"{ft['key_prefix']}{i}")
        a, b = at - start, min(at + r, n) - start
        if buf is None or len(buf) != r * 8 * k:
            s.missing.add("topk_chunks")
        else:
            rows = np.frombuffer(buf, np.uint8).reshape(r, 8 * k)
            ids[a:b] = rows[: b - a, : 4 * k].copy().view(np.int32)
            lps[a:b] = rows[: b - a, 4 * k :].copy().view(np.float32)
            got[a:b] = True
        at += r
    if start + total < n and start < n:
        s.missing.add("topk_rows_short", n - (start + total))
        rows_bad[start + total :] = True
    rows_bad[start : start + len(got)] |= ~got
    ids[~got], lps[~got] = -1, np.nan
    s.topk_ids, s.topk_logprobs, s.topk_start = ids, lps, start
