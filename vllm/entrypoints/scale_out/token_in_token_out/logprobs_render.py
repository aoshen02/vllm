# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object-free rendering of ``/inference/v1/generate`` sample logprobs.

:func:`render_openai_logprobs_parts` reads the engine rows kept by
:class:`ArrayLogprobs` and produces exactly the bytes the legacy path
(``ServingTokens._create_tokens_logprobs`` + ``model_dump`` + Starlette's
``JSONResponse.render``) produces for the same rows, without per-entry
Python/pydantic objects, or returns ``None`` when the rows are irregular and
the caller must use the legacy path.

:func:`render_json_with_fragments` splices pre-rendered values into the JSON
of the remaining (small) response fields.
"""

import json
import secrets
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import msgspec
import numpy as np

from .array_logprobs import ArrayLogprobs

# Rows rendered per batch; bounds the per-entry strings and index arrays.
_RENDER_BLOCK_ROWS = 1024


_SEP_TOP_FIRST = b',"bytes":null,"top_logprobs":['
_SEP_TOP_NEXT = b',"bytes":null},'
_END_WITH_TOP = b',"bytes":null}]}'
_END_NO_TOP = b',"bytes":null,"top_logprobs":[]}'


class _LeadTable:
    """token id -> ``sep + {"token":"token_id:<id>","logprob":`` as a numpy
    object array, so a block of ids maps to leads with one fancy index.

    Covers ids below ``max_ids``: ~100 bytes per id actually seen plus an
    8-byte slot per id up to the largest seen, kept for the process lifetime
    (a few MB per table at a 150k vocabulary). Blocks with other ids are
    formatted per call.
    """

    max_ids = 1 << 18

    def __init__(self, sep: bytes):
        self.sep = sep
        # (values, filled), replaced as one tuple so a lock-free reader always
        # sees a consistent pair.
        self._table: tuple[np.ndarray, np.ndarray] = (
            np.empty(0, dtype=object),
            np.zeros(0, dtype=bool),
        )
        # Renders may run on the event loop and in the worker threads.
        self.lock = threading.Lock()

    def _format(self, token_id: int) -> bytes:
        return self.sep + f'{{"token":"token_id:{token_id}","logprob":'.encode()

    def _formatted(self, ids: np.ndarray) -> np.ndarray:
        result = np.empty(ids.size, dtype=object)
        result[:] = [self._format(i) for i in ids.tolist()]
        return result

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        """Leads of a non-empty array of ids, in its shape."""
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.max_ids:
            return self._formatted(ids.ravel()).reshape(ids.shape)
        values, filled = self._table
        if hi < len(values):
            missing = np.unique(ids[~filled[ids]])
            if not missing.size:
                return values[ids]  # warm path: no lock
        else:
            missing = np.unique(ids)
        # Format outside the lock, publish under it (growth and a few stores).
        # Only bytes per id: they are not GC-tracked.
        formatted = self._formatted(missing)
        with self.lock:
            values, filled = self._table
            if hi >= len(values):
                size = min(self.max_ids, max(hi + 1, 2 * len(values)))
                grown = np.empty(size, dtype=object)
                grown[: len(values)] = values
                grown_filled = np.zeros(size, dtype=bool)
                grown_filled[: len(filled)] = filled
                values, filled = grown, grown_filled
            # Values before flags: a reader that sees a flag sees its value.
            values[missing] = formatted
            filled[missing] = True
            self._table = (values, filled)
            return values[ids]


_NEXT_LEADS = _LeadTable(_SEP_TOP_NEXT)
_PLAIN_LEADS = _LeadTable(b"")


def format_float_reprs(values: np.ndarray) -> list[bytes]:
    """``[repr(float(v)).encode() for v in values]`` for a non-empty 1-D
    float64 array of finite values that float32 represents exactly.

    msgspec's shortest round-trip encoder is used where it gives the same
    digits as ``repr`` (what ``json.dumps`` emits) for float32 values:
    ``1e-4 <= |v| < 1e16`` and zeros; other magnitudes use ``repr``. The fast
    path is disabled if an import-time probe finds a difference.
    """
    floats = values.tolist()
    if not _MSGSPEC_FLOATS_MATCH_REPR:
        return [repr(v).encode("ascii") for v in floats]
    return _format_fast(values, floats)


def _msgspec_floats_match_repr() -> bool:
    """Probe the msgspec fast path against ``repr`` once at import (float32
    boundaries of the fast range, powers of ten, extremes and a fixed
    pseudo-random sample)."""
    probe: list[float] = [0.0, -0.0, 0.1, 0.5, 1.0, 9999.0, 123.456, -1.2e-7]
    for edge in (1e-4, 1e16, 1.0, 1e-3, 1e6, 1e15, 3.4028235e38, 1e-45):
        e = np.float32(edge)
        probe += [
            float(e),
            float(np.nextafter(e, np.float32(0))),
            float(np.nextafter(e, np.float32(3.4028235e38))),
        ]
    probe += [float(np.float32(10.0**p)) for p in range(-45, 39)]
    bits = np.random.default_rng(1234).integers(0, 2**32, 4096, dtype=np.uint64)
    sample = bits.astype(np.uint32).view(np.float32)
    probe += sample[np.isfinite(sample)].astype(np.float64).tolist()
    probe += [-v for v in probe]
    values = np.array(probe, dtype=np.float64)
    values = values[np.isfinite(values)]
    probe = values.tolist()
    try:
        return _format_fast(values, probe) == [repr(v).encode("ascii") for v in probe]
    except Exception:
        return False


def _format_fast(values: np.ndarray, floats: list[float]) -> list[bytes]:
    out = msgspec.json.encode(floats)[1:-1].split(b",")
    magnitude = np.abs(values)
    for i in np.flatnonzero(
        ~(((magnitude >= 1e-4) & (magnitude < 1e16)) | (magnitude == 0))
    ).tolist():
        out[i] = repr(floats[i]).encode("ascii")
    return out


_MSGSPEC_FLOATS_MATCH_REPR = _msgspec_floats_match_repr()


def render_openai_logprobs_parts(
    sampled_token_ids: Sequence[int],
    container: ArrayLogprobs,
    num_output_top_logprobs: int | None,
) -> list[bytes] | None:
    """Render ``ChatCompletionLogProbs`` JSON identical to the legacy path,
    as parts whose concatenation is the JSON.

    Legacy semantics per position (``dict`` built from the row): keys keep
    first-occurrence order, values come from the last occurrence; the sampled
    entry is looked up by the sampled id; ``top_logprobs`` are the first
    ``max(k, 1)`` dict items (none when ``k`` is None); every value is
    ``max(v, -9999.0)``.

    Returns None when a row is irregular (sampled id not in slot 0, repeated
    top-k ids, length mismatch); the caller must then use the legacy path.
    Raises ValueError for values JSON cannot represent (NaN/+inf after the
    clamp), like ``json.dumps(..., allow_nan=False)`` does on the legacy path.
    """
    if not container.is_regular:
        return None
    token_ids, logprobs, _ = container.arrays()
    n = len(sampled_token_ids)
    if n != len(token_ids):
        return None
    if n == 0:
        return [b'{"content":[]}']
    sampled = np.asarray(sampled_token_ids, dtype=np.int64)
    if not np.array_equal(token_ids[:, 0], sampled):
        return None

    num_slots = token_ids.shape[1]
    limit = 0 if num_output_top_logprobs is None else max(num_output_top_logprobs, 1)
    cols = np.arange(limit)
    row_end = _END_WITH_TOP if limit else _END_NO_TOP
    row_sep = row_end + b","
    # Each value is preceded by one "lead" piece: the separator closing the
    # previous entry plus this entry's ``{"token":...,"logprob":`` prefix.
    # Leads of top entries 2..k come from a per-id table, so a block is a
    # single join of alternating leads and float texts. Index arrays are
    # per block, so their memory stays small whatever the response size.
    parts: list[bytes] = [b'{"content":[']
    for start in range(0, n, _RENDER_BLOCK_ROWS):
        ids = token_ids[start : start + _RENDER_BLOCK_ROWS]
        rows = len(ids)
        if num_slots > 2:
            top = np.sort(ids[:, 1:], axis=1)
            if (top[:, 1:] == top[:, :-1]).any():
                return None  # repeated top-k ids
        # Column (>= 1) repeating the sampled id, at most one per row here.
        if num_slots > 1:
            eq = ids[:, 1:] == ids[:, :1]
            has_dup = eq.any(axis=1)
            dup_col = np.where(has_dup, eq.argmax(axis=1) + 1, num_slots)
        else:
            has_dup = np.zeros(rows, dtype=bool)
            dup_col = np.full(rows, num_slots)
        if limit and num_slots - int(has_dup.any()) < limit:
            return None
        # Column 0: sampled entry; columns 1..limit: top entries c = 0..limit-1.
        src = cols[None, :] + (cols[None, :] >= dup_col[:, None])
        id_cols = np.empty((rows, limit + 1), dtype=np.int64)
        id_cols[:, 0] = ids[:, 0]
        id_cols[:, 1:] = np.take_along_axis(ids, src, axis=1)
        # Value source: the duplicate's slot overwrote the sampled entry's.
        value_src = np.empty((rows, limit + 1), dtype=np.int64)
        value_src[:, 0] = np.where(has_dup, dup_col, 0)
        value_src[:, 1:] = np.where(src == 0, value_src[:, :1], src)
        values = np.take_along_axis(
            logprobs[start : start + rows], value_src, axis=1
        ).astype(np.float64)
        values = np.maximum(values, -9999.0).ravel()
        finite = np.isfinite(values)
        if not finite.all():
            # Same message as json.dumps(..., allow_nan=False) for the first
            # offending value in document order (row-major here).
            bad = float(values[int(np.flatnonzero(~finite)[0])])
            raise ValueError(
                f"Out of range float values are not JSON compliant: {bad!r}"
            )

        width = limit + 1
        out: list[bytes] = [b""] * (2 * rows * width)
        if limit > 1:
            out[0::2] = _NEXT_LEADS.lookup(id_cols).ravel().tolist()
        # Flat (1-D) lookups: nested lists would be one GC-tracked list per row.
        firsts = _PLAIN_LEADS.lookup(id_cols[:, 0]).tolist()
        out[0 :: 2 * width] = [row_sep + lead for lead in firsts]
        if start == 0:
            out[0] = firsts[0]
        if limit:
            seconds = _PLAIN_LEADS.lookup(id_cols[:, 1]).tolist()
            out[2 :: 2 * width] = [_SEP_TOP_FIRST + lead for lead in seconds]
        out[1::2] = format_float_reprs(values)
        parts.append(b"".join(out))
    parts.append(row_end + b"]}")
    return parts


def _dumps(content: Any) -> bytes:
    """Same serialization as ``starlette.responses.JSONResponse.render``."""
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")


def render_json_with_fragments(
    content: dict[str, Any], key: str, fragments: Mapping[int, list[bytes]]
) -> list[bytes]:
    """Render ``content`` like ``JSONResponse``, with the value of
    ``content["choices"][i][key]`` the pre-rendered JSON ``fragments[i]``
    (parts to concatenate), as parts whose concatenation is the body. The
    key is appended to choices without it. ``content`` is modified in
    place."""
    token = secrets.token_hex(16)
    for index in fragments:
        content["choices"][index][key] = token
    pieces = _dumps(content).split(f'"{token}"'.encode())
    if len(pieces) != len(fragments) + 1:
        raise AssertionError("Fragment placeholder collision")
    # Choices, hence placeholders, appear in index order.
    out = [pieces[0]]
    for index, piece in zip(sorted(fragments), pieces[1:]):
        out += fragments[index]
        out.append(piece)
    return out
