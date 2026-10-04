# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object-free rendering of ``/inference/v1/generate`` sample logprobs.

Both renderers read the engine rows kept by :class:`ArrayLogprobs` and
produce JSON bytes directly, without per-entry Python/pydantic objects:

* :func:`render_compact_logprobs` produces the opt-in
  ``logprobs_format="compact"`` block (base64 of the raw engine arrays).
* :func:`render_openai_logprobs` produces exactly the bytes the legacy path
  (``ServingTokens._create_tokens_logprobs`` + ``model_dump`` + Starlette's
  ``JSONResponse.render``) produces for the same rows, or returns ``None`` when
  the rows are irregular and the caller must use the legacy path.

:func:`render_json_with_fragments_parts` splices pre-rendered fragments into
the JSON of the remaining (small) response fields and returns the body as a
list of parts, which the router sends without joining them.
"""

import json
import secrets
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import msgspec
import numpy as np
import pybase64

from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.logprobs import ArrayLogprobs

COMPACT_DTYPE_TOKEN_IDS = "int32"
COMPACT_DTYPE_LOGPROBS = "float32"
COMPACT_BYTEORDER = "little"

# Rows rendered per batch; bounds temporary per-entry strings.
_RENDER_BLOCK_ROWS = 1024


def compact_num_slots(container: ArrayLogprobs, num_logprobs: int | None) -> int:
    """Slots per position: the stored width, else ``k + 1`` (0 for ``k=-1``)."""
    stored = container.num_slots
    if stored is not None:
        return stored
    if num_logprobs is None or num_logprobs < 0:
        return 0
    return num_logprobs + 1


def _as_wire_int32(values: np.ndarray, what: str) -> np.ndarray:
    """Cast to little-endian int32, refusing (never wrapping) wider values."""
    if values.dtype == np.dtype("<i4"):
        return values
    info = np.iinfo(np.int32)
    if values.size and (int(values.min()) < info.min or int(values.max()) > info.max):
        raise GenerationError(f"A {what} does not fit the compact int32 wire format")
    return values.astype("<i4")


def _check_compact_rows(
    container: ArrayLogprobs,
    num_logprobs: int | None,
    expected_positions: int | None = None,
    expected_source_positions: int | None = None,
) -> None:
    """Refuse rows the compact format cannot represent (GenerationError, 500):
    inconsistent widths, or a width other than ``k + 1`` for ``k >= 0``
    (e.g. narrower rows when a co-batched request's ``logprob_token_ids``
    replaced the batch's logprob tensors)."""
    if not container.is_regular:
        raise GenerationError(
            "Engine logprob rows have inconsistent widths; the compact "
            "logprobs format cannot represent them"
        )
    if (
        expected_source_positions is not None
        and container.source_positions is not None
        and container.source_positions != expected_source_positions
    ):
        # Streaming: the DELTA slice always has as many rows as new tokens,
        # but if a step had tokens without rows, the slice holds rows of
        # earlier tokens. Compare the cumulative counts instead.
        raise GenerationError(
            f"{container.source_positions} logprob positions for "
            f"{expected_source_positions} generated tokens; the compact "
            "logprobs format cannot represent them"
        )
    if expected_positions is not None and len(container) != expected_positions:
        # E.g. an engine step with tokens but no logprob rows: rows would no
        # longer line up with the generated tokens.
        raise GenerationError(
            f"{len(container)} logprob positions for {expected_positions} "
            "generated tokens; the compact logprobs format cannot represent them"
        )
    stored = container.num_slots
    if (
        stored is not None
        and num_logprobs is not None
        and num_logprobs >= 0
        and stored != num_logprobs + 1
    ):
        raise GenerationError(
            f"Engine logprob rows have {stored} slots, expected "
            f"{num_logprobs + 1}; the compact logprobs format cannot "
            "represent them"
        )


def compact_logprobs_fields(
    container: ArrayLogprobs,
    num_logprobs: int | None,
    expected_positions: int | None = None,
    expected_source_positions: int | None = None,
) -> tuple[int, int, bytes, bytes, bytes]:
    """Return ``(N, S, b64(token_ids), b64(logprobs), b64(ranks))``.

    Raises GenerationError (HTTP 500) when the engine rows cannot be
    represented: inconsistent row widths or ids/ranks beyond int32.
    """
    _check_compact_rows(
        container, num_logprobs, expected_positions, expected_source_positions
    )
    token_ids, logprobs, ranks = container.arrays()
    # Arrays are C-contiguous little-endian (ArrayLogprobs). The wire format
    # is int32/float32; engine data normally already has these dtypes.
    token_ids = _as_wire_int32(token_ids, "token id")
    ranks = _as_wire_int32(ranks, "rank")
    if logprobs.dtype != np.dtype("<f4"):
        logprobs = logprobs.astype("<f4")
    return (
        len(ranks),
        compact_num_slots(container, num_logprobs),
        pybase64.b64encode(token_ids),
        pybase64.b64encode(logprobs),
        pybase64.b64encode(ranks),
    )


def render_compact_logprobs_parts(
    container: ArrayLogprobs,
    num_logprobs: int | None,
    expected_positions: int | None = None,
) -> list[bytes]:
    """The ``compact_logprobs`` JSON object (see the generate SPEC) as parts
    whose concatenation is the JSON; avoids copying the large payloads."""
    token_ids: bytes | list[bytes]
    logprobs: bytes | list[bytes]
    ranks: bytes | list[bytes]
    _check_compact_rows(container, num_logprobs, expected_positions)
    wire = container.wire_parts()
    if wire is not None:
        # Encoded while the rows arrived (ArrayLogprobs wire mode).
        n, stored_slots, token_ids, logprobs, ranks = wire
        s = (
            stored_slots
            if stored_slots is not None
            else compact_num_slots(container, num_logprobs)
        )
    else:
        n, s, token_ids, logprobs, ranks = compact_logprobs_fields(
            container, num_logprobs, expected_positions
        )
    head = (
        f'{{"num_positions":{n},"num_slots":{s},'
        f'"dtype_token_ids":"{COMPACT_DTYPE_TOKEN_IDS}",'
        f'"dtype_logprobs":"{COMPACT_DTYPE_LOGPROBS}",'
        f'"byteorder":"{COMPACT_BYTEORDER}","token_ids":"'
    ).encode("ascii")
    parts = [head]
    for piece, tail in (
        (token_ids, b'","logprobs":"'),
        (logprobs, b'","ranks":"'),
        (ranks, b'"}'),
    ):
        if isinstance(piece, bytes):
            parts.append(piece)
        else:
            parts.extend(piece)
        parts.append(tail)
    return parts


def render_compact_logprobs(
    container: ArrayLogprobs, num_logprobs: int | None
) -> bytes:
    """Render the ``compact_logprobs`` JSON object (see the generate SPEC)."""
    return b"".join(render_compact_logprobs_parts(container, num_logprobs))


_SEP_TOP_FIRST = b',"bytes":null,"top_logprobs":['
_SEP_TOP_NEXT = b',"bytes":null},'
_END_WITH_TOP = b',"bytes":null}]}'
_END_NO_TOP = b',"bytes":null,"top_logprobs":[]}'


class _LeadTable:
    """token id -> ``sep + {"token":"token_id:<id>","logprob":`` as a numpy
    object array, so a block of ids maps to leads with one fancy index.

    Covers ids below ``max_ids`` (~100 bytes per id actually seen, plus an
    8-byte slot per id up to the largest seen); other ids are formatted
    per call.
    """

    max_ids = 1 << 18

    def __init__(self, sep: bytes):
        self.sep = sep
        self.values = np.empty(0, dtype=object)
        self.filled = np.zeros(0, dtype=bool)
        # Renders may run on the event loop and in the worker thread.
        self.lock = threading.Lock()

    def _format(self, token_id: int) -> bytes:
        return self.sep + f'{{"token":"token_id:{token_id}","logprob":'.encode()

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        if ids.size == 0:
            return np.empty(ids.shape, dtype=object)
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.max_ids:
            # Ids outside the table are formatted directly, the rest cached.
            inside = (ids >= 0) & (ids < self.max_ids)
            result = np.empty(ids.shape, dtype=object)
            if inside.any():
                result[inside] = self.lookup(ids[inside])
            outside = [self._format(i) for i in ids[~inside].tolist()]
            column = np.empty(len(outside), dtype=object)
            column[:] = outside
            result[~inside] = column
            return result
        values, filled = self.values, self.filled
        if hi < len(values):
            missing = np.unique(ids[~filled[ids]])
            if not missing.size:
                return values[ids]  # warm path: no lock
        else:
            missing = np.unique(ids)
        # Format outside the lock (the expensive part), publish under it, so
        # a render on the event loop never waits for another thread's
        # formatting; the lock only covers growth and a few stores.
        formatted = [(i, self._format(i)) for i in missing.tolist()]
        with self.lock:
            if hi >= len(self.values):
                size = min(self.max_ids, max(hi + 1, 2 * len(self.values)))
                grown = np.empty(size, dtype=object)
                grown[: len(self.values)] = self.values
                grown_filled = np.zeros(size, dtype=bool)
                grown_filled[: len(self.filled)] = self.filled
                self.values, self.filled = grown, grown_filled
            values, filled = self.values, self.filled
            for token_id, lead in formatted:
                values[token_id] = lead
            filled[missing] = True
            return values[ids]


_NEXT_LEADS = _LeadTable(_SEP_TOP_NEXT)
_PLAIN_LEADS = _LeadTable(b"")


def format_float_reprs(values: np.ndarray, exact_float32: bool) -> list[bytes]:
    """``[repr(float(v)).encode() for v in values]`` for a 1-D float64 array.

    ``repr`` (what ``json.dumps`` emits) costs ~0.3-0.5 us per value. When
    every value is exactly representable in float32, msgspec's shortest
    round-trip encoder is used instead: for ``1e-4 <= |v| < 1e16`` and zeros
    both use the same digits in fixed notation (verified exhaustively over
    all float32 values, see tests); other magnitudes, where the exponent
    notation differs (``1e-05`` vs ``0.00001``), use ``repr``.
    Values must be finite. The fast path is disabled if an import-time
    probe finds msgspec formatting that differs from ``repr``.
    """
    floats = values.tolist()
    if not floats:
        return []
    if not (exact_float32 and _MSGSPEC_FLOATS_MATCH_REPR):
        return [repr(v).encode("ascii") for v in floats]
    return _format_fast(values)


def _msgspec_floats_match_repr() -> bool:
    """Probe the msgspec fast path against ``repr`` once at import.

    The fast path was verified exhaustively for msgspec 0.21.1; this probe
    (float32 boundaries of the fast range, powers of ten, extremes and a
    fixed pseudo-random sample) disables it if another msgspec version
    formats any of them differently.
    """
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
        return _format_fast(values) == [repr(v).encode("ascii") for v in probe]
    except Exception:
        return False


def _format_fast(values: np.ndarray) -> list[bytes]:
    floats = values.tolist()
    out = msgspec.json.encode(floats)[1:-1].split(b",")
    magnitude = np.abs(values)
    for i in np.flatnonzero(
        ~(((magnitude >= 1e-4) & (magnitude < 1e16)) | (magnitude == 0))
    ).tolist():
        out[i] = repr(floats[i]).encode("ascii")
    return out


_MSGSPEC_FLOATS_MATCH_REPR = _msgspec_floats_match_repr()


def _rows_have_unique_slots(token_ids: np.ndarray) -> bool:
    """Whether slots ``1..S-1`` hold distinct ids in every row."""
    if token_ids.shape[1] <= 2:
        return True
    for start in range(0, len(token_ids), 65536):
        block = np.sort(token_ids[start : start + 65536, 1:], axis=1)
        if (block[:, 1:] == block[:, :-1]).any():
            return False
    return True


def render_openai_logprobs(
    sampled_token_ids: Sequence[int],
    container: ArrayLogprobs,
    num_output_top_logprobs: int | None,
) -> bytes | None:
    """Joined :func:`render_openai_logprobs_parts`."""
    parts = render_openai_logprobs_parts(
        sampled_token_ids, container, num_output_top_logprobs
    )
    return None if parts is None else b"".join(parts)


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
    num_slots = token_ids.shape[1]
    sampled = np.asarray(sampled_token_ids, dtype=np.int64)
    if not np.array_equal(token_ids[:, 0], sampled):
        return None
    if not _rows_have_unique_slots(token_ids):
        return None

    limit = 0 if num_output_top_logprobs is None else max(num_output_top_logprobs, 1)
    # Column (>= 1) repeating the sampled id, at most one per row here.
    if num_slots > 1:
        eq = token_ids[:, 1:] == token_ids[:, :1]
        has_dup = eq.any(axis=1)
        dup_col = np.where(has_dup, eq.argmax(axis=1) + 1, num_slots)
    else:
        has_dup = np.zeros(n, dtype=bool)
        dup_col = np.full(n, num_slots)
    if limit and (num_slots - has_dup.astype(np.int64)).min() < limit:
        return None

    # Column 0: sampled entry; columns 1..limit: top entries c = 0..limit-1.
    cols = np.arange(limit)
    src = cols[None, :] + (cols[None, :] >= dup_col[:, None])
    id_cols = np.empty((n, limit + 1), dtype=np.int64)
    id_cols[:, 0] = token_ids[:, 0]
    id_cols[:, 1:] = np.take_along_axis(token_ids, src, axis=1)
    # Value source: the duplicate's slot overwrote the sampled entry's value.
    value_src = np.empty((n, limit + 1), dtype=np.int64)
    value_src[:, 0] = np.where(has_dup, dup_col, 0)
    value_src[:, 1:] = np.where(src == 0, value_src[:, :1], src)
    values = np.maximum(
        np.take_along_axis(logprobs, value_src, axis=1).astype(np.float64), -9999.0
    )
    finite = np.isfinite(values)
    if not finite.all():
        # Same message as json.dumps(..., allow_nan=False) for the first
        # offending value in document order (row-major here).
        bad = values.ravel()[int(np.flatnonzero(~finite.ravel())[0])]
        raise ValueError(
            f"Out of range float values are not JSON compliant: {float(bad)!r}"
        )

    exact_float32 = logprobs.dtype == np.float32
    width = limit + 1
    row_end = _END_WITH_TOP if limit else _END_NO_TOP
    row_sep = row_end + b","
    # Each value is preceded by one "lead" piece: the separator closing the
    # previous entry plus this entry's ``{"token":...,"logprob":`` prefix.
    # Leads of top entries 2..k come from a per-id table, so a block is a
    # single join of alternating leads and float texts.
    parts: list[bytes] = [b'{"content":[']
    for start in range(0, n, _RENDER_BLOCK_ROWS):
        stop = min(start + _RENDER_BLOCK_ROWS, n)
        block_ids = id_cols[start:stop]
        out: list[bytes] = [b""] * (2 * (stop - start) * width)
        if limit > 1:
            out[0::2] = _NEXT_LEADS.lookup(block_ids).ravel().tolist()
        plain = _PLAIN_LEADS.lookup(block_ids[:, : min(width, 2)]).tolist()
        out[0 :: 2 * width] = [row_sep + lead[0] for lead in plain]
        if start == 0:
            out[0] = plain[0][0]
        if limit:
            out[2 :: 2 * width] = [_SEP_TOP_FIRST + lead[1] for lead in plain]
        out[1::2] = format_float_reprs(values[start:stop].ravel(), exact_float32)
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
    content: dict[str, Any],
    choice_fragments: Mapping[int, Mapping[str, bytes | list[bytes]]],
) -> bytes:
    """Joined :func:`render_json_with_fragments_parts`."""
    return b"".join(render_json_with_fragments_parts(content, choice_fragments))


def render_json_with_fragments_parts(
    content: dict[str, Any],
    choice_fragments: Mapping[int, Mapping[str, bytes | list[bytes]]],
) -> list[bytes | memoryview]:
    """Render ``content`` like ``JSONResponse`` with pre-rendered values.

    ``choice_fragments[i][key]`` is the JSON (bytes, or a list of parts to
    concatenate) for ``content["choices"][i][key]``.
    Keys missing from a choice dict are appended (preserving field order of
    trailing optional fields). ``content`` is modified in place.
    """
    if not choice_fragments:
        return [_dumps(content)]
    token = secrets.token_hex(16)
    fragments: list[bytes | list[bytes]] = []
    choices = content["choices"]
    for index, values in choice_fragments.items():
        for key, fragment in values.items():
            choices[index][key] = f"{token}:{len(fragments)}"
            fragments.append(fragment)
    pieces = _dumps(content).split(f'"{token}:'.encode())
    if len(pieces) != len(fragments) + 1:
        raise AssertionError("Fragment placeholder collision")
    out: list[bytes | memoryview] = [pieces[0]]
    for i, piece in enumerate(pieces[1:]):
        marker = f'{i}"'.encode()
        if not piece.startswith(marker):
            raise AssertionError("Fragment placeholder out of order")
        fragment = fragments[i]
        if isinstance(fragment, bytes):
            out.append(fragment)
        else:
            out.extend(fragment)
        out.append(memoryview(piece)[len(marker) :])
    return out
