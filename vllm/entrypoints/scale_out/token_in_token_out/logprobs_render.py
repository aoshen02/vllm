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

:func:`render_json_with_fragments` splices pre-rendered fragments into the
JSON of the remaining (small) response fields.
"""

import json
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pybase64

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


def compact_logprobs_fields(
    container: ArrayLogprobs, num_logprobs: int | None
) -> tuple[int, int, bytes, bytes, bytes]:
    """Return ``(N, S, b64(token_ids), b64(logprobs), b64(ranks))``."""
    token_ids, logprobs, ranks = container.arrays()
    # Arrays are C-contiguous little-endian int32/float32 (ArrayLogprobs).
    return (
        len(ranks),
        compact_num_slots(container, num_logprobs),
        pybase64.b64encode(token_ids),
        pybase64.b64encode(logprobs),
        pybase64.b64encode(ranks),
    )


def render_compact_logprobs(
    container: ArrayLogprobs, num_logprobs: int | None
) -> bytes:
    """Render the ``compact_logprobs`` JSON object (see the generate SPEC)."""
    n, s, token_ids, logprobs, ranks = compact_logprobs_fields(container, num_logprobs)
    head = (
        f'{{"num_positions":{n},"num_slots":{s},'
        f'"dtype_token_ids":"{COMPACT_DTYPE_TOKEN_IDS}",'
        f'"dtype_logprobs":"{COMPACT_DTYPE_LOGPROBS}",'
        f'"byteorder":"{COMPACT_BYTEORDER}","token_ids":"'
    ).encode("ascii")
    return b"".join(
        (head, token_ids, b'","logprobs":"', logprobs, b'","ranks":"', ranks, b'"}')
    )


class _ItemPrefixCache(dict[int, str]):
    """token id -> ``{"token":"token_id:<id>","logprob":`` (bounded)."""

    max_size = 1 << 20

    def __missing__(self, token_id: int) -> str:
        if len(self) >= self.max_size:
            self.clear()
        value = f'{{"token":"token_id:{token_id}","logprob":'
        self[token_id] = value
        return value


_ITEM_PREFIX = _ItemPrefixCache()
_SEP_TOP_FIRST = ',"bytes":null,"top_logprobs":['
_SEP_TOP_NEXT = ',"bytes":null},'
_END_WITH_TOP = ',"bytes":null}]}'
_END_NO_TOP = ',"bytes":null,"top_logprobs":[]}'


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
    """Render ``ChatCompletionLogProbs`` JSON identical to the legacy path.

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
    token_ids, logprobs, _ = container.arrays()
    n = len(sampled_token_ids)
    if n != len(token_ids):
        return None
    if n == 0:
        return b'{"content":[]}'
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
    if not np.isfinite(values).all():
        raise ValueError("Out of range float values are not JSON compliant")

    width = limit + 1
    item = _ITEM_PREFIX.__getitem__
    to_repr = float.__repr__
    concat = str.__add__
    end = _END_WITH_TOP if limit else _END_NO_TOP
    parts: list[bytes] = [b'{"content":[']
    for start in range(0, n, _RENDER_BLOCK_ROWS):
        stop = min(start + _RENDER_BLOCK_ROWS, n)
        entries = list(
            map(
                concat,
                map(item, id_cols[start:stop].ravel().tolist()),
                map(to_repr, values[start:stop].ravel().tolist()),
            )
        )
        block = []
        for offset in range(0, len(entries), width):
            if limit:
                block.append(
                    entries[offset]
                    + _SEP_TOP_FIRST
                    + _SEP_TOP_NEXT.join(entries[offset + 1 : offset + width])
                    + end
                )
            else:
                block.append(entries[offset] + end)
        text = ",".join(block)
        if start:
            parts.append(b",")
        parts.append(text.encode("ascii"))
    parts.append(b"]}")
    return b"".join(parts)


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
    choice_fragments: Mapping[int, Mapping[str, bytes]],
) -> bytes:
    """Render ``content`` like ``JSONResponse`` with pre-rendered values.

    ``choice_fragments[i][key]`` is the JSON for ``content["choices"][i][key]``.
    Keys missing from a choice dict are appended (preserving field order of
    trailing optional fields). ``content`` is modified in place.
    """
    if not choice_fragments:
        return _dumps(content)
    token = secrets.token_hex(16)
    fragments: list[bytes] = []
    choices = content["choices"]
    for index, values in choice_fragments.items():
        for key, fragment in values.items():
            choices[index][key] = f"{token}:{len(fragments)}"
            fragments.append(fragment)
    pieces = _dumps(content).split(f'"{token}:'.encode())
    if len(pieces) != len(fragments) + 1:
        raise AssertionError("Fragment placeholder collision")
    out = [pieces[0]]
    for i, piece in enumerate(pieces[1:]):
        marker = f'{i}"'.encode()
        if not piece.startswith(marker):
            raise AssertionError("Fragment placeholder out of order")
        out.append(fragments[i])
        out.append(memoryview(piece)[len(marker) :])
    return b"".join(out)
