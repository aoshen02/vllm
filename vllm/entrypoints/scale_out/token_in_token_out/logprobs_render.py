# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object-free rendering of ``/inference/v1/generate`` sample logprobs.

Both renderers read the engine rows kept by :class:`ArrayLogprobs` and
produce JSON bytes directly, without per-entry Python/pydantic objects:

* :func:`render_compact_logprobs` produces the opt-in
  ``logprobs_format="compact"`` block (base64 of the raw engine arrays).

:func:`render_json_with_fragments` splices pre-rendered fragments into the
JSON of the remaining (small) response fields.
"""

import json
import secrets
from collections.abc import Mapping
from typing import Any

import pybase64

from vllm.logprobs import ArrayLogprobs

COMPACT_DTYPE_TOKEN_IDS = "int32"
COMPACT_DTYPE_LOGPROBS = "float32"
COMPACT_BYTEORDER = "little"


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
