# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``compact_logprobs`` block (moved from
``vllm/entrypoints/scale_out/token_in_token_out/logprobs_render.py`` at
214f522fbf)."""

import numpy as np
import pybase64

from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.logprobs import SampleLogprobsHandle

from .storage import ArrayLogprobs

COMPACT_DTYPE_TOKEN_IDS = "int32"
COMPACT_DTYPE_LOGPROBS = "float32"
COMPACT_BYTEORDER = "little"


def unwrap_logprobs(container):
    """The plugin's container behind the core handle (outside shared output
    processing); a handle broken for this request fails it like a broken
    container."""
    if type(container) is SampleLogprobsHandle:
        if container.broken:
            raise GenerationError("Compact logprobs encoding failed for this request")
        return container.unwrap()
    return container


def compact_num_slots(container: ArrayLogprobs, num_logprobs: int | None) -> int:
    """Slots per position: the stored width, else ``k + 1`` (0 for ``k=-1``)."""
    stored = container.num_slots
    if stored is not None:
        return stored
    if num_logprobs is None or num_logprobs < 0:
        return 0
    return num_logprobs + 1


def check_compact_rows(
    container: ArrayLogprobs,
    num_logprobs: int | None,
    expected_positions: int | None = None,
    expected_source_positions: int | None = None,
) -> None:
    """Refuse rows the compact format cannot represent (GenerationError, 500):
    inconsistent widths, or a width other than ``k + 1`` for ``k >= 0``
    (e.g. narrower rows when a co-batched request's ``logprob_token_ids``
    replaced the batch's logprob tensors)."""
    if container.broken:
        raise GenerationError("Compact logprobs encoding failed for this request")
    if not container.is_regular:
        raise GenerationError(
            # (Message unchanged from the in-tree implementation.)
            "Engine logprob rows have inconsistent widths or values beyond "
            "int64; the compact logprobs format cannot represent them"
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
) -> tuple[int, int, bytes, bytes]:
    """Return ``(N, k, b64(top-k token_ids), b64(top-k logprobs))``: engine
    slots ``1..k`` of each row (the sampled token is in the choice's
    ``token_ids``).

    Raises GenerationError (HTTP 500) when the engine rows cannot be
    represented (see :func:`check_compact_rows`).
    """
    container = unwrap_logprobs(container)
    check_compact_rows(
        container, num_logprobs, expected_positions, expected_source_positions
    )
    # C-contiguous little-endian int32 / float32: the wire format.
    token_ids, logprobs, ranks = container.arrays()
    return (
        len(ranks),
        max(compact_num_slots(container, num_logprobs) - 1, 0),
        pybase64.b64encode(np.ascontiguousarray(token_ids[:, 1:])),
        pybase64.b64encode(np.ascontiguousarray(logprobs[:, 1:])),
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
    container = unwrap_logprobs(container)
    check_compact_rows(container, num_logprobs, expected_positions)
    wire = container.wire_parts()
    if wire is not None:
        # Encoded while the rows arrived (ArrayLogprobs wire mode).
        n, stored_slots, token_ids, logprobs = wire
        if stored_slots is None:
            stored_slots = compact_num_slots(container, num_logprobs)
        k = max(stored_slots - 1, 0)
    else:
        n, k, token_ids, logprobs = compact_logprobs_fields(
            container, num_logprobs, expected_positions
        )
    parts = [
        (
            f'{{"num_positions":{n},"num_slots":{k},'
            f'"dtype_token_ids":"{COMPACT_DTYPE_TOKEN_IDS}",'
            f'"dtype_logprobs":"{COMPACT_DTYPE_LOGPROBS}",'
            f'"byteorder":"{COMPACT_BYTEORDER}","token_ids":"'
        ).encode("ascii")
    ]
    for piece, end in ((token_ids, b'","logprobs":"'), (logprobs, b'"}')):
        if isinstance(piece, bytes):
            parts.append(piece)
        else:
            parts.extend(piece)
        parts.append(end)
    return parts


def render_compact_logprobs(
    container: ArrayLogprobs, num_logprobs: int | None
) -> bytes:
    """Render the ``compact_logprobs`` JSON object (see the generate SPEC)."""
    return b"".join(render_compact_logprobs_parts(container, num_logprobs))
