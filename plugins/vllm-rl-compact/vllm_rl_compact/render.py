# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``compact_logprobs`` block of a choice."""

from vllm.entrypoints.openai.engine.protocol import GenerationError

from .storage import WireLogprobs


def render_compact_logprobs_parts(
    container: WireLogprobs, num_logprobs: int, num_tokens: int
) -> list[bytes]:
    """The ``compact_logprobs`` JSON object as parts whose concatenation is
    the JSON: the top-k (engine slots ``1..k``) of every position. Rows the
    format cannot represent fail the request (GenerationError, 500)."""
    if not container.is_regular:
        raise GenerationError(
            "Engine logprob rows have inconsistent widths or values beyond "
            "int64; the compact logprobs format cannot represent them"
        )
    if len(container) != num_tokens:
        # E.g. an engine step with tokens but no logprob rows.
        raise GenerationError(
            f"{len(container)} logprob positions for {num_tokens} "
            "generated tokens; the compact logprobs format cannot represent them"
        )
    slots = container.num_slots
    if slots is not None and slots != num_logprobs + 1:
        raise GenerationError(
            f"Engine logprob rows have {slots} slots, expected "
            f"{num_logprobs + 1}; the compact logprobs format cannot "
            "represent them"
        )
    head = (
        f'{{"num_positions":{len(container)},"num_slots":{num_logprobs},'
        '"dtype_token_ids":"int32","dtype_logprobs":"float32",'
        '"byteorder":"little","token_ids":"'
    ).encode("ascii")
    return [
        head,
        *container.token_ids.parts(),
        b'","logprobs":"',
        *container.logprobs.parts(),
        b'"}',
    ]
