# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request/response models of the RL compact format (moved from
``vllm/entrypoints/scale_out/token_in_token_out/protocol.py``)."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
)

__all__ = [
    "CompactGenerateRequest",
    "CompactGenerateResponse",
    "CompactLogprobs",
    "CompactStreamChoice",
    "CompactStreamResponse",
    "RenderedCompactResponse",
]


class CompactGenerateRequest(GenerateRequest):
    """``GenerateRequest`` plus the compact-format switches. Without them it
    is handled by the core ``ServingTokens`` exactly like a core request."""

    logprobs_format: Literal["openai", "compact"] = Field(
        default="openai",
        # Opt-in: omitted when default, so serialized requests are unchanged.
        exclude_if=lambda value: value == "openai",
        description=(
            "Wire format of the sample logprobs. 'openai' (default): "
            "`choices[].logprobs` as today. 'compact': `choices[].logprobs` is "
            "null and `choices[].compact_logprobs` carries the top-k of each "
            "position (engine order) as base64 of little-endian int32 ids and "
            "float32 logprobs; the sampled tokens are `choices[].token_ids`. "
            "Streaming emits one compact block per chunk covering that chunk's "
            "positions."
        ),
    )

    @model_validator(mode="after")
    def _check_compact_logprobs(self) -> "CompactGenerateRequest":
        # logprobs=-1 returns full-vocabulary values without per-slot token ids
        # and ranks, which the compact [N, k+1] layout cannot represent.
        if self.logprobs_format == "compact" and self.sampling_params.logprobs == -1:
            raise ValueError(
                "logprobs_format='compact' does not support sampling_params.logprobs=-1"
            )
        return self


class CompactLogprobs(BaseModel):
    """Sample logprobs in the opt-in ``logprobs_format="compact"`` format:
    the top-k of each position (``num_slots`` = k; the sampled tokens are the
    choice's ``token_ids``).

    Decode: ``np.frombuffer(base64.b64decode(token_ids), "<i4")
    .reshape(num_positions, num_slots)``; ``logprobs`` likewise with
    ``"<f4"``. Values are the raw engine float32 (no clamping; non-finite
    values preserved).
    """

    num_positions: int
    num_slots: int
    dtype_token_ids: Literal["int32"] = "int32"
    dtype_logprobs: Literal["float32"] = "float32"
    byteorder: Literal["little"] = "little"
    token_ids: str
    logprobs: str


class CompactStreamChoice(GenerateResponseStreamChoice):
    # Only present (non-null) for logprobs_format="compact".
    compact_logprobs: CompactLogprobs | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CompactStreamResponse(GenerateStreamResponse):
    # Declared as the subclass so pydantic serializes compact_logprobs.
    choices: list[CompactStreamChoice]


@dataclass
class RenderedCompactResponse:
    """A non-streaming compact response rendered to JSON as ``parts`` that
    concatenate to the ``application/json`` body; the router sends them one
    by one (with a Content-Length) instead of joining them. (Joined bodies
    are the core's ``RenderedGenerateResponse``.)"""

    parts: list[bytes]

    @property
    def body(self) -> bytes:
        """The joined body (copies; for tests and small responses)."""
        return b"".join(self.parts)


class CompactResponseChoice(GenerateResponseChoice):
    # Only present (non-null) for logprobs_format="compact".
    compact_logprobs: CompactLogprobs | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CompactGenerateResponse(GenerateResponse):
    """Response schema (OpenAPI) of the plugin route; full compact responses
    are rendered without model objects."""

    choices: list[CompactResponseChoice]
