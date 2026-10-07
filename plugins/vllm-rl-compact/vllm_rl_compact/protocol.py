# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The request model of the plugin route: ``GenerateRequest`` plus
``logprobs_format``."""

from typing import Literal

from pydantic import Field, model_validator

from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest


class CompactGenerateRequest(GenerateRequest):
    """``GenerateRequest`` plus ``logprobs_format``. Without it the request
    is handled by the core ``ServingTokens`` exactly like a core request."""

    logprobs_format: Literal["openai", "compact"] = Field(
        default="openai",
        # Opt-in: omitted when default, so serialized requests are unchanged.
        exclude_if=lambda value: value == "openai",
        description=(
            "Wire format of the sample logprobs. 'openai' (default): "
            "`choices[].logprobs` as today. 'compact' (non-streaming): "
            "`choices[].logprobs` is null and `choices[].compact_logprobs` "
            "carries the top-k of each position (engine order) as base64 of "
            "little-endian int32 ids and float32 logprobs; the sampled tokens "
            "are `choices[].token_ids`."
        ),
    )

    @model_validator(mode="after")
    def _check_compact_logprobs(self) -> "CompactGenerateRequest":
        if self.logprobs_format != "compact":
            return self
        if self.stream:
            raise ValueError("logprobs_format='compact' does not support stream=true")
        # Full-vocabulary values have no per-slot token ids.
        if self.sampling_params.logprobs == -1:
            raise ValueError(
                "logprobs_format='compact' does not support sampling_params.logprobs=-1"
            )
        return self
