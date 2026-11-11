# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from typing import Self

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    enable_routed_experts_replay: bool = False
    """Let requests replay recorded routed experts during prefill
    (`SamplingParams.routed_experts_replay`). Requires
    `enable_return_routed_experts`."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Capacity of the row cache that serves prefix-cache and KV offload hits,
    or `None` to derive it from the KV cache capacity. Rows of the blocks it
    can hold are retained after their request finishes; a hit on a block whose
    rows were released is recomputed."""

    @model_validator(mode="after")
    def _check_replay(self) -> Self:
        if self.enable_routed_experts_replay and not self.enable_return_routed_experts:
            raise ValueError(
                "enable_routed_experts_replay requires enable_return_routed_experts."
            )
        return self

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        ignored_factors = {
            "max_bytes",
        }
        return get_compile_factors(self, ignored_factors)
