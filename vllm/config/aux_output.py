# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from pydantic import Field

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or ``None`` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    routed_experts_delta: bool = False
    """Optimize for clients that request each turn's routed experts from where
    the previous turn ended (``routed_experts_prompt_start`` > 0 on later
    turns). Such requests take no prefix-cache hit past the block holding their
    start, so every returned row is computed in the request itself and is
    sliced from the step output without being stored. Only first-turn prompt
    rows (start 0) are kept for prefix hits."""

    mask_trimmed_routed_experts: bool = False
    """For models with a bounded decoder replay (DeepSeek-V4.1), let prompt
    rows outside the replay window skip the replay layers as they do without
    routed-experts capture. Their routed experts in those layers are returned
    as -1 (all bits set) instead of being computed."""

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
            "routed_experts_delta",
        }
        return get_compile_factors(self, ignored_factors)
