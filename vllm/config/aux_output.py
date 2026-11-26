# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from typing import Any, Self

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or `None` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    object_key_prefix: str | None = None
    """Key prefix of the shared objects of `SamplingParams.aux_namespace`
    requests; a new one per engine start."""

    enable_return_mm_embeds: bool = False
    """Write the multimodal encoder output of each item an `aux_namespace`
    request consumes as an object and list it in the request's
    `aux_objects`. Requires `enable_return_routed_experts` and
    `object_key_prefix`."""

    object_store: str = "mooncake"
    """Store of those objects: `"mooncake"`, or the qualified name of an
    `AuxObjectStore` class."""

    object_store_extra_config: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments of the object store's `create`, e.g.
    `MooncakeAuxObjectStore.create`."""

    @model_validator(mode="after")
    def _check_object_key_prefix(self) -> Self:
        prefix = self.object_key_prefix
        if prefix is not None and not (prefix and "/" not in prefix):
            raise ValueError("object_key_prefix must be non-empty and without '/'.")
        return self

    def verify_mm_embeds(self) -> None:
        """Reject `enable_return_mm_embeds` without what its objects need:
        the auxiliary-output connector and an object key prefix."""
        if self.enable_return_mm_embeds and not (
            self.enable_return_routed_experts and self.object_key_prefix
        ):
            raise ValueError(
                "enable_return_mm_embeds requires enable_return_routed_experts "
                "and object_key_prefix."
            )

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
            "object_key_prefix",
            "enable_return_mm_embeds",
            "object_store",
            "object_store_extra_config",
        }
        return get_compile_factors(self, ignored_factors)
