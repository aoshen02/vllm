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

    enable_routed_experts_replay: bool = False
    """Let requests replay recorded routed experts during prefill
    (`SamplingParams.routed_experts_replay`). Requires
    `enable_return_routed_experts`."""

    enable_return_indexer_topk: bool = False
    """Capture the sparse-attention indexer's top-k positions of every
    indexer layer. Delivered inline in `CompletionOutput.aux_fields`, or
    through `object_store` for session requests when it is set."""

    enable_return_hidden_states: bool = False
    """Deliver the final hidden states (the input of the LM head) of every
    computed position. Delivered inline in `CompletionOutput.aux_fields`, or
    through `object_store` for session requests when it is set."""

    return_topk_logprobs: int = Field(default=0, ge=0)
    """Deliver the top-k token ids and log-probabilities of the next token at
    every computed position; 0 disables. Delivered inline in
    `CompletionOutput.aux_fields`, or through `object_store` for session
    requests when it is set, where prefix-cache hits keep the rows of the
    request that computed them."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Capacity of the row cache that serves prefix-cache and KV offload hits,
    or `None` to derive it from the KV cache capacity. Rows of the blocks it
    can hold are retained after their request finishes; a hit on a block whose
    rows were released is recomputed."""

    object_store: dict[str, Any] | None = None
    """Deliver the enabled outputs of requests that set
    `SamplingParams.aux_session` through a shared object store: the response
    lists the keys of the objects that cover its rows, one object per KV cache
    block, followed by the rows past the last full block. Keys: `backend`
    (`"mooncake"`, configured by `MOONCAKE_CONFIG_PATH`, or `"local"` for a
    single process) and `key_prefix` (default `"vllm-aux"`), which should be
    unique per job. Routed experts are optional here, so dense models can
    deliver hidden states and top-k logprobs."""

    session_idle_timeout_s: float = Field(default=3600.0, gt=0)
    """Seconds after which the engine forgets a session with no running
    request: its next request hits only objects it publishes again."""

    @property
    def object_key_prefix(self) -> str:
        """Prefix of every shared object key."""
        return (self.object_store or {}).get("key_prefix", "vllm-aux")

    @model_validator(mode="after")
    def _check_object_store(self) -> Self:
        if self.object_store is None:
            return self
        if not (
            self.enable_return_routed_experts
            or self.enable_return_indexer_topk
            or self.enable_return_hidden_states
            or self.return_topk_logprobs
        ):
            raise ValueError("object_store requires at least one output.")
        if self.object_store.get("backend") not in ("mooncake", "local"):
            raise ValueError(
                "object_store backend must be 'mooncake' or 'local', got "
                f"{self.object_store.get('backend')!r}."
            )
        return self

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
        return (
            self.enable_return_routed_experts
            or self.enable_return_indexer_topk
            or self.enable_return_hidden_states
            or bool(self.return_topk_logprobs)
            or self.object_store is not None
        )

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        ignored_factors = {
            "max_bytes",
            "object_store",
            "return_topk_logprobs",
            "session_idle_timeout_s",
        }
        return get_compile_factors(self, ignored_factors)
