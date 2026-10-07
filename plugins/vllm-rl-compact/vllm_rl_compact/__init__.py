# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL compact logprobs for ``/inference/v1/generate`` as a vLLM endpoint plugin.

Registered under the ``vllm.endpoint_plugins`` entry point group as
``rl_compact``; loaded only when named in ``VLLM_PLUGINS``. Adds the opt-in
``logprobs_format="compact"`` request switch: the top-k ids and float32
logprobs of each position as base64 arrays (see the README for the core APIs
it uses).
"""

from argparse import Namespace

from fastapi import FastAPI
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.logprobs import register_sample_logprobs_container

from . import router
from .serving import ARRAY_CONTAINER, WIRE_CONTAINER, CompactServingTokens
from .storage import ArrayLogprobs

__all__ = ["ArrayLogprobs", "RLCompactPlugin", "register_containers"]


def _array(params) -> ArrayLogprobs:
    return ArrayLogprobs()


def _wire(params) -> ArrayLogprobs:
    return ArrayLogprobs(wire_base64=True)


def register_containers() -> None:
    """Idempotent (the same factories may be registered again)."""
    # Generate responses carry no sampled text (detokenized for stop strings).
    for name, factory in (
        (ARRAY_CONTAINER, _array),
        (WIRE_CONTAINER, _wire),
    ):
        register_sample_logprobs_container(name, factory, skip_sampled_text=True)


class RLCompactPlugin:
    name = "rl_compact"
    required_tasks = ("generate",)

    def __init__(self) -> None:
        # Containers are created by the OutputProcessor of this (API server)
        # process; register them here, before any request.
        register_containers()

    def attach_router(self, app: FastAPI) -> None:
        router.attach_router(app)

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        core = getattr(state, "serving_tokens", None)
        middleware = getattr(args, "middleware", None)
        state.rl_compact_serving = (
            CompactServingTokens.from_core(core, user_middleware=bool(middleware))
            if core is not None
            else None
        )
