# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL compact logprobs for ``/inference/v1/generate`` as a vLLM endpoint plugin
(entry point group ``vllm.endpoint_plugins``, name ``rl_compact``; loaded only
when named in ``VLLM_PLUGINS``): ``logprobs_format="compact"`` returns the top-k
ids and float32 logprobs of each position as base64 arrays."""

from argparse import Namespace

from fastapi import FastAPI
from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.logprobs import register_sample_logprobs_container

from . import router
from .serving import COMPACT_CONTAINER, CompactServingTokens
from .storage import WireLogprobs


def _container(params) -> WireLogprobs:
    return WireLogprobs()


class RLCompactPlugin:
    name = "rl_compact"
    required_tasks = ("generate",)

    def __init__(self) -> None:
        # Containers are created by the OutputProcessor of this (API server)
        # process: registered before any request.
        register_sample_logprobs_container(
            COMPACT_CONTAINER, _container, skip_sampled_text=True
        )

    def attach_router(self, app: FastAPI) -> None:
        router.attach_router(app)

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        core = getattr(state, "serving_tokens", None)
        state.rl_compact_serving = CompactServingTokens(core) if core else None
