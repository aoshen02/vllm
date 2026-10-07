# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /inference/v1/generate`` with ``logprobs_format``: the plugin's route
takes the core route's place (precedence and the single OpenAPI operation are
kept); requests without ``logprobs_format="compact"`` are passed to the core
endpoint unchanged."""

from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import load_aware_call, with_cancellation
from vllm.logprobs import set_sample_logprobs_container

from .protocol import CompactGenerateRequest
from .serving import COMPACT_CONTAINER

GENERATE_PATH = "/inference/v1/generate"


class RenderedJSONResponse(JSONResponse):
    """A JSON body rendered as parts (headers as ``JSONResponse``), sent as
    consecutive messages (small parts coalesced) instead of joined: no copy
    of the whole body, parts released as they are sent."""

    COALESCE_BYTES = 1 << 20

    def __init__(self, parts: list[bytes]) -> None:
        self._parts = parts
        super().__init__(None, headers={"content-length": str(sum(map(len, parts)))})

    def render(self, content: Any) -> bytes:
        return b""

    def _chunks(self) -> Iterator[bytes]:
        parts, self._parts = self._parts[::-1], []  # pop() releases them in order
        buffer = bytearray()
        while parts:
            part = parts.pop()
            if len(part) < self.COALESCE_BYTES:
                buffer += part
                if len(buffer) < self.COALESCE_BYTES:
                    continue
                part, buffer = bytes(buffer), bytearray()
            elif buffer:
                yield bytes(buffer)
                buffer = bytearray()
            yield part
        if buffer:
            yield bytes(buffer)

    async def __call__(self, scope, receive, send) -> None:
        start = {"type": "http.response.start", "status": self.status_code}
        await send({**start, "headers": self.raw_headers})
        # Hold one chunk back so the last one carries more_body=False.
        pending = b""
        for index, chunk in enumerate(self._chunks()):
            if index:
                await send(
                    {"type": "http.response.body", "body": pending, "more_body": True}
                )
            pending = chunk
        await send({"type": "http.response.body", "body": pending, "more_body": False})
        if self.background is not None:
            await self.background()


@with_cancellation
@load_aware_call
async def generate_compact(request: CompactGenerateRequest, raw_request: Request):
    handler = getattr(raw_request.app.state, "rl_compact_serving", None)
    if handler is None:
        raise NotImplementedError("The model does not support generate tokens API")
    set_sample_logprobs_container(request.sampling_params, COMPACT_CONTAINER)
    response = await handler.serve_tokens(request, raw_request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )
    if isinstance(response, list):
        return RenderedJSONResponse(response)
    return JSONResponse(content=response.model_dump())


def attach_router(app: FastAPI) -> None:
    routes = app.router.routes
    index, core = next(
        (i, r)
        for i, r in enumerate(routes)
        if isinstance(r, APIRoute) and r.path == GENERATE_PATH and "POST" in r.methods
    )

    async def generate(request: CompactGenerateRequest, raw_request: Request):
        if request.logprobs_format == "compact":
            return await generate_compact(request, raw_request)
        return await core.endpoint(request, raw_request)

    app.router.add_api_route(
        GENERATE_PATH,
        generate,
        methods=["POST"],
        # add_api_route prepends app.router's own dependencies again.
        dependencies=core.dependencies[len(app.router.dependencies) :],
        response_class=core.response_class,
        name=core.name,
        responses=core.responses,
    )
    routes[index] = routes.pop()
    app.openapi_schema = None
