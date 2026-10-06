# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``POST /inference/v1/generate`` with the compact switches.

Replaces the core route with ``replace_route`` (in place, so precedence
relative to other routes is kept, and OpenAPI shows one operation with the
plugin's request schema): requests without ``logprobs_format="compact"`` are
passed to the replaced core endpoint unchanged (after validation with this
plugin's request model, and the core route's dependencies, which the new route
inherits); compact requests go to
:class:`~vllm_rl_compact.serving.CompactServingTokens`.
"""

from collections.abc import Callable, Iterator
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateResponse,
    RenderedGenerateResponse,
)
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logprobs import set_sample_logprobs_container
from vllm.plugins.endpoint_plugins.routing import replace_route

from .protocol import (
    CompactGenerateRequest,
    CompactGenerateResponse,
    RenderedCompactResponse,
)
from .serving import compact_container

GENERATE_PATH = "/inference/v1/generate"


class RenderedJSONResponse(JSONResponse):
    """A pre-rendered JSON body (``JSONResponse`` subclass, so
    ``load_aware_call`` bookkeeping and the headers are those of the core
    route). A joined body is one message; parts are consecutive
    ``http.response.body`` messages (small parts coalesced, the last with
    ``more_body=False``): no full-size join, parts released as sent.
    Single-use."""

    COALESCE_BYTES = 1 << 20

    def __init__(
        self, rendered: RenderedCompactResponse | RenderedGenerateResponse
    ) -> None:
        self._single_message = isinstance(rendered, RenderedGenerateResponse)
        parts = [rendered.body] if self._single_message else rendered.parts
        self._parts: list[bytes] | None = parts
        super().__init__(
            content=None,
            headers={"content-length": str(sum(len(p) for p in parts))},
        )

    def render(self, content: Any) -> bytes:
        return b""

    def _chunks(self) -> Iterator[bytes]:
        if self._parts is None:
            raise RuntimeError("A rendered generate response can only be sent once")
        parts = self._parts
        self._parts = None
        if self._single_message:
            yield parts[0]
            return
        parts.reverse()  # pop() from the end releases parts in order
        buffer = bytearray()
        while parts:
            part = parts.pop()
            if len(part) < self.COALESCE_BYTES:
                buffer += part
                if len(buffer) < self.COALESCE_BYTES:
                    continue
                part, buffer = buffer, bytearray()
            elif buffer:
                yield bytes(buffer)
                buffer = bytearray()
            yield bytes(part)
        if buffer:
            yield bytes(buffer)

    async def __call__(self, scope, receive, send) -> None:
        if self._parts is None:
            raise RuntimeError("A rendered generate response can only be sent once")
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
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
    set_sample_logprobs_container(request.sampling_params, compact_container(request))
    generator = await handler.serve_tokens(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    if isinstance(generator, (RenderedCompactResponse, RenderedGenerateResponse)):
        return RenderedJSONResponse(generator)
    if isinstance(generator, GenerateResponse):
        return JSONResponse(content=generator.model_dump())
    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI) -> None:
    core_generate: list[Callable[..., Any]] = []

    async def generate(request: CompactGenerateRequest, raw_request: Request):
        if request.logprobs_format == "compact":
            return await generate_compact(request, raw_request)
        # The core endpoint, unchanged (its own cancellation / load tracking).
        return await core_generate[0](request, raw_request)

    core_generate.append(
        replace_route(
            app,
            GENERATE_PATH,
            generate,
            methods=["POST"],
            responses={
                HTTPStatus.OK.value: {
                    "model": CompactGenerateResponse,
                    "content": {"text/event-stream": {}},
                },
                HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
                HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
                HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            },
        )
    )
