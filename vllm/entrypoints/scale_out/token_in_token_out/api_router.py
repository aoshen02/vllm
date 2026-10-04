# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import json
from collections.abc import Iterator
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

from .protocol import (
    GenerateRequest,
    GenerateResponse,
    RenderedGenerateResponse,
)
from .serving import ServingTokens

logger = init_logger(__name__)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def generate_tokens(request: Request) -> ServingTokens | None:
    return request.app.state.serving_tokens


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


class _RenderedJSONResponse(JSONResponse):
    """``JSONResponse`` for an already-rendered body given as parts.

    Subclassing JSONResponse keeps ``load_aware_call`` bookkeeping and the
    headers (``content-type``, ``content-length``) identical to the default
    path. The parts are sent as consecutive ``http.response.body`` messages
    (small ones coalesced) instead of one joined body: no full-size copy,
    the server's write flow control applies, and each part is released
    once sent. Bodies up to ``COALESCE_BYTES`` go out as a single message,
    exactly like ``JSONResponse``; larger ones are a stream of messages whose
    last carries ``more_body=False`` (body-rewriting middleware such as
    GZipMiddleware then sees a streaming response). Single-use: the parts are
    consumed by the first send.
    """

    COALESCE_BYTES = 1 << 20

    def __init__(self, rendered: RenderedGenerateResponse) -> None:
        self._parts: list[bytes | memoryview] | None = rendered.parts
        # Same header order as JSONResponse: content-length, content-type.
        super().__init__(
            content=None,
            headers={"content-length": str(rendered.content_length)},
        )

    def render(self, content: Any) -> bytes:
        # The body is sent from ``self._parts`` in ``__call__``.
        return b""

    def _chunks(self) -> Iterator[bytes]:
        if self._parts is None:
            raise RuntimeError("A rendered generate response can only be sent once")
        parts = self._parts
        self._parts = None
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
            yield part if isinstance(part, bytes) else bytes(part)
        if buffer:
            yield bytes(buffer)

    async def __call__(self, scope, receive, send) -> None:
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


@router.post(
    "/inference/v1/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def generate(request: GenerateRequest, raw_request: Request):
    handler = generate_tokens(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support generate tokens API")

    generator = await handler.serve_tokens(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, RenderedGenerateResponse):
        return _RenderedJSONResponse(generator)

    elif isinstance(generator, GenerateResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


def attach_router(app: FastAPI):
    if getattr(app.state.args, "tokens_only", False):

        @router.post("/abort_requests")
        async def abort_requests(raw_request: Request):
            """
            Abort one or more requests. To be used in a
            Disaggregated Everything setup.
            """
            try:
                body = await raw_request.json()
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail=f"JSON decode error: {e}",
                ) from e
            request_ids = body.get("request_ids")
            if request_ids is None:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="Missing 'request_ids' in request body",
                )
            # Abort requests in background
            asyncio.create_task(engine_client(raw_request).abort(request_ids))
            return Response(status_code=200)

    app.include_router(router)
