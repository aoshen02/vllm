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

from . import serving as serving_module
from .logprobs_render import join_parts
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


# Compact + --middleware: bodies up to this size are joined on the event loop
# (a ~1 MiB memcpy is well under a millisecond), larger ones off-loop.
INLINE_JOIN_MAX_BYTES = 1 << 20


class _RenderedJSONResponse(JSONResponse):
    """``JSONResponse`` for an already-rendered body given as parts.

    Subclassing JSONResponse keeps ``load_aware_call`` bookkeeping and the
    headers (``content-type``, ``content-length``) identical to the default
    path. Framing:

    * single message (exactly like ``JSONResponse``): the default format
      always (``RenderedGenerateResponse.single_message``; the parts were
      joined off the event loop), and any response when user
      ``--middleware`` is configured, so body-transforming middleware such
      as GZipMiddleware sees a regular response (Content-Length, size
      thresholds);
    * otherwise (opt-in compact format) consecutive ``http.response.body``
      messages, small parts coalesced, the last with ``more_body=False``: no
      full-size join, write flow control, parts released as sent.

    Single-use: the parts are consumed by the first send; a second send
    raises RuntimeError before anything is sent.
    """

    COALESCE_BYTES = 1 << 20

    def __init__(
        self, rendered: RenderedGenerateResponse, single_message: bool = False
    ) -> None:
        self._parts: list[bytes | memoryview] | None = rendered.parts
        # Join into one message, exactly like JSONResponse, e.g. when user
        # middleware may transform bodies (GZipMiddleware keeps a
        # Content-Length only for single-message bodies).
        self._single_message = single_message
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
        if self._parts is None:
            raise RuntimeError("A rendered generate response can only be sent once")
        if self._single_message:
            parts, self._parts = self._parts, None
            # Already a single part when joined by serving / generate().
            body = (
                parts[0]
                if len(parts) == 1 and type(parts[0]) is bytes
                else (join_parts(parts))
            )
            del parts
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            await send({"type": "http.response.body", "body": body})
            if self.background is not None:
                await self.background()
            return
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
        # User middleware (--middleware) may transform bodies; keep the
        # JSONResponse framing (one message) for exact compatibility then.
        args = getattr(raw_request.app.state, "args", None)
        user_middleware = bool(getattr(args, "middleware", None))
        if user_middleware and not generator.single_message:
            # Compact under user middleware: one message as well. A small
            # body is joined inline (negligible GIL hold), so it never queues
            # behind an unrelated large build in the single response-builder
            # thread; only a large body is joined there.
            if generator.content_length <= INLINE_JOIN_MAX_BYTES:
                body = join_parts(generator.parts)
            else:
                body = await asyncio.get_running_loop().run_in_executor(
                    serving_module._RESPONSE_BUILDER, join_parts, generator.parts
                )
            generator = RenderedGenerateResponse([body], single_message=True)
        return _RenderedJSONResponse(generator, single_message=generator.single_message)

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
