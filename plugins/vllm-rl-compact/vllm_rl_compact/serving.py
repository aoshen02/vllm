# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses of ``logprobs_format="compact"`` requests.

Input processing and the engine call are the core handler's
(``ServingTokens.start_generate``); the response build and the streaming loop
are the in-tree implementation of 214f522fbf, restricted to compact.
"""

import asyncio
import functools
import time
from collections.abc import AsyncGenerator

from fastapi import Request

from vllm.entrypoints.generate.base.serving import clamp_prompt_logprobs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.logprobs_render import (
    render_json_with_fragments,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    RenderedGenerateResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import (
    ServingTokens,
    build_response_off_loop,
)
from vllm.entrypoints.serve.utils.api_utils import should_include_usage
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.collection_utils import as_list
from vllm.utils.serial_utils import numpy2base64

from .protocol import CompactLogprobs, RenderedCompactResponse
from .protocol import CompactStreamChoice as GenerateResponseStreamChoice
from .protocol import CompactStreamResponse as GenerateStreamResponse
from .render import (
    check_compact_rows,
    compact_logprobs_fields,
    render_compact_logprobs_parts,
    unwrap_logprobs,
)
from .storage import ArrayLogprobs

logger = init_logger(__name__)

ARRAY_CONTAINER = "rl_compact.array"
WIRE_CONTAINER = "rl_compact.wire"


def compact_container(request) -> str:
    """The sample-logprobs container of a compact request."""
    if request.stream:
        # DELTA slices of array rows, one compact block per chunk.
        return ARRAY_CONTAINER
    # Encode rows to the wire format while they arrive, so the post-abort
    # response build only stitches segments.
    return WIRE_CONTAINER


def _raise_if_error(finish_reason: str | None, request_id: str) -> None:
    """As the core handler: a request that failed in the engine."""
    if finish_reason == "error":
        logger.error(
            "Request %s failed with an internal error during generation",
            request_id,
        )
        raise GenerationError("Internal server error")


class CompactServingTokens:
    """Serves compact requests with the configured core handler (engine
    client, models, renderer, defaults): its ``start_generate``, then the
    compact response."""

    def __init__(self, core: ServingTokens, user_middleware: bool = False) -> None:
        self.core = core
        # User --middleware may transform bodies: JSONResponse framing then.
        self.user_middleware = user_middleware

    @classmethod
    def from_core(
        cls, core: ServingTokens, user_middleware: bool = False
    ) -> "CompactServingTokens":
        return cls(core, user_middleware)

    async def serve_tokens(
        self, request: GenerateRequest, raw_request: Request | None = None
    ) -> (
        ErrorResponse
        | GenerateResponse
        | RenderedCompactResponse
        | RenderedGenerateResponse
        | AsyncGenerator[str, None]
    ):
        start = await self.core.start_generate(request, raw_request)
        if isinstance(start, ErrorResponse):
            return start
        args = (
            start.result_generator,
            start.request_id,
            start.model_name,
            start.request_metadata,
        )
        if request.stream:
            return self.serve_tokens_stream_generator(request, *args)
        return await self.serve_tokens_full_generator(request, *args)

    @staticmethod
    def _require_array_logprobs(logprobs: object) -> ArrayLogprobs:
        logprobs = unwrap_logprobs(logprobs)  # core handle -> ArrayLogprobs
        if not isinstance(logprobs, ArrayLogprobs):
            raise TypeError(
                "logprobs_format='compact' requires ArrayLogprobs sample "
                f"logprobs, got {type(logprobs).__name__}"
            )
        return logprobs

    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> (
        ErrorResponse
        | GenerateResponse
        | RenderedCompactResponse
        | RenderedGenerateResponse
    ):
        join_compact = self.user_middleware
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.core.create_error_response("Client disconnected")

        assert final_res is not None

        # As in the core: large bodies are built in a worker thread.
        response, usage, choice_meta = await build_response_off_loop(
            self._num_logprob_entries(request, final_res),
            functools.partial(
                self._build_full_response,
                request,
                final_res,
                request_id,
                model_name,
                created_time,
                join_compact=join_compact,
            ),
        )
        # Shared-state side effects stay on the event loop.
        request_metadata.final_usage_info = usage
        self._log_full_response(request_id, final_res, choice_meta)
        return response

    def _log_full_response(
        self,
        request_id: str,
        final_res: RequestOutput,
        choice_meta: list[tuple[int, str | None]],
    ) -> None:
        # Log complete response if output logging is enabled
        core = self.core
        if core.enable_log_outputs and core.request_logger:
            for index, finish_reason in choice_meta:
                # Get the corresponding output token IDs
                output_token_ids = None
                if index < len(final_res.outputs):
                    output_token_ids = final_res.outputs[index].token_ids

                if output_token_ids:
                    # Log token_ids only.
                    core.request_logger.log_outputs(
                        request_id=request_id,
                        outputs="",
                        output_token_ids=output_token_ids,
                        finish_reason=finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

    @staticmethod
    def _num_logprob_entries(request: GenerateRequest, final_res: RequestOutput) -> int:
        num_logprobs = request.sampling_params.num_logprobs
        if num_logprobs is None:
            return 0
        total = 0
        for output in final_res.outputs:
            try:
                logprobs = unwrap_logprobs(output.logprobs)
            except GenerationError:
                return 0  # broken for this request: fails fast at render
            if not isinstance(logprobs, ArrayLogprobs):
                return 0  # legacy containers: unchanged (inline) behavior
            total += len(logprobs) * (logprobs.num_slots or 1)
        return total

    def _build_full_response(
        self,
        request: GenerateRequest,
        final_res: RequestOutput,
        request_id: str,
        model_name: str,
        created_time: int,
        join_compact: bool = False,
    ) -> tuple[
        GenerateResponse | RenderedCompactResponse | RenderedGenerateResponse,
        UsageInfo,
        list[tuple[int, str | None]],
    ]:
        """Build the final response, its usage and (index, finish_reason) per
        choice. CPU only, no shared-state side effects: may run in a worker
        thread. ``join_compact``: also join a compact body into one message
        (user middleware configured)."""
        sampling_params: SamplingParams = request.sampling_params
        # choice position -> pre-rendered JSON of its compact_logprobs
        fragments: dict[int, list[bytes]] = {}

        choices: list[GenerateResponseChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            _raise_if_error(output.finish_reason, request_id)

            token_ids = output.token_ids
            out_logprobs = output.logprobs

            # Compact carries whatever sample logprobs the engine returns
            # (``logprobs`` and/or ``logprob_token_ids``).
            if sampling_params.num_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                fragments[len(choices)] = render_compact_logprobs_parts(
                    self._require_array_logprobs(out_logprobs),
                    sampling_params.num_logprobs,
                    expected_positions=len(token_ids),
                )
            routed_experts_b64 = (
                numpy2base64(output.routed_experts)
                if output.routed_experts is not None
                else None
            )

            sampling_mask = None
            if output.sampling_mask is not None:
                sampling_mask = output.sampling_mask.token_ids

            choice_data = GenerateResponseChoice(
                index=output.index,
                logprobs=None,
                finish_reason=output.finish_reason if output.finish_reason else "stop",
                token_ids=as_list(output.token_ids),
                routed_experts=routed_experts_b64,
                sampling_mask=sampling_mask,
            )

            choices.append(choice_data)
            num_generated_tokens += len(output.token_ids)

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if (
            self.core.enable_prompt_tokens_details
            and final_res.num_cached_tokens is not None
        ):
            # This info is not available at the /coordinator level
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        response = GenerateResponse(
            request_id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            kv_transfer_params=final_res.kv_transfer_params,
            ec_transfer_params=final_res.ec_transfer_params,
        )

        choice_meta = [(choice.index, choice.finish_reason) for choice in choices]
        if fragments:
            parts = render_json_with_fragments(
                response.model_dump(), "compact_logprobs", fragments
            )
            if join_compact:
                # One message like JSONResponse (compatibility with
                # body-transforming user middleware). Joined here, where the
                # build ran: inline for small builds, in the same
                # worker-thread job for large ones (no second queue trip).
                body = b"".join(parts)
                del parts
                return RenderedGenerateResponse(body), usage, choice_meta
            return RenderedCompactResponse(parts), usage, choice_meta
        return response, usage, choice_meta

    async def serve_tokens_stream_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_prompt_tokens = 0
        num_generated_tokens: list[int] = []
        first_iteration = True
        num_cached_tokens = None
        sampling_params: SamplingParams = request.sampling_params

        include_usage, include_continuous_usage = should_include_usage(
            request.stream_options, False
        )

        try:
            async for res in result_generator:
                if first_iteration:
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)
                    num_cached_tokens = res.num_cached_tokens
                    num_generated_tokens = [0] * len(res.outputs)
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    delta_token_ids = output.token_ids
                    num_generated_tokens[i] += len(delta_token_ids)

                    finish_reason = output.finish_reason
                    _raise_if_error(finish_reason, request_id)
                    if not delta_token_ids:
                        # A storage failure may come without new tokens (e.g.
                        # a failed slice for a terminal empty delta): check
                        # the storage before such an output is skipped.
                        if sampling_params.num_logprobs is not None:
                            check_compact_rows(
                                self._require_array_logprobs(output.logprobs),
                                sampling_params.num_logprobs,
                                expected_positions=0,
                                expected_source_positions=num_generated_tokens[i],
                            )
                        continue

                    compact_logprobs = None
                    if sampling_params.num_logprobs is not None:
                        out_logprobs = output.logprobs
                        assert out_logprobs is not None, "Did not output logprobs"
                        compact_logprobs = self._compact_logprobs_model(
                            self._require_array_logprobs(out_logprobs),
                            sampling_params.num_logprobs,
                            expected_positions=len(delta_token_ids),
                            expected_source_positions=num_generated_tokens[i],
                        )

                    routed_experts_b64 = (
                        numpy2base64(output.routed_experts)
                        if output.routed_experts is not None
                        else None
                    )

                    chunk = GenerateStreamResponse(
                        request_id=request_id,
                        choices=[
                            GenerateResponseStreamChoice(
                                index=i,
                                logprobs=None,
                                finish_reason=finish_reason,
                                token_ids=as_list(delta_token_ids),
                                routed_experts=routed_experts_b64,
                                compact_logprobs=compact_logprobs,
                            )
                        ],
                    )
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=num_generated_tokens[i],
                            total_tokens=(num_prompt_tokens + num_generated_tokens[i]),
                        )

                    yield f"data: {chunk.model_dump_json()}\n\n"

            total_completion_tokens = sum(num_generated_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=num_prompt_tokens + total_completion_tokens,
            )

            if self.core.enable_prompt_tokens_details and num_cached_tokens is not None:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )

            if include_usage:
                final_chunk = GenerateStreamResponse(
                    request_id=request_id,
                    choices=[],
                    usage=final_usage_info,
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

            request_metadata.final_usage_info = final_usage_info

        except GenerationError as e:
            data = self.core.create_streaming_error_response(
                str(e), err_type="InternalServerError", status_code=e.status_code
            )
            yield f"data: {data}\n\n"
        except Exception as e:
            logger.exception("Error in token generation stream.")
            data = self.core.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _compact_logprobs_model(
        logprobs: ArrayLogprobs,
        num_logprobs: int | None,
        expected_positions: int | None = None,
        expected_source_positions: int | None = None,
    ) -> CompactLogprobs:
        n, k, token_ids, values = compact_logprobs_fields(
            logprobs, num_logprobs, expected_positions, expected_source_positions
        )
        return CompactLogprobs(
            num_positions=n,
            num_slots=k,
            token_ids=token_ids.decode("ascii"),
            logprobs=values.decode("ascii"),
        )
