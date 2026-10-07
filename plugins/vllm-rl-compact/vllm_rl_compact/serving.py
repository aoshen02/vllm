# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-streaming responses of ``logprobs_format="compact"`` requests: input
processing and the engine call are the core handler's
(``ServingTokens.start_generate``)."""

import asyncio
import functools
import time

from fastapi import Request

from vllm.entrypoints.generate.base.serving import clamp_prompt_logprobs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PromptTokenUsageInfo,
    UsageInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.logprobs_render import (
    render_json_with_fragments,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import (
    ServingTokens,
    build_response_off_loop,
)
from vllm.logger import init_logger
from vllm.logprobs import SampleLogprobsHandle
from vllm.outputs import RequestOutput
from vllm.utils.collection_utils import as_list
from vllm.utils.serial_utils import numpy2base64

from .render import render_compact_logprobs_parts

logger = init_logger(__name__)

# The sample-logprobs container of compact requests (see __init__).
COMPACT_CONTAINER = "rl_compact.wire"


def _entries(final_res: RequestOutput) -> int:
    """Logprob entries (positions x slots) of ``final_res``."""
    total = 0
    for output in final_res.outputs:
        handle = output.logprobs
        if type(handle) is SampleLogprobsHandle and not handle.broken:
            container = handle.unwrap()
            total += len(container) * (container.num_slots or 1)
    return total


class CompactServingTokens:
    """Serves compact requests with the configured core handler (engine
    client, models, renderer, defaults)."""

    def __init__(self, core: ServingTokens) -> None:
        self.core = core

    async def serve_tokens(
        self, request: GenerateRequest, raw_request: Request | None = None
    ) -> ErrorResponse | GenerateResponse | list[bytes]:
        """The response; a body rendered as parts (to be sent unjoined) when
        it has compact logprobs."""
        start = await self.core.start_generate(request, raw_request)
        if isinstance(start, ErrorResponse):
            return start
        created_time = int(time.time())
        final_res: RequestOutput | None = None
        try:
            async for res in start.result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.core.create_error_response("Client disconnected")
        assert final_res is not None

        response, usage = await build_response_off_loop(
            _entries(final_res),
            functools.partial(
                self._build_full_response,
                request,
                final_res,
                start.request_id,
                start.model_name,
                created_time,
            ),
        )
        start.request_metadata.final_usage_info = usage
        core = self.core
        if core.enable_log_outputs and core.request_logger:
            for output in final_res.outputs:
                if output.token_ids:
                    core.request_logger.log_outputs(
                        request_id=start.request_id,
                        outputs="",
                        output_token_ids=output.token_ids,
                        finish_reason=output.finish_reason or "stop",
                        is_streaming=False,
                        delta=False,
                    )
        return response

    def _build_full_response(
        self,
        request: GenerateRequest,
        final_res: RequestOutput,
        request_id: str,
        model_name: str,
        created_time: int,
    ) -> tuple[GenerateResponse | list[bytes], UsageInfo]:
        """The response and its usage. CPU only, no shared-state side effects:
        may run in a worker thread."""
        num_logprobs = request.sampling_params.num_logprobs
        # choice position -> pre-rendered JSON of its compact_logprobs
        fragments: dict[int, list[bytes]] = {}
        choices: list[GenerateResponseChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            if output.finish_reason == "error":
                logger.error(
                    "Request %s failed with an internal error during generation",
                    request_id,
                )
                raise GenerationError("Internal server error")
            # Compact carries whatever sample logprobs the engine returns
            # (``logprobs`` and/or ``logprob_token_ids``).
            if num_logprobs is not None:
                handle = output.logprobs
                if type(handle) is not SampleLogprobsHandle or handle.broken:
                    raise GenerationError(
                        "Compact logprobs encoding failed for this request"
                    )
                fragments[len(choices)] = render_compact_logprobs_parts(
                    handle.unwrap(), num_logprobs, len(output.token_ids)
                )
            choices.append(
                GenerateResponseChoice(
                    index=output.index,
                    logprobs=None,
                    finish_reason=output.finish_reason or "stop",
                    token_ids=as_list(output.token_ids),
                    routed_experts=(
                        numpy2base64(output.routed_experts)
                        if output.routed_experts is not None
                        else None
                    ),
                    sampling_mask=(
                        output.sampling_mask.token_ids
                        if output.sampling_mask is not None
                        else None
                    ),
                )
            )
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
        if not fragments:
            return response, usage
        parts = render_json_with_fragments(
            response.model_dump(), "compact_logprobs", fragments
        )
        return parts, usage
