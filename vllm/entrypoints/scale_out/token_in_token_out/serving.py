# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import contextvars
import functools
import time
from collections.abc import AsyncGenerator, Callable
from collections.abc import Sequence as GenericSequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import msgspec
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker
from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput, TokensPrompt, mm_input
from vllm.logger import init_logger
from vllm.logprobs import (
    Logprob,
    SampleLogprobsHandle,
    register_sample_logprobs_container,
    set_sample_logprobs_container,
)
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.collection_utils import as_list
from vllm.utils.serial_utils import numpy2base64

from .array_logprobs import ArrayLogprobs
from .logprobs_render import render_json_with_fragments, render_openai_logprobs_parts
from .mm_serde import decode_mm_kwargs_item
from .protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
    RenderedGenerateResponse,
)

logger = init_logger(__name__)

T = TypeVar("T")

# Non-streaming requests with sample logprobs keep them as the engine's rows
# (ArrayLogprobs) instead of a Logprob object per entry, and the response is
# rendered from the rows (same bytes).
ARRAY_LOGPROBS_CONTAINER = "generate.array_logprobs"


def _array_logprobs(params: SamplingParams) -> ArrayLogprobs:
    return ArrayLogprobs()


# Generate responses carry no sampled text (detokenized only for stop strings).
register_sample_logprobs_container(
    ARRAY_LOGPROBS_CONTAINER, _array_logprobs, skip_sampled_text=True
)

# Logprob entries (positions x slots) from which the response is built in a
# worker thread rather than on the event loop (about 0.2 us per entry).
OFFLOAD_MIN_LOGPROB_ENTRIES = 1 << 15
# Large builds run in two threads (more only add memory); smaller ones have
# their own thread, so they never queue behind a multi-second large build.
LARGE_LOGPROB_ENTRIES = 1 << 20
_LARGE_RESPONSE_BUILDER = ThreadPoolExecutor(2, thread_name_prefix="generate-response")
_RESPONSE_BUILDER = ThreadPoolExecutor(1, thread_name_prefix="generate-response-mid")


async def build_response_off_loop(entries: int, build: Callable[[], T]) -> T:
    """Run ``build`` (CPU only, no shared-state side effects) inline when the
    response holds fewer than ``OFFLOAD_MIN_LOGPROB_ENTRIES`` logprob entries,
    else in a worker thread, so that other requests keep being served."""
    if entries < OFFLOAD_MIN_LOGPROB_ENTRIES:
        return build()
    pool = (
        _LARGE_RESPONSE_BUILDER
        if entries >= LARGE_LOGPROB_ENTRIES
        else _RESPONSE_BUILDER
    )
    return await asyncio.get_running_loop().run_in_executor(
        pool, functools.partial(contextvars.copy_context().run, build)
    )


def _array_rows(handle: SampleLogprobsHandle) -> ArrayLogprobs:
    """The rows behind a handle; GenerationError (500) if they failed."""
    rows = None if handle.broken else handle.unwrap()
    if type(rows) is not ArrayLogprobs:
        raise GenerationError("Logprobs encoding failed for this request")
    return rows


def _array_logprob_entries(final_res: RequestOutput) -> int:
    """Logprob entries (positions x slots) held as rows by ``final_res``."""
    total = 0
    for output in final_res.outputs:
        handle = output.logprobs
        if type(handle) is SampleLogprobsHandle and not handle.broken:
            rows = handle.unwrap()
            if isinstance(rows, ArrayLogprobs):
                total += len(rows) * (rows.num_slots or 1)
    return total


@dataclass
class GenerateStart:
    """A prepared generate request (see :meth:`ServingTokens.start_generate`).

    ``result_generator`` is the engine's lazy output stream: the request is
    submitted to the engine when it is first iterated. The caller must consume
    it to the end or close it (``aclose()``), as ``serve_tokens`` does."""

    request_id: str
    model_name: str
    request_metadata: RequestResponseMetadata
    result_generator: AsyncGenerator[RequestOutput, None]


class ServingTokens(GenerateBaseServing):
    """Provides Tokens IN <> Tokens OUT functionality to vLLM API."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        online_renderer: OnlineRenderer,
        *,
        request_logger: RequestLogger | None,
        force_no_detokenize: bool = False,
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_log_outputs: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )
        self.online_renderer = online_renderer
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_log_outputs = enable_log_outputs
        self.force_no_detokenize = force_no_detokenize
        if force_no_detokenize:
            logger.info(
                "Tokens-only mode is enabled, skipping detokenization "
                "step for incoming requests."
            )

        # Mirrors ``OpenAIServingChat`` so we can apply server-side
        # ``max_tokens`` defaulting when the client omits it. Without this,
        # ``SamplingParams.max_tokens`` falls back to its dataclass default
        # of 16 and silently truncates every generation.
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> (
        GenerateResponse
        | RenderedGenerateResponse
        | ErrorResponse
        | AsyncGenerator[str, None]
    ):
        # Here, not in start_generate: its callers get per-entry logprobs.
        sampling_params = request.sampling_params
        if (
            not request.stream
            and sampling_params.logprobs is not None
            # (another component, e.g. an endpoint plugin, chose a container)
            and sampling_params._sample_logprobs_container is None
        ):
            set_sample_logprobs_container(sampling_params, ARRAY_LOGPROBS_CONTAINER)
        start = await self.start_generate(request, raw_request)
        if isinstance(start, ErrorResponse):
            return start
        if request.stream:
            return self.serve_tokens_stream_generator(
                request,
                start.result_generator,
                start.request_id,
                start.model_name,
                start.request_metadata,
            )
        return await self.serve_tokens_full_generator(
            request,
            start.result_generator,
            start.request_id,
            start.model_name,
            start.request_metadata,
        )

    async def start_generate(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateStart | ErrorResponse:
        """Validate and preprocess ``request`` and prepare the engine call,
        like :meth:`serve_tokens`, without building the response: for
        components (e.g. endpoint plugins) that render their own response from
        the ``RequestOutput`` stream. Nothing is submitted to the engine until
        ``GenerateStart.result_generator`` is iterated; when an
        ``ErrorResponse`` is returned, the engine was not called."""
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        lora_request = None
        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        request_id = (
            f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        sampling_params = request.sampling_params
        max_num_seqs = self.engine_client.vllm_config.scheduler_config.max_num_seqs
        if sampling_params.n > max_num_seqs:
            return self.create_error_response(
                f"sampling_params.n must be at most the server's max_num_seqs "
                f"({max_num_seqs}), got {sampling_params.n}."
            )
        try:
            msgspec.msgpack.encode(sampling_params)
        except (OverflowError, TypeError, ValueError) as e:
            return self.create_error_response(e)

        engine_input: EngineInput
        if request.content_parts:
            tracker = AsyncMultiModalItemTracker(self.model_config)
            mm_parser = tracker.create_parser()
            for part in request.content_parts:
                ptype = part.get("type", "")
                url = part.get("url")
                uuid = part.get("uuid")
                if ptype == "image_url":
                    mm_parser.parse_image(url, uuid)
                elif ptype == "audio_url":
                    mm_parser.parse_audio(url, uuid)
                elif ptype == "video_url":
                    mm_parser.parse_video(url, uuid)
            mm_data, mm_uuids = await tracker.resolve_items()
            prompt = TokensPrompt(prompt_token_ids=request.token_ids)
            if mm_data:
                prompt["multi_modal_data"] = mm_data
            if mm_uuids:
                prompt["multi_modal_uuids"] = mm_uuids
            (engine_input,) = await self.online_renderer.renderer.render_cmpl_async(
                [prompt]
            )
        elif features := request.features:
            # Convert PlaceholderRangeInfo → PlaceholderRange per modality.
            mm_placeholders: dict[str, list[PlaceholderRange]] = {
                modality: [
                    PlaceholderRange(offset=p.offset, length=p.length) for p in ranges
                ]
                for modality, ranges in features.mm_placeholders.items()
            }

            # Deserialize tensor data when present; None → cache hit.
            mm_kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
            if features.kwargs_data is not None:
                for modality, items in features.kwargs_data.items():
                    mm_kwargs[modality] = [
                        decode_mm_kwargs_item(item) if item is not None else None
                        for item in items
                    ]
            else:
                for modality, hashes in features.mm_hashes.items():
                    mm_kwargs[modality] = [None] * len(hashes)

            engine_input = mm_input(
                prompt_token_ids=request.token_ids,
                mm_kwargs=MultiModalKwargsItems(mm_kwargs),
                mm_hashes=features.mm_hashes,
                mm_placeholders=mm_placeholders,
                cache_salt=request.cache_salt,
            )
        else:
            (engine_input,) = await self.online_renderer.preprocess_completion(
                request,
                prompt_input=request.token_ids,
                prompt_embeds=None,
                skip_mm_cache=True,
            )

        # The engine's output stream (the request is submitted when it is
        # first iterated).
        result_generator: AsyncGenerator[RequestOutput, None] | None = None

        # Pass disaggregated-serving parameters through to the engine.
        if request.kv_transfer_params is not None:
            extra = sampling_params.extra_args or {}
            extra["kv_transfer_params"] = request.kv_transfer_params
            sampling_params.extra_args = extra
        if request.ec_transfer_params is not None:
            extra = sampling_params.extra_args or {}
            extra["ec_transfer_params"] = request.ec_transfer_params
            sampling_params.extra_args = extra

        # Apply server-side ``max_tokens`` defaulting when the client did
        # not set it, matching the OpenAI-compat endpoints. ``SamplingParams``
        # defaults ``max_tokens`` to 16, which would otherwise silently cap
        # every generation that omits the field.
        if not request.is_sampling_param_provided("max_tokens"):
            sampling_params.max_tokens = get_max_tokens(
                max_model_len=self.model_config.max_model_len,
                max_tokens=None,
                input_length=self._extract_prompt_len(engine_input),
                default_sampling_params=self.default_sampling_params,
                override_max_tokens=self.override_max_tokens,
            )

        if self.force_no_detokenize:
            sampling_params.detokenize = False
        sampling_params.output_kind = (
            RequestOutputKind.DELTA if request.stream else RequestOutputKind.FINAL_ONLY
        )

        self._log_inputs(
            request_id,
            engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )

        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)
        session_id = self._get_session_id_from_headers(raw_request)

        result_generator = self.engine_client.generate(
            engine_input,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=data_parallel_rank,
            session_id=session_id,
        )

        assert result_generator is not None
        return GenerateStart(request_id, model_name, request_metadata, result_generator)

    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse | RenderedGenerateResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        assert final_res is not None

        response, usage, choice_meta = await build_response_off_loop(
            _array_logprob_entries(final_res),
            functools.partial(
                self._build_full_response,
                request,
                final_res,
                request_id,
                model_name,
                created_time,
            ),
        )
        # Shared-state side effects stay on the event loop.
        request_metadata.final_usage_info = usage

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for index, finish_reason in choice_meta:
                # Get the corresponding output token IDs
                output_token_ids = None
                if index < len(final_res.outputs):
                    output_token_ids = final_res.outputs[index].token_ids

                if output_token_ids:
                    # Log token_ids only.
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs="",
                        output_token_ids=output_token_ids,
                        finish_reason=finish_reason,
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
    ) -> tuple[
        GenerateResponse | RenderedGenerateResponse,
        UsageInfo,
        list[tuple[int, str | None]],
    ]:
        """The response, its usage and (index, finish_reason) per choice.
        CPU only, no shared-state side effects: may run in a worker thread."""
        sampling_params: SamplingParams = request.sampling_params
        # choice position -> pre-rendered JSON of its logprobs
        fragments: dict[int, list[bytes]] = {}
        choices: list[GenerateResponseChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            self._raise_if_error(output.finish_reason, request_id)

            token_ids = output.token_ids
            out_logprobs = output.logprobs

            # This is top_logprobs in completions API
            logprobs = None
            if sampling_params.logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                if type(out_logprobs) is SampleLogprobsHandle:
                    rows = _array_rows(out_logprobs)
                    rendered = render_openai_logprobs_parts(
                        token_ids, rows, sampling_params.logprobs
                    )
                    if rendered is not None:
                        fragments[len(choices)] = rendered
                    else:  # irregular rows: the legacy path, in one pass
                        logprobs = self._create_tokens_logprobs(
                            token_ids=token_ids,
                            top_logprobs=list(rows),
                            num_output_top_logprobs=sampling_params.logprobs,
                        )
                else:
                    logprobs = self._create_tokens_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        num_output_top_logprobs=sampling_params.logprobs,
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
                logprobs=logprobs,
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
            self.enable_prompt_tokens_details
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
                response.model_dump(), "logprobs", fragments
            )
            # Joined here (exact bytes: the copy releases the GIL), so a
            # large body is never copied on the event loop.
            return RenderedGenerateResponse(b"".join(parts)), usage, choice_meta
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
                    self._raise_if_error(finish_reason, request_id)

                    if not delta_token_ids:
                        continue

                    if sampling_params.logprobs is not None:
                        out_logprobs = output.logprobs
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_tokens_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=sampling_params.logprobs,
                        )
                    else:
                        logprobs = None

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
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                token_ids=as_list(delta_token_ids),
                                routed_experts=routed_experts_b64,
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

            if self.enable_prompt_tokens_details and num_cached_tokens is not None:
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
            yield (
                f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
            )
        except Exception as e:
            logger.exception("Error in token generation stream.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def _create_tokens_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            token = f"token_id:{token_id}"
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        logprob=max(step_token.logprob, -9999.0),
                        top_logprobs=[
                            ChatCompletionLogProb(
                                token=f"token_id:{token_id}",
                                logprob=max(logprob.logprob, -9999.0),
                            )
                            for i, (token_id, logprob) in enumerate(
                                step_top_logprobs.items()
                            )
                            if num_output_top_logprobs is not None
                            and i < max(num_output_top_logprobs, 1)
                        ],
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)
