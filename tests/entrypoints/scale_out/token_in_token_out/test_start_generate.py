# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ServingTokens.start_generate`: the request scheduled on the engine, for
components that build their own response."""

from unittest.mock import MagicMock

import pytest

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _make_request_output,
    _mock_engine,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.scale_out.token_in_token_out.serving import GenerateStart
from vllm.logprobs import SampleLogprobsHandle, create_sample_logprobs
from vllm.sampling_params import RequestOutputKind, SamplingParams


def _serving():
    engine = _mock_engine()

    async def mock_generate(*args, **kwargs):
        yield _make_request_output(
            "req-1", token_ids=[10], finish_reason="stop", finished=True
        )

    engine.generate = MagicMock(side_effect=mock_generate)
    return engine, _build_serving_tokens(engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_start_generate_schedules_without_building_a_response(stream):
    engine, serving = _serving()
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        model=MODEL_NAME,
        stream=stream,
        request_id="fixed",
    )
    raw_request = MagicMock()
    raw_request.headers = {}
    start = await serving.start_generate(request, raw_request)
    assert isinstance(start, GenerateStart)
    assert start.request_id == "generate-tokens-fixed"
    assert start.model_name == MODEL_NAME
    assert raw_request.state.request_metadata is start.request_metadata
    assert engine.generate.call_args.args[2] == start.request_id
    params = engine.generate.call_args.args[1]
    assert params.output_kind == (
        RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY
    )
    outputs = [out async for out in start.result_generator]
    assert [out.outputs[0].token_ids for out in outputs] == [[10]]


@pytest.mark.asyncio
async def test_start_generate_returns_validation_errors():
    engine, serving = _serving()
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, n=129),  # > max_num_seqs
        model=MODEL_NAME,
    )
    assert isinstance(await serving.start_generate(request), ErrorResponse)
    assert isinstance(await serving.serve_tokens(request), ErrorResponse)
    engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_start_generate_keeps_per_entry_logprobs():
    """The core's array storage is chosen by serve_tokens, so
    callers of start_generate (e.g. a plugin rendering its own response) get
    the iterable per-entry logprobs, as before."""
    engine, serving = _serving()
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, logprobs=3),
        model=MODEL_NAME,
    )
    start = await serving.start_generate(request)
    assert isinstance(start, GenerateStart)
    params = engine.generate.call_args.args[1]
    assert params._sample_logprobs_container is None
    logprobs = create_sample_logprobs(False, params)
    assert not isinstance(logprobs, SampleLogprobsHandle) and list(logprobs) == []
    [out async for out in start.result_generator]
