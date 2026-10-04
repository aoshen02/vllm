# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``logprobs_format`` option of ``/inference/v1/generate``."""

import json
import time
from argparse import Namespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pybase64 as base64
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm.entrypoints.scale_out.token_in_token_out import api_router
from vllm.entrypoints.scale_out.token_in_token_out.logprobs_render import (
    render_compact_logprobs,
    render_json_with_fragments,
    render_openai_logprobs,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    CompactLogprobs,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
    RenderedGenerateResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.logprobs import ArrayLogprobs, append_logprobs_for_next_position
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.outputs import LogprobsLists

from .test_generate_stream import (
    _build_serving_tokens,
    _mock_engine,
    _parse_sse_chunks,
)


def _engine_rows(start: int, count: int, width: int, seed: int = 0):
    """Engine-like rows: slot 0 = sampled, slots 1.. = distinct top ids.

    Even positions repeat the sampled id inside the top-k (like a real
    sampled token), odd positions do not.
    """
    rng = np.random.default_rng(seed + start)
    top = np.stack(
        [rng.choice(50_000, size=width - 1, replace=False) for _ in range(count)]
    ).astype(np.int64)
    sampled = np.where(
        np.arange(start, start + count) % 2 == 0,
        top[:, min(1, width - 2)],
        50_000 + np.arange(start, start + count),
    )
    token_ids = np.column_stack((sampled, top))
    logprobs = -rng.random((count, width), dtype=np.float32) * 20
    ranks = rng.integers(1, 1000, size=count).astype(np.int64)
    return token_ids, logprobs.astype(np.float32), ranks


class _OutputProcessorEngine:
    """Feeds engine rows through the real OutputProcessor."""

    def __init__(self, chunks, finish: FinishReason | None = FinishReason.ABORT):
        self.chunks = chunks
        self.finish = finish
        self.sampling_params: Any = None

    def generate(self, engine_input, sampling_params, request_id, **kwargs):
        self.sampling_params = sampling_params

        async def _gen():
            processor = OutputProcessor(tokenizer=None, log_stats=False)
            request = EngineCoreRequest(
                request_id=request_id + "-int",
                external_req_id=request_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=None,
                arrival_time=0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                sampling_params=sampling_params,
                pooling_params=None,
            )
            queue = RequestOutputCollector(sampling_params.output_kind, request_id)
            processor.add_request(request, None, queue=queue)
            for token_ids, logprobs, ranks in self.chunks:
                processor.process_outputs(
                    [
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=token_ids[:, 0].tolist(),
                            new_logprobs=(
                                None
                                if sampling_params.num_logprobs is None
                                else LogprobsLists(token_ids, logprobs, ranks)
                            ),
                        )
                    ]
                )
                if (out := queue.get_nowait()) is not None:
                    yield out
            if self.finish == FinishReason.ABORT:
                processor.abort_requests([request.request_id], internal=True)
            else:
                processor.process_outputs(
                    [
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=self.finish,
                        )
                    ]
                )
            if (out := queue.get_nowait()) is not None:
                yield out

        return _gen()


def _serving(feeder: _OutputProcessorEngine):
    engine = _mock_engine()
    engine.generate = MagicMock(side_effect=feeder.generate)
    return _build_serving_tokens(engine)


def _request(logprobs: int | None = 3, **kwargs) -> GenerateRequest:
    sampling = {"max_tokens": 100, "logprobs": logprobs}
    sampling.update(kwargs.pop("sampling", {}))
    return GenerateRequest.model_validate(
        {"token_ids": [1, 2, 3], "sampling_params": sampling, **kwargs}
    )


def _decode(block: dict):
    n, s = block["num_positions"], block["num_slots"]
    token_ids = np.frombuffer(base64.b64decode(block["token_ids"]), "<i4")
    logprobs = np.frombuffer(base64.b64decode(block["logprobs"]), "<f4")
    ranks = np.frombuffer(base64.b64decode(block["ranks"]), "<i4")
    return token_ids.reshape(n, s), logprobs.reshape(n, s), ranks


def _expected(chunks, num_slots):
    token_ids = np.concatenate([c[0] for c in chunks])[:, :num_slots]
    logprobs = np.concatenate([c[1] for c in chunks])[:, :num_slots]
    ranks = np.concatenate([c[2] for c in chunks])
    return token_ids.astype("<i4"), logprobs.astype("<f4"), ranks.astype("<i4")


# ---------------------------------------------------------------- protocol


def test_logprobs_format_defaults_and_rejects_unknown():
    assert _request().logprobs_format == "openai"
    assert _request(logprobs_format="compact").logprobs_format == "compact"
    with pytest.raises(ValidationError):
        _request(logprobs_format="msgpack")


def test_unknown_logprobs_format_is_http_400():
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False, tokens_only=False)
    app.state.serving_tokens = MagicMock()
    init_exception_handler(app)
    api_router.attach_router(app)
    with TestClient(app) as client:
        result = client.post(
            "/inference/v1/generate",
            json={
                "token_ids": [1],
                "sampling_params": {"logprobs": 1},
                "logprobs_format": "nope",
            },
        )
    assert result.status_code == 400
    assert "logprobs_format" in result.text
    app.state.serving_tokens.serve_tokens.assert_not_called()


def test_default_choice_serialization_has_no_compact_key():
    choice = GenerateResponseChoice(index=0, token_ids=[1])
    assert "compact_logprobs" not in choice.model_dump()
    assert "compact_logprobs" not in json.loads(choice.model_dump_json())
    stream_choice = GenerateResponseStreamChoice(index=0, token_ids=[1])
    assert "compact_logprobs" not in json.loads(stream_choice.model_dump_json())
    assert (
        "compact_logprobs"
        not in GenerateStreamResponse(choices=[stream_choice]).model_dump(
            exclude_none=True
        )["choices"][0]
    )


def test_render_without_fragments_matches_json_response():
    response = GenerateResponse(
        request_id="中文",
        choices=[GenerateResponseChoice(index=0, token_ids=[1, 2])],
        kv_transfer_params={"x": [1.5, "é"]},
    )
    content = response.model_dump()
    assert render_json_with_fragments(content, {}) == JSONResponse(content).body


def test_render_with_fragments_is_valid_and_ordered():
    response = GenerateResponse(
        request_id="r",
        choices=[
            GenerateResponseChoice(index=0, token_ids=[1]),
            GenerateResponseChoice(index=1, token_ids=[2]),
        ],
    )
    body = render_json_with_fragments(
        response.model_dump(),
        {0: {"logprobs": b'{"content":[]}'}, 1: {"compact_logprobs": b"[7]"}},
    )
    data = json.loads(body)
    assert data["choices"][0]["logprobs"] == {"content": []}
    assert data["choices"][1]["logprobs"] is None
    assert data["choices"][1]["compact_logprobs"] == [7]
    assert list(data["choices"][1])[-1] == "compact_logprobs"


def test_compact_model_and_fast_render_agree():
    container = ArrayLogprobs()
    container.append_rows(*_engine_rows(0, 5, 4))
    fast = render_compact_logprobs(container, 3)
    data = json.loads(fast)
    model = CompactLogprobs(**data)
    assert model.model_dump_json().encode() == fast
    assert list(data) == [
        "num_positions",
        "num_slots",
        "dtype_token_ids",
        "dtype_logprobs",
        "byteorder",
        "token_ids",
        "logprobs",
        "ranks",
    ]


# -------------------------------------------------------------- container


def test_array_logprobs_sequence_semantics_match_list_path():
    token_ids, logprobs, ranks = _engine_rows(0, 6, 5)
    container = ArrayLogprobs()
    container.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    container.append_rows(token_ids[2:], logprobs[2:], ranks[2:])
    legacy: list = []
    for i in range(6):
        append_logprobs_for_next_position(
            legacy,
            token_ids[i].tolist(),
            logprobs[i].tolist(),
            [None] * 5,
            int(ranks[i]),
            4,
        )
    assert len(container) == 6
    assert list(container) == legacy
    assert container[-1] == legacy[-1]
    tail = container[-4:]
    assert isinstance(tail, ArrayLogprobs) and list(tail) == legacy[-4:]
    assert len(container[:0]) == 0
    merged = ArrayLogprobs()
    merged.extend(container[:3])
    merged.extend(container[3:])
    assert list(merged) == legacy
    # arrays() consolidates and keeps exact engine values.
    ids, values, rk = container.arrays()
    np.testing.assert_array_equal(ids, token_ids)
    assert values.tobytes() == logprobs.tobytes()
    np.testing.assert_array_equal(rk, ranks)


@pytest.mark.parametrize("num_logprobs", [0, 2, -1])
def test_processor_array_path_matches_flat_path(num_logprobs):
    width = 4
    chunks = [_engine_rows(0, 3, width), _engine_rows(3, 2, width)]

    def processor(array: bool):
        request = MagicMock(spec=EngineCoreRequest)
        request.sampling_params = SamplingParams(
            logprobs=num_logprobs, flat_logprobs=not array, array_logprobs=array
        )
        return LogprobsProcessor.from_new_request(None, request)

    array, flat = processor(True), processor(False)
    for token_ids, logprobs, ranks in chunks:
        lists = LogprobsLists(token_ids, logprobs, ranks)
        for proc in (array, flat):
            proc._update_sample_logprobs(lists)
    assert isinstance(array.logprobs, ArrayLogprobs)
    assert len(array.logprobs) == len(flat.logprobs) == 5
    assert list(array.logprobs) == list(flat.logprobs)
    assert array.cumulative_logprob == flat.cumulative_logprob
    slots = width if num_logprobs == -1 else num_logprobs + 1
    assert array.logprobs.num_slots == slots


# ----------------------------------------------------------------- serving


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_width", [4, 6])
async def test_compact_abort_with_partial_output(engine_width):
    """Abort after partial output returns all accumulated rows bit-exactly.

    ``engine_width`` 6 emulates rows padded to the batch-wide max top-k.
    """
    chunks = [_engine_rows(0, 5, engine_width), _engine_rows(5, 3, engine_width)]
    # Non-finite values (incl. a NaN payload) must survive bit-exactly.
    chunks[0][1][0, 1] = np.float32(-np.inf)
    chunks[0][1][1, 0] = np.float32(np.inf)
    chunks[1][1][2, 2] = np.array([0x7FC00123], dtype=np.uint32).view(np.float32)[0]
    chunks[1][1][0, 3] = np.float32(-1e30)
    feeder = _OutputProcessorEngine(chunks)
    serving = _serving(feeder)

    response = await serving.serve_tokens(_request(logprobs_format="compact"))

    assert feeder.sampling_params.array_logprobs is True
    assert feeder.sampling_params.detokenize is False
    assert isinstance(response, RenderedGenerateResponse)
    data = json.loads(response.body)
    (choice,) = data["choices"]
    assert choice["logprobs"] is None
    assert choice["finish_reason"] == "abort"
    expected_ids, expected_lps, expected_ranks = _expected(chunks, 4)
    assert choice["token_ids"] == expected_ids[:, 0].tolist()
    block = choice["compact_logprobs"]
    assert block["num_positions"] == 8 and block["num_slots"] == 4
    assert block["dtype_token_ids"] == "int32"
    assert block["dtype_logprobs"] == "float32"
    assert block["byteorder"] == "little"
    ids, lps, ranks = _decode(block)
    np.testing.assert_array_equal(ids, expected_ids)
    assert lps.tobytes() == expected_lps.tobytes()
    np.testing.assert_array_equal(ranks, expected_ranks)
    # Slot 0 is the sampled token; ranks are the engine's sampled ranks.
    np.testing.assert_array_equal(ids[:, 0], choice["token_ids"])
    assert data["usage"]["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_compact_abort_before_any_output():
    feeder = _OutputProcessorEngine([])
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=0, logprobs_format="compact")
    )
    choice = json.loads(response.body)["choices"][0]
    assert choice["token_ids"] == []
    assert choice["compact_logprobs"] == {
        "num_positions": 0,
        "num_slots": 1,
        "dtype_token_ids": "int32",
        "dtype_logprobs": "float32",
        "byteorder": "little",
        "token_ids": "",
        "logprobs": "",
        "ranks": "",
    }


@pytest.mark.asyncio
async def test_compact_keeps_detokenizer_for_stop_strings():
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)])
    await _serving(feeder).serve_tokens(
        _request(logprobs_format="compact", sampling={"stop": ["x"]})
    )
    assert feeder.sampling_params.array_logprobs is True
    assert feeder.sampling_params.detokenize is True


@pytest.mark.asyncio
async def test_compact_without_logprobs_has_no_block():
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)])
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=None, logprobs_format="compact")
    )
    assert isinstance(response, GenerateResponse)
    assert "compact_logprobs" not in response.model_dump()["choices"][0]


@pytest.mark.asyncio
async def test_default_stream_ignores_client_array_flag():
    chunks = [_engine_rows(0, 3, 4)]
    feeder = _OutputProcessorEngine(chunks, finish=FinishReason.LENGTH)
    generator = await _serving(feeder).serve_tokens(
        _request(stream=True, sampling={"array_logprobs": True})
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert feeder.sampling_params.array_logprobs is False
    assert feeder.sampling_params.detokenize is True
    content = events[0]["choices"][0]["logprobs"]["content"]
    assert [c["token"] for c in content] == [
        f"token_id:{t}" for t in chunks[0][0][:, 0].tolist()
    ]


async def _full_body(monkeypatch, chunks, request, array: bool) -> bytes:
    """Full non-streaming body through the router's rendering rules."""
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    if not array:
        monkeypatch.setattr(
            ServingTokens, "_use_array_logprobs", staticmethod(lambda r: False)
        )
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(request.model_copy(deep=True))
    monkeypatch.undo()
    assert feeder.sampling_params.array_logprobs is array
    if array:
        assert isinstance(response, RenderedGenerateResponse)
        return response.body
    assert isinstance(response, GenerateResponse)
    return JSONResponse(content=response.model_dump()).body


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs", [0, 1, 3])
async def test_default_full_response_is_byte_identical(monkeypatch, logprobs):
    chunks = [_engine_rows(0, 5, 6), _engine_rows(5, 4, 6)]
    chunks[0][1][2, 0] = np.float32(-np.inf)  # clamped to -9999.0
    chunks[1][1][1, 2] = np.float32(-1e30)
    chunks[1][1][3, 1] = np.float32(-0.0)
    request = _request(logprobs=logprobs, request_id="fixed-id")
    fast = await _full_body(monkeypatch, chunks, request, array=True)
    legacy = await _full_body(monkeypatch, chunks, request, array=False)
    assert fast == legacy


def _legacy_logprobs_bytes(token_ids, legacy, k) -> bytes:
    model = ServingTokens._create_tokens_logprobs(
        cast(Any, None),
        token_ids=token_ids,
        top_logprobs=legacy,
        num_output_top_logprobs=k,
    )
    return JSONResponse(content=model.model_dump()).body


def _containers(token_ids, logprobs, ranks, num_logprobs):
    container = ArrayLogprobs()
    width = token_ids.shape[1]
    slots = width if num_logprobs == -1 else num_logprobs + 1
    container.append_rows(token_ids[:, :slots], logprobs[:, :slots], ranks)
    legacy: list = []
    for i in range(len(ranks)):
        append_logprobs_for_next_position(
            legacy,
            token_ids[i].tolist(),
            logprobs[i].tolist(),
            [None] * width,
            int(ranks[i]),
            num_logprobs,
        )
    return container, legacy


@pytest.mark.parametrize("num_logprobs", [0, 1, 2, 5, -1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_openai_renderer_matches_legacy(num_logprobs, seed):
    width = 6
    token_ids, logprobs, ranks = _engine_rows(seed * 100, 40, width, seed=seed)
    special = np.array(
        [-np.inf, -1e30, -9999.0, -9999.5, -0.0, -1.2e-7, -5e-45, -1e-5, -123.456],
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed)
    logprobs.flat[rng.choice(logprobs.size, len(special), replace=False)] = special
    container, legacy = _containers(token_ids, logprobs, ranks, num_logprobs)
    sampled = token_ids[:, 0].tolist()
    fast = render_openai_logprobs(sampled, container, num_logprobs)
    assert fast == _legacy_logprobs_bytes(sampled, legacy, num_logprobs)
    # Also exactly what the full legacy response embeds.
    assert render_openai_logprobs([], ArrayLogprobs(), num_logprobs) == (
        _legacy_logprobs_bytes([], [], num_logprobs)
    )


def test_openai_renderer_without_top_k_count():
    token_ids, logprobs, ranks = _engine_rows(0, 4, 3)
    container, legacy = _containers(token_ids, logprobs, ranks, 2)
    sampled = token_ids[:, 0].tolist()
    fast = render_openai_logprobs(sampled, container, None)
    assert fast == _legacy_logprobs_bytes(sampled, legacy, None)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_openai_renderer_nonfinite_raises_like_legacy(value):
    token_ids, logprobs, ranks = _engine_rows(0, 3, 4)
    logprobs[1, 1] = np.float32(value)
    container, legacy = _containers(token_ids, logprobs, ranks, 3)
    sampled = token_ids[:, 0].tolist()
    with pytest.raises(ValueError, match="Out of range float values"):
        render_openai_logprobs(sampled, container, 3)
    with pytest.raises(ValueError, match="Out of range float values"):
        _legacy_logprobs_bytes(sampled, legacy, 3)


@pytest.mark.parametrize("case", ["dup_top", "sampled_mismatch", "length"])
def test_openai_renderer_defers_irregular_rows(case):
    token_ids, logprobs, ranks = _engine_rows(0, 4, 5)
    sampled = token_ids[:, 0].tolist()
    if case == "dup_top":
        token_ids[2, 3] = token_ids[2, 4]
    elif case == "sampled_mismatch":
        sampled[1] += 1
    else:
        sampled = sampled[:-1]
    container, legacy = _containers(token_ids, logprobs, ranks, 4)
    assert render_openai_logprobs(sampled, container, 4) is None
    if case == "dup_top":
        # The legacy fallback through ArrayLogprobs positional access still
        # yields the legacy bytes.
        assert _legacy_logprobs_bytes(sampled, container, 4) == _legacy_logprobs_bytes(
            sampled, legacy, 4
        )


@pytest.mark.asyncio
async def test_default_full_response_irregular_rows_fall_back(monkeypatch):
    chunks = [_engine_rows(0, 4, 5)]
    chunks[0][0][1, 2] = chunks[0][0][1, 3]
    request = _request(logprobs=4, request_id="fixed-id")
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(request.model_copy(deep=True))
    monkeypatch.undo()
    assert isinstance(response, GenerateResponse)
    legacy = await _full_body(monkeypatch, chunks, request, array=False)
    assert JSONResponse(content=response.model_dump()).body == legacy


@pytest.mark.asyncio
async def test_compact_stream_emits_per_chunk_blocks():
    chunks = [_engine_rows(0, 3, 5), _engine_rows(3, 4, 5)]
    chunks[1][1][1, 1] = np.float32(np.nan)
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert events[-1] == "[DONE]"
    data_events = [e for e in events[:-1] if e["choices"]]
    assert len(data_events) == 2
    decoded = []
    for event, chunk in zip(data_events, chunks):
        (choice,) = event["choices"]
        assert choice["logprobs"] is None
        block = choice["compact_logprobs"]
        assert block["num_positions"] == len(choice["token_ids"]) == len(chunk[2])
        decoded.append(_decode(block))
    expected_ids, expected_lps, expected_ranks = _expected(chunks, 4)
    np.testing.assert_array_equal(np.concatenate([d[0] for d in decoded]), expected_ids)
    assert np.concatenate([d[1] for d in decoded]).tobytes() == expected_lps.tobytes()
    np.testing.assert_array_equal(
        np.concatenate([d[2] for d in decoded]), expected_ranks
    )


@pytest.mark.asyncio
async def test_default_stream_has_no_compact_key():
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)])
    generator = await _serving(feeder).serve_tokens(_request(stream=True))
    events = _parse_sse_chunks([chunk async for chunk in generator])
    choice = events[0]["choices"][0]
    assert "compact_logprobs" not in choice
    assert len(choice["logprobs"]["content"]) == 2


def test_router_renders_body_and_releases_load_counter():
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False, tokens_only=False)
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    serving = MagicMock()

    async def _serve(request, raw_request):
        return RenderedGenerateResponse(b'{"choices":[]}')

    serving.serve_tokens = _serve
    app.state.serving_tokens = serving
    init_exception_handler(app)
    api_router.attach_router(app)
    with TestClient(app) as client:
        result = client.post(
            "/inference/v1/generate",
            json={
                "token_ids": [1],
                "sampling_params": {},
                "logprobs_format": "compact",
            },
        )
    assert result.status_code == 200
    assert result.content == b'{"choices":[]}'
    assert result.headers["content-type"] == "application/json"
    assert int(result.headers["content-length"]) == len(result.content)
    assert app.state.server_load_metrics == 0
