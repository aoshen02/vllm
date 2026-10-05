# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``logprobs_format`` option of ``/inference/v1/generate``."""

import contextvars
import hashlib
import json
import threading
import time
from argparse import Namespace
from typing import Any, cast
from unittest.mock import MagicMock

import msgspec
import numpy as np
import pybase64 as base64
import pytest
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.scale_out.token_in_token_out import (
    api_router,
    logprobs_render,
)
from vllm.entrypoints.scale_out.token_in_token_out.logprobs_render import (
    format_float_reprs,
    render_compact_logprobs,
    render_compact_logprobs_parts,
    render_json_with_fragments,
    render_json_with_fragments_parts,
    render_openai_logprobs,
    render_openai_logprobs_parts,
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
from vllm.logprobs import ArrayLogprobs, Logprob, append_logprobs_for_next_position
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import (
    BaseIncrementalDetokenizer,
    IncrementalDetokenizer,
)
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.serial_utils import MsgpackEncoder

from .test_generate_stream import (
    MODEL_NAME,
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
    top = (
        np.stack(
            [rng.choice(50_000, size=width - 1, replace=False) for _ in range(count)]
        )
        .astype(np.int64)
        .reshape(count, width - 1)
    )
    positions = np.arange(start, start + count)
    if width == 1:  # logprobs=0: sampled slot only
        sampled = 50_000 + positions
    else:
        sampled = np.where(
            positions % 2 == 0, top[:, min(1, width - 2)], 50_000 + positions
        )
    token_ids = np.column_stack((sampled, top))
    logprobs = -rng.random((count, width), dtype=np.float32) * 20
    ranks = rng.integers(1, 1000, size=count).astype(np.int64)
    return token_ids, logprobs.astype(np.float32), ranks


class _OutputProcessorEngine:
    """Feeds engine rows through the real OutputProcessor."""

    def __init__(
        self,
        chunks,
        finish: FinishReason | None = FinishReason.ABORT,
        tokenizer=None,
        routed=None,
    ):
        self.routed = routed
        self.chunks = chunks
        self.finish = finish
        self.tokenizer = tokenizer
        self.sampling_params: Any = None
        self.engine_request: Any = None
        self.detokenizer: Any = None

    def generate(self, engine_input, sampling_params, request_id, **kwargs):
        self.sampling_params = sampling_params

        async def _gen():
            processor = OutputProcessor(tokenizer=self.tokenizer, log_stats=False)
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
            self.engine_request = request
            self.detokenizer = processor.request_states[request.request_id].detokenizer
            prompt_k = sampling_params.prompt_logprobs
            for i, (token_ids, logprobs, ranks) in enumerate(self.chunks):
                processor.process_outputs(
                    [
                        EngineCoreOutput(
                            request_id=request.request_id,
                            routed_experts=self.routed[i] if self.routed else None,
                            new_token_ids=token_ids[:, 0].tolist(),
                            new_logprobs=(
                                None
                                if sampling_params.num_logprobs is None
                                or logprobs is None
                                else LogprobsLists(token_ids, logprobs, ranks)
                            ),
                            new_prompt_logprobs_tensors=(
                                _prompt_logprobs_tensors(prompt_k)
                                if i == 0 and prompt_k is not None
                                else None
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


def _prompt_logprobs_tensors(k: int) -> LogprobsTensors:
    """Prompt logprobs for prompt [1, 2, 3] (positions 1 and 2)."""
    ids = torch.tensor([[2, 1000 + k, 3000][: k + 1], [3, 5000, 7000][: k + 1]])
    values = torch.tensor([[-0.5, -1.5, -2.5][: k + 1], [-0.25, -3.0, -4.0][: k + 1]])
    return LogprobsTensors(ids, values, torch.tensor([4, 1]))


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return get_tokenizer(MODEL_NAME)


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
    assert feeder.sampling_params.detokenize is True
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


async def _full_body(
    monkeypatch, chunks, request, array: bool, tokenizer=None
) -> bytes:
    """Full non-streaming body through the router's rendering rules."""
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    if not array:
        monkeypatch.setattr(
            ServingTokens, "_use_array_logprobs", staticmethod(lambda r: False)
        )
    feeder = _OutputProcessorEngine(chunks, tokenizer=tokenizer)
    response = await _serving(feeder).serve_tokens(request.model_copy(deep=True))
    monkeypatch.undo()
    assert feeder.sampling_params.array_logprobs is array
    if isinstance(response, RenderedGenerateResponse):
        assert array
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
    with pytest.raises(ValueError, match="Out of range float values") as fast:
        render_openai_logprobs(sampled, container, 3)
    with pytest.raises(ValueError, match="Out of range float values") as slow:
        _legacy_logprobs_bytes(sampled, legacy, 3)
    # Same message (incl. ": nan" / ": inf"), so the 400 body is unchanged.
    assert str(fast.value) == str(slow.value)


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


def test_float_reprs_match_python_repr():
    """Sampled check; scripts/claude-genopt-py-float-exhaustive.py checked all
    finite float32 values."""
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2**32, size=200_000, dtype=np.uint64).astype(np.uint32)
    values = bits.view(np.float32)
    edges = []
    for edge in (1e-4, 1e16, 9999.0, 1.0):
        e = np.float32(edge)
        edges += [e, np.nextafter(e, np.float32(0)), np.nextafter(e, np.float32(1e30))]
    special = np.array(
        edges + [0.0, 1e-45, 3.4028235e38, 1.2e-7, 0.1, 5e-5, 123.456],
        dtype=np.float32,
    )
    values = np.concatenate([values[np.isfinite(values)], special, -special])
    as_double = values.astype(np.float64)
    expected = [repr(v).encode() for v in as_double.tolist()]
    assert format_float_reprs(as_double, True) == expected
    doubles = rng.standard_normal(2000) * np.logspace(-8, 8, 2000)
    assert format_float_reprs(doubles, False) == [
        repr(v).encode() for v in doubles.tolist()
    ]
    assert format_float_reprs(np.empty(0), True) == []


def test_openai_renderer_float64_engine_values_match_legacy():
    token_ids, logprobs, ranks = _engine_rows(0, 6, 4)
    values = logprobs.astype(np.float64) + 1e-9  # not float32-representable
    container, legacy = _containers(token_ids, values, ranks, 3)
    assert container.arrays()[1].dtype == np.float64
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 3) == (
        _legacy_logprobs_bytes(sampled, legacy, 3)
    )


def test_parts_renderers_join_to_bytes():
    container = ArrayLogprobs()
    token_ids, logprobs, ranks = _engine_rows(0, 3, 4)
    container.append_rows(token_ids, logprobs, ranks)
    sampled = token_ids[:, 0].tolist()
    assert b"".join(render_compact_logprobs_parts(container, 3)) == (
        render_compact_logprobs(container, 3)
    )
    parts = render_openai_logprobs_parts(sampled, container, 3)
    assert parts is not None
    assert b"".join(parts) == render_openai_logprobs(sampled, container, 3)


def test_array_logprobs_coalesces_single_row_steps(monkeypatch):
    """One row per engine step (the common decode case) must not create one
    numpy array per row, and suffix slices must stay correct."""
    token_ids, logprobs, ranks = _engine_rows(0, 11, 3)
    row_bytes = 3 * 8 + 4
    monkeypatch.setattr(ArrayLogprobs, "BLOCK_BYTES", 4 * row_bytes)
    container = ArrayLogprobs()
    for i in range(11):
        container.append_rows(
            token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
        )
        tail = container[-1:]
        np.testing.assert_array_equal(tail.arrays()[0], token_ids[i : i + 1])
    assert len(container) == 11
    # Geometric 1, 2, 4 then capped at 4 rows: 1 + 2 + 4 + 4 = 11.
    assert [len(r) for r in container.rank_chunks] == [1, 2, 4, 4]
    middle = container[2:9]
    np.testing.assert_array_equal(middle.arrays()[0], token_ids[2:9])
    assert middle.arrays()[1].tobytes() == logprobs[2:9].tobytes()
    ids, values, rk = container.arrays()
    np.testing.assert_array_equal(ids, token_ids)
    assert values.tobytes() == logprobs.tobytes()
    np.testing.assert_array_equal(rk, ranks)
    assert len(container.rank_chunks) == 1
    # Appending after consolidation keeps working.
    container.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    assert len(container) == 13
    np.testing.assert_array_equal(container.arrays()[0][11:], token_ids[:2])


# ------------------------------------------------- round-2 audit regressions


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs", [None, 2])
@pytest.mark.parametrize("prompt_logprobs", [0, 2])
async def test_default_prompt_logprobs_keep_decoded_tokens(
    monkeypatch, gpt2_tokenizer, logprobs, prompt_logprobs
):
    """Audit #1: a default request with prompt logprobs and no stop strings
    must keep detokenization (decoded_token text), byte-identical to legacy."""
    chunks = [_engine_rows(0, 3, 3), _engine_rows(3, 2, 3)]
    request = _request(
        logprobs=logprobs,
        request_id="fixed-id",
        sampling={"prompt_logprobs": prompt_logprobs},
    )
    fast = await _full_body(monkeypatch, chunks, request, True, gpt2_tokenizer)
    legacy = await _full_body(monkeypatch, chunks, request, False, gpt2_tokenizer)
    assert fast == legacy
    prompt = json.loads(fast)["prompt_logprobs"]
    assert prompt[0] is None
    decoded = [entry["decoded_token"] for entry in prompt[1].values()]
    assert decoded[0] == gpt2_tokenizer.decode([2])
    assert all(isinstance(text, str) for text in decoded)


@pytest.mark.asyncio
async def test_default_keeps_requested_detokenize(gpt2_tokenizer):
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 3)], tokenizer=gpt2_tokenizer)
    await _serving(feeder).serve_tokens(_request(logprobs=2))
    assert feeder.sampling_params.array_logprobs is True
    assert feeder.sampling_params.detokenize is True


@pytest.mark.asyncio
async def test_compact_with_prompt_logprobs_keeps_decoded_tokens(gpt2_tokenizer):
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 3)], tokenizer=gpt2_tokenizer)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=2, logprobs_format="compact", sampling={"prompt_logprobs": 1})
    )
    assert feeder.sampling_params.detokenize is True
    prompt = json.loads(response.body)["prompt_logprobs"]
    assert prompt[1]["2"]["decoded_token"] == gpt2_tokenizer.decode([2])


def test_array_logprobs_short_wide_request_reserves_little():
    """Audit #2: a 1-row wide request must not reserve thousands of rows."""
    width = 1024
    token_ids, logprobs, ranks = _engine_rows(0, 1, width)
    container = ArrayLogprobs()
    container.append_rows(token_ids, logprobs, ranks)
    row_bytes = width * 8 + 4
    assert container.reserved_bytes() == row_bytes
    for i in range(1, 40):
        container.append_rows(token_ids, logprobs, ranks)
        used = (i + 1) * row_bytes
        assert container.reserved_bytes() <= 2 * used
    big = ArrayLogprobs()
    big.append_rows(*_engine_rows(0, 3000, 8))
    assert big.reserved_bytes() == 3000 * (8 * 8 + 4)


def test_array_logprobs_block_bytes_cap(monkeypatch):
    monkeypatch.setattr(ArrayLogprobs, "BLOCK_BYTES", 10 * (4 * 8 + 4))
    container = ArrayLogprobs()
    rows = _engine_rows(0, 1, 4)
    for _ in range(100):
        container.append_rows(*rows)
    assert max(len(r) for r in container.rank_chunks) == 10


def test_array_logprobs_single_token_delta_slice_allocates_nothing():
    """Audit #2: DELTA ``[-1:]`` slices are exact-sized views."""
    token_ids, logprobs, ranks = _engine_rows(0, 300, 129)
    container = ArrayLogprobs()
    for i in range(300):
        container.append_rows(
            token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
        )
        tail = container[-1:]
        assert tail.reserved_bytes() == 129 * 8 + 4
        assert np.shares_memory(tail.token_id_chunks[0], container.token_id_chunks[-1])
    tail.append_rows(token_ids[:1], logprobs[:1], ranks[:1])
    # Appending to a slice never writes into the source container.
    np.testing.assert_array_equal(container.arrays()[0], token_ids)
    np.testing.assert_array_equal(tail.arrays()[0], token_ids[[299, 0]])


def test_array_logprobs_suffix_slice_visits_only_needed_blocks(monkeypatch):
    """Audit #3: a last-row slice touches only its own block."""
    monkeypatch.setattr(ArrayLogprobs, "BLOCK_BYTES", 2 * (3 * 8 + 4))
    token_ids, logprobs, ranks = _engine_rows(0, 20, 3)
    container = ArrayLogprobs()
    for i in range(20):
        container.append_rows(
            token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
        )
    assert len(container.rank_chunks) >= 10
    visited: list[int] = []
    original = ArrayLogprobs._filled_block

    def counting(self, i):
        visited.append(i)
        return original(self, i)

    monkeypatch.setattr(ArrayLogprobs, "_filled_block", counting)
    tail = container[-1:]
    assert visited == [len(container.rank_chunks) - 1]
    np.testing.assert_array_equal(tail.arrays()[0], token_ids[-1:])
    visited.clear()
    suffix = container[-5:]
    # 5 rows in blocks of at most 2 rows: at most 4 blocks are touched.
    assert len(visited) <= 4
    np.testing.assert_array_equal(suffix.arrays()[0], token_ids[-5:])


@pytest.mark.parametrize("first_rows", [1, 3])
def test_array_logprobs_mixed_float_dtypes_are_lossless(first_rows):
    """Audit #4: float64 rows after float32 rows keep full precision, both
    inside a block with free capacity and across a block boundary."""
    token_ids, logprobs32, ranks = _engine_rows(0, first_rows + 2, 4)
    container = ArrayLogprobs()
    container.append_rows(
        token_ids[:first_rows], logprobs32[:first_rows], ranks[:first_rows]
    )
    values64 = logprobs32[first_rows:].astype(np.float64)
    values64[0, 0] = -1.000000001
    values64[1, 2] = -2.000000003
    container.append_rows(token_ids[first_rows:], values64, ranks[first_rows:])
    assert container[first_rows][int(token_ids[first_rows, 0])].logprob == (
        -1.000000001
    )
    legacy: list = []
    expected = np.concatenate([logprobs32[:first_rows].astype(np.float64), values64])
    for i in range(first_rows + 2):
        append_logprobs_for_next_position(
            legacy,
            token_ids[i].tolist(),
            expected[i].tolist(),
            [None] * 4,
            int(ranks[i]),
            3,
        )
    assert list(container) == legacy
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 3) == _legacy_logprobs_bytes(
        sampled, legacy, 3
    )
    assert container.arrays()[1].dtype == np.float64


@pytest.mark.parametrize("consolidate", [False, True])
def test_array_logprobs_self_extend_doubles(consolidate):
    """Audit #5: ``c.extend(c)`` terminates and doubles, like a list."""
    token_ids, logprobs, ranks = _engine_rows(0, 7, 3)
    container = ArrayLogprobs()
    for i in range(7):
        container.append_rows(
            token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
        )
    if consolidate:
        container.arrays()
    container.extend(container)
    assert len(container) == 14
    np.testing.assert_array_equal(
        container.arrays()[0], np.concatenate([token_ids, token_ids])
    )


def test_array_logprobs_flag_stays_off_engine_wire():
    """Audit #6: array_logprobs is frontend-only; the request sent to
    EngineCore after add_request does not carry it."""
    params = SamplingParams(
        logprobs=2,
        array_logprobs=True,
        array_logprobs_base64=True,
        array_logprobs_wire_topk_only=True,
    )
    request = EngineCoreRequest(
        request_id="r-int",
        external_req_id="r",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=params,
        pooling_params=None,
    )
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(request, None, queue=None)
    state = processor.request_states["r-int"]
    assert isinstance(state.logprobs_processor.logprobs, ArrayLogprobs)
    assert state.logprobs_processor.logprobs.wire_parts() is not None
    assert b"array_logprobs" not in MsgpackEncoder().encode(request)[0]
    assert request.sampling_params.array_logprobs is False
    assert request.sampling_params.array_logprobs_base64 is False
    assert request.sampling_params.array_logprobs_wire_topk_only is False
    # The caller's (possibly shared) params object is not mutated.
    assert params.array_logprobs is True
    assert params.array_logprobs_base64 is True


def test_float_fast_path_guard(monkeypatch):
    """Kimi #7a: the msgspec fast path is probed against repr at import and
    is not used when the probe fails."""
    assert logprobs_render._MSGSPEC_FLOATS_MATCH_REPR is True
    assert logprobs_render._msgspec_floats_match_repr() is True
    values = np.array([-1e-5, -0.5, -1e16, -0.037000000476837158])
    expected = [repr(v).encode() for v in values.tolist()]
    real_encode = msgspec.json.encode
    monkeypatch.setattr(
        logprobs_render.msgspec.json,
        "encode",
        lambda obj: real_encode(obj).replace(b"0.5", b"5e-1"),
    )
    assert logprobs_render._msgspec_floats_match_repr() is False
    monkeypatch.setattr(logprobs_render, "_MSGSPEC_FLOATS_MATCH_REPR", False)
    assert format_float_reprs(values, True) == expected


def test_out_of_int32_ids_never_wrap():
    """Kimi #7b: ids/ranks beyond int32 are stored exactly (default path) and
    rejected by the int32 compact wire format instead of wrapping."""
    token_ids, logprobs, ranks = _engine_rows(0, 3, 4)
    token_ids = token_ids.astype(np.int64)
    token_ids[1, 2] = 2**31 + 5
    ranks = ranks.astype(np.int64)
    container, legacy = _containers(token_ids, logprobs, ranks, 3)
    assert list(container) == legacy
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 3) == _legacy_logprobs_bytes(
        sampled, legacy, 3
    )
    with pytest.raises(GenerationError, match="int32"):
        render_compact_logprobs(container, 3)
    rank_container = ArrayLogprobs()
    big_ranks = np.full(3, 2**31, dtype=np.int64)
    rank_container.append_rows(token_ids[:, :1] * 0, logprobs[:, :1], big_ranks)
    assert rank_container[0][0].rank == 2**31
    with pytest.raises(GenerationError, match="int32"):
        render_compact_logprobs(rank_container, 0)


def _narrow_steps(k: int, widths: list[int]):
    """Engine rows whose width changes between steps (e.g. a co-batched
    request's logprob_token_ids replaced the batch's logprob tensors)."""
    chunks = []
    for step, width in enumerate(widths):
        ids = np.array([[7 + step] + list(range(100, 100 + width - 1))])
        lps = np.full((1, width), -np.inf, dtype=np.float32)
        lps[0, 0] = -1.0 - step
        lps[0, 1:] = -2.0 - np.arange(width - 1, dtype=np.float32)
        chunks.append((ids, lps, np.array([1 + step])))
    return chunks


def _process(k: int, chunks, array: bool, kind=RequestOutputKind.FINAL_ONLY):
    params = SamplingParams(
        max_tokens=10, logprobs=k, detokenize=False, array_logprobs=array
    )
    params.output_kind = kind
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[1, 2],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=params,
        pooling_params=None,
    )
    queue = RequestOutputCollector(kind, "r")
    processor.add_request(request, None, queue=queue)
    for ids, lps, ranks in chunks:
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="r",
                    new_token_ids=ids[:, 0].tolist(),
                    new_logprobs=LogprobsLists(ids, lps, ranks),
                )
            ]
        )
    processor.abort_requests(["r"], internal=True)
    out = queue.get_nowait()
    assert out is not None
    return out.outputs[0].logprobs


@pytest.mark.parametrize(
    "k,widths",
    [(5, [6, 2]), (5, [2, 6, 3]), (-1, [7, 3]), (-1, [3, 7]), (2, [6, 6])],
)
@pytest.mark.parametrize(
    "kind", [RequestOutputKind.FINAL_ONLY, RequestOutputKind.DELTA]
)
def test_irregular_row_widths_never_raise(k, widths, kind):
    """Claude audit A: narrow or varying engine rows must not raise inside
    OutputProcessor (that kills AsyncLLM's output handler); the stored
    positions equal the legacy list path's."""
    chunks = _narrow_steps(k, widths)
    array = _process(k, chunks, True, kind)
    legacy = _process(k, chunks, False, kind)
    assert isinstance(array, ArrayLogprobs)
    assert list(array) == list(legacy)
    assert array.is_regular == (len(set(widths)) == 1 or (k >= 0 and min(widths) > k))


@pytest.mark.asyncio
async def test_irregular_rows_default_body_matches_legacy(monkeypatch):
    chunks = [_engine_rows(0, 3, 6), _engine_rows(3, 2, 2), _engine_rows(5, 1, 6)]
    request = _request(logprobs=5, request_id="fixed-id")
    fast = await _full_body(monkeypatch, chunks, request, array=True)
    legacy = await _full_body(monkeypatch, chunks, request, array=False)
    assert fast == legacy


@pytest.mark.asyncio
async def test_irregular_rows_compact_is_a_request_error():
    chunks = [_engine_rows(0, 3, 6), _engine_rows(3, 2, 2)]
    feeder = _OutputProcessorEngine(chunks)
    with pytest.raises(GenerationError, match="inconsistent widths"):
        await _serving(feeder).serve_tokens(
            _request(logprobs=5, logprobs_format="compact")
        )
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs=5, logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert events[-1] == "[DONE]"
    assert any("error" in e for e in events[:-1] if isinstance(e, dict))


def test_array_logprobs_accepts_numpy_integer_index():
    token_ids, logprobs, ranks = _engine_rows(0, 3, 4)
    container, legacy = _containers(token_ids, logprobs, ranks, 3)
    assert container[np.int64(1)] == legacy[1]
    assert container[np.int32(-1)] == legacy[-1]
    with pytest.raises(TypeError):
        container[1.0]  # type: ignore[index]


def test_generate_request_omits_default_logprobs_format():
    """Claude audit C: render endpoints serialize GenerateRequest; the
    opt-in field must not appear unless set."""
    request = _request()
    assert "logprobs_format" not in request.model_dump()
    assert "logprobs_format" not in json.loads(request.model_dump_json())
    compact = _request(logprobs_format="compact")
    assert compact.model_dump()["logprobs_format"] == "compact"
    restored = GenerateRequest.model_validate_json(compact.model_dump_json())
    assert restored.logprobs_format == "compact"


def test_plain_leads_cover_large_vocab(monkeypatch):
    """Claude r2 #5: entry prefixes are cached for every id below the table
    bound (no 131k cap); larger ids are formatted directly."""
    table = logprobs_render._LeadTable(b"")
    monkeypatch.setattr(logprobs_render, "_PLAIN_LEADS", table)
    token_ids, logprobs, ranks = _engine_rows(0, 20, 4)
    # Sampled ids (plain-lead column) beyond the old 131,072-entry cap; keep
    # the sampled-in-top-k duplicates consistent.
    duplicate = token_ids[:, 1:] == token_ids[:, :1]
    token_ids[:, 0] += 200_000
    token_ids[:, 1:][duplicate] += 200_000
    token_ids[3, 0] = 2**20 + 3  # beyond the table bound
    container, legacy = _containers(token_ids, logprobs, ranks, 3)
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 3) == _legacy_logprobs_bytes(
        sampled, legacy, 3
    )
    assert table.filled[int(token_ids[1, 0])]
    assert len(table.filled) <= logprobs_render._LeadTable.max_ids


# sha256 of the bodies the pre-change implementation (456c93187a, i.e. base
# 0fd2e8d503 + listener patch) returns for golden_generate_scenarios.py,
# produced with agent_run/scripts/claude-genopt-py-golden.py (real gpt2
# tokenizer, prompt logprobs, sampled-in-top-k rows, -inf, narrow rows).
GOLDEN_SHA256 = {
    "lp2_plp2_length": (
        "45d9a05e207128846a472d7397f795f0030a52ee77757fb3eb6381b98deefe18"
    ),
    "lpNone_plp1_abort": (
        "3c0cb3e93edca68e3f551a703bf9c0ac295e240b4cc41f8ded9535c65175cf4b"
    ),
    "lp0_abort": "cf982779105509e6f147eef4405361db959b3c0a4e629fb7227fc63794862d82",
    "lp5_narrow_rows": (
        "f4f2ff4bde6d73f2dc7eed672018c64c4a3025eb9c0250e795652a5ff251590a"
    ),
    "r3_lp3_u8_abort": (
        "d3adb076e65ce213a76938c9b959d4fb0506b9753e8d72a1c94c50a63dde7476"
    ),
    "r3_nolp_u8_61layers": (
        "3a00780860d635011eb1c28157eb69d2217e8f192dd6685b532487e76c696e25"
    ),
    "r3_lp1_u16_length": (
        "19f224bf794e0bfd83e2b1346a97f7434bf864e73c17db0d06bbbe22ab67c5a3"
    ),
    "r3_abort_before_forward": (
        "80412cf24f437b1f10fb4a32517fb0b3a681d91287c3179884fd70209792c4f6"
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(GOLDEN_SHA256))
async def test_default_body_matches_pre_change_golden(name):
    """Claude audit E: compare with the old implementation's bytes, not with
    this tree's legacy branch."""
    from .golden_generate_scenarios import SCENARIOS, render_body

    assert set(SCENARIOS) == set(GOLDEN_SHA256)
    body = await render_body(name)
    assert hashlib.sha256(body).hexdigest() == GOLDEN_SHA256[name]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold,offloaded", [(0, True), (1 << 40, False)])
@pytest.mark.parametrize("fmt", ["openai", "compact"])
async def test_large_responses_are_built_off_the_event_loop(
    monkeypatch, threshold, offloaded, fmt
):
    """Large array-logprob responses are built in a worker thread so the
    event loop keeps serving other requests; the body is unchanged."""
    from vllm.entrypoints.scale_out.token_in_token_out import serving as serving_mod

    threads: list[Any] = []
    original = ServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ServingTokens, "_build_full_response", recording)
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    bodies = []
    for limit in (threshold, 1 << 40):
        monkeypatch.setattr(serving_mod, "OFFLOAD_MIN_LOGPROB_ENTRIES", limit)
        feeder = _OutputProcessorEngine([_engine_rows(0, 4, 4)])
        response = await _serving(feeder).serve_tokens(
            _request(logprobs=3, request_id="fixed-id", logprobs_format=fmt)
        )
        bodies.append(response.body)
    assert (threads[0] != threading.get_ident()) is offloaded
    assert threads[1] == threading.get_ident()
    assert bodies[0] == bodies[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["openai", "compact"])
@pytest.mark.parametrize("stop", [None, ["x"]])
@pytest.mark.parametrize("prompt_logprobs", [None, 1])
async def test_sampled_text_detokenizer_only_for_stop_strings(
    gpt2_tokenizer, fmt, stop, prompt_logprobs
):
    """Generate responses carry no text: the sampled-token detokenizer runs
    only for stop strings, while prompt logprobs keep decoded tokens."""
    sampling: dict[str, Any] = {}
    if stop:
        sampling["stop"] = stop
    if prompt_logprobs is not None:
        sampling["prompt_logprobs"] = prompt_logprobs
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 3)], tokenizer=gpt2_tokenizer)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=2, logprobs_format=fmt, sampling=sampling)
    )
    assert feeder.sampling_params.detokenize is True
    assert isinstance(feeder.detokenizer, IncrementalDetokenizer)
    assert isinstance(feeder.detokenizer, BaseIncrementalDetokenizer) == bool(stop)
    if prompt_logprobs is not None:
        prompt = json.loads(response.body)["prompt_logprobs"]
        assert prompt[1]["2"]["decoded_token"] == gpt2_tokenizer.decode([2])


# ------------------------------------------------------- round 3: wire mode


def _wire_container(chunks) -> ArrayLogprobs:
    container = ArrayLogprobs(wire_base64=True)
    for token_ids, logprobs, ranks in chunks:
        container.append_rows(token_ids, logprobs, ranks)
    return container


@pytest.mark.parametrize("flush", [3, 7, 64, 1 << 20])
@pytest.mark.parametrize("step", [1, 3, 1024])
def test_wire_mode_matches_one_shot_compact(monkeypatch, flush, step):
    """Incremental base64 at append time == one-shot encoding of the arrays,
    for any engine step size and segment size (carry handling)."""
    from vllm import logprobs as logprobs_mod

    monkeypatch.setattr(logprobs_mod._Base64Stream, "FLUSH_BYTES", flush)
    token_ids, logprobs, ranks = _engine_rows(0, 50, 5)
    logprobs[3, 2] = np.array([0x7FC00123], dtype=np.uint32).view(np.float32)[0]
    logprobs[7, 0] = np.float32(-np.inf)
    chunks = [
        (token_ids[i : i + step], logprobs[i : i + step], ranks[i : i + step])
        for i in range(0, 50, step)
    ]
    wire = _wire_container(chunks)
    plain = ArrayLogprobs()
    for chunk in chunks:
        plain.append_rows(*chunk)
    assert len(wire) == 50
    assert wire.num_slots == 5
    assert wire.reserved_bytes() == 0  # no array storage in wire mode
    assert wire.wire_parts() is not None
    fast = b"".join(render_compact_logprobs_parts(wire, 4))
    assert fast == render_compact_logprobs(plain, 4)
    block = json.loads(fast)
    ids, lps, rk = _decode(block)
    np.testing.assert_array_equal(ids, token_ids)
    assert lps.tobytes() == logprobs.tobytes()
    np.testing.assert_array_equal(rk, ranks)
    # Rendering does not consume: render twice, append more, render again.
    assert b"".join(render_compact_logprobs_parts(wire, 4)) == fast
    wire.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    plain.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    assert b"".join(render_compact_logprobs_parts(wire, 4)) == (
        render_compact_logprobs(plain, 4)
    )


def test_wire_mode_positional_access_decodes():
    token_ids, logprobs, ranks = _engine_rows(0, 6, 4)
    wire = _wire_container([(token_ids, logprobs, ranks)])
    container, legacy = _containers(token_ids, logprobs, ranks, 3)
    assert wire[2] == legacy[2]  # leaves wire mode, decodes rows
    assert wire.wire_parts() is None
    assert list(wire) == legacy
    assert list(wire[-3:]) == legacy[-3:]
    assert render_compact_logprobs(wire, 3) == render_compact_logprobs(container, 3)


@pytest.mark.parametrize("case", ["width", "int64"])
def test_wire_mode_unrepresentable_rows_fall_back(case):
    """Rows the wire cannot carry leave wire mode; compact then fails the
    request (500) exactly as without wire mode, and positions stay exact."""
    token_ids, logprobs, ranks = _engine_rows(0, 4, 4)
    wire = _wire_container([(token_ids[:2], logprobs[:2], ranks[:2])])
    if case == "width":
        wire.append_rows(token_ids[2:, :3], logprobs[2:, :3], ranks[2:])
        assert not wire.is_regular
    else:
        big = token_ids[2:].astype(np.int64)
        big[0, 1] = 2**31 + 7
        wire.append_rows(big, logprobs[2:], ranks[2:])
        assert wire[2][2**31 + 7].logprob == float(logprobs[2, 1])
    assert len(wire) == 4
    assert wire.wire_parts() is None
    with pytest.raises(GenerationError):
        render_compact_logprobs_parts(wire, 3)


@pytest.mark.asyncio
async def test_compact_full_response_uses_wire_mode_and_same_bytes(monkeypatch):
    chunks = [_engine_rows(0, 5, 6), _engine_rows(5, 3, 6)]
    request = _request(logprobs=4, logprobs_format="compact", request_id="fixed")
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    feeder = _OutputProcessorEngine(chunks)
    wire_body = (
        await _serving(feeder).serve_tokens(request.model_copy(deep=True))
    ).body
    assert feeder.sampling_params.array_logprobs_base64 is True
    original = ServingTokens._configure_logprobs.__func__

    def no_wire(cls, req, params):
        original(cls, req, params)
        params.array_logprobs_base64 = False

    monkeypatch.setattr(ServingTokens, "_configure_logprobs", classmethod(no_wire))
    feeder = _OutputProcessorEngine(chunks)
    plain_body = (
        await _serving(feeder).serve_tokens(request.model_copy(deep=True))
    ).body
    assert feeder.sampling_params.array_logprobs_base64 is False
    assert wire_body == plain_body


@pytest.mark.asyncio
async def test_compact_stream_does_not_use_wire_mode():
    feeder = _OutputProcessorEngine([_engine_rows(0, 3, 4)])
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    _ = [chunk async for chunk in generator]
    assert feeder.sampling_params.array_logprobs_base64 is False


def test_router_sends_parts_with_content_length():
    """The rendered body is sent as several body messages (no joined copy)
    with the same bytes and headers as a joined body."""
    big = [b"x" * (3 << 20), memoryview(b'","y":'), b"z" * 10, b"w" * (2 << 20)]
    expected = b"".join(big)
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False, tokens_only=False)
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    serving = MagicMock()

    async def _serve(request, raw_request):
        return RenderedGenerateResponse(list(big))

    serving.serve_tokens = _serve
    app.state.serving_tokens = serving
    init_exception_handler(app)
    api_router.attach_router(app)
    messages: list[dict] = []
    inner = app.build_middleware_stack()

    async def recording_app(scope, receive, send):
        async def _send(message):
            messages.append(message)
            await send(message)

        await inner(scope, receive, _send)

    app.build_middleware_stack = lambda: recording_app  # type: ignore[method-assign]
    with TestClient(app) as client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {}},
        )
    assert result.status_code == 200
    assert result.content == expected
    assert int(result.headers["content-length"]) == len(expected)
    assert result.headers["content-type"] == "application/json"
    bodies = [m for m in messages if m["type"] == "http.response.body"]
    assert len(bodies) >= 3
    assert b"".join(m["body"] for m in bodies) == expected
    assert bodies[-1]["more_body"] is False
    assert app.state.server_load_metrics == 0


def test_rendered_response_accepts_bytes():
    rendered = RenderedGenerateResponse(b"{}")
    assert rendered.parts == [b"{}"] and rendered.body == b"{}"
    assert rendered.content_length == 2


def test_lead_table_large_ids_and_growth(monkeypatch):
    """Leads for ids beyond the table bound are formatted directly; the
    table grows on demand; bytes stay identical to the legacy path."""
    table = logprobs_render._LeadTable(b"|")
    monkeypatch.setattr(logprobs_render._LeadTable, "max_ids", 64)
    small = np.array([[3, 9], [63, 3]])
    assert table.lookup(small).tolist() == [
        [b'|{"token":"token_id:3","logprob":', b'|{"token":"token_id:9","logprob":'],
        [b'|{"token":"token_id:63","logprob":', b'|{"token":"token_id:3","logprob":'],
    ]
    assert len(table.values) == 64
    big = np.array([[64, 2**31 + 1]])
    assert table.lookup(big).tolist() == [
        [
            b'|{"token":"token_id:64","logprob":',
            b'|{"token":"token_id:2147483649","logprob":',
        ],
    ]
    monkeypatch.setattr(
        logprobs_render,
        "_NEXT_LEADS",
        logprobs_render._LeadTable(logprobs_render._SEP_TOP_NEXT),
    )
    monkeypatch.setattr(logprobs_render._LeadTable, "max_ids", 1000)
    token_ids, logprobs, ranks = _engine_rows(0, 5, 6)
    container, legacy = _containers(token_ids, logprobs, ranks, 5)
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 5) == _legacy_logprobs_bytes(
        sampled, legacy, 5
    )


# --------------------------------------------- round-2 audit regressions


def _single_row_appends(container, token_ids, logprobs, ranks, count):
    for i in range(count):
        container.append_rows(
            token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
        )


@pytest.mark.parametrize("widen", ["float64", "int64_id", "int64_rank"])
def test_dtype_widening_mid_block_keeps_only_initialized_rows(widen):
    """Codex/Claude r2 #1: 4 single-row appends leave the third block (cap 4)
    with 1 initialized row; a widening append must not expose the rest."""
    token_ids, logprobs32, ranks = _engine_rows(0, 5, 4)
    token_ids = token_ids.astype(np.int64)
    ranks = ranks.astype(np.int64)
    logprobs: np.ndarray = logprobs32
    if widen == "float64":
        logprobs = logprobs32.astype(np.float64)
        logprobs[4, 0] = -1.000000001
    elif widen == "int64_id":
        token_ids[4, 2] = 2**31 + 11
    else:
        ranks[4] = 2**31 + 13
    container = ArrayLogprobs()
    _single_row_appends(container, token_ids, logprobs32, ranks, 4)
    assert [len(r) for r in container.rank_chunks] == [1, 2, 4]
    container.append_rows(token_ids[4:], logprobs[4:], ranks[4:])
    legacy: list = []
    for i in range(5):
        values = (logprobs if i == 4 else logprobs32)[i].tolist()
        append_logprobs_for_next_position(
            legacy, token_ids[i].tolist(), values, [None] * 4, int(ranks[i]), 3
        )
    assert len(container) == 5
    assert container[4] == legacy[4]
    assert list(container) == legacy
    assert [list(container[i:]) for i in range(5)] == [legacy[i:] for i in range(5)]
    for _ in range(2):
        ids, values, rk = container.arrays()
        assert ids.shape == (5, 4) and values.shape == (5, 4) and rk.shape == (5,)
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 3) == _legacy_logprobs_bytes(
        sampled, legacy, 3
    )
    if widen == "float64":
        block = json.loads(render_compact_logprobs(container, 3))
        assert block["num_positions"] == 5
        np.testing.assert_array_equal(_decode(block)[0], token_ids)
    else:
        with pytest.raises(GenerationError, match="int32"):
            render_compact_logprobs(container, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_compact_uniformly_narrow_rows_are_a_request_error(stream):
    """Codex/Claude r2 #2: every row narrower than k+1 (co-batched
    logprob_token_ids) is not representable: 500, not a num_slots=2 block."""
    chunks = [_engine_rows(0, 3, 2), _engine_rows(3, 2, 2)]
    feeder = _OutputProcessorEngine(chunks)
    request = _request(logprobs=5, logprobs_format="compact", stream=stream)
    if not stream:
        with pytest.raises(GenerationError, match="2 slots, expected 6"):
            await _serving(feeder).serve_tokens(request)
        return
    generator = await _serving(feeder).serve_tokens(request)
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert events[-1] == "[DONE]"
    data = [e for e in events[:-1] if isinstance(e, dict)]
    assert "error" in data[0]  # already the first chunk
    assert not any("compact_logprobs" in str(e) for e in data)


@pytest.mark.asyncio
async def test_compact_logprob_token_ids_without_logprobs():
    """Rust parity: logprob_token_ids alone gives a compact block with
    S = len(ids) + 1 (num_logprobs); the default format stays as before."""
    chunks = [_engine_rows(0, 3, 3)]
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(
        _request(
            logprobs=None,
            logprobs_format="compact",
            sampling={"logprob_token_ids": [5, 9]},
        )
    )
    choice = json.loads(response.body)["choices"][0]
    assert choice["logprobs"] is None
    assert choice["compact_logprobs"]["num_slots"] == 3
    np.testing.assert_array_equal(_decode(choice["compact_logprobs"])[0], chunks[0][0])
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=None, sampling={"logprob_token_ids": [5, 9]})
    )
    dump = response.model_dump()["choices"][0]
    assert dump["logprobs"] is None and "compact_logprobs" not in dump


def test_irregular_fallback_is_linear(monkeypatch):
    """Codex r2 #3: legacy materialization visits each block once."""
    monkeypatch.setattr(ArrayLogprobs, "BLOCK_BYTES", 2 * (3 * 8 + 4))
    token_ids, logprobs, ranks = _engine_rows(0, 64, 3)
    container = ArrayLogprobs()
    _single_row_appends(container, token_ids, logprobs, ranks, 64)
    container.append_rows(token_ids[:1, :2], logprobs[:1, :2], ranks[:1])
    assert not container.is_regular
    visits: list[int] = []
    original = ArrayLogprobs._filled_block

    def counting(self, i):
        visits.append(i)
        return original(self, i)

    monkeypatch.setattr(ArrayLogprobs, "_filled_block", counting)
    positions = list(container)
    assert len(positions) == 65
    assert len(visits) == len(container.rank_chunks)


@pytest.mark.asyncio
async def test_offloaded_build_side_effects_on_loop_thread(monkeypatch):
    """Kimi r2: usage metadata and output logging happen on the event loop
    thread; contextvars are visible in the worker (Claude r2 #6)."""
    from vllm.entrypoints.scale_out.token_in_token_out import serving as serving_mod

    monkeypatch.setattr(serving_mod, "OFFLOAD_MIN_LOGPROB_ENTRIES", 0)
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("m", default="-")
    marker.set("caller")
    seen: dict[str, Any] = {}
    original = ServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        seen["builder_thread"] = threading.get_ident()
        seen["builder_ctx"] = marker.get()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ServingTokens, "_build_full_response", recording)
    feeder = _OutputProcessorEngine([_engine_rows(0, 3, 4)])
    serving = _serving(feeder)
    logger = MagicMock()
    logger.log_outputs.side_effect = lambda **kw: seen.setdefault(
        "log_thread", threading.get_ident()
    )
    serving.request_logger = logger
    serving.enable_log_outputs = True
    metadata_threads: list[int] = []

    class Metadata:
        def __setattr__(self, name, value):
            metadata_threads.append(threading.get_ident())
            object.__setattr__(self, name, value)

    metadata = Metadata()
    request = _request(logprobs=3)
    ServingTokens._configure_logprobs(request, request.sampling_params)
    response = await serving.serve_tokens_full_generator(
        request,
        feeder.generate(None, request.sampling_params, "r"),
        "r",
        "m",
        metadata,  # type: ignore[arg-type]
    )
    assert isinstance(response, RenderedGenerateResponse)
    loop_thread = threading.get_ident()
    assert seen["builder_thread"] != loop_thread
    assert seen["builder_ctx"] == "caller"
    assert seen["log_thread"] == loop_thread
    assert metadata_threads == [loop_thread]
    assert metadata.final_usage_info.completion_tokens == 3  # type: ignore[attr-defined]


def test_generate_request_schema_keeps_all_fields():
    """Claude r2 #3: excluding the default logprobs_format must not empty the
    serialization schema (it is the render endpoints' response model)."""
    for mode in ("serialization", "validation"):
        props = GenerateRequest.model_json_schema(mode=mode)["properties"]
        assert set(GenerateRequest.model_fields) <= set(props) | {
            name
            for name, f in GenerateRequest.model_fields.items()
            if f.alias and f.alias in props
        }
        assert "logprobs_format" in props
    for model in (GenerateResponseChoice, GenerateResponseStreamChoice):
        props = model.model_json_schema(mode="serialization")["properties"]
        assert "compact_logprobs" in props and "token_ids" in props


# ------------------------------------------- SPEC amendments (round 3)


@pytest.mark.parametrize("stream", [False, True])
def test_compact_full_vocab_logprobs_is_http_400(stream):
    """SPEC: compact + logprobs=-1 is rejected at request validation."""
    with pytest.raises(ValidationError, match="logprobs=-1"):
        _request(logprobs=-1, logprobs_format="compact", stream=stream)
    assert _request(logprobs=-1).sampling_params.logprobs == -1  # default ok
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
                "sampling_params": {"logprobs": -1},
                "logprobs_format": "compact",
                "stream": stream,
            },
        )
    assert result.status_code == 400
    app.state.serving_tokens.serve_tokens.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_compact_positions_must_match_tokens(stream):
    """SPEC: a step with tokens but no logprob rows misaligns the rows; the
    compact response fails (500 / error chunk) instead of rendering."""
    ids, _, _ = _engine_rows(0, 1, 4)
    chunks = [(ids, None, None), _engine_rows(1, 2, 4)]
    feeder = _OutputProcessorEngine(chunks)
    request = _request(logprobs=3, logprobs_format="compact", stream=stream)
    if not stream:
        with pytest.raises(GenerationError, match="2 logprob positions for 3"):
            await _serving(feeder).serve_tokens(request)
        return
    generator = await _serving(feeder).serve_tokens(request)
    events = _parse_sse_chunks([chunk async for chunk in generator])
    data = [e for e in events if isinstance(e, dict)]
    assert "error" in data[0]
    assert "0 logprob positions for 1" in json.dumps(data[0])


@pytest.mark.asyncio
async def test_compact_stream_logprob_token_ids_only_and_null_logprobs():
    """SPEC: blocks are emitted iff logprobs or logprob_token_ids are
    requested (S = len(ids)+1); stream choices keep "logprobs": null."""
    chunks = [_engine_rows(0, 2, 3), _engine_rows(2, 1, 3)]
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serving(feeder).serve_tokens(
        _request(
            logprobs=None,
            logprobs_format="compact",
            stream=True,
            sampling={"logprob_token_ids": [5, 9]},
        )
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    data = [e for e in events if isinstance(e, dict) and e["choices"]]
    assert len(data) == 2
    for event, chunk in zip(data, chunks):
        choice = event["choices"][0]
        assert "logprobs" in choice and choice["logprobs"] is None
        assert choice["compact_logprobs"]["num_slots"] == 3
        np.testing.assert_array_equal(_decode(choice["compact_logprobs"])[0], chunk[0])
    # Neither logprobs nor logprob_token_ids: no compact key at all.
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs=None, logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    for event in events:
        if isinstance(event, dict):
            for choice in event["choices"]:
                assert "compact_logprobs" not in choice


@pytest.mark.asyncio
async def test_compact_zero_token_abort_stream_vs_full():
    """Zero-token abort: the stream emits no chunk carrying a compact key
    (zero-token deltas are skipped, as on the base path); the full response
    carries an empty block."""
    feeder = _OutputProcessorEngine([])
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert events == ["[DONE]"]
    feeder = _OutputProcessorEngine([])
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format="compact")
    )
    block = json.loads(response.body)["choices"][0]["compact_logprobs"]
    assert block["num_positions"] == 0 and block["num_slots"] == 4
    assert block["token_ids"] == block["logprobs"] == block["ranks"] == ""


# ------------------------------------------------- round-3 audit regressions


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_step", [1, 2])
async def test_compact_stream_step_without_rows_is_an_error(missing_step):
    """Claude r3 #1: a step with a token but no logprob rows must not make a
    later chunk silently carry an earlier token's row."""
    chunks: list[Any] = [
        _engine_rows(0, 2, 4),
        _engine_rows(2, 1, 4),
        _engine_rows(3, 2, 4),
    ]
    ids = chunks[missing_step][0]
    chunks[missing_step] = (ids, None, None)
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    data = [e for e in events if isinstance(e, dict)]
    for event, chunk in zip(data[:missing_step], chunks):
        np.testing.assert_array_equal(
            _decode(event["choices"][0]["compact_logprobs"])[0], chunk[0]
        )
    assert "error" in data[missing_step]
    assert "generated tokens" in json.dumps(data[missing_step])
    for event in data:
        for choice in event.get("choices", []):
            block = choice.get("compact_logprobs")
            if block:
                assert _decode(block)[0][:, 0].tolist() == choice["token_ids"]


def test_delta_slices_record_source_positions():
    token_ids, logprobs, ranks = _engine_rows(0, 5, 4)
    container = ArrayLogprobs()
    container.append_rows(token_ids, logprobs, ranks)
    tail = container[-2:]
    assert tail.source_positions == 5
    merged = container[-3:-2]
    merged.extend(tail)
    assert merged.source_positions == 5 and len(merged) == 3
    assert container.source_positions is None


def _gzip_client(parts, minimum_size=500):
    from starlette.middleware.gzip import GZipMiddleware

    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False, tokens_only=False)
    serving = MagicMock()

    async def _serve(request, raw_request):
        return RenderedGenerateResponse(list(parts))

    serving.serve_tokens = _serve
    app.state.serving_tokens = serving
    init_exception_handler(app)
    api_router.attach_router(app)
    app.add_middleware(GZipMiddleware, minimum_size=minimum_size)
    return TestClient(app)


def test_small_rendered_body_behaves_like_json_response_under_gzip():
    """Claude/Codex r3 #2: a body up to 1 MiB is a single message, so
    GZipMiddleware keeps Content-Length and minimum_size like JSONResponse."""
    parts = [b'{"choices":[', memoryview(b'{"index":0}'), b"]}"]
    with _gzip_client(parts) as client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {}},
            headers={"accept-encoding": "gzip"},
        )
    assert result.content == b"".join(parts)
    assert "content-encoding" not in result.headers
    assert int(result.headers["content-length"]) == len(result.content)


def test_large_rendered_body_under_gzip_decodes():
    big = [b'{"a":"' + b"x" * (3 << 20), b'"}']
    with _gzip_client(big) as client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {}},
            headers={"accept-encoding": "gzip"},
        )
    assert result.content == b"".join(big)
    assert result.headers["content-encoding"] == "gzip"


@pytest.mark.asyncio
async def test_rendered_response_messages_and_single_use():
    rendered = RenderedGenerateResponse([b"x" * (3 << 20), b"tail"])
    response = api_router._RenderedJSONResponse(rendered)
    messages: list[dict] = []

    async def send(message):
        messages.append(message)

    await response({"type": "http"}, None, send)
    bodies = [m for m in messages if m["type"] == "http.response.body"]
    assert [m["more_body"] for m in bodies] == [True, False]
    assert bodies[-1]["body"] == b"tail"
    assert b"".join(m["body"] for m in bodies) == b"x" * (3 << 20) + b"tail"
    with pytest.raises(RuntimeError, match="only be sent once"):
        await response({"type": "http"}, None, send)
    messages.clear()
    small = api_router._RenderedJSONResponse(RenderedGenerateResponse([b"{", b"}"]))
    await small({"type": "http"}, None, send)
    assert [m.get("more_body") for m in messages[1:]] == [False]
    messages.clear()
    empty = api_router._RenderedJSONResponse(RenderedGenerateResponse([]))
    await empty({"type": "http"}, None, send)
    assert messages[1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }


def test_log_response_logs_whole_multipart_body(monkeypatch):
    """Codex r3 #2: the debug response logger joins all body parts."""
    from vllm.entrypoints.serve.middleware import log_response as log_mod

    logged: list[str] = []
    monkeypatch.setattr(
        log_mod.logger, "info", lambda fmt, *args: logged.append(fmt % args)
    )
    log_mod._log_non_streaming_response([b'{"choices":[', b'{"token_ids":[1]}]}'])
    assert logged == ['response_body={{"choices":[{"token_ids":[1]}]}}']


def test_wire_mode_refuses_float64_and_stays_lossless():
    """Claude r3 #4: float64 rows leave wire mode (lossless array storage);
    decoded wire rows are writable."""
    token_ids, logprobs, ranks = _engine_rows(0, 4, 3)
    wire = ArrayLogprobs(wire_base64=True)
    wire.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    values64 = logprobs[2:].astype(np.float64)
    values64[0, 1] = -1e-50
    wire.append_rows(token_ids[2:], values64, ranks[2:])
    assert wire.wire_parts() is None
    assert wire[2][int(token_ids[2, 1])].logprob == -1e-50
    _, values, _ = wire.arrays()
    assert values.dtype == np.float64 and values.flags.writeable
    first = ArrayLogprobs(wire_base64=True)
    first.append_rows(token_ids[:1], logprobs[:1].astype(np.float64), ranks[:1])
    assert first.wire_parts() is None


def test_lead_table_warm_lookups_do_not_take_the_lock():
    """Claude r3 #5: warm lookups never wait on the lock."""
    table = logprobs_render._LeadTable(b"")
    ids = np.array([[1, 2], [3, 4]])
    table.lookup(ids)
    with table.lock:
        result = table.lookup(ids)  # would deadlock if it took the lock
    assert result[1, 1] == b'{"token":"token_id:4","logprob":'


@pytest.mark.parametrize("size", [1, 8, 9, 129])
def test_storage_int_dtype_fast_paths(size):
    from vllm import logprobs as logprobs_mod

    check = logprobs_mod._storage_int_dtype
    for dtype in (np.int8, np.int16, np.int32, np.uint8, np.uint16):
        assert check(np.ones(size, dtype=dtype)) == np.dtype("<i4")
    values = np.zeros(size, dtype=np.int64)
    assert check(values) == np.dtype("<i4")
    values[-1] = 2**31
    assert check(values) == np.dtype("<i8")
    values[-1] = -(2**31) - 1
    assert check(values) == np.dtype("<i8")
    values[-1] = -(2**31)
    assert check(values) == np.dtype("<i4")
    assert check(np.full(size, 2**31 - 1, dtype=np.uint32)) == np.dtype("<i4")
    assert check(np.full(size, 2**31, dtype=np.uint32)) == np.dtype("<i8")


def test_tail_fast_path_still_widens(monkeypatch):
    token_ids, logprobs, ranks = _engine_rows(0, 3, 4)
    container = ArrayLogprobs()
    container.append_rows(token_ids[:1].astype(np.int32), logprobs[:1], ranks[:1])
    container.append_rows(token_ids[1:2].astype(np.int32), logprobs[1:2], ranks[1:2])
    wide = token_ids[2:3].copy()
    wide[0, 1] = 2**33
    container.append_rows(wide, logprobs[2:3], ranks[2:3])
    assert container[2][2**33].logprob == float(logprobs[2, 1])
    assert container.arrays()[0].dtype == np.int64


def test_openai_render_allocates_few_gc_tracked_objects():
    """Round 4 A: rendering must not allocate per-row GC-tracked containers
    (nested lists triggered hundreds of collections per request; their
    pauses hold the GIL and showed up as ~0.16-0.2 s event-loop stalls)."""
    import gc

    token_ids, logprobs, ranks = _engine_rows(0, 16384, 9)
    container, _ = _containers(token_ids, logprobs, ranks, 8)
    sampled = token_ids[:, 0].tolist()
    render_openai_logprobs(sampled, container, 8)  # warm the lead tables
    collections: list[int] = []

    def callback(phase, info):
        if phase == "start":
            collections.append(info["generation"])

    gc.collect()
    gc.callbacks.append(callback)
    try:
        render_openai_logprobs(sampled, container, 8)
    finally:
        gc.callbacks.remove(callback)
    assert len(collections) <= 2, collections


# ------------------------------------------------- round 5: SPEC v3 and R3


def _decode_block(block: dict):
    n, s = block["num_positions"], block["num_slots"]
    ids = np.frombuffer(base64.b64decode(block["token_ids"]), "<i4")
    lps = np.frombuffer(base64.b64decode(block["logprobs"]), "<f4")
    return ids.reshape(n, s), lps.reshape(n, s)


@pytest.mark.parametrize("field", ["compact_include_sampled", "compact_include_ranks"])
def test_compact_switches_require_compact(field):
    """SPEC v3: the switches are only valid with logprobs_format=compact."""
    assert (
        getattr(_request(logprobs_format="compact", **{field: False}), field) is False
    )
    assert getattr(_request(**{field: True}), field) is True  # default value ok
    with pytest.raises(ValidationError, match="require"):
        _request(**{field: False})
    assert field not in _request().model_dump()
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False, tokens_only=False)
    app.state.serving_tokens = MagicMock()
    init_exception_handler(app)
    api_router.attach_router(app)
    with TestClient(app) as client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {"logprobs": 1}, field: False},
        )
    assert result.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sampled", [True, False])
@pytest.mark.parametrize("ranks", [True, False])
async def test_compact_include_switches(stream, sampled, ranks):
    """SPEC v3: include_sampled=false -> top-k slots only, num_slots=k,
    "sampled_slot": false; include_ranks=false -> no ranks key. Defaults keep
    today's bytes (no new keys)."""
    chunks = [_engine_rows(0, 3, 5), _engine_rows(3, 2, 5)]
    chunks[0][1][1, 2] = np.float32(-np.inf)
    feeder = _OutputProcessorEngine(chunks)
    extra = {}
    if not sampled:
        extra["compact_include_sampled"] = False
    if not ranks:
        extra["compact_include_ranks"] = False
    request = _request(logprobs=4, logprobs_format="compact", stream=stream, **extra)
    serving = _serving(feeder)
    result = await serving.serve_tokens(request)
    if stream:
        events = _parse_sse_chunks([chunk async for chunk in result])
        blocks = [
            e["choices"][0]["compact_logprobs"]
            for e in events
            if isinstance(e, dict) and e["choices"]
        ]
    else:
        assert feeder.sampling_params.array_logprobs_wire_topk_only is (not sampled)
        blocks = [json.loads(result.body)["choices"][0]["compact_logprobs"]]
    ids = np.concatenate([_decode_block(b)[0] for b in blocks])
    lps = np.concatenate([_decode_block(b)[1] for b in blocks])
    all_ids, all_lps, all_ranks = _expected(chunks, 5)
    first = 0 if sampled else 1
    np.testing.assert_array_equal(ids, all_ids[:, first:])
    assert lps.tobytes() == np.ascontiguousarray(all_lps[:, first:]).tobytes()
    for block in blocks:
        assert block["num_slots"] == (5 if sampled else 4)
        if sampled:
            assert "sampled_slot" not in block
        else:
            assert block["sampled_slot"] is False
            assert list(block).index("sampled_slot") == 5
        assert ("ranks" in block) is ranks
    if ranks:
        got = np.concatenate(
            [np.frombuffer(base64.b64decode(b["ranks"]), "<i4") for b in blocks]
        )
        np.testing.assert_array_equal(got, all_ranks)


@pytest.mark.parametrize("include_sampled", [True, False])
def test_wire_topk_only_matches_array_render_and_decodes(include_sampled):
    token_ids, logprobs, ranks = _engine_rows(0, 9, 5)
    wire = ArrayLogprobs(wire_base64=True, wire_topk_only=True)
    plain = ArrayLogprobs()
    for i in range(0, 9, 2):
        wire.append_rows(token_ids[i : i + 2], logprobs[i : i + 2], ranks[i : i + 2])
        plain.append_rows(token_ids[i : i + 2], logprobs[i : i + 2], ranks[i : i + 2])
    for include_ranks in (True, False):
        assert render_compact_logprobs(
            wire, 4, include_sampled=include_sampled, include_ranks=include_ranks
        ) == render_compact_logprobs(
            plain, 4, include_sampled=include_sampled, include_ranks=include_ranks
        )
    # Lossless decode of the top-k-only wire container (slot 0 side streams).
    wire2 = ArrayLogprobs(wire_base64=True, wire_topk_only=True)
    wire2.append_rows(token_ids, logprobs, ranks)
    ids, values, rk = wire2.arrays()
    np.testing.assert_array_equal(ids, token_ids)
    assert values.tobytes() == logprobs.tobytes()
    np.testing.assert_array_equal(rk, ranks)


@pytest.mark.asyncio
async def test_compact_topk_only_narrow_rows_still_500():
    chunks = [_engine_rows(0, 3, 2)]
    feeder = _OutputProcessorEngine(chunks)
    with pytest.raises(GenerationError, match="2 slots, expected 6"):
        await _serving(feeder).serve_tokens(
            _request(
                logprobs=5, logprobs_format="compact", compact_include_sampled=False
            )
        )


def _r3_chunks(layers, topk, dtype, sizes, seed=0):
    rng = np.random.default_rng(seed)
    high = 256 if dtype == np.uint8 else 1000
    return [rng.integers(0, high, size=(n, layers, topk)).astype(dtype) for n in sizes]


def _r3_request_state():
    params = SamplingParams(max_tokens=100, logprobs=2)
    params.output_kind = RequestOutputKind.FINAL_ONLY
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=params,
        pooling_params=None,
    )
    queue = RequestOutputCollector(params.output_kind, "r")
    processor.add_request(request, None, queue=queue)
    return processor, queue


@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_routed_experts_survive_abort(mixed_dtype):
    """R3 is returned on abort (FINAL_ONLY), byte-identical to base."""
    from vllm.utils.serial_utils import numpy2base64

    processor, queue = _r3_request_state()
    chunks = _r3_chunks(4, 8, np.uint8, [3, 1, 1])
    if mixed_dtype:
        chunks[2] = chunks[2].astype(np.uint16)
    token_ids, logprobs, ranks = _engine_rows(0, 3, 3)
    for i, chunk in enumerate(chunks):
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="r",
                    new_token_ids=[int(token_ids[i, 0])],
                    new_logprobs=LogprobsLists(
                        token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
                    ),
                    routed_experts=chunk,
                )
            ]
        )
    processor.abort_requests(["r"], internal=True)
    output = queue.get_nowait().outputs[0]
    expected = numpy2base64(np.concatenate(chunks, axis=0))
    assert numpy2base64(output.routed_experts) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["openai", "compact"])
async def test_routed_experts_in_full_responses(fmt):
    """R3 is present in default and compact responses (legacy chunk path,
    spliced as a JSON fragment); null without any forward."""
    from vllm.utils.serial_utils import numpy2base64

    chunks = [_engine_rows(0, 2, 4), _engine_rows(2, 1, 4)]
    routed = _r3_chunks(5, 8, np.uint8, [4, 1])

    feeder = _OutputProcessorEngine(chunks, routed=routed)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format=fmt)
    )
    choice = json.loads(response.body)["choices"][0]
    assert choice["routed_experts"] == numpy2base64(np.concatenate(routed))
    feeder = _OutputProcessorEngine([])
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format=fmt)
    )
    body = response.body if hasattr(response, "parts") else None
    dump = json.loads(body) if body else response.model_dump()
    assert dump["choices"][0]["routed_experts"] is None
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format=fmt, stream=True)
    )
    _ = [chunk async for chunk in generator]


def test_fragments_spliced_in_document_order():
    content = {"choices": [{"a": None, "b": None, "c": 1}]}
    body = render_json_with_fragments(
        content, {0: {"b": b'"B"', "a": [b'"', b"A", b'"']}}
    )
    assert body == b'{"choices":[{"a":"A","b":"B","c":1}]}'


def test_wire_layout_defaults():
    """The wire container encodes full rows unless asked for top-k only (a
    property named like the InitVar once shadowed its default)."""
    assert not ArrayLogprobs(wire_base64=True).is_wire_topk_only
    assert ArrayLogprobs(wire_base64=True, wire_topk_only=True).is_wire_topk_only
    assert not ArrayLogprobs().is_wire_topk_only


# --------------------------------------------- round 4+5 audit regressions


def _two_request_processor(params_a: SamplingParams, params_b: SamplingParams):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    queues = {}
    for rid, params in (("a", params_a), ("b", params_b)):
        params.output_kind = RequestOutputKind.FINAL_ONLY
        request = EngineCoreRequest(
            request_id=rid,
            external_req_id=rid,
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            sampling_params=params,
            pooling_params=None,
        )
        queues[rid] = RequestOutputCollector(params.output_kind, rid)
        processor.add_request(request, None, queue=queues[rid])
    return processor, queues


def _step(rid: str, token_ids, logprobs, ranks, routed=None):
    return EngineCoreOutput(
        request_id=rid,
        new_token_ids=token_ids[:, 0].tolist(),
        new_logprobs=LogprobsLists(token_ids, logprobs, ranks),
        routed_experts=routed,
    )


def test_topk_only_with_zero_logprobs_does_not_crash_process_outputs():
    """r45 BLOCKER: compact + include_sampled=false + logprobs=0 wrote a
    (n, 0) array; memoryview.cast raised inside process_outputs, killing the
    output handler for every request."""
    compact = SamplingParams(
        max_tokens=10,
        logprobs=0,
        array_logprobs=True,
        array_logprobs_base64=True,
        array_logprobs_wire_topk_only=True,
    )
    other = SamplingParams(max_tokens=10, logprobs=2, array_logprobs=True)
    processor, queues = _two_request_processor(compact, other)
    ids1, lps1, ranks1 = _engine_rows(0, 2, 1)
    ids3, lps3, ranks3 = _engine_rows(0, 2, 3)
    for i in range(2):
        processor.process_outputs(
            [
                _step("a", ids1[i : i + 1], lps1[i : i + 1], ranks1[i : i + 1]),
                _step("b", ids3[i : i + 1], lps3[i : i + 1], ranks3[i : i + 1]),
            ]
        )
    processor.abort_requests(["a", "b"], internal=True)
    out_a = queues["a"].get_nowait().outputs[0]
    out_b = queues["b"].get_nowait().outputs[0]
    block = json.loads(
        render_compact_logprobs(out_a.logprobs, 0, include_sampled=False)
    )
    assert block["num_positions"] == 2 and block["num_slots"] == 0
    assert block["token_ids"] == block["logprobs"] == ""
    assert block["sampled_slot"] is False
    plain = ArrayLogprobs()
    plain.append_rows(ids1, lps1, ranks1)
    assert render_compact_logprobs(
        out_a.logprobs, 0, include_sampled=False
    ) == render_compact_logprobs(plain, 0, include_sampled=False)
    assert len(out_b.logprobs) == 2


@pytest.mark.asyncio
async def test_topk_only_zero_logprobs_full_request():
    chunks = [_engine_rows(0, 2, 1), _engine_rows(2, 1, 1)]
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=0, logprobs_format="compact", compact_include_sampled=False)
    )
    block = json.loads(response.body)["choices"][0]["compact_logprobs"]
    assert block["num_positions"] == 3 and block["num_slots"] == 0


def test_encoder_failures_fail_only_their_request(monkeypatch):
    """Frontend-only storage never raises out of process_outputs: a failing
    wire encode fails that request's compact render (500); others are
    unaffected."""
    from vllm import logprobs as logprobs_mod

    compact = SamplingParams(
        max_tokens=10, logprobs=2, array_logprobs=True, array_logprobs_base64=True
    )
    other = SamplingParams(max_tokens=10, logprobs=2, array_logprobs=True)
    processor, queues = _two_request_processor(compact, other)
    ids, lps, ranks = _engine_rows(0, 3, 3)
    processor.process_outputs([_step("a", ids[:1], lps[:1], ranks[:1])])

    def boom(self, data):
        raise RuntimeError("encoder failure")

    monkeypatch.setattr(logprobs_mod._Base64Stream, "write", boom)
    processor.process_outputs(
        [
            _step("a", ids[1:2], lps[1:2], ranks[1:2]),
            _step("b", ids[1:2], lps[1:2], ranks[1:2]),
        ]
    )
    monkeypatch.undo()
    processor.abort_requests(["a", "b"], internal=True)
    out_a = queues["a"].get_nowait().outputs[0]
    out_b = queues["b"].get_nowait().outputs[0]
    assert out_a.logprobs.broken and len(out_a.logprobs) == 2
    with pytest.raises(GenerationError, match="encoding failed"):
        render_compact_logprobs(out_a.logprobs, 2)
    assert out_b.finish_reason == "abort"
    assert len(out_b.logprobs) == 1


@pytest.mark.parametrize("failing", ["decode"])
def test_unwire_failure_is_contained(monkeypatch, failing):
    """Codex r6 #1: a failure while leaving wire mode (float64 fallback) is
    contained: the request's storage is discarded and marked broken, and the
    rest of the batch is processed."""
    from vllm import logprobs as logprobs_mod

    compact = SamplingParams(
        max_tokens=10, logprobs=2, array_logprobs=True, array_logprobs_base64=True
    )
    other = SamplingParams(max_tokens=10, logprobs=2, array_logprobs=True)
    processor, queues = _two_request_processor(compact, other)
    ids, lps, ranks = _engine_rows(0, 3, 3)
    processor.process_outputs([_step("a", ids[:1], lps[:1], ranks[:1])])

    def boom(*args, **kwargs):
        raise MemoryError("injected")

    monkeypatch.setattr(logprobs_mod._WireEncoder, "decode", boom)
    processor.process_outputs(
        [
            _step("a", ids[1:2], lps[1:2].astype(np.float64), ranks[1:2]),
            _step("b", ids[1:2], lps[1:2], ranks[1:2]),
        ]
    )
    monkeypatch.undo()
    processor.process_outputs(
        [
            _step("a", ids[2:3], lps[2:3], ranks[2:3]),
            _step("b", ids[2:3], lps[2:3], ranks[2:3]),
        ]
    )
    processor.abort_requests(["a", "b"], internal=True)
    out_a = queues["a"].get_nowait().outputs[0]
    out_b = queues["b"].get_nowait().outputs[0]
    assert out_a.logprobs.broken and len(out_a.logprobs) == 3
    with pytest.raises(GenerationError):
        render_compact_logprobs(out_a.logprobs, 2)
    assert len(out_b.logprobs) == 2
    np.testing.assert_array_equal(out_b.logprobs.arrays()[0], ids[1:3])


def test_broken_container_semantics():
    """Claude r6 NIT 2: a broken container raises a clear error on access,
    and slices/merges stay broken (no exception in DELTA slicing)."""
    token_ids, logprobs, ranks = _engine_rows(0, 3, 3)
    container = ArrayLogprobs()
    container.append_rows(token_ids, logprobs, ranks)
    container.mark_broken()
    container.num_positions = 3
    with pytest.raises(ValueError, match="broken"):
        container[0]
    with pytest.raises(ValueError, match="broken"):
        list(container)
    with pytest.raises(ValueError, match="broken"):
        container.arrays()
    tail = container[-2:]
    assert tail.broken and len(tail) == 2 and not tail.is_regular
    merged = ArrayLogprobs()
    merged.append_rows(token_ids[:1], logprobs[:1], ranks[:1])
    merged.extend(tail)
    assert merged.broken and len(merged) == 3
    container.append_rows(token_ids[:1], logprobs[:1], ranks[:1])
    assert len(container) == 4


@pytest.mark.parametrize("wire", [False, True])
def test_non_native_byte_order_keeps_precision(wire):
    """Codex r6 #3: storage dtype by kind and size, not equality with native
    dtypes (``>f8`` used to be stored as float32)."""
    token_ids, logprobs, ranks = _engine_rows(0, 3, 4)
    values = logprobs.astype(np.float64)
    values[1, 2] = -0.12345678901234566
    container = ArrayLogprobs(wire_base64=wire)
    container.append_rows(
        token_ids.astype(">i8"), values.astype(">f8"), ranks.astype(">i8")
    )
    native, legacy = _containers(token_ids, values, ranks, 3)
    assert container[1][int(token_ids[1, 2])].logprob == -0.12345678901234566
    assert list(container) == legacy
    sampled = token_ids[:, 0].tolist()
    assert render_openai_logprobs(sampled, container, 3) == _legacy_logprobs_bytes(
        sampled, legacy, 3
    )
    stored = container.arrays()
    assert stored[1].dtype == np.dtype("<f8") and stored[0].dtype.isnative
    # Big-endian float32 stays in wire mode and encodes like native rows.
    wire32 = ArrayLogprobs(wire_base64=True)
    wire32.append_rows(token_ids.astype(">i4"), logprobs.astype(">f4"), ranks)
    assert wire32.wire_parts() is not None
    plain = ArrayLogprobs()
    plain.append_rows(token_ids, logprobs, ranks)
    assert render_compact_logprobs(wire32, 3) == render_compact_logprobs(plain, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["openai", "compact"])
async def test_default_format_is_sent_as_one_message(fmt):
    """Round 7 decision: the default format is always one body message (like
    JSONResponse, joined off-loop); multi-part only for compact."""
    chunks = [_engine_rows(0, 3, 4)]
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format=fmt)
    )
    assert isinstance(response, RenderedGenerateResponse)
    if fmt == "openai":
        assert response.single_message and len(response.parts) == 1
    else:
        assert not response.single_message and len(response.parts) > 1
    big = [b'{"a":"' + b"x" * (3 << 20), b'"}']
    client = _gzip_client(big)

    async def _serve(request, raw_request):
        return RenderedGenerateResponse(list(big), single_message=fmt == "openai")

    client.app.state.serving_tokens.serve_tokens = _serve
    with client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {}},
            headers={"accept-encoding": "gzip"},
        )
    assert result.content == b"".join(big)
    assert ("content-length" in result.headers) is (fmt == "openai")


def test_completion_output_positional_fields_unchanged():
    """Codex r45 #2: new fields are appended, so positional construction
    matches the base signature."""
    import dataclasses

    from vllm.outputs import CompletionOutput

    base = [
        "index",
        "text",
        "token_ids",
        "cumulative_logprob",
        "logprobs",
        "routed_experts",
        "finish_reason",
        "stop_reason",
        "lora_request",
        "sampling_mask",
        "spec_decode_metrics",
    ]
    names = [f.name for f in dataclasses.fields(CompletionOutput)]
    assert names[: len(base)] == base
    output = CompletionOutput(0, "", [1], None, None, None, "stop")
    assert output.finish_reason == "stop" and output.finished()


def test_lead_table_reader_sees_consistent_snapshot():
    """Claude r45 #2: a growth between the warm reader's loads must not pair
    old values with new flags (None leads)."""

    class Interleaved(logprobs_render._LeadTable):
        armed = False

        @property
        def _table(self):
            table = self.__dict__["_table"]
            if self.armed:
                self.armed = False
                # Another thread grows the table and fills id 5 right after
                # this reader loaded the (old) snapshot.
                logprobs_render._LeadTable.lookup(self, np.array([5, 40]))
            return table

        @_table.setter
        def _table(self, value):
            self.__dict__["_table"] = value

        # The pre-fix reader loaded ``filled`` separately (after ``values``):
        # interleave the same concurrent growth there too, so the test also
        # reproduces the race on that implementation.
        @property
        def filled(self):
            filled = self.__dict__.get("filled")
            if filled is None:
                return self._table[1]
            if self.armed:
                self.armed = False
                logprobs_render._LeadTable.lookup(self, np.array([5, 40]))
                return self.__dict__["filled"]
            return filled

        @filled.setter
        def filled(self, value):
            self.__dict__["filled"] = value

    table = Interleaved(b"")
    table.lookup(np.array([1, 2, 7]))
    table.armed = True
    result = table.lookup(np.array([5, 1]))
    assert result.tolist() == [
        b'{"token":"token_id:5","logprob":',
        b'{"token":"token_id:1","logprob":',
    ]


def test_uint64_beyond_int64_is_never_wrapped():
    """Codex r45 #3: uint64 ids/ranks beyond int64 are kept exactly (legacy
    positions) and compact rejects them, in array and wire mode."""
    token_ids, logprobs, ranks = _engine_rows(0, 2, 3)
    big = token_ids.astype(np.uint64)
    big[1, 2] = np.uint64(2**64 - 1)
    for wire in (False, True):
        container = ArrayLogprobs(wire_base64=wire)
        container.append_rows(big[:1], logprobs[:1], ranks[:1])
        container.append_rows(big[1:], logprobs[1:], ranks[1:])
        assert container[1][2**64 - 1].logprob == float(logprobs[1, 2])
        with pytest.raises(GenerationError):
            render_compact_logprobs(container, 2)
    ranks_big = ranks.astype(np.uint64)
    ranks_big[1] = np.uint64(2**63)  # row 1: sampled id not in its top-k
    container = ArrayLogprobs()
    container.append_rows(token_ids, logprobs, ranks_big)
    assert container[1][int(token_ids[1, 0])].rank == 2**63


def test_user_middleware_gets_single_message_bodies():
    """Codex r45 #5: with user --middleware (e.g. gzip), the body is one
    message like JSONResponse, so GZipMiddleware keeps Content-Length."""
    big = [b'{"a":"' + b"x" * (3 << 20), b'"}']
    client = _gzip_client(big)
    client.app.state.args.middleware = ["some.module.GZip"]
    with client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {}},
            headers={"accept-encoding": "gzip"},
        )
    assert result.content == b"".join(big)
    assert result.headers["content-encoding"] == "gzip"
    assert "content-length" in result.headers


@pytest.mark.asyncio
async def test_single_use_guard_before_response_start():
    response = api_router._RenderedJSONResponse(RenderedGenerateResponse([b"{}"]))
    messages: list[dict] = []

    async def send(message):
        messages.append(message)

    await response({"type": "http"}, None, send)
    messages.clear()
    with pytest.raises(RuntimeError, match="only be sent once"):
        await response({"type": "http"}, None, send)
    assert messages == []


# ------------------------------------------------ round-7 audit regressions


def _delta_two_request_processor():
    params = {
        rid: SamplingParams(
            max_tokens=10,
            logprobs=2,
            array_logprobs=True,
            array_logprobs_base64=False,
        )
        for rid in ("a", "b")
    }
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    queues = {}
    for rid, sp in params.items():
        sp.output_kind = RequestOutputKind.DELTA
        request = EngineCoreRequest(
            request_id=rid,
            external_req_id=rid,
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            sampling_params=sp,
            pooling_params=None,
        )
        queues[rid] = RequestOutputCollector(sp.output_kind, rid)
        processor.add_request(request, None, queue=queues[rid])
    return processor, queues


def test_delta_slicing_failure_is_contained(monkeypatch):
    """Codex r7 #1: a failure while slicing the DELTA output (inside
    RequestState._new_completion_output) fails only that request; the
    co-batched request is processed."""
    from vllm import logprobs as logprobs_mod

    processor, queues = _delta_two_request_processor()
    ids, lps, ranks = _engine_rows(0, 2, 3)
    original = logprobs_mod.ArrayLogprobs._slice
    calls = {"n": 0}

    def failing_slice(self, index):
        calls["n"] += 1
        if calls["n"] == 1:  # request "a" (first in the batch)
            raise MemoryError("injected")
        return original(self, index)

    monkeypatch.setattr(logprobs_mod.ArrayLogprobs, "_slice", failing_slice)
    processor.process_outputs(
        [
            _step("a", ids[:1], lps[:1], ranks[:1]),
            _step("b", ids[:1], lps[:1], ranks[:1]),
        ]
    )
    monkeypatch.undo()
    out_a = queues["a"].get_nowait().outputs[0]
    out_b = queues["b"].get_nowait().outputs[0]
    assert out_a.logprobs.broken and len(out_a.logprobs) == 1
    with pytest.raises(GenerationError):
        render_compact_logprobs(out_a.logprobs, 2)
    assert not out_b.logprobs.broken
    np.testing.assert_array_equal(out_b.logprobs.arrays()[0], ids[:1])


def test_delta_merge_failure_is_contained(monkeypatch):
    """Codex r7 #2: a failure while merging queued DELTA outputs
    (RequestOutputCollector.put -> ArrayLogprobs.extend) fails only that
    request; the co-batched request is merged normally."""
    from vllm import logprobs as logprobs_mod

    processor, queues = _delta_two_request_processor()
    ids, lps, ranks = _engine_rows(0, 3, 3)
    # Two steps without get(): the second output is merged into the first.
    processor.process_outputs(
        [
            _step("a", ids[:1], lps[:1], ranks[:1]),
            _step("b", ids[:1], lps[:1], ranks[:1]),
        ]
    )
    original = logprobs_mod.ArrayLogprobs._filled_blocks
    calls = {"n": 0}

    def failing_blocks(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("injected")
        return original(self)

    monkeypatch.setattr(logprobs_mod.ArrayLogprobs, "_filled_blocks", failing_blocks)
    processor.process_outputs(
        [
            _step("a", ids[1:2], lps[1:2], ranks[1:2]),
            _step("b", ids[1:2], lps[1:2], ranks[1:2]),
        ]
    )
    monkeypatch.undo()
    out_a = queues["a"].get_nowait().outputs[0]
    out_b = queues["b"].get_nowait().outputs[0]
    assert out_a.logprobs.broken and len(out_a.logprobs) == 2
    assert list(out_a.token_ids) == ids[:2, 0].tolist()
    assert not out_b.logprobs.broken
    np.testing.assert_array_equal(out_b.logprobs.arrays()[0], ids[:2])


def test_unwire_failure_keeps_wire_data():
    """A failed decode leaves the wire storage intact (no silent loss)."""
    from unittest.mock import patch

    from vllm import logprobs as logprobs_mod

    token_ids, logprobs, ranks = _engine_rows(0, 3, 3)
    container = ArrayLogprobs(wire_base64=True)
    container.append_rows(token_ids, logprobs, ranks)
    with (
        patch.object(logprobs_mod._WireEncoder, "decode", side_effect=MemoryError("x")),
        pytest.raises(MemoryError),
    ):
        container._unwire()
    assert container.wire_parts() is not None
    np.testing.assert_array_equal(container.arrays()[0], token_ids)


@pytest.mark.parametrize("broken", [False, True])
def test_stepped_slices_raise(broken):
    """Kimi r7 NIT: stepped slices raise, also for broken containers (no
    misleading length)."""
    token_ids, logprobs, ranks = _engine_rows(0, 4, 3)
    container = ArrayLogprobs()
    container.append_rows(token_ids, logprobs, ranks)
    if broken:
        container.mark_broken()
        container.num_positions = 4
    with pytest.raises(ValueError, match="contiguous"):
        container[::2]
    assert len(container[1:3]) == 2


# --------------------------------------------- Claude round-7 audit items


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["openai", "compact"])
async def test_rendered_parts_are_exact_bytes(fmt):
    """Claude r7 #1: every body part is exact bytes, so the off-loop join of
    the default body releases the GIL (a memoryview part made bytes.join hold
    it for the whole 2.2 GB copy: ~217 ms event-loop stalls)."""
    chunks = [_engine_rows(0, 3, 4)]
    feeder = _OutputProcessorEngine(chunks)
    response = await _serving(feeder).serve_tokens(
        _request(logprobs=3, logprobs_format=fmt)
    )
    assert all(type(part) is bytes for part in response.parts)
    parts = render_json_with_fragments_parts(
        {"choices": [{"a": None, "b": 1}]}, {0: {"a": [b"1", b"2"]}}
    )
    assert all(type(part) is bytes for part in parts)
    assert logprobs_render.join_parts([b"x", memoryview(b"yz")]) == b"xyz"


def test_second_failure_on_broken_container_keeps_counting(monkeypatch):
    """Claude r7 NIT 2: positions keep counting on an already broken
    container."""
    request = MagicMock(spec=EngineCoreRequest)
    request.sampling_params = SamplingParams(logprobs=2, array_logprobs=True)
    processor = LogprobsProcessor.from_new_request(None, request)

    def boom(*args, **kwargs):
        raise MemoryError("injected")

    monkeypatch.setattr(LogprobsProcessor, "_append_array_logprobs", boom)
    for step in range(3):
        ids, lps, ranks = _engine_rows(step * 2, 2, 3)
        processor._update_sample_logprobs(LogprobsLists(ids, lps, ranks))
    assert processor.logprobs.broken and len(processor.logprobs) == 6


def test_compact_with_user_middleware_joins_off_loop(monkeypatch):
    """Claude r7 NIT 3: a large compact + --middleware body is joined in the
    builder thread, not on the event loop."""
    threads: list[Any] = []
    real_join = logprobs_render.join_parts

    def recording_join(parts):
        threads.append(threading.current_thread().name)
        return real_join(parts)

    monkeypatch.setattr(api_router, "join_parts", recording_join)
    parts = [b'{"a":"', b"x" * (api_router.INLINE_JOIN_MAX_BYTES + 1), b'"}']
    client = _gzip_client(parts)
    client.app.state.args.middleware = ["some.module.Middleware"]
    with client:
        result = client.post(
            "/inference/v1/generate",
            json={"token_ids": [1], "sampling_params": {}},
        )
    assert result.content == b"".join(parts)
    assert int(result.headers["content-length"]) == len(result.content)
    assert len(threads) == 1
    assert threads[0].startswith("generate-response")


def test_small_compact_with_user_middleware_skips_busy_builder():
    """Claude r9 MINOR: a small compact + --middleware body is joined inline,
    so it does not wait behind an unrelated build occupying the single
    response-builder thread."""
    release = threading.Event()
    busy = api_router.serving_module._RESPONSE_BUILDER.submit(release.wait, 10)
    try:
        parts = [b'{"a":"', b"x" * 10, b'"}']
        client = _gzip_client(parts)
        client.app.state.args.middleware = ["some.module.Middleware"]
        with client:
            t0 = time.perf_counter()
            result = client.post(
                "/inference/v1/generate",
                json={"token_ids": [1], "sampling_params": {}},
            )
            elapsed = time.perf_counter() - t0
        assert not busy.done()
        assert result.content == b"".join(parts)
        assert int(result.headers["content-length"]) == len(result.content)
        assert elapsed < 5
    finally:
        release.set()
        busy.result(timeout=30)


@pytest.mark.parametrize("broken_side", ["target", "source"])
def test_merge_with_broken_side_copies_nothing(monkeypatch, broken_side):
    """Claude r8 NIT: merging into/from a broken container only counts
    positions; nothing is unwired, snapshotted or re-attached."""
    token_ids, logprobs, ranks = _engine_rows(0, 4, 3)
    target = ArrayLogprobs()
    target.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    source = ArrayLogprobs(wire_base64=True)
    source.append_rows(token_ids[2:], logprobs[2:], ranks[2:])
    source._append_legacy([{1: Logprob(-1.0)}])
    (target if broken_side == "target" else source).mark_broken()
    if broken_side == "target":
        target.num_positions = 2
    else:
        source.num_positions = 3
    calls: list[str] = []
    monkeypatch.setattr(ArrayLogprobs, "_unwire", lambda self: calls.append("unwire"))

    def record_blocks(self):
        calls.append("blocks")
        return []

    monkeypatch.setattr(ArrayLogprobs, "_filled_blocks", record_blocks)
    target.extend(source)
    assert calls == []
    assert target.broken and len(target) == 5
    assert target._legacy is None and target.token_id_chunks == []
