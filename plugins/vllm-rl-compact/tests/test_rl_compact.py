# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The compact block (``logprobs_format="compact"``) through the real
OutputProcessor, the core handle and the plugin's response build."""

import json
import threading
import time
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pybase64 as base64
import pytest
import torch
import vllm_rl_compact.serving as serving_mod
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vllm_rl_compact import RLCompactPlugin
from vllm_rl_compact import router as plugin_router
from vllm_rl_compact.protocol import CompactGenerateRequest
from vllm_rl_compact.serving import CompactServingTokens

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _mock_engine,
)
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.scale_out.token_in_token_out import api_router
from vllm.entrypoints.scale_out.token_in_token_out import serving as core_serving
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.logprobs import set_sample_logprobs_container
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

RLCompactPlugin()  # registers the container


def _engine_rows(start: int, count: int, width: int):
    """Engine-like rows: slot 0 = sampled, slots 1.. = distinct top ids; even
    positions repeat the sampled id inside the top-k."""
    rng = np.random.default_rng(start)
    top = np.stack(
        [rng.choice(50_000, size=width - 1, replace=False) for _ in range(count)]
    ).reshape(count, width - 1)
    positions = np.arange(start, start + count)
    sampled = (
        50_000 + positions
        if width == 1
        else np.where(positions % 2 == 0, top[:, 0], 50_000 + positions)
    )
    token_ids = np.column_stack((sampled, top)).astype(np.int32)
    logprobs = (-rng.random((count, width)) * 20).astype(np.float32)
    return token_ids, logprobs, rng.integers(1, 1000, size=count)


class _Engine:
    """Feeds engine rows (or None: a step without rows) through the real
    OutputProcessor, then aborts or finishes the request."""

    def __init__(self, chunks, finish=FinishReason.ABORT, tokenizer=None):
        self.chunks, self.finish, self.tokenizer = chunks, finish, tokenizer

    def generate(self, engine_input, sampling_params, request_id, **kwargs):
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
            prompt_k = sampling_params.prompt_logprobs
            for i, (token_ids, logprobs, ranks) in enumerate(self.chunks):
                processor.process_outputs(
                    [
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=token_ids[:, 0].tolist(),
                            new_logprobs=None
                            if logprobs is None or sampling_params.num_logprobs is None
                            else LogprobsLists(token_ids, logprobs, ranks),
                            new_prompt_logprobs_tensors=(
                                LogprobsTensors(
                                    torch.tensor([[2, 1001], [3, 5000]]),
                                    torch.tensor([[-0.5, -1.5], [-0.25, -3.0]]),
                                    torch.tensor([4, 1]),
                                )
                                if i == 0 and prompt_k is not None
                                else None
                            ),
                        )
                    ]
                )
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
            yield queue.get_nowait()

        return _gen()


def _request(logprobs: int | None = 3, **kwargs) -> CompactGenerateRequest:
    sampling = {"max_tokens": 100, "logprobs": logprobs, **kwargs.pop("sampling", {})}
    return CompactGenerateRequest.model_validate(
        {
            "token_ids": [1, 2, 3],
            "sampling_params": sampling,
            "logprobs_format": "compact",
            **kwargs,
        }
    )


async def _serve(chunks, request, tokenizer=None):
    engine = _mock_engine()
    engine.generate = MagicMock(
        side_effect=_Engine(chunks, tokenizer=tokenizer).generate
    )
    set_sample_logprobs_container(
        request.sampling_params, serving_mod.COMPACT_CONTAINER
    )
    return await CompactServingTokens(_build_serving_tokens(engine)).serve_tokens(
        request
    )


def _decode(block: dict):
    n, k = block["num_positions"], block["num_slots"]
    ids = np.frombuffer(base64.b64decode(block["token_ids"]), "<i4")
    lps = np.frombuffer(base64.b64decode(block["logprobs"]), "<f4")
    return ids.reshape(n, k), lps.reshape(n, k)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_width", [4, 6])  # 6: rows padded to a larger k
async def test_abort_with_partial_output(engine_width):
    """All accumulated rows, bit-exact (non-finite values and NaN payloads
    included), in the documented key order."""
    chunks = [_engine_rows(0, 5, engine_width), _engine_rows(5, 3, engine_width)]
    chunks[0][1][0, 1] = np.float32(-np.inf)
    chunks[1][1][2, 2] = np.array([0x7FC00123], dtype=np.uint32).view(np.float32)[0]
    response = await _serve(chunks, _request())
    (choice,) = json.loads(b"".join(response))["choices"]
    assert choice["logprobs"] is None and choice["finish_reason"] == "abort"
    block = choice["compact_logprobs"]
    assert list(block) == [
        "num_positions",
        "num_slots",
        "dtype_token_ids",
        "dtype_logprobs",
        "byteorder",
        "token_ids",
        "logprobs",
    ]
    ids, lps = _decode(block)
    all_ids = np.concatenate([c[0] for c in chunks])
    all_lps = np.concatenate([c[1] for c in chunks])
    assert choice["token_ids"] == all_ids[:, 0].tolist()
    np.testing.assert_array_equal(ids, all_ids[:, 1:4])
    assert lps.tobytes() == np.ascontiguousarray(all_lps[:, 1:4]).tobytes()


@pytest.mark.asyncio
async def test_abort_before_any_output():
    response = await _serve([], _request(logprobs=0))
    choice = json.loads(b"".join(response))["choices"][0]
    assert choice["token_ids"] == []
    assert choice["compact_logprobs"]["num_positions"] == 0
    assert choice["compact_logprobs"]["token_ids"] == ""


@pytest.mark.asyncio
async def test_logprob_token_ids_without_logprobs():
    chunks = [_engine_rows(0, 3, 3)]
    response = await _serve(
        chunks, _request(logprobs=None, sampling={"logprob_token_ids": [5, 9]})
    )
    block = json.loads(b"".join(response))["choices"][0]["compact_logprobs"]
    assert block["num_slots"] == 2
    np.testing.assert_array_equal(_decode(block)[0], chunks[0][0][:, 1:])


@pytest.mark.asyncio
async def test_without_logprobs_there_is_no_block():
    response = await _serve([_engine_rows(0, 2, 4)], _request(logprobs=None))
    assert isinstance(response, GenerateResponse)
    assert "compact_logprobs" not in response.model_dump()["choices"][0]


@pytest.mark.asyncio
async def test_prompt_logprobs_keep_decoded_tokens():
    tokenizer = get_tokenizer(MODEL_NAME)
    response = await _serve(
        [_engine_rows(0, 2, 3)],
        _request(logprobs=2, sampling={"prompt_logprobs": 1}),
        tokenizer=tokenizer,
    )
    prompt = json.loads(b"".join(response))["prompt_logprobs"]
    assert prompt[1]["2"]["decoded_token"] == tokenizer.decode([2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks,k,match",
    [
        ([_engine_rows(0, 3, 6), _engine_rows(3, 2, 2)], 5, "inconsistent widths"),
        ([_engine_rows(0, 3, 2)], 5, "2 slots, expected 6"),
        (
            [(_engine_rows(0, 1, 4)[0], None, None), _engine_rows(1, 2, 4)],
            3,
            "2 logprob positions for 3",
        ),
        ([_engine_rows(0, 2, 3)[:2] + (np.array([1]),)], 2, "encoding failed"),
    ],
    ids=["irregular", "narrow", "step without rows", "row counts"],
)
async def test_rows_the_format_cannot_represent_fail_the_request(chunks, k, match):
    with pytest.raises(GenerationError, match=match):
        await _serve(chunks, _request(logprobs=k))


@pytest.mark.parametrize(
    "body,status",
    [
        ({"sampling_params": {"logprobs": -1}, "logprobs_format": "compact"}, 400),
        (
            {
                "sampling_params": {"logprobs": 1},
                "logprobs_format": "compact",
                "stream": True,
            },
            400,
        ),
        ({"sampling_params": {}, "logprobs_format": "msgpack"}, 400),
    ],
)
def test_requests_the_format_does_not_serve_are_rejected(body, status):
    with pytest.raises(ValidationError):
        CompactGenerateRequest.model_validate({"token_ids": [1], **body})
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False, tokens_only=False)
    app.state.serving_tokens = MagicMock()
    init_exception_handler(app)
    api_router.attach_router(app)
    plugin_router.attach_router(app)
    with TestClient(app) as client:
        result = client.post("/inference/v1/generate", json={"token_ids": [1], **body})
    assert result.status_code == status


@pytest.mark.asyncio
async def test_large_responses_are_built_off_the_event_loop(monkeypatch):
    threads: list[Any] = []
    real = CompactServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        threads.append(threading.current_thread().name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CompactServingTokens, "_build_full_response", recording)
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    bodies = []
    for limit in (16, 1 << 40):  # 4 positions x 4 slots = 16 entries
        monkeypatch.setattr(core_serving, "OFFLOAD_MIN_LOGPROB_ENTRIES", limit)
        response = await _serve([_engine_rows(0, 4, 4)], _request(request_id="r"))
        bodies.append(b"".join(response))
    assert threads[0] != threading.current_thread().name
    assert threads[1] == threading.current_thread().name
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("flush", [3, 7, 64, 1 << 20])
@pytest.mark.parametrize("step", [1, 3, 1024])
def test_rows_encoded_as_they_arrive_equal_one_shot_encoding(monkeypatch, flush, step):
    from vllm_rl_compact import storage

    monkeypatch.setattr(storage._Base64Stream, "FLUSH_BYTES", flush)
    token_ids, logprobs, ranks = _engine_rows(0, 50, 5)
    wire = storage.WireLogprobs()
    for i in range(0, 50, step):
        wire.append_rows(
            token_ids[i : i + step], logprobs[i : i + step], ranks[i : i + step]
        )
    assert b"".join(wire.token_ids.parts()) == base64.b64encode(
        np.ascontiguousarray(token_ids[:, 1:])
    )
    assert b"".join(wire.logprobs.parts()) == base64.b64encode(
        np.ascontiguousarray(logprobs[:, 1:])
    )
