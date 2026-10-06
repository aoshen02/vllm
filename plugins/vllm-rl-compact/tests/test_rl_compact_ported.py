# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact-format and ``ArrayLogprobs`` storage tests of the in-tree implementation
(214f522fbf ``tests/entrypoints/scale_out/token_in_token_out/test_logprobs_format.py``),
ported to the design-C ``vllm_rl_compact`` plugin through a thin adapter layer.
Default-format fast-path (T3) tests are not part of this plugin."""

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
import vllm_rl_compact.storage as logprobs_mod
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from vllm_rl_compact import register_containers
from vllm_rl_compact import router as plugin_router
from vllm_rl_compact.protocol import CompactGenerateRequest as GenerateRequest
from vllm_rl_compact.protocol import (
    CompactLogprobs,
    CompactResponseChoice,
    RenderedCompactResponse,
)
from vllm_rl_compact.protocol import CompactStreamChoice as GenerateResponseStreamChoice
from vllm_rl_compact.protocol import CompactStreamResponse as GenerateStreamResponse
from vllm_rl_compact.render import (
    render_compact_logprobs,
    render_compact_logprobs_parts,
)
from vllm_rl_compact.serving import (
    ARRAY_CONTAINER,
    WIRE_CONTAINER,
)
from vllm_rl_compact.serving import CompactServingTokens as ServingTokens
from vllm_rl_compact.storage import ArrayLogprobs

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _mock_engine,
    _parse_sse_chunks,
)
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.scale_out.token_in_token_out import api_router
from vllm.entrypoints.scale_out.token_in_token_out import serving as core_serving
from vllm.entrypoints.scale_out.token_in_token_out.logprobs_render import (
    render_json_with_fragments,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateResponse,
    GenerateResponseChoice,
    RenderedGenerateResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import (
    ServingTokens as CoreServingTokens,
)
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.logprobs import (
    Logprob,
    SampleLogprobsHandle,
    append_logprobs_for_next_position,
    set_sample_logprobs_container,
)
from vllm.sampling_params import RequestOutputKind
from vllm.sampling_params import SamplingParams as CoreSamplingParams
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

register_containers()


# ---------------------------------------------------------------- adapters
def SamplingParams(  # noqa: N802 (constructor shim: in-tree flag names)
    *args,
    array_logprobs=False,
    array_logprobs_base64=False,
    **kwargs,
):
    params = CoreSamplingParams(*args, **kwargs)
    if array_logprobs_base64:
        set_sample_logprobs_container(params, WIRE_CONTAINER)
    elif array_logprobs:
        set_sample_logprobs_container(params, ARRAY_CONTAINER)
    return params


def _container_of(params):
    return params._sample_logprobs_container


def _is_wire(params) -> bool:
    return _container_of(params) == WIRE_CONTAINER


def _attach(app):
    core = getattr(app.state, "serving_tokens", None)
    if isinstance(core, ServingTokens):  # a plugin handler from _serving()
        core = app.state.serving_tokens = core.core
    api_router.attach_router(app)
    plugin_router.attach_router(app)
    middleware = getattr(getattr(app.state, "args", None), "middleware", None)
    app.state.rl_compact_serving = (
        ServingTokens.from_core(core, user_middleware=bool(middleware))
        if isinstance(core, CoreServingTokens)
        else core
    )


async def _serve_compact(serving, request):
    """Compact requests are served by the plugin route, which selects the
    container server-side before calling serve_tokens."""
    set_sample_logprobs_container(
        request.sampling_params, serving_mod.compact_container(request)
    )
    return await serving.serve_tokens(request)


def _inner(logprobs):
    """Core hands sample logprobs out as SampleLogprobsHandle :
    the plugin's ArrayLogprobs behind it."""
    if type(logprobs) is SampleLogprobsHandle:
        return None if logprobs.broken else logprobs.unwrap()
    return logprobs


def _broken(logprobs):
    inner = _inner(logprobs)
    return inner is None or bool(getattr(inner, "broken", False))


async def _serve_tokens_any(serving, request, *args):
    """Like the plugin route: compact requests get their container and the
    plugin handler; others the unchanged core handler."""
    if getattr(request, "logprobs_format", "openai") == "compact":
        set_sample_logprobs_container(
            request.sampling_params, serving_mod.compact_container(request)
        )
        return await serving.serve_tokens(request, *args)
    return await serving.core.serve_tokens(request, *args)


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
    return ServingTokens.from_core(_build_serving_tokens(engine))


def _request(logprobs: int | None = 3, **kwargs) -> GenerateRequest:
    sampling = {"max_tokens": 100, "logprobs": logprobs}
    sampling.update(kwargs.pop("sampling", {}))
    return GenerateRequest.model_validate(
        {"token_ids": [1, 2, 3], "sampling_params": sampling, **kwargs}
    )


def _decode(block: dict):
    """(top-k ids [N, k], top-k logprobs [N, k]) of a compact block (the one
    format: no sampled slot, no ranks)."""
    assert "sampled_slot" not in block and "ranks" not in block
    n, k = block["num_positions"], block["num_slots"]
    token_ids = np.frombuffer(base64.b64decode(block["token_ids"]), "<i4")
    logprobs = np.frombuffer(base64.b64decode(block["logprobs"]), "<f4")
    return token_ids.reshape(n, k), logprobs.reshape(n, k)


def _expected(chunks, num_slots):
    """(top-k ids, top-k logprobs, sampled ids) of the engine rows truncated
    to ``num_slots`` = k + 1."""
    token_ids = np.concatenate([c[0] for c in chunks])[:, :num_slots]
    logprobs = np.concatenate([c[1] for c in chunks])[:, :num_slots]
    return (
        np.ascontiguousarray(token_ids[:, 1:]).astype("<i4"),
        np.ascontiguousarray(logprobs[:, 1:]).astype("<f4"),
        token_ids[:, 0].tolist(),
    )


def test_logprobs_format_defaults_and_rejects_unknown():
    assert _request().logprobs_format == "openai"
    assert _request(logprobs_format="compact").logprobs_format == "compact"
    with pytest.raises(ValidationError):
        _request(logprobs_format="msgpack")


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
    ]


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
    assert isinstance(_inner(array.logprobs), ArrayLogprobs)
    assert len(array.logprobs) == len(flat.logprobs) == 5
    assert list(_inner(array.logprobs)) == list(flat.logprobs)
    assert array.cumulative_logprob == flat.cumulative_logprob
    array_rows = _inner(array.logprobs)
    slots = width if num_logprobs == -1 else num_logprobs + 1
    assert array_rows.num_slots == slots


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

    response = await _serve_tokens_any(serving, _request(logprobs_format="compact"))

    assert _container_of(feeder.sampling_params) is not None
    assert feeder.sampling_params.detokenize is True
    assert isinstance(response, RenderedCompactResponse)
    data = json.loads(response.body)
    (choice,) = data["choices"]
    assert choice["logprobs"] is None
    assert choice["finish_reason"] == "abort"
    expected_ids, expected_lps, sampled = _expected(chunks, 4)
    assert choice["token_ids"] == sampled
    block = choice["compact_logprobs"]
    assert block["num_positions"] == 8 and block["num_slots"] == 3
    assert block["dtype_token_ids"] == "int32"
    assert block["dtype_logprobs"] == "float32"
    assert block["byteorder"] == "little"
    ids, lps = _decode(block)
    np.testing.assert_array_equal(ids, expected_ids)
    assert lps.tobytes() == expected_lps.tobytes()
    assert data["usage"]["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_compact_abort_before_any_output():
    feeder = _OutputProcessorEngine([])
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=0, logprobs_format="compact")
    )
    choice = json.loads(response.body)["choices"][0]
    assert choice["token_ids"] == []
    assert choice["compact_logprobs"] == {
        "num_positions": 0,
        "num_slots": 0,
        "dtype_token_ids": "int32",
        "dtype_logprobs": "float32",
        "byteorder": "little",
        "token_ids": "",
        "logprobs": "",
    }


@pytest.mark.asyncio
async def test_compact_keeps_detokenizer_for_stop_strings():
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)])
    await _serve_tokens_any(
        _serving(feeder), _request(logprobs_format="compact", sampling={"stop": ["x"]})
    )
    assert _container_of(feeder.sampling_params) is not None
    assert feeder.sampling_params.detokenize is True


@pytest.mark.asyncio
async def test_compact_without_logprobs_has_no_block():
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)])
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=None, logprobs_format="compact")
    )
    assert isinstance(response, GenerateResponse)
    assert "compact_logprobs" not in response.model_dump()["choices"][0]


@pytest.mark.asyncio
async def test_default_stream_ignores_client_array_flag():
    chunks = [_engine_rows(0, 3, 4)]
    feeder = _OutputProcessorEngine(chunks, finish=FinishReason.LENGTH)
    generator = await _serve_tokens_any(
        _serving(feeder), _request(stream=True, sampling={"array_logprobs": True})
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert _container_of(feeder.sampling_params) is None
    assert feeder.sampling_params.detokenize is True
    content = events[0]["choices"][0]["logprobs"]["content"]
    assert [c["token"] for c in content] == [
        f"token_id:{t}" for t in chunks[0][0][:, 0].tolist()
    ]


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


@pytest.mark.asyncio
async def test_compact_stream_emits_per_chunk_blocks():
    chunks = [_engine_rows(0, 3, 5), _engine_rows(3, 4, 5)]
    chunks[1][1][1, 1] = np.float32(np.nan)
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serve_tokens_any(
        _serving(feeder), _request(logprobs_format="compact", stream=True)
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
    expected_ids, expected_lps, _ = _expected(chunks, 4)
    np.testing.assert_array_equal(np.concatenate([d[0] for d in decoded]), expected_ids)
    assert np.concatenate([d[1] for d in decoded]).tobytes() == expected_lps.tobytes()


@pytest.mark.asyncio
async def test_default_stream_has_no_compact_key():
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)])
    generator = await _serve_tokens_any(_serving(feeder), _request(stream=True))
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
    _attach(app)
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


@pytest.mark.asyncio
async def test_compact_with_prompt_logprobs_keeps_decoded_tokens(gpt2_tokenizer):
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 3)], tokenizer=gpt2_tokenizer)
    response = await _serve_tokens_any(
        _serving(feeder),
        _request(
            logprobs=2, logprobs_format="compact", sampling={"prompt_logprobs": 1}
        ),
    )
    assert feeder.sampling_params.detokenize is True
    prompt = json.loads(response.body)["prompt_logprobs"]
    assert prompt[1]["2"]["decoded_token"] == gpt2_tokenizer.decode([2])


def test_array_logprobs_short_wide_request_reserves_little():
    """A 1-row wide request must not reserve thousands of rows."""
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
    """DELTA ``[-1:]`` slices are exact-sized views."""
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
    """A last-row slice touches only its own block."""
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


@pytest.mark.parametrize("consolidate", [False, True])
def test_array_logprobs_self_extend_doubles(consolidate):
    """``c.extend(c)`` terminates and doubles, like a list."""
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
    """Array_logprobs is frontend-only; the request sent to
    EngineCore after add_request does not carry it."""
    params = SamplingParams(
        logprobs=2,
        array_logprobs=True,
        array_logprobs_base64=True,
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
    assert isinstance(_inner(state.logprobs_processor.logprobs), ArrayLogprobs)
    assert _inner(state.logprobs_processor.logprobs).wire_parts() is not None
    assert b"rl_compact" not in MsgpackEncoder().encode(request)[0]
    assert _container_of(request.sampling_params) is None
    # The caller's (possibly shared) params object is not mutated.
    assert _container_of(params) == WIRE_CONTAINER


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
    """Narrow or varying engine rows must not raise inside
    OutputProcessor (that kills AsyncLLM's output handler); the stored
    positions equal the legacy list path's."""
    chunks = _narrow_steps(k, widths)
    array = _process(k, chunks, True, kind)
    legacy = _process(k, chunks, False, kind)
    array = _inner(array)
    assert isinstance(array, ArrayLogprobs)
    assert list(array) == list(legacy)
    assert array.is_regular == (len(set(widths)) == 1 or (k >= 0 and min(widths) > k))


@pytest.mark.asyncio
async def test_irregular_rows_compact_is_a_request_error():
    chunks = [_engine_rows(0, 3, 6), _engine_rows(3, 2, 2)]
    feeder = _OutputProcessorEngine(chunks)
    with pytest.raises(GenerationError, match="inconsistent widths"):
        await _serve_tokens_any(
            _serving(feeder), _request(logprobs=5, logprobs_format="compact")
        )
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=5, logprobs_format="compact", stream=True)
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
    """Render endpoints serialize GenerateRequest; the
    opt-in field must not appear unless set."""
    request = _request()
    assert "logprobs_format" not in request.model_dump()
    assert "logprobs_format" not in json.loads(request.model_dump_json())
    compact = _request(logprobs_format="compact")
    assert compact.model_dump()["logprobs_format"] == "compact"
    restored = GenerateRequest.model_validate_json(compact.model_dump_json())
    assert restored.logprobs_format == "compact"


@pytest.mark.asyncio
# 4 positions x (3 + 1) slots = 16 entries: the threshold boundary checks the
# entry count itself (through the core handle), not just the offload switch.
@pytest.mark.parametrize("threshold,offloaded", [(16, True), (17, False)])
@pytest.mark.parametrize("fmt", ["compact"])
async def test_large_responses_are_built_off_the_event_loop(
    monkeypatch, threshold, offloaded, fmt
):
    """Large array-logprob responses are built in a worker thread so the
    event loop keeps serving other requests; the body is unchanged."""

    threads: list[Any] = []
    original = ServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ServingTokens, "_build_full_response", recording)
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    bodies = []
    for limit in (threshold, 1 << 40):
        monkeypatch.setattr(core_serving, "OFFLOAD_MIN_LOGPROB_ENTRIES", limit)
        feeder = _OutputProcessorEngine([_engine_rows(0, 4, 4)])
        response = await _serve_tokens_any(
            _serving(feeder),
            _request(logprobs=3, request_id="fixed-id", logprobs_format=fmt),
        )
        bodies.append(response.body)
    assert (threads[0] != threading.get_ident()) is offloaded
    assert threads[1] == threading.get_ident()
    assert bodies[0] == bodies[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["compact"])
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
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=2, logprobs_format=fmt, sampling=sampling)
    )
    assert feeder.sampling_params.detokenize is True
    assert isinstance(feeder.detokenizer, IncrementalDetokenizer)
    assert isinstance(feeder.detokenizer, BaseIncrementalDetokenizer) == bool(stop)
    if prompt_logprobs is not None:
        prompt = json.loads(response.body)["prompt_logprobs"]
        assert prompt[1]["2"]["decoded_token"] == gpt2_tokenizer.decode([2])


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
    ids, lps = _decode(block)
    np.testing.assert_array_equal(ids, token_ids[:, 1:])
    assert lps.tobytes() == np.ascontiguousarray(logprobs[:, 1:]).tobytes()
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
    """Rows of another width leave wire mode (irregular), ids beyond int32
    break the container; compact then fails the request (500) exactly as
    without wire mode, and positions stay exact."""
    token_ids, logprobs, ranks = _engine_rows(0, 4, 4)
    wire = _wire_container([(token_ids[:2], logprobs[:2], ranks[:2])])
    if case == "width":
        wire.append_rows(token_ids[2:, :3], logprobs[2:, :3], ranks[2:])
        assert not wire.is_regular and not wire.broken
    else:
        big = token_ids[2:].astype(np.int64)
        big[0, 1] = 2**31 + 7
        wire.append_rows(big, logprobs[2:], ranks[2:])
        assert wire.broken
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
        await _serve_tokens_any(_serving(feeder), request.model_copy(deep=True))
    ).body
    assert _is_wire(feeder.sampling_params)
    monkeypatch.setattr(serving_mod, "compact_container", lambda req: ARRAY_CONTAINER)
    feeder = _OutputProcessorEngine(chunks)
    plain_body = (
        await _serve_tokens_any(_serving(feeder), request.model_copy(deep=True))
    ).body
    assert not _is_wire(feeder.sampling_params)
    assert wire_body == plain_body


@pytest.mark.asyncio
async def test_compact_stream_does_not_use_wire_mode():
    feeder = _OutputProcessorEngine([_engine_rows(0, 3, 4)])
    generator = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    _ = [chunk async for chunk in generator]
    assert not _is_wire(feeder.sampling_params)


def _single_row_appends(container, token_ids, logprobs, ranks, count):
    for i in range(count):
        container.append_rows(
            token_ids[i : i + 1], logprobs[i : i + 1], ranks[i : i + 1]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_compact_uniformly_narrow_rows_are_a_request_error(stream):
    """Every row narrower than k+1 (co-batched
    logprob_token_ids) is not representable: 500, not a num_slots=2 block."""
    chunks = [_engine_rows(0, 3, 2), _engine_rows(3, 2, 2)]
    feeder = _OutputProcessorEngine(chunks)
    request = _request(logprobs=5, logprobs_format="compact", stream=stream)
    if not stream:
        with pytest.raises(GenerationError, match="2 slots, expected 6"):
            await _serve_tokens_any(_serving(feeder), request)
        return
    generator = await _serve_tokens_any(_serving(feeder), request)
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
    response = await _serve_tokens_any(
        _serving(feeder),
        _request(
            logprobs=None,
            logprobs_format="compact",
            sampling={"logprob_token_ids": [5, 9]},
        ),
    )
    choice = json.loads(response.body)["choices"][0]
    assert choice["logprobs"] is None
    assert choice["compact_logprobs"]["num_slots"] == 2
    np.testing.assert_array_equal(
        _decode(choice["compact_logprobs"])[0], chunks[0][0][:, 1:]
    )
    feeder = _OutputProcessorEngine(chunks)
    response = await _serve_tokens_any(
        _serving(feeder),
        _request(logprobs=None, sampling={"logprob_token_ids": [5, 9]}),
    )
    dump = response.model_dump()["choices"][0]
    assert dump["logprobs"] is None and "compact_logprobs" not in dump


def test_irregular_fallback_is_linear(monkeypatch):
    """Legacy materialization visits each block once."""
    monkeypatch.setattr(ArrayLogprobs, "BLOCK_BYTES", 2 * (3 * 8 + 4))
    token_ids, logprobs, ranks = _engine_rows(0, 64, 3)
    container = ArrayLogprobs()
    _single_row_appends(container, token_ids, logprobs, ranks, 64)
    container.append_rows(token_ids[:1, :2], logprobs[:1, :2], ranks[:1])
    assert not container.is_regular
    visits: list[int] = []
    original = ArrayLogprobs._filled_blocks

    def counting(self):
        for block in original(self):
            visits.append(len(block[2]))
            yield block

    monkeypatch.setattr(ArrayLogprobs, "_filled_blocks", counting)
    positions = list(container)
    assert len(positions) == 65
    # One pass: each block once (64 array rows, then the legacy row).
    assert len(visits) == len(container.rank_chunks) and sum(visits) == 64


def test_generate_request_schema_keeps_all_fields():
    """Excluding the default logprobs_format must not empty the
    serialization schema (it is the render endpoints' response model)."""
    for mode in ("serialization", "validation"):
        props = GenerateRequest.model_json_schema(mode=mode)["properties"]
        assert set(GenerateRequest.model_fields) <= set(props) | {
            name
            for name, f in GenerateRequest.model_fields.items()
            if f.alias and f.alias in props
        }
        assert "logprobs_format" in props
    for model in (CompactResponseChoice, GenerateResponseStreamChoice):
        props = model.model_json_schema(mode="serialization")["properties"]
        assert "compact_logprobs" in props and "token_ids" in props


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
    _attach(app)
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
            await _serve_tokens_any(_serving(feeder), request)
        return
    generator = await _serve_tokens_any(_serving(feeder), request)
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
    generator = await _serve_tokens_any(
        _serving(feeder),
        _request(
            logprobs=None,
            logprobs_format="compact",
            stream=True,
            sampling={"logprob_token_ids": [5, 9]},
        ),
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    data = [e for e in events if isinstance(e, dict) and e["choices"]]
    assert len(data) == 2
    for event, chunk in zip(data, chunks):
        choice = event["choices"][0]
        assert "logprobs" in choice and choice["logprobs"] is None
        assert choice["compact_logprobs"]["num_slots"] == 2
        np.testing.assert_array_equal(
            _decode(choice["compact_logprobs"])[0], chunk[0][:, 1:]
        )
    # Neither logprobs nor logprob_token_ids: no compact key at all.
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serve_tokens_any(
        _serving(feeder),
        _request(logprobs=None, logprobs_format="compact", stream=True),
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
    generator = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    assert events == ["[DONE]"]
    feeder = _OutputProcessorEngine([])
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format="compact")
    )
    block = json.loads(response.body)["choices"][0]["compact_logprobs"]
    assert block["num_positions"] == 0 and block["num_slots"] == 3
    assert block["token_ids"] == block["logprobs"] == "" and "ranks" not in block


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_step", [1, 2])
async def test_compact_stream_step_without_rows_is_an_error(missing_step):
    """A step with a token but no logprob rows must not make a
    later chunk silently carry an earlier token's row."""
    chunks: list[Any] = [
        _engine_rows(0, 2, 4),
        _engine_rows(2, 1, 4),
        _engine_rows(3, 2, 4),
    ]
    ids = chunks[missing_step][0]
    chunks[missing_step] = (ids, None, None)
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    events = _parse_sse_chunks([chunk async for chunk in generator])
    data = [e for e in events if isinstance(e, dict)]
    for event, chunk in zip(data[:missing_step], chunks):
        np.testing.assert_array_equal(
            _decode(event["choices"][0]["compact_logprobs"])[0], chunk[0][:, 1:]
        )
    assert "error" in data[missing_step]
    assert "generated tokens" in json.dumps(data[missing_step])
    for event in data:
        for choice in event.get("choices", []):
            block = choice.get("compact_logprobs")
            if block:
                assert len(_decode(block)[0]) == len(choice["token_ids"])


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


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["compact"])
async def test_routed_experts_in_full_responses(fmt):
    """R3 is present in default and compact responses (legacy chunk path);
    null without any forward."""
    from vllm.utils.serial_utils import numpy2base64

    chunks = [_engine_rows(0, 2, 4), _engine_rows(2, 1, 4)]
    routed = _r3_chunks(5, 8, np.uint8, [4, 1])

    feeder = _OutputProcessorEngine(chunks, routed=routed)
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format=fmt)
    )
    choice = json.loads(response.body)["choices"][0]
    assert choice["routed_experts"] == numpy2base64(np.concatenate(routed))
    feeder = _OutputProcessorEngine([])
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format=fmt)
    )
    body = response.body if hasattr(response, "parts") else None
    dump = json.loads(body) if body else response.model_dump()
    assert dump["choices"][0]["routed_experts"] is None
    feeder = _OutputProcessorEngine(chunks)
    generator = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format=fmt, stream=True)
    )
    _ = [chunk async for chunk in generator]


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


@pytest.mark.parametrize("wire", [False, True])
def test_row_count_mismatch_fails_only_its_request(wire):
    """Fewer token-id rows than ranks (here 1 vs 2, with two
    logprob rows) were broadcast by numpy: the default render repeated rows
    and compact advertised more entries than it carried. Now the container
    is broken (request-local 500); the co-batched request is unaffected."""
    params = SamplingParams(
        max_tokens=10, logprobs=2, array_logprobs=True, array_logprobs_base64=wire
    )
    other = SamplingParams(max_tokens=10, logprobs=2, array_logprobs=True)
    processor, queues = _two_request_processor(params, other)
    ids, lps, ranks = _engine_rows(0, 2, 3)
    bad = EngineCoreOutput(
        request_id="a",
        new_token_ids=[5, 5],
        new_logprobs=LogprobsLists(
            np.array([[5, 6, 7]], dtype=np.int64),
            np.array([[-0.5, -1.0, -2.0], [-0.25, -1.5, -3.0]], dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
        ),
    )
    processor.process_outputs([bad, _step("b", ids, lps, ranks)])
    processor.abort_requests(["a", "b"], internal=True)
    out_a = queues["a"].get_nowait().outputs[0]
    out_b = queues["b"].get_nowait().outputs[0]
    assert out_a.token_ids == [5, 5]
    assert _broken(out_a.logprobs) and len(out_a.logprobs) == 2
    with pytest.raises(GenerationError, match="encoding failed"):
        render_compact_logprobs(out_a.logprobs, 2)
    assert not _broken(out_b.logprobs) and len(out_b.logprobs) == 2
    np.testing.assert_array_equal(_inner(out_b.logprobs).arrays()[0], ids)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 3), (2, 3), (2,)),  # fewer id rows
        ((2, 3), (1, 3), (2,)),  # fewer logprob rows
        ((2, 3), (2, 2), (2,)),  # width mismatch
        ((2, 3), (2, 3), (2, 1)),  # 2-D ranks
        ((3,), (3,), (1,)),  # 1-D rows
    ],
)
def test_append_rows_rejects_inconsistent_shapes(wire, shapes):
    id_shape, lp_shape, rank_shape = shapes
    container = ArrayLogprobs(wire_base64=wire)
    ok_ids, ok_lps, ok_ranks = _engine_rows(0, 1, 3)
    container.append_rows(ok_ids, ok_lps, ok_ranks)
    container.append_rows(
        np.ones(id_shape, dtype=np.int64),
        -np.ones(lp_shape, dtype=np.float32),
        np.ones(rank_shape, dtype=np.int64),
    )
    assert container.broken
    assert len(container) == 1 + rank_shape[0]


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("bad", ["list_ids", "list_ranks", "zero_d_ranks"])
def test_append_rows_non_array_inputs_break_the_container(wire, bad):
    """List inputs fail with a clear error, and a 0-d
    ranks array (len() raises) is contained too, never raised."""
    container = ArrayLogprobs(wire_base64=wire)
    ids, lps, ranks = _engine_rows(0, 2, 3)
    container.append_rows(ids[:1], lps[:1], ranks[:1])
    if bad == "list_ids":
        container.append_rows(ids[1:].tolist(), lps[1:], ranks[1:])
        assert len(container) == 2
    elif bad == "list_ranks":
        container.append_rows(ids[1:], lps[1:], ranks[1:].tolist())
        assert len(container) == 2
    else:
        container.append_rows(ids[1:], lps[1:], np.array(3))
        assert len(container) == 1  # positions unknown
    assert container.broken


def test_logprobs_processor_contains_zero_d_ranks():
    request = MagicMock(spec=EngineCoreRequest)
    request.sampling_params = SamplingParams(logprobs=2, array_logprobs=True)
    processor = LogprobsProcessor.from_new_request(None, request)
    ids, lps, ranks = _engine_rows(0, 1, 3)
    processor._update_sample_logprobs(LogprobsLists(ids, lps, ranks))
    processor._update_sample_logprobs(LogprobsLists(ids, lps, np.array(1)))
    assert _broken(processor.logprobs) and len(processor.logprobs) == 1


def test_topk_only_with_zero_logprobs_does_not_crash_process_outputs():
    """Compact (top-k only) + logprobs=0 wrote a (n, 0) array;
    memoryview.cast raised inside process_outputs, killing the output handler
    for every request."""
    compact = SamplingParams(
        max_tokens=10,
        logprobs=0,
        array_logprobs=True,
        array_logprobs_base64=True,
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
    block = json.loads(render_compact_logprobs(out_a.logprobs, 0))
    assert block["num_positions"] == 2 and block["num_slots"] == 0
    assert block["token_ids"] == block["logprobs"] == ""
    plain = ArrayLogprobs()
    plain.append_rows(ids1, lps1, ranks1)
    assert render_compact_logprobs(out_a.logprobs, 0) == render_compact_logprobs(
        plain, 0
    )
    assert len(out_b.logprobs) == 2


@pytest.mark.asyncio
async def test_topk_only_zero_logprobs_full_request():
    chunks = [_engine_rows(0, 2, 1), _engine_rows(2, 1, 1)]
    feeder = _OutputProcessorEngine(chunks)
    response = await _serve_tokens_any(
        _serving(feeder),
        _request(logprobs=0, logprobs_format="compact"),
    )
    block = json.loads(response.body)["choices"][0]["compact_logprobs"]
    assert block["num_positions"] == 3 and block["num_slots"] == 0


def test_encoder_failures_fail_only_their_request(monkeypatch):
    """Frontend-only storage never raises out of process_outputs: a failing
    wire encode fails that request's compact render (500); others are
    unaffected."""

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
    assert _broken(out_a.logprobs) and len(out_a.logprobs) == 2
    with pytest.raises(GenerationError, match="encoding failed"):
        render_compact_logprobs(out_a.logprobs, 2)
    assert out_b.finish_reason == "abort"
    assert len(out_b.logprobs) == 1


@pytest.mark.parametrize("failing", ["decode"])
def test_unwire_failure_is_contained(monkeypatch, failing):
    """A failure while leaving wire mode (a width change) is
    contained: the request's storage is discarded and marked broken, and the
    rest of the batch is processed."""

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
            _step("a", ids[1:2, :2], lps[1:2, :2], ranks[1:2]),
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
    assert _broken(out_a.logprobs) and len(out_a.logprobs) == 3
    with pytest.raises(GenerationError):
        render_compact_logprobs(out_a.logprobs, 2)
    assert len(out_b.logprobs) == 2
    np.testing.assert_array_equal(_inner(out_b.logprobs).arrays()[0], ids[1:3])


def test_broken_container_semantics():
    """A broken container raises a clear error on access,
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
    """A failure while slicing the DELTA output (inside
    RequestState._new_completion_output) fails only that request; the
    co-batched request is processed."""

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
    assert _broken(out_a.logprobs) and len(out_a.logprobs) == 1
    with pytest.raises(GenerationError):
        render_compact_logprobs(out_a.logprobs, 2)
    assert not _broken(out_b.logprobs)
    np.testing.assert_array_equal(_inner(out_b.logprobs).arrays()[0], ids[:1])


def test_delta_merge_failure_is_contained(monkeypatch):
    """A failure while merging queued DELTA outputs
    (RequestOutputCollector.put -> ArrayLogprobs.extend) fails only that
    request; the co-batched request is merged normally."""

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
    assert _broken(out_a.logprobs) and len(out_a.logprobs) == 2
    assert list(out_a.token_ids) == ids[:2, 0].tolist()
    assert not _broken(out_b.logprobs)
    np.testing.assert_array_equal(_inner(out_b.logprobs).arrays()[0], ids[:2])


def test_unwire_failure_keeps_wire_data():
    """A failed decode leaves the wire storage intact (no silent loss)."""
    from unittest.mock import patch

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
    """Stepped slices raise, also for broken containers (no
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


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["compact"])
async def test_rendered_parts_are_exact_bytes(fmt):
    """Every body part is exact bytes, so the off-loop join of
    the default body releases the GIL (a memoryview part made bytes.join hold
    it for the whole 2.2 GB copy: ~217 ms event-loop stalls)."""
    chunks = [_engine_rows(0, 3, 4)]
    feeder = _OutputProcessorEngine(chunks)
    response = await _serve_tokens_any(
        _serving(feeder), _request(logprobs=3, logprobs_format=fmt)
    )
    assert all(type(part) is bytes for part in response.parts)
    parts = render_json_with_fragments(
        {"choices": [{"a": None, "b": 1}]}, "a", {0: [b"1", b"2"]}
    )
    assert all(type(part) is bytes for part in parts)
    assert b"".join(parts) == b'{"choices":[{"a":12,"b":1}]}'


def test_second_failure_on_broken_container_keeps_counting(monkeypatch):
    """Positions keep counting on an already broken
    container."""
    request = MagicMock(spec=EngineCoreRequest)
    request.sampling_params = SamplingParams(logprobs=2, array_logprobs=True)
    processor = LogprobsProcessor.from_new_request(None, request)

    def boom(*args, **kwargs):
        raise MemoryError("injected")

    # The core passes rows to the container, whose append_rows
    # contains failures.
    monkeypatch.setattr(ArrayLogprobs, "_append_rows", boom)
    for step in range(3):
        ids, lps, ranks = _engine_rows(step * 2, 2, 3)
        processor._update_sample_logprobs(LogprobsLists(ids, lps, ranks))
    assert _broken(processor.logprobs) and len(processor.logprobs) == 6


def _middleware_app(feeder: _OutputProcessorEngine) -> TestClient:
    """Real router + ServingTokens, user --middleware configured."""
    app = FastAPI()
    app.state.args = Namespace(
        log_error_stack=False, tokens_only=False, middleware=["some.Middleware"]
    )
    app.state.serving_tokens = _serving(feeder)
    init_exception_handler(app)
    _attach(app)
    return TestClient(app)


def _compact_post(client: TestClient, positions: int):
    return client.post(
        "/inference/v1/generate",
        json={
            "token_ids": [1, 2, 3],
            "sampling_params": {"max_tokens": positions, "logprobs": 128},
            "logprobs_format": "compact",
        },
    )


def _check_compact_body(result, chunks):
    assert result.status_code == 200
    assert int(result.headers["content-length"]) == len(result.content)
    block = result.json()["choices"][0]["compact_logprobs"]
    token_ids, logprobs = _decode(block)
    expected = _expected(chunks, 129)
    np.testing.assert_array_equal(token_ids, expected[0])
    np.testing.assert_array_equal(logprobs, expected[1])


def _recording_builds(monkeypatch) -> list[str]:
    """Threads the compact responses are built (and joined) in."""
    threads: list[str] = []
    original = ServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        threads.append(threading.current_thread().name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ServingTokens, "_build_full_response", recording)
    return threads


@pytest.mark.parametrize("positions", [1500, 5500])
def test_compact_middleware_build_skips_busy_large_builder(monkeypatch, positions):
    """A compact + --middleware body below
    LARGE_LOGPROB_ENTRIES (here ~2 and ~7.5 MB) is built and joined in one
    job off the large-build thread, so it does not wait behind an unrelated
    large build occupying it."""

    assert positions * 129 < core_serving.LARGE_LOGPROB_ENTRIES
    builds = _recording_builds(monkeypatch)
    chunks = [_engine_rows(0, positions, 129)]
    release = threading.Event()
    busy = core_serving._LARGE_RESPONSE_BUILDER.submit(lambda: release.wait(20))
    try:
        with _middleware_app(_OutputProcessorEngine(chunks)) as client:
            t0 = time.perf_counter()
            result = _compact_post(client, positions)
            elapsed = time.perf_counter() - t0
        assert not busy.done()
        assert elapsed < 10
    finally:
        release.set()
        busy.result(timeout=30)
    assert 2_000_000 < len(result.content) < 8_000_000
    _check_compact_body(result, chunks)
    assert builds == ["generate-response-mid-0"]


@pytest.mark.asyncio
async def test_stream_fails_on_broken_handle_without_new_tokens():
    """A terminal delta without new tokens whose logprobs slice
    failed (a broken zero-position handle) fails the stream; it is not skipped
    as an empty output."""
    from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
    from vllm.outputs import CompletionOutput, RequestOutput

    serving = _serving(_OutputProcessorEngine([]))
    token_ids, logprobs, ranks = _engine_rows(0, 2, 4)
    healthy = SampleLogprobsHandle(ArrayLogprobs(), 0)
    healthy.append_rows(token_ids, logprobs, ranks)

    def output(handle, tokens, finish_reason):
        return RequestOutput(
            request_id="r",
            prompt=None,
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    0, "", tokens, None, handle, finish_reason=finish_reason
                )
            ],
            finished=finish_reason is not None,
        )

    async def results():
        yield output(healthy, token_ids[:, 0].tolist(), None)
        yield output(SampleLogprobsHandle(None, 0), [], "stop")

    request = _request(logprobs=3, stream=True, logprobs_format="compact")
    chunks = [
        chunk
        async for chunk in serving.serve_tokens_stream_generator(
            request, results(), "r", "m", RequestResponseMetadata(request_id="r")
        )
    ]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert "compact_logprobs" in chunks[0]
    assert "Compact logprobs encoding failed" in chunks[-2]


@pytest.mark.asyncio
async def test_stream_fails_on_a_failed_terminal_slice(monkeypatch):
    """The storage's own slice failure for a terminal delta
    without new tokens (a broken ArrayLogprobs inside a healthy core handle,
    through the real OutputProcessor) fails the stream instead of ending it
    with [DONE]."""
    real_slice = ArrayLogprobs._slice

    def failing_slice(self, index):
        start, stop, _ = index.indices(self.num_positions)
        if stop == start:
            raise ValueError("terminal slice failure")
        return real_slice(self, index)

    monkeypatch.setattr(ArrayLogprobs, "_slice", failing_slice)
    feeder = _OutputProcessorEngine([_engine_rows(0, 2, 4)], finish=FinishReason.STOP)
    generator = await _serve_compact(
        _serving(feeder), _request(logprobs=3, logprobs_format="compact", stream=True)
    )
    chunks = [chunk async for chunk in generator]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert "compact_logprobs" in chunks[0]
    assert "Compact logprobs encoding failed" in chunks[-2]


def test_compact_middleware_offloaded_build_joins_in_same_job(monkeypatch):
    """An offloaded compact + --middleware build is joined inside the same
    response-builder job: one builder trip, no join on the event loop."""

    monkeypatch.setattr(core_serving, "OFFLOAD_MIN_LOGPROB_ENTRIES", 0)
    builds = _recording_builds(monkeypatch)
    submits: list[Any] = []
    builder = core_serving._RESPONSE_BUILDER
    real_submit = builder.submit

    def counting_submit(fn):
        submits.append(fn)
        return real_submit(fn)

    monkeypatch.setattr(builder, "submit", counting_submit)
    chunks = [_engine_rows(0, 40, 129)]
    with _middleware_app(_OutputProcessorEngine(chunks)) as client:
        result = _compact_post(client, 40)
    _check_compact_body(result, chunks)
    assert len(submits) == 1
    assert builds == ["generate-response-mid-0"]


@pytest.mark.parametrize("broken_side", ["target", "source"])
def test_merge_with_broken_side_copies_nothing(monkeypatch, broken_side):
    """Merging into/from a broken container only counts
    positions; nothing is unwired, snapshotted or re-attached."""
    token_ids, logprobs, ranks = _engine_rows(0, 4, 3)
    target = ArrayLogprobs()
    target.append_rows(token_ids[:2], logprobs[:2], ranks[:2])
    source = ArrayLogprobs(wire_base64=True)
    source.append_rows(token_ids[2:], logprobs[2:], ranks[2:])
    source._legacy = [{1: Logprob(-1.0)}]
    source.num_positions += 1
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
