# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample logprobs kept as engine rows (ArrayLogprobs) and the object-free
default-format render of /inference/v1/generate: same bytes as the per-entry
path, chosen only for non-streaming requests with sample logprobs."""

import itertools
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.responses import JSONResponse

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _make_request_output,
    _mock_engine,
)
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.scale_out.token_in_token_out import serving as serving_mod
from vllm.entrypoints.scale_out.token_in_token_out.array_logprobs import (
    ArrayLogprobs,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    RenderedGenerateResponse,
)
from vllm.logprobs import (
    Logprob,
    SampleLogprobsHandle,
    append_logprobs_for_next_position,
    create_sample_logprobs,
    register_sample_logprobs_container,
    set_sample_logprobs_container,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

ARRAY = serving_mod.ARRAY_LOGPROBS_CONTAINER


def _rows(n, slots, seed=0, vocab=50_000):
    """Engine-like rows: distinct candidates; on even rows the sampled token
    (slot 0) is also one of the top-k, with another value; one -inf value
    (clamped to -9999). Ranks are int64, as the engine's."""
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(vocab, slots, replace=False) for _ in range(n)])
    if slots > 1:
        for i in range(0, n, 2):
            ids[i, 0] = ids[i, 1 + i % (slots - 1)]
    lps = (-rng.random((n, slots)) * 20).astype(np.float32)
    lps[0, -1] = -np.inf
    return ids.astype(np.int32), lps, rng.integers(1, 100, n)


def _handle(k, *steps):
    """The engine rows of ``steps`` fed to the array container through the
    core handle, as the LogprobsProcessor does."""
    params = SamplingParams(logprobs=k)
    params.output_kind = RequestOutputKind.FINAL_ONLY
    set_sample_logprobs_container(params, ARRAY)
    handle = create_sample_logprobs(False, params)
    for ids, lps, ranks in steps:
        handle.append_engine_rows(ids, lps, ranks, None if k == -1 else k + 1, 0.0)
    return handle


def _per_entry(k, *steps):
    """The same rows stored by the base per-entry path."""
    stored: list = []
    for ids, lps, ranks in steps:
        for i in range(len(ranks)):
            append_logprobs_for_next_position(
                stored,
                ids[i].tolist(),
                lps[i].tolist(),
                itertools.repeat(None),
                int(ranks[i]),
                k,
            )
    return stored


def _final(outputs, finish_reason="length"):
    return RequestOutput(
        request_id="r",
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(i, "", tokens, None, logprobs, finish_reason=finish_reason)
            for i, (tokens, logprobs) in enumerate(outputs)
        ],
        finished=True,
    )


def _body(serving, k, outputs, finish_reason="length"):
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10, logprobs=k),
        model=MODEL_NAME,
    )
    response = serving._build_full_response(
        request, _final(outputs, finish_reason), "r", MODEL_NAME, 1700000000
    )[0]
    if isinstance(response, RenderedGenerateResponse):
        return response.body, True
    assert isinstance(response, GenerateResponse)
    return JSONResponse(content=response.model_dump()).body, False


def _outcome(serving, k, outputs, finish_reason="length"):
    try:
        return _body(serving, k, outputs, finish_reason)[0]
    except ValueError as e:
        return f"{type(e).__name__}: {e}"


@pytest.mark.parametrize("k", [0, 1, 3, 8])
@pytest.mark.parametrize("n", [1, 37, 1500])  # 1500: across render blocks
def test_rendered_bytes_equal_the_per_entry_path(k, n):
    serving = _build_serving_tokens(_mock_engine())
    rows = _rows(n, k + 1, seed=k * 7 + n)
    token_ids = rows[0][:, 0].tolist()
    fast, rendered = _body(serving, k, [(token_ids, _handle(k, rows))])
    legacy, _ = _body(serving, k, [(token_ids, _per_entry(k, rows))])
    assert rendered and fast == legacy


def _cases():
    """(name, k, choices as (token_ids, steps)) of irregular and special rows."""
    cases = []
    for k in (0, 1, 3):
        for label, value, slot in [
            ("nan sampled", np.nan, 0),
            ("nan top", np.nan, -1),
            ("+inf sampled", np.inf, 0),
            ("-0.0", -0.0, 0),
            ("1e-5", -1e-5, 0),
            ("denormal", -1e-45, -1),
            ("-9999.5", -9999.5, 0),
        ]:
            rows = _rows(5, k + 1, seed=k)
            rows[1][2, slot] = value
            cases.append((f"k={k} {label}", k, [(rows[0][:, 0].tolist(), [rows])]))
        wide = _rows(20, k + 5, seed=10 + k)  # truncated to k + 1 slots
        wide[0][3, 0] = wide[0][3, k + 3]  # sampled id only beyond k
        cases.append((f"k={k} wider", k, [(wide[0][:, 0].tolist(), [wide])]))
        a, b = _rows(7, k + 1, seed=20 + k), _rows(9, k + 1, seed=30 + k)
        both = np.concatenate([a[0], b[0]])[:, 0].tolist()
        cases.append((f"k={k} n=2", k, [(both, [a, b]), (a[0][:, 0].tolist(), [a])]))
        mismatch = _rows(4, k + 1, seed=40 + k)
        tokens = mismatch[0][:, 0].tolist()
        tokens[1] = 49_999 if tokens[1] != 49_999 else 49_998
        cases.append((f"k={k} sampled id not in slot 0", k, [(tokens, [mismatch])]))
        extra = _rows(4, k + 1, seed=50 + k)
        cases.append((f"k={k} more rows", k, [(extra[0][:3, 0].tolist(), [extra])]))
        big = _rows(5, k + 1, seed=60 + k, vocab=300_000)
        big[0][:, 0] = [262_143, 262_144, 299_999, 0, 5]  # beyond the lead table
        cases.append((f"k={k} large ids", k, [(big[0][:, 0].tolist(), [big])]))
    dup = _rows(4, 4, seed=70)
    dup[0][1, 2] = dup[0][1, 1]  # repeated top-k id
    cases.append(("repeated top-k", 3, [(dup[0][:, 0].tolist(), [dup])]))
    full = _rows(6, 40, seed=80)
    cases.append(("k=-1", -1, [(full[0][:, 0].tolist(), [full])]))
    return cases


@pytest.mark.parametrize("name,k,choices", _cases(), ids=lambda c: str(c))
def test_special_and_irregular_rows_match_the_per_entry_path(name, k, choices):
    """Base's own storage (append_logprobs_for_next_position) on one side,
    the engine rows through the handle on the other: same bytes, or the same
    error."""
    serving = _build_serving_tokens(_mock_engine())
    legacy = _outcome(
        serving, k, [(tokens, _per_entry(k, *steps)) for tokens, steps in choices]
    )
    fast = _outcome(
        serving, k, [(tokens, _handle(k, *steps)) for tokens, steps in choices]
    )
    assert fast == legacy


def test_aborted_choice_without_tokens():
    serving = _build_serving_tokens(_mock_engine())
    legacy = _outcome(serving, 3, [([], [])], finish_reason="abort")
    assert _outcome(serving, 3, [([], _handle(3))], finish_reason="abort") == legacy


def test_n2_and_irregular_rows():
    """Two choices; a choice whose rows changed width (irregular) falls back
    to the per-entry path for that choice, with the same bytes."""
    serving = _build_serving_tokens(_mock_engine())
    a = _rows(5, 4, seed=1)
    b1, b2 = _rows(3, 4, seed=2), _rows(2, 3, seed=3)
    regular, irregular = _handle(3, a), _handle(3, b1, b2)
    assert not irregular.unwrap().is_regular
    tok_a = a[0][:, 0].tolist()
    tok_b = b1[0][:, 0].tolist() + b2[0][:, 0].tolist()
    fast, _ = _body(serving, 3, [(tok_a, regular), (tok_b, irregular)])
    legacy, _ = _body(
        serving, 3, [(tok_a, _per_entry(3, a)), (tok_b, _per_entry(3, b1, b2))]
    )
    assert fast == legacy


def test_broken_handle_fails_the_request():
    serving = _build_serving_tokens(_mock_engine())
    with pytest.raises(GenerationError):
        _body(serving, 1, [([1, 2], SampleLogprobsHandle(None))])


def _other(params):
    return ArrayLogprobs()


register_sample_logprobs_container(
    "test.other_container", _other, skip_sampled_text=True
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream,logprobs,preset,expected",
    [
        (False, 3, None, ARRAY),
        (False, 0, None, ARRAY),
        (True, 3, None, None),
        (False, None, None, None),
        (False, 3, "test.other_container", "test.other_container"),
    ],
)
async def test_container_selection(stream, logprobs, preset, expected):
    """Non-streaming requests with sample logprobs only, unless another
    component (e.g. an endpoint plugin) already chose a container."""
    engine = _mock_engine()
    seen = []

    async def generate(engine_input, params, *args, **kwargs):
        seen.append(params._sample_logprobs_container)
        yield _make_request_output(
            "r",
            [10],
            finish_reason="stop",
            finished=True,
            logprobs=None if logprobs is None else [{10: Logprob(-0.5)}],
        )

    engine.generate = MagicMock(side_effect=generate)
    serving = _build_serving_tokens(engine)
    params = SamplingParams(max_tokens=1, logprobs=logprobs)
    if preset:
        set_sample_logprobs_container(params, preset)
    request = GenerateRequest(
        token_ids=[1, 2, 3], sampling_params=params, model=MODEL_NAME, stream=stream
    )
    out = await serving.serve_tokens(request)
    if stream:
        [chunk async for chunk in out]
    assert seen == [expected]


def _full_request(n=40):
    return GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=n, logprobs=3),
        model=MODEL_NAME,
    )


async def _serve_full(serving, rows):
    """serve_tokens_full_generator for one choice of ``rows`` (k=3)."""

    async def results():
        yield _final([(rows[0][:, 0].tolist(), _handle(3, rows))])

    return await serving.serve_tokens_full_generator(
        _full_request(len(rows[2])), results(), "r", MODEL_NAME, MagicMock()
    )


@pytest.mark.asyncio
async def test_large_builds_run_off_the_event_loop(monkeypatch):
    serving = _build_serving_tokens(_mock_engine())
    rows = _rows(40, 4)
    threads, real = [], serving_mod.ServingTokens._build_full_response

    def recording(self, *args, **kwargs):
        threads.append(threading.current_thread().name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(serving_mod.time, "time", lambda: 1700000000.0)
    inline = await _serve_full(serving, rows)
    monkeypatch.setattr(serving_mod.ServingTokens, "_build_full_response", recording)
    monkeypatch.setattr(serving_mod, "OFFLOAD_MIN_LOGPROB_ENTRIES", 160)  # 40 x 4
    monkeypatch.setattr(serving_mod, "LARGE_LOGPROB_ENTRIES", 160)
    response = await _serve_full(serving, rows)
    assert threads[0].startswith("generate-response_")
    assert response.body == inline.body


def test_storage_blocks_and_rows(monkeypatch):
    monkeypatch.setattr(ArrayLogprobs, "BLOCK_BYTES", 256)  # many small blocks
    ids, lps, ranks = _rows(50, 4)
    store = ArrayLogprobs()
    for start in range(0, 50, 7):
        chunk = slice(start, start + 7)
        # Engine ids are int32 (int64 in some runners), ranks int64.
        store.append_rows(ids[chunk].astype(np.int64), lps[chunk], ranks[chunk])
    assert len(store) == 50 and len(store.rank_chunks) > 1
    assert list(store) == _per_entry(-1, (ids, lps, ranks))
    got_ids, got_lps, got_ranks = store.arrays()
    assert (got_ids.dtype, got_lps.dtype, got_ranks.dtype) == ("<i4", "<f4", "<i4")
    np.testing.assert_array_equal(got_ids, ids)
    np.testing.assert_array_equal(got_lps, lps)
    np.testing.assert_array_equal(got_ranks, ranks)


@pytest.mark.parametrize(
    "ids,lps,ranks",
    [
        (np.array([[1, 2]]), np.array([[-1.0, -2.0]]), np.array([1])),  # float64
        (np.array([[1, 2**31]]), np.zeros((1, 2), np.float32), np.array([1])),
        (np.array([[1, 2]]), np.zeros((1, 2), np.float32), np.array([2**40])),
        (np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), np.array([1])),
        (np.array([[1, 2]]), np.zeros((2, 2), np.float32), np.array([1])),
        (np.array([[1, 2]]), np.zeros((1, 2), np.float32), [1]),
    ],
)
def test_rows_that_are_not_the_engines_fail_the_request(ids, lps, ranks):
    """Dtypes or shapes other than the engine's: the core handle breaks."""
    handle = _handle(1)
    handle.append_engine_rows(ids, lps, ranks, None, 0.0)
    assert handle.broken and len(handle) == 1
