# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The frontend-only sample-logprobs container hook (``vllm.logprobs``)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm import logprobs as logprobs_mod
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.logprobs import (
    SampleLogprobsHandle,
    create_sample_logprobs,
    register_sample_logprobs_container,
    set_sample_logprobs_container,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.outputs import LogprobsLists
from vllm.v1.serial_utils import MsgpackEncoder


class RowsContainer:
    def __init__(self, params=None):
        self.rows: list = []

    def append_rows(self, token_ids, logprobs, ranks):
        self.rows.append((token_ids, logprobs, ranks))


class FailingContainer(RowsContainer):
    def append_rows(self, token_ids, logprobs, ranks):
        raise RuntimeError("container failure")


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(logprobs_mod, "_SAMPLE_LOGPROBS_CONTAINERS", {})
    register_sample_logprobs_container("rows", RowsContainer, skip_sampled_text=True)
    register_sample_logprobs_container(
        "failing", FailingContainer, skip_sampled_text=True
    )


def _params(container=None, kind=RequestOutputKind.FINAL_ONLY, **kwargs):
    params = SamplingParams(max_tokens=10, logprobs=2, **kwargs)
    params.output_kind = kind
    if container:
        set_sample_logprobs_container(params, container)
    return params


def _request(rid, params):
    return EngineCoreRequest(
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


def _rows(n, width, start=0):
    rng = np.random.default_rng(start)
    ids = rng.integers(0, 1000, (n, width)).astype(np.int64)
    lps = (-rng.random((n, width))).astype(np.float32)
    return ids, lps, rng.integers(1, 9, n).astype(np.int64)


def _process(params_by_id, rows_by_id):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    requests = {}
    for rid, params in params_by_id.items():
        requests[rid] = _request(rid, params)
        processor.add_request(
            requests[rid], None, queue=RequestOutputCollector(params.output_kind, rid)
        )
    processor.process_outputs(
        [
            EngineCoreOutput(
                request_id=rid,
                new_token_ids=ids[:, 0].tolist(),
                new_logprobs=LogprobsLists(ids, lps, ranks),
            )
            for rid, (ids, lps, ranks) in rows_by_id.items()
        ]
    )
    return processor, requests


def test_rows_reach_the_container_and_stay_off_the_engine_wire():
    params = _params("rows")
    ids, lps, ranks = _rows(3, 5)  # engine rows padded beyond k + 1 = 3 slots
    processor, requests = _process(
        {"a": params, "b": _params()}, {"a": (ids, lps, ranks), "b": (ids, lps, ranks)}
    )
    assert requests["a"].sampling_params._sample_logprobs_container is None
    assert params._sample_logprobs_container == "rows"
    assert b"rows" not in b"".join(
        bytes(b) for b in MsgpackEncoder().encode(requests["a"])
    )
    state, default = processor.request_states["a"], processor.request_states["b"]
    handle = state.logprobs_processor.logprobs
    assert type(handle) is SampleLogprobsHandle and len(handle) == 3
    ((got_ids, got_lps, got_ranks),) = handle.unwrap().rows
    np.testing.assert_array_equal(got_ids, ids[:, :3])
    np.testing.assert_array_equal(got_lps, lps[:, :3])
    np.testing.assert_array_equal(got_ranks, ranks)
    assert (
        state.logprobs_processor.cumulative_logprob
        == default.logprobs_processor.cumulative_logprob
    )


@pytest.mark.parametrize("case", ["container", "factory", "row counts"])
def test_failures_fail_only_their_request(monkeypatch, case):
    ids, lps, ranks = _rows(2, 3)
    params, rows = _params("rows"), (ids, lps, ranks)
    if case == "container":
        params = _params("failing")
    elif case == "factory":
        monkeypatch.setitem(
            logprobs_mod._SAMPLE_LOGPROBS_CONTAINERS, "rows", (lambda p: 1 / 0, True)
        )
    else:
        rows = (ids, lps, ranks[:1])
    processor, _ = _process(
        {"a": params, "b": _params()}, {"a": rows, "b": (ids, lps, ranks)}
    )
    handle = processor.request_states["a"].logprobs_processor.logprobs
    assert handle.broken and len(handle) == len(rows[2])
    with pytest.raises(ValueError):
        handle.unwrap()
    assert len(processor.request_states["b"].logprobs_processor.logprobs) == 2


def test_streaming_requests_keep_the_default_storage():
    ids, lps, ranks = _rows(2, 3)
    params = _params("rows", kind=RequestOutputKind.DELTA)
    processor, _ = _process({"a": params}, {"a": (ids, lps, ranks)})
    assert type(processor.request_states["a"].logprobs_processor.logprobs) is list


def test_registration_and_selection():
    register_sample_logprobs_container("rows", RowsContainer, skip_sampled_text=True)
    with pytest.raises(ValueError, match="already registered"):
        register_sample_logprobs_container(
            "rows", lambda p: RowsContainer(), skip_sampled_text=True
        )
    with pytest.raises(ValueError, match="Unknown"):
        set_sample_logprobs_container(SamplingParams(), "nope")
    params = SamplingParams(logprobs=2)
    assert create_sample_logprobs(False, params) == []
    request = GenerateRequest.model_validate(
        {"token_ids": [1], "sampling_params": {"_sample_logprobs_container": "rows"}}
    )
    assert request.sampling_params._sample_logprobs_container is None


@pytest.mark.parametrize("stop", [None, ["x"]])
@pytest.mark.parametrize("container", [None, "rows"])
def test_sampled_text_is_skipped_only_without_stop_strings(
    monkeypatch, stop, container
):
    monkeypatch.setattr(
        "vllm.v1.engine.output_processor.IncrementalDetokenizer.from_new_request",
        lambda tokenizer, request: tokenizer,
    )
    tokenizer = MagicMock()
    processor = OutputProcessor(tokenizer=tokenizer, log_stats=False)
    processor.add_request(_request("r", _params(container, stop=stop)), None)
    skipped = container is not None and not stop
    assert processor.request_states["r"].detokenizer is (None if skipped else tokenizer)
