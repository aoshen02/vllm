# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup

# ---------------------------------------------------------------------------
# SamplingParams field
# ---------------------------------------------------------------------------


def test_sampling_params_trace_field_defaults_to_none():
    params = SamplingParams(max_tokens=10)
    assert params.trace_decode_token_ids is None


def test_sampling_params_trace_field_accepts_list():
    ids = [100, 200, 300]
    params = SamplingParams(trace_decode_token_ids=ids)
    assert params.trace_decode_token_ids == ids


def test_sampling_params_trace_field_preserved_by_clone():
    ids = [1, 2, 3]
    params = SamplingParams(trace_decode_token_ids=ids)
    cloned = params.clone()
    assert cloned.trace_decode_token_ids == ids
    assert cloned.trace_decode_token_ids is not params.trace_decode_token_ids


def test_sampling_params_trace_field_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        SamplingParams(trace_decode_token_ids=[])


@pytest.mark.parametrize("invalid_ids", [[-1, 5], [1, "2"]])
def test_sampling_params_trace_field_rejects_invalid_token_ids(invalid_ids):
    with pytest.raises(ValueError, match="non-negative integers"):
        SamplingParams(trace_decode_token_ids=invalid_ids)


def _make_model_config(vocab_size: int):
    from unittest.mock import Mock

    model_config = Mock()
    model_config.get_vocab_size = lambda: vocab_size
    return model_config


def test_validate_trace_decode_token_ids_accepts_in_vocab():
    params = SamplingParams(trace_decode_token_ids=[0, 50, 99])
    # Should not raise.
    params._validate_trace_decode_token_ids(_make_model_config(vocab_size=100))


def test_validate_trace_decode_token_ids_rejects_out_of_vocab():
    # The non-negative check passes at construction, but the token id exceeds
    # the vocabulary; verify() must reject it before it reaches the sampler.
    params = SamplingParams(trace_decode_token_ids=[0, 100])
    with pytest.raises(ValueError, match="out-of-vocab"):
        params._validate_trace_decode_token_ids(_make_model_config(vocab_size=100))


def test_validate_trace_decode_token_ids_noop_when_unset():
    params = SamplingParams(max_tokens=4)
    # Should not raise when the field is unset.
    params._validate_trace_decode_token_ids(_make_model_config(vocab_size=100))


def test_trace_replay_requires_v2_model_runner(monkeypatch):
    input_processor = InputProcessor.__new__(InputProcessor)
    input_processor.model_config = Mock()
    input_processor.speculative_config = None
    input_processor.structured_outputs_config = None
    input_processor.renderer = Mock(tokenizer=None)
    input_processor.use_v2_model_runner = False
    monkeypatch.setattr(SamplingParams, "verify", Mock())

    with pytest.raises(ValueError, match="only supported by the V2 model runner"):
        input_processor._validate_params(
            SamplingParams(trace_decode_token_ids=[1]), ("generate",)
        )
