# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


@pytest.mark.parametrize("top_k", [0, -1])
def test_sampling_mask_replay_requires_finite_top_k(
    top_k: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(enable_return_sampling_mask=True)
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.renderer = SimpleNamespace(tokenizer=None)
    monkeypatch.setattr(SamplingParams, "verify", lambda *_: None)

    with pytest.raises(ValueError, match="top_k > 0"):
        processor._validate_params(SamplingParams(top_k=top_k), ("generate",))
