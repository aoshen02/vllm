# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic /inference/v1/generate scenarios for golden-body tests.

Only uses APIs that also exist before the ``logprobs_format`` change, so the
same scenarios can be rendered by the old implementation to produce the
golden hashes in ``test_logprobs_format.py`` (see ``GOLDEN_SHA256`` there).
"""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from fastapi.responses import JSONResponse

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _mock_engine,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

PROMPT = [464, 3290, 318, 257]


def _rows(start: int, count: int, width: int):
    """Rows of real gpt2 ids; even positions repeat the sampled id in top-k."""
    rng = np.random.default_rng(1000 + start)
    top = np.stack(
        [rng.choice(50_000, size=width - 1, replace=False) for _ in range(count)]
    )
    sampled = np.where(
        np.arange(start, start + count) % 2 == 0,
        top[:, min(1, width - 2)],
        40_000 + start,
    )
    ids = np.column_stack((sampled, top)).astype(np.int64)
    lps = (-rng.random((count, width)) * 12).astype(np.float32)
    lps[0, -1] = -np.inf
    ranks = rng.integers(1, 500, size=count)
    return ids, lps, ranks


def _prompt_tensors(k: int) -> LogprobsTensors:
    n = len(PROMPT) - 1
    ids = torch.tensor(
        [[PROMPT[i + 1]] + [11 + j + 7 * i for j in range(k)] for i in range(n)]
    )
    values = torch.tensor(
        [[-0.5 - i - 0.25 * j for j in range(k + 1)] for i in range(n)],
        dtype=torch.float32,
    )
    return LogprobsTensors(ids, values, torch.tensor([3, 1, 2]))


class _Feeder:
    def __init__(self, chunks, finish, tokenizer):
        self.chunks, self.finish, self.tokenizer = chunks, finish, tokenizer

    def generate(self, engine_input, sampling_params, request_id, **kwargs):
        async def _gen():
            processor = OutputProcessor(tokenizer=self.tokenizer, log_stats=False)
            request = EngineCoreRequest(
                request_id=request_id + "-int",
                external_req_id=request_id,
                prompt_token_ids=list(PROMPT),
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
            k_prompt = sampling_params.prompt_logprobs
            for i, (ids, lps, ranks) in enumerate(self.chunks):
                processor.process_outputs(
                    [
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=ids[:, 0].tolist(),
                            new_logprobs=None
                            if sampling_params.logprobs is None
                            else LogprobsLists(ids, lps, ranks),
                            new_prompt_logprobs_tensors=_prompt_tensors(k_prompt)
                            if i == 0 and k_prompt is not None
                            else None,
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
            out = queue.get_nowait()
            assert out is not None
            yield out

        return _gen()


SCENARIOS: dict[str, dict[str, Any]] = {
    "lp2_plp2_length": dict(
        logprobs=2,
        prompt_logprobs=2,
        finish=FinishReason.LENGTH,
        chunks=[(0, 3, 6), (3, 2, 6)],
    ),
    "lpNone_plp1_abort": dict(
        logprobs=None, prompt_logprobs=1, finish=FinishReason.ABORT, chunks=[(0, 2, 4)]
    ),
    "lp0_abort": dict(
        logprobs=0,
        prompt_logprobs=None,
        finish=FinishReason.ABORT,
        chunks=[(0, 4, 3), (4, 1, 3)],
    ),
    "lp5_narrow_rows": dict(
        logprobs=5,
        prompt_logprobs=None,
        finish=FinishReason.ABORT,
        chunks=[(0, 2, 6), (2, 1, 2), (3, 2, 6)],
    ),
}


async def render_body(name: str) -> bytes:
    """Body bytes exactly as api_router.generate would send them."""
    spec = SCENARIOS[name]
    chunks = [_rows(*c) for c in spec["chunks"]]
    tokenizer = get_tokenizer(MODEL_NAME)
    engine = _mock_engine()
    feeder = _Feeder(chunks, spec["finish"], tokenizer)
    engine.generate = MagicMock(side_effect=feeder.generate)
    serving = _build_serving_tokens(engine)
    sampling = {"max_tokens": 50, "logprobs": spec["logprobs"]}
    if spec["prompt_logprobs"] is not None:
        sampling["prompt_logprobs"] = spec["prompt_logprobs"]
    request = GenerateRequest.model_validate(
        {"request_id": "golden", "token_ids": PROMPT, "sampling_params": sampling}
    )
    with patch.object(time, "time", lambda: 1700000000.0):
        response = await serving.serve_tokens(request)
    body = getattr(response, "body", None)
    if isinstance(body, bytes) and not hasattr(response, "model_dump"):
        return body
    return JSONResponse(content=response.model_dump()).body
