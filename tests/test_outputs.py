# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.outputs import CompletionOutput, RequestOutput, SamplingMask

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_request_output_aggregation_preserves_sampling_mask_logprobs():
    def make_output(token_id: int, support: list[int]) -> RequestOutput:
        return RequestOutput(
            request_id="test",
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[token_id],
                    cumulative_logprob=None,
                    logprobs=None,
                    sampling_mask=SamplingMask([support], [[-0.1] * len(support)]),
                )
            ],
            finished=False,
        )

    output = make_output(10, [10, 11])
    output.add(make_output(20, [20, 21]), aggregate=True)

    assert output.outputs[0].token_ids == [10, 20]
    assert output.outputs[0].sampling_mask == SamplingMask(
        [[10, 11], [20, 21]], [[-0.1, -0.1], [-0.1, -0.1]]
    )
