# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogprobsProcessor.

These tests exercise the truncation invariant that the MRV2 sampler relies
on: when the sampler returns a row wider than a request's own
`num_logprobs + 1` (because another request in the batch needed a wider
row), the trailing positions are populated with sentinel values
(`token_id=0`, `logprob=-inf`). LogprobsProcessor must read only the first
`num_logprobs + 1` entries so those sentinels never reach the user.
"""

import itertools

import numpy as np
import pytest
import torch

from vllm.logprobs import (
    FlatLogprobs,
    append_logprobs_for_next_position,
    create_sample_logprobs,
)
from vllm.v1.engine import EngineCoreOutput, LogprobsWire
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsLists
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def _make_processor(
    num_logprobs: int, flat_logprobs: bool = False
) -> LogprobsProcessor:
    return LogprobsProcessor(
        tokenizer=None,
        logprobs=create_sample_logprobs(flat_logprobs=flat_logprobs),
        prompt_logprobs=None,
        cumulative_logprob=0.0,
        num_logprobs=num_logprobs,
        num_prompt_logprobs=None,
    )


def test_drops_trailing_sentinel_columns():
    """A request that asked for 3 custom token logprobs but ended up in a
    batch padded to width 5 must not surface the trailing -inf entries."""
    processor = _make_processor(num_logprobs=3)

    sampled = 42
    # Layout: [sampled, custom_1, custom_2, custom_3, SENTINEL, SENTINEL]
    # Use float32-exact values so cumulative_logprob compares cleanly.
    token_ids = np.array([[sampled, 100, 200, 300, 0, 0]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.0, -2.0, -3.0, -np.inf, -np.inf]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(
        LogprobsWire(token_ids.tolist(), logprobs.tolist(), ranks.tolist())
    )

    assert len(processor.logprobs) == 1
    pos = processor.logprobs[0]
    # Exactly sampled + 3 requested tokens; trailing sentinels dropped.
    assert set(pos.keys()) == {sampled, 100, 200, 300}
    assert 0 not in pos
    assert all(np.isfinite(lp.logprob) for lp in pos.values())
    # cumulative_logprob comes from the sampled token's logprob only.
    assert processor.cumulative_logprob == -0.5


def test_accepts_exactly_sized_row():
    """When the row is exactly num_logprobs+1, no truncation needed."""
    processor = _make_processor(num_logprobs=2)

    token_ids = np.array([[7, 11, 13]], dtype=np.int32)
    logprobs = np.array([[-0.5, -1.5, -2.5]], dtype=np.float32)
    ranks = np.array([1], dtype=np.int32)

    processor._update_sample_logprobs(
        LogprobsWire(token_ids.tolist(), logprobs.tolist(), ranks.tolist())
    )

    pos = processor.logprobs[0]
    assert set(pos.keys()) == {7, 11, 13}


def test_prompt_token_id_logprobs_are_popped_once():
    """DELTA outputs carry fixed-ID prompt scores exactly once."""
    processor = _make_processor(num_logprobs=1)
    assert processor.pop_prompt_token_id_logprobs() is None

    processor.update_from_output(
        EngineCoreOutput(
            request_id="req-0",
            new_token_ids=[],
            prompt_token_id_logprobs=torch.tensor(
                [[-0.5, -1.5], [-2.5, -3.5]], dtype=torch.float32
            ),
        )
    )

    scores = processor.pop_prompt_token_id_logprobs()
    assert scores is not None
    assert scores.tolist() == [[-0.5, -1.5], [-2.5, -3.5]]
    assert processor.pop_prompt_token_id_logprobs() is None


def _as_tuples(sample_logprobs) -> list[list[tuple]]:
    """Flatten logprobs to comparable tuples, with logprobs as float bits."""
    return [
        [
            (tid, lp.logprob.hex(), lp.rank, lp.decoded_token)
            for tid, lp in position.items()
        ]
        for position in sample_logprobs
    ]


@pytest.mark.parametrize("flat_logprobs", [False, True])
@pytest.mark.parametrize("num_logprobs", [0, 2])
@pytest.mark.parametrize("num_positions", [1, 3])
def test_list_wire_matches_ndarray_path(
    num_logprobs: int, flat_logprobs: bool, num_positions: int
):
    """Logprobs sent as lists over msgpack equal, bit for bit, what the
    previous per-request ndarray slices produced; rows are wider than
    `num_logprobs + 1` as in a mixed batch."""
    rng = np.random.default_rng(num_logprobs * 10 + num_positions)
    num_rows, width = 8, 5
    batch = LogprobsLists(
        logprob_token_ids=rng.integers(0, 50000, (num_rows, width)),
        logprobs=(rng.standard_normal((num_rows, width)) * 5).astype(np.float32),
        sampled_token_ranks=rng.integers(1, 50000, num_rows),
    )
    batch.logprobs[0, 0] = -np.inf
    wire = LogprobsWire(
        batch.logprob_token_ids.tolist(),
        batch.logprobs.tolist(),
        batch.sampled_token_ranks.tolist(),
    )
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(EngineCoreOutput)

    processor = _make_processor(num_logprobs, flat_logprobs)
    expected = create_sample_logprobs(flat_logprobs)
    expected_cumulative = 0.0
    for start in range(0, num_rows, num_positions):
        end = start + num_positions
        output = EngineCoreOutput(
            request_id="req-0",
            new_token_ids=[0] * num_positions,
            new_logprobs=LogprobsWire(
                wire.logprob_token_ids[start:end],
                wire.logprobs[start:end],
                wire.sampled_token_ranks[start:end],
            ),
        )
        processor.update_from_output(decoder.decode(encoder.encode(output)))

        # The previous path: ndarray slices, converted row by row.
        ref = batch.slice_request(start, num_positions)
        for rank, logprobs, token_ids in zip(
            ref.sampled_token_ranks, ref.logprobs, ref.logprob_token_ids
        ):
            expected_cumulative += logprobs.tolist()[0]
            append_logprobs_for_next_position(
                expected,
                token_ids.tolist(),
                logprobs.tolist(),
                itertools.repeat(None),
                rank.tolist(),
                num_logprobs,
            )

    assert isinstance(processor.logprobs, FlatLogprobs) == flat_logprobs
    assert _as_tuples(processor.logprobs) == _as_tuples(expected)
    assert processor.cumulative_logprob == expected_cumulative
    if flat_logprobs:
        assert processor.logprobs.start_indices == expected.start_indices
        assert processor.logprobs.end_indices == expected.end_indices
