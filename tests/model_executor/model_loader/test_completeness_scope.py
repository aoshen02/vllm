# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A streamed update suspends the checks that assume one `load_weights` call
sees the whole checkpoint."""

import torch

from vllm.model_executor.model_loader.completeness import (
    completeness_checks_enabled,
    streaming_a_checkpoint,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _Tied(torch.nn.Module):
    """`lm_head.weight` is `embed.weight`, so the loader skips the alias and
    relies on the canonical name arriving."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)
        self.lm_head.weight = self.embed.weight
        self.embed.weight.weight_loader = default_weight_loader

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        loader.aliased_params = {"lm_head.weight": "embed.weight"}
        return loader.load_weights(weights)


def test_the_scope_is_off_by_default_and_restored_after():
    assert completeness_checks_enabled()
    with streaming_a_checkpoint():
        assert not completeness_checks_enabled()
    assert completeness_checks_enabled()


def test_the_mtp_names_follow_the_general_scope():
    """Eleven model files consult the MTP spelling; one scope drives them all."""
    with streaming_a_checkpoint():
        assert not is_mtp_completeness_check_enabled()
    assert is_mtp_completeness_check_enabled()


def test_a_batch_without_the_canonical_name_is_refused_outside_the_scope():
    """The check is right at cold start: nothing ever wrote the tied weight."""
    model = _Tied()
    try:
        model.load_weights([("lm_head.weight", torch.full((4, 4), 5.0))])
    except ValueError as error:
        assert "embed.weight" in str(error)
    else:
        raise AssertionError("the cold-start check should have refused")


def test_a_streamed_update_may_split_an_alias_from_its_canonical_name():
    """Each batch is one `load_weights` call, and the two names land in
    different batches; judging a batch against the whole checkpoint refused
    every update to a model with tied word embeddings."""
    model = _Tied()
    with streaming_a_checkpoint():
        model.load_weights([("lm_head.weight", torch.full((4, 4), 5.0))])
        model.load_weights([("embed.weight", torch.full((4, 4), 5.0))])

    assert torch.equal(model.embed.weight, torch.full((4, 4), 5.0))
