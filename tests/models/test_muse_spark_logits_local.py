# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.muse_spark.amd.model import MuseSpark1xForCausalLM


def test_compute_logits_local_applies_output_transform() -> None:
    model = MuseSpark1xForCausalLM.__new__(MuseSpark1xForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    model.logits_processor = lambda lm_head, h, skip_gather=False: lm_head(h)
    model.metap_output_multiplier = 0.5
    model.output_soft_cap_temp = 20.0
    model.mul_tanh_soft_cap = None
    hidden = torch.randn(3, 4, dtype=torch.bfloat16) * 50

    torch.testing.assert_close(
        model.compute_logits_local(hidden), model.compute_logits(hidden)
    )


def test_nvidia_router_multiplier_folds_once_per_load():
    """The gate weight holds the folded multiplier after every load, a reload
    of the raw weight included, and a post-load rerun alone never folds twice."""
    from types import SimpleNamespace

    from vllm.models.muse_spark.nvidia.model import MuseSpark1xMoE

    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    nn.Module.__init__(moe)
    moe.router = nn.Linear(4, 3, bias=False)
    moe.router.weight_loader = lambda param, w: param.data.copy_(w)
    moe.moe_router = SimpleNamespace(router_output_multiplier=2.0)
    moe.router_output_multiplier = 2.0
    moe._router_multiplier_folded = False
    moe.router.weight.weight_loader = moe._load_router_weight  # as __init__ wires
    raw = torch.randn(3, 4)

    def load_and_fold() -> torch.Tensor:
        moe.router.weight.weight_loader(moe.router.weight, raw)
        moe.fold_router_output_multiplier()
        return moe.router.weight.detach().clone()

    cold = load_and_fold()
    torch.testing.assert_close(cold, raw * 2.0)
    assert moe.moe_router.router_output_multiplier is None
    moe.fold_router_output_multiplier()  # rerun without a reload
    torch.testing.assert_close(moe.router.weight.detach(), cold)
    torch.testing.assert_close(load_and_fold(), cold)  # reload
