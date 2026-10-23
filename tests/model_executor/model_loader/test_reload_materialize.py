# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialized reload temporaries start zeroed: rows no checkpoint tensor covers
(alignment padding a module zeroed at init) must not carry stale memory."""

import pytest
import torch

from vllm.model_executor.model_loader.reload.meta import (
    capture_layer_to_meta,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_params_buffers


class TestMaterializeMetaTensor:
    @pytest.mark.parametrize(
        "shape,transpose,dtype",
        [
            ((8, 16), False, torch.bfloat16),
            ((8, 16), True, torch.float32),
            ((4, 6, 10), False, torch.uint8),
        ],
        ids=["bf16", "strided-fp32", "packed-uint8"],
    )
    def test_zero_filled(self, shape, transpose, dtype):
        param = torch.nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False)
        if transpose:
            param = torch.nn.Parameter(param.data.t(), requires_grad=False)
        param.marker = "kept"

        out = materialize_meta_tensor(to_meta_tensor(param))

        assert not out.is_meta
        assert out.stride() == param.stride()
        assert out.marker == "kept"
        assert type(out) is type(param)
        assert torch.count_nonzero(out) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kernel_device", ["cpu", "cuda"], ids=["host", "device"])
def test_reload_temporary_follows_kernel_device(kernel_device):
    """A host-offloaded weight (e.g. a CPU embedding table) reloads on the host,
    not on the device, which may not fit a second copy."""
    layer = torch.nn.Linear(4, 4, bias=False, device=kernel_device)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cuda"),
    )
    info.kernel_tensors = get_layer_params_buffers(layer)
    restore_layer_on_meta(layer, info)

    materialize_layer(layer, info)

    assert layer.weight.device.type == kernel_device
