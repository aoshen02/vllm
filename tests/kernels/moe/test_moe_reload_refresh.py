# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle tests for refresh_after_weight_reload.

After an RL weight reload, kernel objects whose tensors may be captured by
CUDA graphs must be preserved and their runtime-derived state rewritten in
place: same storage (data_ptr), new values. These tests guard that contract
without running the flashinfer kernels themselves.
"""

from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsModular,
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsBase,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
from vllm.model_executor.layers.quantization.online.nvfp4 import Nvfp4OnlineMoEMethod
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA required", allow_module_level=True)

NUM_EXPERTS = 4


def make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )


def make_per_tensor_config(scale: float):
    return fp8_w8a8_moe_quant_config(
        w1_scale=torch.full((NUM_EXPERTS,), scale, device="cuda"),
        w2_scale=torch.full((NUM_EXPERTS,), 2 * scale, device="cuda"),
        a1_scale=torch.tensor(scale, device="cuda"),
        a2_scale=torch.tensor(2 * scale, device="cuda"),
    )


def test_per_tensor_fp8_refresh_updates_values_in_place():
    """After a reload, the derived output scales must take their values from
    the freshly loaded config while keeping the original storage."""
    experts = TrtLlmFp8ExpertsMonolithic(make_moe_config(), make_per_tensor_config(0.5))
    ptrs = {
        name: getattr(experts, name).data_ptr()
        for name in ("_g1_alphas", "_g2_alphas", "_g1_scale_c")
    }
    old = {name: getattr(experts, name).clone() for name in ptrs}

    fresh = make_per_tensor_config(0.25)
    experts.refresh_after_weight_reload(fresh)

    expected = experts._compute_output_scales(fresh)
    for (name, ptr), want in zip(ptrs.items(), expected):
        got = getattr(experts, name)
        assert got.data_ptr() == ptr, f"{name} was rebound"
        assert not torch.equal(got, old[name]), f"{name} kept stale values"
        torch.testing.assert_close(got, want)


def test_per_tensor_fp8_refresh_requires_fresh_config():
    experts = TrtLlmFp8ExpertsMonolithic(make_moe_config(), make_per_tensor_config(0.5))
    with pytest.raises(AssertionError, match="fresh quant"):
        experts.refresh_after_weight_reload()


@pytest.mark.parametrize(
    "experts_cls", [TrtLlmFp8ExpertsModular, TrtLlmFp8ExpertsMonolithic]
)
def test_fp8_swiglu_constants_restored_in_place(experts_cls):
    """The block/MXFP8 SwiGLU constants live on the shared base; after a
    simulated sleep-level-2 wipe, refresh must restore them without
    reallocating."""
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=torch.ones(NUM_EXPERTS, 1, 1, device="cuda"),
        w2_scale=torch.ones(NUM_EXPERTS, 1, 1, device="cuda"),
        block_shape=[1, 32],
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = experts_cls(make_moe_config(), quant_config)

    names = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
    ptrs = {name: getattr(experts, name).data_ptr() for name in names}
    for name in names:
        getattr(experts, name).zero_()

    experts.refresh_after_weight_reload()

    for name, want in zip(names, (1.702, 1.0, 7.0)):
        got = getattr(experts, name)
        assert got.data_ptr() == ptrs[name], f"{name} was rebound"
        torch.testing.assert_close(got, torch.full((NUM_EXPERTS,), want, device="cuda"))


def test_mxfp4_swiglu_constants_restored_in_place():
    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_scale=torch.ones(NUM_EXPERTS, 1, 1, device="cuda"),
        w2_scale=torch.ones(NUM_EXPERTS, 1, 1, device="cuda"),
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = TrtLlmMxfp4ExpertsBase(make_moe_config(), quant_config)

    names = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
    ptrs = {name: getattr(experts, name).data_ptr() for name in names}
    for name in names:
        getattr(experts, name).zero_()

    experts.refresh_after_weight_reload()

    for name, want in zip(names, (1.702, 1.0, 7.0)):
        got = getattr(experts, name)
        assert got.data_ptr() == ptrs[name], f"{name} was rebound"
        torch.testing.assert_close(got, torch.full((NUM_EXPERTS,), want, device="cuda"))


def _make_nvfp4_layer():
    layer = torch.nn.Module()
    for name, val in (
        ("w13_weight_scale_2", 0.5),
        ("w2_weight_scale_2", 0.25),
        ("w13_input_scale", 1.0),
        ("w2_input_scale", 2.0),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.full((NUM_EXPERTS,), val, device="cuda"), requires_grad=False
            ),
        )
    return layer


def test_nvfp4_pwal_publishes_in_place_without_double_folding():
    """A second process_weights_after_loading (weight reload) must rewrite
    g1_scale_c and the folded SwiGLU constants into the same storage, folding
    from the raw values rather than the already-folded ones."""
    layer = _make_nvfp4_layer()
    moe_config = make_moe_config()
    moe_config.swiglu_alpha = 1.702
    moe_config.swiglu_beta = 1.0
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=layer.w13_weight_scale_2,
        g2_alphas=layer.w2_weight_scale_2,
        a1_gscale=1.0 / layer.w13_input_scale,
        a2_gscale=1.0 / layer.w2_input_scale,
        w1_scale=torch.ones(NUM_EXPERTS, 1, 1, device="cuda"),
        w2_scale=torch.ones(NUM_EXPERTS, 1, 1, device="cuda"),
        gemm1_clamp_limit=7.0,
    )
    experts = TrtLlmNvFp4ExpertsMonolithic(moe_config, quant_config)
    experts.process_weights_after_loading(layer)

    names = ("g1_scale_c", "gemm1_clamp_limit", "gemm1_beta", "gemm1_alpha")
    ptrs = {name: getattr(layer, name).data_ptr() for name in names}

    for name, val in (
        ("w13_weight_scale_2", 0.125),
        ("w2_weight_scale_2", 0.0625),
        ("w13_input_scale", 3.0),
        ("w2_input_scale", 4.0),
    ):
        setattr(
            layer,
            name,
            torch.nn.Parameter(
                torch.full((NUM_EXPERTS,), val, device="cuda"), requires_grad=False
            ),
        )
    experts.process_weights_after_loading(layer)

    fused_g1 = 0.125 * 3.0
    expected = {
        "g1_scale_c": fused_g1 * (1.0 / 4.0),
        "gemm1_clamp_limit": 7.0 / fused_g1,
        "gemm1_beta": 1.0 / fused_g1,
        "gemm1_alpha": 1.702,
    }
    for name in names:
        got = getattr(layer, name)
        assert got.data_ptr() == ptrs[name], f"{name} was rebound"
        assert getattr(experts, name) is got, f"expert {name} points elsewhere"
        torch.testing.assert_close(
            got.data, torch.full((NUM_EXPERTS,), expected[name], device="cuda")
        )


def _run_nvfp4_setup_kernel_twice(nvfp4_backend: NvFp4MoeBackend):
    method = object.__new__(Nvfp4OnlineMoEMethod)
    method.nvfp4_backend = nvfp4_backend
    method.moe = make_moe_config()
    method.moe_quant_config = None
    method.moe_kernel = None
    method.experts_cls = mock.Mock()

    layer = mock.Mock()
    nvfp4_mod = "vllm.model_executor.layers.quantization.online.nvfp4"
    with (
        mock.patch(
            f"{nvfp4_mod}.convert_to_nvfp4_moe_kernel_format",
            return_value=tuple(mock.Mock() for _ in range(8)),
        ),
        mock.patch(f"{nvfp4_mod}.replace_parameter"),
        mock.patch(
            f"{nvfp4_mod}.make_nvfp4_moe_kernel",
            side_effect=lambda **kw: mock.Mock(),
        ),
        mock.patch.object(
            method, "get_fused_moe_quant_config", return_value=mock.Mock()
        ),
    ):
        kernels = []
        for _ in range(2):
            method._setup_kernel(layer)
            kernels.append(method.moe_kernel)
    return kernels


def test_online_nvfp4_trtllm_keeps_kernel_and_reprocesses_weights():
    first, second = _run_nvfp4_setup_kernel_twice(NvFp4MoeBackend.FLASHINFER_TRTLLM)
    assert second is first
    assert first.fused_experts.process_weights_after_loading.call_count == 2


def test_online_nvfp4_other_backends_still_rebuild_kernel():
    first, second = _run_nvfp4_setup_kernel_twice(NvFp4MoeBackend.FLASHINFER_CUTLASS)
    assert second is not first


def _run_setup_kernel_twice(fp8_backend: Fp8MoeBackend):
    """Drive Fp8MoEMethod._setup_kernel twice with the format/build helpers
    mocked out, returning (first_kernel, second_kernel)."""
    method = object.__new__(Fp8MoEMethod)
    method.fp8_backend = fp8_backend
    method.moe = make_moe_config()
    method.moe_quant_config = None
    method.moe_kernel = None
    method.experts_cls = mock.Mock()
    method.weight_scale_name = "weight_scale"

    layer = mock.Mock()
    w = torch.zeros(1)
    fp8_mod = "vllm.model_executor.layers.quantization.fp8"
    with (
        mock.patch(
            f"{fp8_mod}.convert_to_fp8_moe_kernel_format",
            side_effect=lambda **kw: (
                kw["w13"],
                kw["w2"],
                kw["w13_scale"],
                kw["w2_scale"],
            ),
        ),
        mock.patch(f"{fp8_mod}.replace_parameter"),
        mock.patch(
            f"{fp8_mod}.make_fp8_moe_kernel", side_effect=lambda **kw: mock.Mock()
        ),
        mock.patch.object(
            method, "get_fused_moe_quant_config", return_value=mock.Mock()
        ),
    ):
        kernels = []
        for _ in range(2):
            method._setup_kernel(layer, w, w, w, w, None, None)
            kernels.append(method.moe_kernel)
    return kernels


def test_non_trtllm_fp8_backend_still_rebuilds_kernel():
    first, second = _run_setup_kernel_twice(Fp8MoeBackend.TRITON)
    assert second is not first
    first.refresh_after_weight_reload.assert_not_called()


def test_trtllm_fp8_backend_keeps_kernel_and_refreshes():
    first, second = _run_setup_kernel_twice(Fp8MoeBackend.FLASHINFER_TRTLLM)
    assert second is first
    assert first.refresh_after_weight_reload.call_count == 1
