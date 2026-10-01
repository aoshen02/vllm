# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize(
    "shared,batch_invariant", [(True, True), (True, False), (False, True)]
)
def test_nemotron_shared_kernel_selection_is_model_local(
    monkeypatch, shared, batch_invariant, default_vllm_config
):
    """Only BI shared W4A16 linears select the aligned FlashInfer kernel."""
    from types import SimpleNamespace

    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        NemotronSharedNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
    from vllm.model_executor.models import nemotron_h

    original_kernels = []

    def linear(**kwargs):
        method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
        method.spec = SimpleNamespace(weight=kNvfp4Static, activation=None)
        method.kernel = object()
        original_kernels.append(method.kernel)
        return SimpleNamespace(quant_method=method)

    monkeypatch.setattr(nemotron_h, "ColumnParallelLinear", linear)
    monkeypatch.setattr(nemotron_h, "RowParallelLinear", linear)
    monkeypatch.setattr(nemotron_h, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        NemotronSharedNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls: (True, None)),
    )
    prefix = "backbone.layers.1.mixer." + ("shared_experts" if shared else "mlp")
    mlp = nemotron_h.NemotronHMLP(None, 2688, 1856, prefix=prefix)
    for layer, original in zip((mlp.up_proj, mlp.down_proj), original_kernels):
        if shared and batch_invariant:
            assert type(layer.quant_method.kernel) is NemotronSharedNvFp4LinearKernel
        else:
            assert layer.quant_method.kernel is original


def test_nemotron_shared_scale_preserves_humming_roundtrip(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        FlashInferCuteDslNvFp4W4A16LinearKernel,
        NemotronSharedNvFp4LinearKernel,
    )

    scale = torch.tensor([0.1234567], dtype=torch.float32)
    expected = 1.0 / (1.0 / scale)
    layer = SimpleNamespace(weight_global_scale=scale.clone())

    def prepare(self, layer):
        layer.weight_global_scale = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(
        FlashInferCuteDslNvFp4W4A16LinearKernel,
        "process_weights_after_loading",
        prepare,
    )
    kernel = NemotronSharedNvFp4LinearKernel.__new__(NemotronSharedNvFp4LinearKernel)
    kernel.process_weights_after_loading(layer)
    assert torch.equal(
        layer.weight_global_scale.view(torch.int32), expected.view(torch.int32)
    )


def test_nemotron_humming_schedule_preserves_reduction_and_input():
    import copy

    from vllm.model_executor.models.nemotron_h_moe import nemotron_humming_schedule

    def entry(lower, upper, n):
        return [
            lower,
            upper,
            dict(
                block_shape=[16, n, 32],
                warp_shape=[16, 64, 32],
                use_stream_k=False,
                mma_type="mma",
                use_f16_accum=False,
                num_stages=3,
            ),
        ]

    original = [entry(0, 448, 256), entry(448, 1024, 256), entry(1024, 2048, 128)]
    saved = copy.deepcopy(original)
    actual = nemotron_humming_schedule(original)
    assert original == saved
    assert actual[0][2]["warp_shape"] == [16, 32, 32]
    assert actual[1][2]["warp_shape"] == [16, 64, 32]
    assert actual[0][2]["num_stages"] == actual[1][2]["num_stages"] == 4
    assert actual[2] == original[2]
    for old, new in zip(original, actual):
        assert old[:2] == new[:2]
        assert old[2]["block_shape"] == new[2]["block_shape"]
        assert new[2]["use_f16_accum"] is False
    original[0][2]["block_shape"][2] = 64
    with pytest.raises(ValueError, match="reduction recipe"):
        nemotron_humming_schedule(original)


def test_nemotron_proxy_decode_matches_prefill_without_custom_worker(monkeypatch):
    """Explicit EP4 proxy gate; compare generated logprobs to teacher forcing."""
    import math
    import os
    import struct

    from tests.utils import RemoteOpenAIServer

    model = os.getenv("VLLM_TEST_NEMOTRON_PROXY")
    if not model:
        pytest.skip("Set VLLM_TEST_NEMOTRON_PROXY to an operator-covering checkpoint")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", "indexed")
    args = [
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--moe-backend",
        "humming",
        "--all2all-backend",
        "flashinfer_nvlink_one_sided",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--max-model-len",
        "9216",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "16384",
        "--block-size",
        "6768",
        "--mamba-block-size",
        "6768",
    ]
    with RemoteOpenAIServer(model, args, max_wait_seconds=900) as server:
        client = server.get_client()
        prompt_ids = [100 + i % 1000 for i in range(8192)]
        options = {"ignore_eos": True, "return_token_ids": True}
        generated = client.completions.create(
            model=model,
            prompt=prompt_ids,
            temperature=0,
            max_tokens=1024,
            logprobs=1,
            extra_body=options,
        ).choices[0]
        tokens = generated.model_extra["token_ids"]
        assert len(tokens) == 1024
        replay = client.completions.create(
            model=model,
            prompt=prompt_ids + tokens[:-1],
            temperature=0,
            max_tokens=1,
            logprobs=1,
            echo=True,
            extra_body=options,
        ).choices[0]
        assert replay.model_extra["token_ids"] == tokens[-1:]
        replay_scores = replay.logprobs.token_logprobs[8192:]
        assert len(replay_scores) == 1024
        for decode, prefill in zip(
            generated.logprobs.token_logprobs, replay_scores, strict=True
        ):
            assert math.isfinite(decode) and math.isfinite(prefill)
            assert struct.pack("f", decode) == struct.pack("f", prefill)


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_nemotron_copy_specs_cover_all_replay_buffers(monkeypatch, batch_invariant):
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncCalculator,
    )
    from vllm.model_executor.models import nemotron_h

    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    funcs = nemotron_h.NemotronHForCausalLM.get_mamba_state_copy_func()
    base = MambaStateCopyFuncCalculator.mamba2_state_copy_func()
    assert funcs[:2] == base
    assert len(funcs) == (5 if batch_invariant else 2)
    state = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    for copy_func in funcs[2:]:
        spec = copy_func(state, [0, 2, 1], 1, 1)
        assert spec.start_addr == state[2].data_ptr()
        assert spec.num_elements == state[2].numel()
    assert MambaStateCopyFuncCalculator.mamba2_state_copy_func() == base


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("ll_eligible", [False, True])
def test_nemotron_gate_bi_keeps_default_and_other_model_dispatch_isolated(
    monkeypatch, batch_invariant, ll_eligible
):
    """CPU constructor policy only; GPU arithmetic is a separate router gate."""
    from vllm.model_executor.models import nemotron_h

    inherited = {
        "allow_ll_bf16_gemm": ll_eligible,
        "allow_specialized_router_gemm": True,
        "allow_fp32_router_gemm": False,
        "allow_bf16x3_router_gemm": False,
        "allow_cublas_router_gemm": True,
    }
    constructor_calls = []

    def fake_parent_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        constructor_calls.append((args, kwargs))
        for name, value in inherited.items():
            setattr(self, name, value)

    monkeypatch.setattr(nemotron_h.GateLinear, "__init__", fake_parent_init)
    monkeypatch.setattr(nemotron_h.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    arguments = dict(
        out_dtype=torch.float32,
        force_fp32_compute=True,
        prefix="backbone.layers.1.mixer.gate",
    )
    before = nemotron_h.GateLinear(2688, 128, **arguments)
    candidate = nemotron_h.NemotronHGateLinear(2688, 128, **arguments)
    after = nemotron_h.GateLinear(2688, 128, **arguments)
    assert constructor_calls == [((2688, 128), arguments)] * 3
    expected = dict(inherited)
    if batch_invariant:
        expected["allow_ll_bf16_gemm"] = False
    for name in inherited:
        assert getattr(candidate, name) == expected[name]
        assert getattr(before, name) == inherited[name]
        assert getattr(after, name) == inherited[name]


def test_nemotron_h_lm_head_receives_quant_config():
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.dtype = None
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch("vllm.model_executor.models.nemotron_h.NemotronHModel") as MockModel,
        patch("vllm.model_executor.models.nemotron_h.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.nemotron_h.LogitsProcessor"),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockModel.return_value.has_moe = False

        NemotronHForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_relu2_fp8_fusion_uses_registry():
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = NemotronHMLP.__new__(NemotronHMLP)
    torch.nn.Module.__init__(mlp)
    mlp.up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.nemotron_h.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused
