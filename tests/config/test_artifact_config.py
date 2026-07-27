# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ArtifactConfig, VllmConfig

pytestmark = pytest.mark.cpu_test


def _verify_routed_experts_config(**overrides):
    model_config = SimpleNamespace(
        enable_return_routed_experts=True,
        is_moe=True,
        runner_type="generate",
        is_encoder_decoder=False,
        is_multimodal_model=False,
        is_diffusion=False,
    )
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
        ubatch_size=0,
        enable_elastic_ep=False,
    )
    device_config = SimpleNamespace(device_type="cuda")
    cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    artifact_config = ArtifactConfig()
    configs = {
        "model": model_config,
        "parallel": parallel_config,
        "device": device_config,
        "cache": cache_config,
        "artifact": artifact_config,
    }
    for name, value in overrides.items():
        target, attribute = name.split("__", 1)
        setattr(configs[target], attribute, value)

    config = VllmConfig.__new__(VllmConfig)
    config.model_config = model_config
    config.parallel_config = parallel_config
    config.device_config = device_config
    config.cache_config = cache_config
    config.artifact_config = artifact_config
    config.speculative_config = None
    config.ec_transfer_config = None
    config._verify_return_routed_experts_compatibility()


def test_artifact_config_defaults_to_shm_and_bounds_mooncake_memory():
    config = ArtifactConfig()

    assert config.backend == "shm"
    assert config.shm_dir == "/dev/shm/vllm-artifacts"
    assert config.mooncake_staging_buffer_bytes == 64 << 20


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"device__device_type": "cpu"}, "device_type='cpu'"),
        ({"model__is_moe": False}, "non-MoE models"),
        ({"model__runner_type": "pooling"}, "runner_type='pooling'"),
        ({"model__is_encoder_decoder": True}, "encoder-decoder models"),
        ({"model__is_multimodal_model": True}, "multimodal models"),
        ({"model__is_diffusion": True}, "discrete diffusion models"),
        ({"parallel__pipeline_parallel_size": 2}, "PP=2"),
        ({"parallel__decode_context_parallel_size": 2}, "DCP=2"),
        ({"parallel__prefill_context_parallel_size": 2}, "PCP=2"),
        ({"parallel__enable_dbo": True}, "dual batch overlap"),
        ({"parallel__ubatch_size": 2}, "executor microbatching"),
        ({"parallel__enable_elastic_ep": True}, "elastic expert parallelism"),
        ({"cache__kv_sharing_fast_prefill": True}, "KV sharing fast prefill"),
        ({"artifact__shm_dir": "/tmp/artifacts"}, "outside /dev/shm"),
    ],
)
def test_artifact_connector_rejects_unsupported_modes(overrides, error):
    with pytest.raises(ValueError, match=error):
        _verify_routed_experts_config(**overrides)


def test_artifact_connector_accepts_tp_execution():
    _verify_routed_experts_config()


def test_mooncake_backend_does_not_require_a_shm_object_directory():
    _verify_routed_experts_config(
        artifact__backend="mooncake",
        artifact__shm_dir="/tmp/not-used",
    )


def test_artifact_connector_accepts_speculative_decoding():
    config = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace(
        enable_return_routed_experts=True,
        is_moe=True,
        runner_type="generate",
        is_encoder_decoder=False,
        is_multimodal_model=False,
        is_diffusion=False,
    )
    config.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
        ubatch_size=0,
        enable_elastic_ep=False,
    )
    config.device_config = SimpleNamespace(device_type="cuda")
    config.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    config.artifact_config = ArtifactConfig()
    config.speculative_config = SimpleNamespace()
    config.ec_transfer_config = None

    config._verify_return_routed_experts_compatibility()


@pytest.mark.parametrize(
    ("connector", "role", "extra_config"),
    [
        ("NixlConnector", "kv_both", {}),
        ("MooncakeConnector", "kv_both", {}),
        ("OffloadingConnector", "kv_producer", {}),
        ("OffloadingConnector", "kv_both", {"spec_name": "OtherSpec"}),
    ],
)
def test_artifact_connector_rejects_unsupported_kv_transfer(
    connector, role, extra_config
):
    config = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace(enable_return_routed_experts=True)
    config.kv_transfer_config = SimpleNamespace(
        is_kv_transfer_instance=True,
        kv_connector=connector,
        kv_role=role,
        kv_connector_extra_config=extra_config,
    )

    with pytest.raises(ValueError, match="only supports the CPU KV offload"):
        config._verify_return_routed_experts_kv_compatibility()


def test_artifact_connector_accepts_supported_cpu_kv_offload():
    config = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace(enable_return_routed_experts=True)
    config.kv_transfer_config = SimpleNamespace(
        is_kv_transfer_instance=True,
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"spec_name": "CPUOffloadingSpec"},
    )

    config._verify_return_routed_experts_kv_compatibility()


def test_artifact_guards_are_inactive_when_capture_is_disabled():
    config = VllmConfig.__new__(VllmConfig)
    config.model_config = SimpleNamespace(enable_return_routed_experts=False)

    config._verify_return_routed_experts_compatibility()
