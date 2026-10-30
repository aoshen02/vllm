# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 LoRA adapter-name contract (CPU only)."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import process_packed_modules_mapping
from vllm.model_executor.models.pixtral import PixtralForConditionalGeneration
from vllm.models.mistral_large4.nvidia.model import (
    _LORA_PACKED_MODULES,
    MistralLarge4ForCausalLM,
)

NUM_EXPERTS = 2
RANK = 4

# Module suffixes vLLM finds for ML4 (LinearBase / MoERunner).
ML4_LORA_MODULES = [
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "wq_b",
    "wk_weights_proj",
    "gate",
    "gate_up_proj",
    "down_proj",
    "experts",
]

# Adapter module (without prefix) -> (in, out).
ADAPTER_MODULES = {
    "layers.0.self_attn.q_proj": (16, 32),
    "layers.0.self_attn.kv_a_proj_with_mqa": (16, 12),
    "layers.0.self_attn.o_proj": (32, 16),
    "layers.0.mlp.gate_proj": (16, 24),
    "layers.0.mlp.up_proj": (16, 24),
    "layers.0.mlp.down_proj": (24, 16),
    "layers.1.mlp.shared_experts.gate_proj": (16, 8),
    "layers.1.mlp.shared_experts.up_proj": (16, 8),
    "layers.1.mlp.shared_experts.down_proj": (8, 16),
    **{
        f"layers.1.mlp.experts.{e}.{proj}": shape
        for e in range(NUM_EXPERTS)
        for proj, shape in (
            ("gate_proj", (16, 8)),
            ("up_proj", (16, 8)),
            ("down_proj", (8, 16)),
        )
    },
}


class _ModelStub(nn.Module):
    is_3d_moe_weight = False
    get_expert_mapping = MistralLarge4ForCausalLM.get_expert_mapping

    def __init__(self, packed_modules_mapping: dict[str, list[str]]) -> None:
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping
        self.config = SimpleNamespace(n_routed_experts=NUM_EXPERTS)


def _expected_lora_modules(packed_modules_mapping: dict[str, list[str]]) -> set[str]:
    with mock.patch("vllm.lora.utils.is_moe_model", return_value=True):
        packed = process_packed_modules_mapping(_ModelStub(packed_modules_mapping))
    # As LoRAModelManager / WorkerLoRAManager._load_adapter.
    expected: list[str] = []
    for module in ML4_LORA_MODULES:
        expected.extend(packed.get(module, [module]))
        if module == "experts":
            expected.append(module)
    return set(expected)


def _write_adapter(path, modules: dict[str, tuple[int, int]], prefix: str) -> None:
    tensors = {}
    for name, (n_in, n_out) in modules.items():
        key = f"base_model.model.{prefix}{name}"
        tensors[f"{key}.lora_A.weight"] = torch.randn(RANK, n_in)
        tensors[f"{key}.lora_B.weight"] = torch.randn(n_out, RANK)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    config = {"r": RANK, "lora_alpha": 2 * RANK, "target_modules": "all-linear"}
    (path / "adapter_config.json").write_text(json.dumps(config))


def _load(path, packed_modules_mapping, weights_mapper) -> LoRAModel:
    return LoRAModel.from_local_checkpoint(
        str(path),
        _expected_lora_modules(packed_modules_mapping),
        peft_helper=PEFTHelper.from_local_dir(str(path), None),
        lora_model_id=1,
        device="cpu",
        weights_mapper=weights_mapper,
    )


# (packed mapping of the served model, adapter key prefix, weights mapper,
#  vLLM module prefix)
SERVED_MODELS = [
    (_LORA_PACKED_MODULES, "model.", None, "model."),
    (
        PixtralForConditionalGeneration.packed_modules_mapping,
        "language_model.model.",
        PixtralForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper(),
        "language_model.model.",
    ),
    (
        PixtralForConditionalGeneration.packed_modules_mapping,
        "model.language_model.",
        PixtralForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper(),
        "language_model.model.",
    ),
]
SERVED_IDS = ["text", "pixtral-vllm-names", "pixtral-hf-names"]


class TestML4LoRANames:
    def test_expert_mapping_names(self):
        names = {
            weight_name
            for _, weight_name, _, _ in _ModelStub(
                _LORA_PACKED_MODULES
            ).get_expert_mapping()
        }
        assert names == {
            f"experts.{e}.{proj}."
            for e in range(NUM_EXPERTS)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }

    def test_pixtral_wrapper_agrees_on_packed_names(self):
        pixtral = PixtralForConditionalGeneration.packed_modules_mapping
        for name, children in _LORA_PACKED_MODULES.items():
            assert pixtral[name] == children

    @pytest.mark.parametrize(
        "packed,key_prefix,mapper,module_prefix", SERVED_MODELS, ids=SERVED_IDS
    )
    def test_contract_keys_load(
        self, tmp_path, packed, key_prefix, mapper, module_prefix
    ):
        _write_adapter(tmp_path, ADAPTER_MODULES, key_prefix)
        lora = _load(tmp_path, packed, mapper)
        assert set(lora.loras) == {module_prefix + m for m in ADAPTER_MODULES}

    @pytest.mark.parametrize(
        "module",
        [
            "layers.0.mlp.gate_up_proj",
            "layers.1.mlp.shared_experts.gate_up_proj",
            "layers.1.mlp.experts.0.w1",
            "layers.1.mlp.experts.0.w2",
            "layers.1.mlp.experts.0.w3",
        ],
        ids=["dense-fused", "shared-fused", "expert-w1", "expert-w2", "expert-w3"],
    )
    @pytest.mark.parametrize(
        "packed,key_prefix,mapper,module_prefix", SERVED_MODELS, ids=SERVED_IDS
    )
    def test_checkpoint_style_keys_rejected(
        self, tmp_path, module, packed, key_prefix, mapper, module_prefix
    ):
        _write_adapter(tmp_path, {module: (16, 16)}, key_prefix)
        with pytest.raises(ValueError, match="expected target modules"):
            _load(tmp_path, packed, mapper)
