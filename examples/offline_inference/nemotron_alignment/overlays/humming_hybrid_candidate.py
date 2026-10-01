"""Private pre-capture Nemotron indexed hybrid tuning."""

import copy
import hashlib
import json
from pathlib import Path

from humming_n128_contract import config_record, selected_schedule
from humming_hybrid_contract import hybrid_config

RECIPE = "humming-indexed-n256-stage4-small-warpn32-v1"
TENSOR_NAMES = (
    "w13_weight",
    "w2_weight",
    "w1_scale",
    "w2_scale",
    "g1_alphas",
    "g2_alphas",
)
GEOMETRY = {"w13": (2048, 2688, 192, 0), "w2": (2816, 1920, 128, 64)}


def source_sha(name):
    return hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()


def validate_serialized_tuning(table, serialized):
    if serialized != json.dumps(table):
        raise ValueError("Original serialized tuning was stale")


def tensor_inventory(layer, experts):
    tensors = {name: getattr(layer, name) for name in TENSOR_NAMES[:2]}
    tensors.update(
        {name: getattr(experts.quant_config, name) for name in TENSOR_NAMES[2:]}
    )
    return tensors, {
        name: dict(
            object_id=id(value),
            pointer=value.data_ptr(),
            storage_pointer=value.untyped_storage().data_ptr(),
            storage_offset=value.storage_offset(),
            shape=list(value.shape),
            stride=list(value.stride()),
            dtype=str(value.dtype),
            device=str(value.device),
        )
        for name, value in tensors.items()
    }


def validate_base(configs):
    if set(configs) != set(GEOMETRY):
        raise ValueError("Unexpected routed projections")
    for name, config in configs.items():
        actual = tuple(
            config[key] for key in ("shape_n", "shape_k", "pad_shape_n", "pad_shape_k")
        )
        if actual != GEOMETRY[name]:
            raise ValueError("Unsupported packed/logical geometry")
        required = dict(
            sm_version=100,
            num_experts=32,
            b_dtype="float4e2m1",
            a_dtype="bfloat16",
            c_dtype="bfloat16",
            bs_dtype="float8e4m3",
            as_dtype=None,
            input_quant_mode="none",
            input_scale_group_size=0,
            weight_scale_group_size=16,
            weight_scale_group_size_n=0,
            weight_scale_type="group",
            weight_scale_2_type="tensor",
            use_int_weight_scale=False,
            use_fused_e8m0_scale=False,
            has_zero_point=False,
            is_fp_zero_point=False,
            has_bias=False,
            mma_type="umma",
            use_packed_k_layout=False,
        )
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("Unsupported W4A16 packed recipe")


def validate_inventory(record, expected_moe_layers):
    if expected_moe_layers not in (2, 23):
        raise ValueError("Only reviewed proxy/full layer counts are supported")
    if (
        record["recipe"] != RECIPE
        or not record["installed_before_capture"]
        or record["actual_model_runner_module"] != "vllm.v1.worker.gpu.model_runner"
        or record["actual_model_runner_class"] != "GPUModelRunner"
        or record["bi"] is not True
        or record["tp"] != 1
        or record["ubatch_enabled"] is not False
    ):
        raise ValueError("Unsupported execution recipe")
    layers = record["layers"]
    if len(layers) != expected_moe_layers or len(
        {row["name"] for row in layers}
    ) != len(layers):
        raise ValueError("Wrong or duplicated routed layer inventory")
    for row in layers:
        if (
            row["local_experts"] != 32
            or row["global_experts"] != 128
            or row["activation"] != "relu2_no_mul"
        ):
            raise ValueError("Wrong expert geometry/activation")
        validate_base(row["base_configs"])
        compute = row["compute_config"]
        if row.get("compute_serialized_synced") is not True:
            raise ValueError("Compute JSON and Python recipe differ")
        if (
            compute.get("use_batch_invariant") is not True
            or compute.get("use_f16_accum") is not False
        ):
            raise ValueError("Require BI FP32 accumulation")
        expected = {
            name: hybrid_config(table) for name, table in row["original"].items()
        }
        if set(expected) != set(GEOMETRY) or row["candidate"] != expected:
            raise ValueError("Non-hybrid tuning mutation")
        changes = {
            name: sum(a != b for a, b in zip(row["original"][name], table, strict=True))
            for name, table in expected.items()
        }
        if changes != {"w13": 40, "w2": 40} or row["changed_intervals"] != changes:
            raise ValueError("Unexpected hybrid interval inventory")
        if not row["serialized_synced"]:
            raise ValueError("Tuning JSON and Python table differ")
        if (
            set(row["tensors_before"]) != set(TENSOR_NAMES)
            or row["tensors_before"] != row["tensors_after"]
        ):
            raise ValueError("Packed tensors or scales replaced")
        for state in row["tensors_before"].values():
            if state["pointer"] <= 0 or not state["device"].startswith("cuda"):
                raise ValueError("Expected live CUDA storage")
        for observation in row.get("observed_selections", []):
            if (
                observation["calls"] <= 0
                or type(observation["valid_shape_m"]) is not int
            ):
                raise ValueError("Invalid host selection observation")
            for name in GEOMETRY:
                expected_selection = selected_schedule(
                    row["candidate"][name], observation["valid_shape_m"]
                )
                if observation["selected"][name] != expected_selection:
                    raise ValueError("Actual selected interval mismatch")
                if (
                    observation["tuning_sha256"][name]
                    != hashlib.sha256(
                        json.dumps(row["candidate"][name]).encode()
                    ).hexdigest()
                ):
                    raise ValueError("Actual serialized tuning mismatch")
    return record


def install(model_runner, expected_moe_layers):
    from vllm import envs
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    from vllm.model_executor.models.nemotron_h import NemotronHMoE

    if (
        type(model_runner).__module__ != "vllm.v1.worker.gpu.model_runner"
        or type(model_runner).__name__ != "GPUModelRunner"
        or model_runner.cudagraph_manager is not None
        or model_runner.ubatch_runner is not None
        or hasattr(model_runner, "_humming_stage4_state")
    ):
        raise ValueError("Install once after V2 model load and before capture")
    if not envs.VLLM_BATCH_INVARIANT or get_tensor_model_parallel_world_size() != 1:
        raise ValueError("Only BI1/TP1 may install stage4")
    model = model_runner.get_model()
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if type(module) is NemotronHMoE
    ]
    if len(selected) != expected_moe_layers:
        raise ValueError("Unexpected actual Nemotron MoE count")
    record = dict(
        recipe=RECIPE,
        adapter_sha256=source_sha(Path(__file__).name),
        transform_sha256=source_sha("humming_hybrid_contract.py"),
        installed_before_capture=True,
        actual_model_runner_module=type(model_runner).__module__,
        actual_model_runner_class=type(model_runner).__name__,
        bi=True,
        tp=1,
        ubatch_enabled=False,
        layers=[],
        scope="instance tuning only; tensor identities, not tensor content hashes",
    )
    state = []
    for name, module in selected:
        layer = module.experts.routed_experts
        method = layer.quant_method
        if (
            type(method) is not ModelOptNvFp4FusedMoE
            or method.nvfp4_backend.value != "HUMMING"
            or not method.use_a16
        ):
            raise ValueError("Require original Humming W4A16 quant method")
        experts = method.moe_kernel.fused_experts
        if type(experts) is not HummingIndexedExperts:
            raise ValueError("Require actual indexed Humming implementation")
        if json.loads(experts.compute_config_str) != experts.compute_config:
            raise ValueError("Original serialized compute recipe was stale")
        original = {
            name: copy.deepcopy(getattr(experts, name + "_tuning_config"))
            for name in GEOMETRY
        }
        for projection in GEOMETRY:
            validate_serialized_tuning(
                original[projection],
                getattr(experts, projection + "_tuning_config_str"),
            )
        candidate = {name: hybrid_config(table) for name, table in original.items()}
        tensors, before = tensor_inventory(layer, experts)
        row = dict(
            name=name,
            local_experts=experts.num_experts,
            global_experts=experts.global_num_experts,
            activation=layer.activation.value,
            base_configs={
                name: config_record(experts.humming_configs[name]) for name in GEOMETRY
            },
            compute_config=copy.deepcopy(experts.compute_config),
            compute_serialized_synced=True,
            original=original,
            candidate=candidate,
            changed_intervals={
                key: sum(a != b for a, b in zip(original[key], value, strict=True))
                for key, value in candidate.items()
            },
            tensors_before=before,
            tensors_after=copy.deepcopy(before),
            serialized_synced=True,
            observed_selections=[],
        )
        record["layers"].append(row)
        state.append(
            dict(layer=layer, method=method, experts=experts, tensors=tensors, row=row)
        )
    validate_inventory(record, expected_moe_layers)
    for item in state:
        for name, table in item["row"]["candidate"].items():
            setattr(item["experts"], name + "_tuning_config", copy.deepcopy(table))
            setattr(item["experts"], name + "_tuning_config_str", json.dumps(table))
        item["row"]["tensors_after"] = tensor_inventory(item["layer"], item["experts"])[
            1
        ]
    validate_inventory(record, expected_moe_layers)
    model_runner._humming_stage4_state = state
    return record


def audit(model_runner, record, expected_moe_layers):
    for item, row in zip(
        model_runner._humming_stage4_state, record["layers"], strict=True
    ):
        experts = item["experts"]
        if (
            item["layer"].quant_method is not item["method"]
            or item["layer"].quant_method.moe_kernel.fused_experts is not experts
        ):
            raise ValueError("Installed expert implementation replaced")
        if experts.compute_config != row["compute_config"]:
            raise ValueError("Compute recipe changed after install")
        if json.loads(experts.compute_config_str) != row["compute_config"]:
            raise ValueError("Serialized compute recipe changed after install")
        for name, table in row["candidate"].items():
            if (
                config_record(experts.humming_configs[name])
                != row["base_configs"][name]
            ):
                raise ValueError("Packed geometry changed after install")
            if getattr(experts, name + "_tuning_config") != table or getattr(
                experts, name + "_tuning_config_str"
            ) != json.dumps(table):
                raise ValueError("Stage4 tuning changed after install")
        row["tensors_after"] = tensor_inventory(item["layer"], experts)[1]
    return validate_inventory(record, expected_moe_layers)


def observe_proxy_selections(model_runner):
    """Proxy-only host observer; never installed in a performance worker."""
    for item in model_runner._humming_stage4_state:
        experts, row = item["experts"], item["row"]
        native = experts.prepare_humming_moe_kwargs
        seen = {}

        def observe(*args, _native=native, _row=row, _seen=seen, **kwargs):
            result = _native(*args, **kwargs)
            estimates = [kw["valid_shape_m"] for kw in result[:2]]
            if type(estimates[0]) is not int or estimates[0] != estimates[1]:
                raise ValueError("Unexpected actual host valid_shape_m")
            estimate = estimates[0]
            if estimate not in _seen:
                observation = dict(
                    valid_shape_m=estimate, calls=0, selected={}, tuning_sha256={}
                )
                for name, kw in zip(GEOMETRY, result[:2], strict=True):
                    serialized = kw["tuning_config"]
                    if serialized != json.dumps(_row["candidate"][name]):
                        raise ValueError("Returned tuning is not installed stage4")
                    observation["tuning_sha256"][name] = hashlib.sha256(
                        serialized.encode()
                    ).hexdigest()
                    observation["selected"][name] = selected_schedule(
                        _row["candidate"][name], estimate
                    )
                _seen[estimate] = observation
                _row["observed_selections"].append(observation)
            _seen[estimate]["calls"] += 1
            return result

        experts.prepare_humming_moe_kwargs = observe
    return "proxy host warmup/capture/eager selection; not Graph replay counts"
