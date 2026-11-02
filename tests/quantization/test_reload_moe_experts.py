# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct reload keeps every MoE experts kernel tensor at its cold-start address.

Direct reload reruns post-load processing on every weight update, and that
rebuilds the experts object. CUDA graphs captured at startup keep reading the
cold-start addresses, so a tensor the rerun allocates anew is read stale (or
freed). `test_reload_keeps_experts_tensors` runs one case per (quantization,
`moe_backend`, experts class), enumerated from the MoE oracles rather than
listed by hand, and checks every tensor the kernel reads.

`test_every_experts_class_is_covered` (no GPU) fails when a concrete
`FusedMoEExperts` subclass is neither reached by those cases nor listed in
`UNCOVERED_EXPERTS` with a reason. If it fails for a new class:
  * reach it: map its backend in the oracle and add a quantization to
    `QUANTS` if none routes there; or
  * list it in `UNCOVERED_EXPERTS` with the reason it cannot run here.
Kept constants (`kept_constants` + `keep_constants(layer)`) are what makes
the new class pass; see `FusedMoEExperts.keep_constants`.
"""

import ast
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest
import torch

import vllm
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsMxfp4,
)
from vllm.model_executor.layers.fused_moe.oracle import (
    fp8,
    int8,
    int_wna16,
    mxfp4,
    mxfp8,
    nvfp4,
    unquantized,
    w4a8,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.platforms import current_platform

from .test_keep_param_objects import (
    A8_FP8_TOKEN,
    CT_MXFP4,
    CT_NVFP4,
    GPT_OSS_SWIGLU,
    INT,
    NVFP4,
    W4_GROUP,
    W8_CHANNEL,
    _cold_start,
    _ct,
    _ct_int,
    _kernel_tensors,
    _load,
    _moe,
    _same_bytes,
)

ExpertsCls = type[mk.FusedMoEExperts]


def _one(to_cls, to_backend) -> Callable[[str], list[ExpertsCls]]:
    return lambda name: to_cls(to_backend(name))


# oracle -> the experts classes an explicit `moe_backend` selects from
ORACLES: dict[str, Callable[[str], list[ExpertsCls]]] = {
    "unquantized": _one(
        unquantized.backend_to_kernel_cls, unquantized.map_unquantized_backend
    ),
    "fp8": _one(fp8.backend_to_kernel_cls, fp8.map_fp8_backend),
    "nvfp4": _one(nvfp4.backend_to_kernel_cls, nvfp4.map_nvfp4_backend),
    "mxfp4": lambda name: [
        cls
        for backend in mxfp4.map_mxfp4_backend(name)
        for cls in mxfp4.backend_to_kernel_cls(backend)
    ],
    "mxfp8": _one(
        mxfp8._mxfp8_backend_to_kernel_cls, lambda n: mxfp8._BACKEND_NAME_MAP[n]
    ),
    "int8": _one(int8.backend_to_kernel_cls, int8.map_int8_backend),
    "wna16": _one(int_wna16.backend_to_kernel_cls, int_wna16.map_wna16_backend),
    "w4a8": _one(w4a8.backend_to_kernel_cls, lambda n: w4a8.W4A8MoeBackend[n.upper()]),
}

A8_TOKEN = INT | dict(num_bits=8, strategy="token", dynamic=True)
W8_FP8_CHANNEL = dict(
    num_bits=8, type="float", strategy="channel", symmetric=True, dynamic=False
)
# (id, oracle, MoE layer builder): one checkpoint format per post-load path
QUANTS: list[tuple[str, str, Callable]] = [
    ("unquantized", "unquantized", lambda: _moe(None)),
    ("fp8-static", "fp8", lambda: _moe(Fp8Config(activation_scheme="static"))),
    ("fp8-block", "fp8", lambda: _moe(Fp8Config(weight_block_size=[128, 128]))),
    # the FP8 path that allows vLLM CUTLASS (TritonOrCutlassExperts)
    (
        "ct-w8a8-fp8",
        "fp8",
        lambda: _moe(_ct_int(W8_FP8_CHANNEL, A8_FP8_TOKEN, "float-quantized")),
    ),
    (
        "modelopt-nvfp4",
        "nvfp4",
        lambda: _moe(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
    ),
    ("ct-nvfp4", "nvfp4", lambda: _moe(_ct(*CT_NVFP4))),
    ("mxfp4", "mxfp4", lambda: _moe(Mxfp4Config(), has_bias=True, **GPT_OSS_SWIGLU)),
    (
        "modelopt-mxfp8",
        "mxfp8",
        lambda: _moe(ModelOptMxFp8Config.from_config({"quant_method": "mxfp8"})),
    ),
    (
        "ct-w8a8-int8",
        "int8",
        lambda: _moe(_ct_int(W8_CHANNEL, A8_TOKEN, "int-quantized")),
    ),
    ("ct-w4a16", "wna16", lambda: _moe(_ct_int(W4_GROUP))),
    ("ct-w4a8-fp8", "w4a8", lambda: _moe(_ct_int(W4_GROUP, A8_FP8_TOKEN))),
]


# Quantization methods that pick the experts class themselves:
# (id, MoE layer builder, moe_backend, experts class)
SELF_SELECTED: list[tuple[str, Callable, str, ExpertsCls]] = [
    ("ct-mxfp4", lambda: _moe(_ct(*CT_MXFP4)), "auto", CutlassExpertsMxfp4),
]


def _candidates(oracle: str | None, backend: str) -> list[ExpertsCls]:
    if oracle is None:
        return []
    try:
        return ORACLES[oracle](backend)
    except (KeyError, ValueError):  # the oracle does not take this backend
        return []


def _name(cls: type) -> str:
    return f"{cls.__module__.rsplit('.', 1)[-1]}.{cls.__qualname__}"


def _cases() -> list:
    cases = []
    for quant, oracle, build in QUANTS:
        for backend in get_args(MoEBackend):
            for cls in dict.fromkeys(_candidates(oracle, backend)):
                case_id = f"{quant}-{backend}-{cls.__name__}"
                cases.append(_param(case_id, build, oracle, backend, cls))
    for quant, build, backend, cls in SELF_SELECTED:
        case_id = f"{quant}-{backend}-{cls.__name__}"
        cases.append(_param(case_id, build, None, backend, cls))
    return cases


def _param(case_id: str, *args):
    marks = []
    if reason := KNOWN_STALE.get(case_id):
        marks.append(pytest.mark.xfail(reason=reason, strict=True))
    return pytest.param(*args, id=case_id, marks=marks)


_BATCHED = "not covered yet: batched activation format (DP/EP all2all only)"
_CPU = "CPU platform only"
_XPU = "XPU platform only"
_LORA = "not covered yet: selected only with LoRA enabled"
_MUSE = "not covered yet: built by the Muse Spark MoE runner, not by an oracle"
# Concrete experts classes no case above reaches ("module.Class"), and why.
UNCOVERED_EXPERTS: dict[str, str] = {
    "batched_deep_gemm_moe.BatchedDeepGemmExperts": _BATCHED,
    "cutlass_moe.CutlassBatchedExpertsFp8": _BATCHED,
    "flashinfer_cutedsl_batched_moe.FlashInferCuteDSLBatchedExperts": _BATCHED,
    "fused_batched_moe.NaiveBatchedExperts": _BATCHED,
    "marlin_moe.BatchedMarlinExperts": _BATCHED,
    "cpu_int4_moe.CPUExpertsInt4": _CPU,
    "cpu_moe.ArmCPUExpertsInt8": _CPU,
    "cpu_moe.ArmCPUUnquantizedExperts": _CPU,
    "cpu_moe.CPUExpertsFp8": _CPU,
    "cpu_moe.CPUExpertsFp8W8A8": _CPU,
    "cpu_moe.CPUExpertsInt4": _CPU,
    "cpu_moe.CPUExpertsInt8": _CPU,
    "cpu_moe.CPUExpertsMxfp4": _CPU,
    "cpu_moe.CPUUnquantizedExperts": _CPU,
    "cpu_moe.PowerCPUExpertsInt8": _CPU,
    "cpu_moe.PowerCPUUnquantizedExperts": _CPU,
    "cpu_moe.X86CPUUnquantizedExperts": _CPU,
    "cpu_moe.ZenCPUExpertsInt8": _CPU,
    "zentorch_moe.ZentorchExpertsInt4": _CPU,
    "xpu_moe.XPUExperts": _XPU,
    "xpu_moe.XPUExpertsBlockFp8": _XPU,
    "xpu_moe.XPUExpertsFp8": _XPU,
    "xpu_moe.XPUExpertsMxFp4": _XPU,
    "xpu_moe.XPUExpertsMxFp8": _XPU,
    "xpu_moe.XPUExpertsWNA16": _XPU,
    "trtllm_lora_moe.TrtLlmBf16LoRAExperts": _LORA,
    "trtllm_lora_moe.TrtLlmMxfp4LoRAExperts": _LORA,
    "muse_moe_runner.MuseSparkAiterExperts": _MUSE,
    "muse_moe_runner.MuseSparkTritonExperts": _MUSE,
    "muse_moe_runner.MuseSparkTrtLlmNvFp4Experts": _MUSE,
    "muse_moe_runner.MuseSparkTrtLlmNvFp4ExpertsModular": _MUSE,
    "cutlass_moe.CutlassExpertsFp8": "runs inside TritonOrCutlassExperts (checked)",
    "moonep_experts.MoonEPExperts": "not covered yet: MoonEP kernels (EP) only",
    "mxfp8_emulation_moe.Mxfp8EmulationTritonExperts": (
        "not covered yet: auto-selected only without a native MXFP8 kernel"
    ),
    "mxfp8_emulation_moe.Mxfp8TritonExpertsBase": "base class: no oracle selects it",
}

# Cases that fail today; a fix drops the entry (strict: a fixed case fails
# until its entry is removed).
_LOCKS = "stale on rerun: experts.locks (a new Humming locks buffer per build)"
KNOWN_STALE: dict[str, str] = {
    "fp8-static-humming-HummingIndexedExperts": _LOCKS,
    "ct-w8a8-fp8-humming-HummingIndexedExperts": _LOCKS,
    "modelopt-nvfp4-humming-HummingIndexedExperts": _LOCKS,
    "mxfp4-humming-HummingIndexedExperts": _LOCKS,
    "modelopt-mxfp8-humming-HummingIndexedExperts": _LOCKS,
    "ct-w8a8-int8-humming-HummingIndexedExperts": _LOCKS,
    "ct-w4a8-fp8-humming-HummingIndexedExperts": _LOCKS,
    # the rerun fails before the address check
    "fp8-block-humming-HummingIndexedExperts": "rerun: KeyError 'weight_scale_inv'",
    "ct-nvfp4-humming-HummingIndexedExperts": "rerun: KeyError 'weight_global_scale'",
    "ct-w4a16-humming-HummingIndexedExperts": "rerun: KeyError 'weight_packed'",
}


def _force(cls: ExpertsCls, candidates: list[ExpertsCls], monkeypatch) -> None:
    """Make the oracle select `cls` from `candidates`: every other candidate
    reports the config unsupported, with `cls`'s reason if it has one."""
    own = inspect.getattr_static(cls, "is_supported_config")
    monkeypatch.setattr(cls, "is_supported_config", own)

    def unsupported(_, *args):
        reason = own.__func__(cls, *args)[1]
        return False, f"{cls.__name__}: {reason or 'supported, not selected'}"

    for other in candidates:
        if other is not cls and not issubclass(cls, other):
            monkeypatch.setattr(other, "is_supported_config", staticmethod(unsupported))


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="KNOWN_STALE and the skips were measured on SM100",
)
@pytest.mark.parametrize("build,oracle,backend,cls", _cases())
def test_reload_keeps_experts_tensors(
    build: Callable,
    oracle: str | None,
    backend: str,
    cls: ExpertsCls,
    default_vllm_config,
    dist_init,
    monkeypatch,
) -> None:
    """After a direct reload, every tensor the kernel reads (layer parameters,
    experts object, quant config) keeps its cold-start address and holds what
    a cold start on the new weights computes."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    _force(cls, _candidates(oracle, backend), monkeypatch)
    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.moe_backend = backend
    try:
        layer, method, names = _cold_start(build, default_vllm_config, seed=0)
    except Exception as e:  # hardware, missing library, unsupported config
        pytest.skip(f"cold start: {type(e).__name__}: {str(e)[:300]}")
    if type(selected := method.moe_kernel.fused_experts) is not cls:
        pytest.skip(f"the quantization method picks {type(selected).__name__}")
    cold = {n: t.detach() for n, t in _kernel_tensors(layer, method).items()}

    begin_direct_reload(layer)
    torch.manual_seed(1)
    _load(layer, names)
    finish_direct_reload(layer, default_vllm_config.model_config, torch.device("cuda"))
    reloaded = _kernel_tensors(layer, method)
    assert {n: t.data_ptr() for n, t in reloaded.items()} == {
        n: t.data_ptr() for n, t in cold.items()
    }

    fresh = _kernel_tensors(*_cold_start(build, default_vllm_config, seed=1)[:2])
    # bytes: random FP8/E8M0 payloads include NaN encodings
    assert [n for n in reloaded if not _same_bytes(reloaded[n], fresh[n])] == []


def _experts_modules() -> set[str]:
    """Modules defining a `FusedMoEExperts` subclass, found from the source
    (by base class name) so unrelated modules are not imported."""
    root = Path(vllm.__file__).parent
    bases: dict[str, set[str]] = {}  # class name -> base names
    where: dict[str, set[str]] = {}  # class name -> modules
    for path in root.rglob("*.py"):
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.ClassDef):
                names = {
                    ast.unparse(b).split("[")[0].split(".")[-1] for b in node.bases
                }
                bases.setdefault(node.name, set()).update(names)
                where.setdefault(node.name, set()).add(module)
    experts = {"FusedMoEExperts"}
    while new := {c for c, b in bases.items() if b & experts} - experts:
        experts |= new
    return {m for c in experts for m in where[c]}


def _all_experts_classes() -> set[str]:
    """Concrete `FusedMoEExperts` subclasses in vLLM."""
    for module in sorted(_experts_modules()):
        importlib.import_module(module)
    found, todo = set(), [mk.FusedMoEExperts]
    while todo:
        cls = todo.pop()
        todo += cls.__subclasses__()
        if cls.__module__.startswith("vllm.") and not inspect.isabstract(cls):
            found.add(_name(cls))
    return found


def test_every_experts_class_is_covered() -> None:
    reached = {
        _name(cls)
        for _, oracle, _ in QUANTS
        for backend in get_args(MoEBackend)
        for cls in _candidates(oracle, backend)
    } | {_name(cls) for *_, cls in SELF_SELECTED}
    classes = _all_experts_classes()
    missing = sorted(classes - reached - UNCOVERED_EXPERTS.keys())
    assert not missing, (
        "MoE experts classes not covered by test_reload_keeps_experts_tensors:"
        + "".join(f"\n  {name}" for name in missing)
        + "\nReach each one (map its backend in the MoE oracle and add a "
        "quantization to QUANTS if none routes there), or add it to "
        "UNCOVERED_EXPERTS in this file with the reason it cannot run there."
    )
    stale = sorted(UNCOVERED_EXPERTS.keys() & reached)
    gone = sorted(UNCOVERED_EXPERTS.keys() - classes)
    assert not stale + gone, (
        f"remove from UNCOVERED_EXPERTS: {stale} (now covered), {gone} (no "
        "longer a concrete experts class)"
    )
