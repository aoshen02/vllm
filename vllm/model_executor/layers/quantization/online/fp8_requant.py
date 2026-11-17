# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 routed experts re-quantized to MXFP4 at load (ROCm).

`VLLM_FP8_MOE_REQUANT=mxfp4` loads the checkpoint's block-FP8 expert
weights and, as each expert's weight and block scales arrive, dequantizes
them to bf16 and quantizes them to MXFP4 (e2m1 values, e8m0 block-32 scales)
for the aiter MXFP4 MoE kernels: half the expert bytes of FP8.
`VLLM_FP8_MOE_REQUANT_ACT` selects
the activation dtype: a8w4 (fp8 per-token; aiter's interleaved FlyDSL kernels,
the `AITER_MXFP4_BF16` backend; default), a4w4 (MXFP4 activations,
`AITER_MXFP4_MXFP4`) or a16w4 (bf16, `AITER_TRITON_MXFP4_BF16`).
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.online.loading import _load_quantized
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def dequant_block_fp8_to_bf16(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: list[int]
) -> torch.Tensor:
    """`[N, K]` fp8 x `[ceil(N/bn), ceil(K/bk)]` fp32 block scales -> bf16."""
    n, k = weight.shape
    bn, bk = block_size
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:, :k]
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


class Fp8BlockExpertsAsMxfp4FusedMoE(Mxfp4OnlineMoEMethod):
    """Block-FP8 routed experts stored as MXFP4 (VLLM_FP8_MOE_REQUANT=mxfp4).

    The experts are MXFP4 parameters. A checkpoint expert's fp8 weight and its
    block `weight_scale_inv` arrive in separate loads: the first waits for the
    other, then the pair is re-quantized and loaded (peak memory is the MXFP4
    model plus the unpaired experts). Rounding the values to the e2m1 grid with
    power-of-two block-32 scales is lossy: the weight error is of the order of
    an MXFP4 quantization of the bf16 weights.
    """

    # online quantization: the parameters do not map to checkpoint tensors
    uses_meta_device = True

    def __init__(self, quant_config, layer: RoutedExperts) -> None:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        act = envs.VLLM_FP8_MOE_REQUANT_ACT.lower()
        moe_config = layer.moe_config
        # a4w4 asks for MXFP4 activations (AITER_MXFP4_MXFP4). With bf16
        # activations the oracle picks AITER_MXFP4_BF16, whose interleaved
        # weight layout makes aiter quantize activations to fp8 (a8w4); true
        # bf16 activations need the aiter Triton backend.
        self.activation_quant_key = kMxfp4Dynamic if act == "a4w4" else None  # type: ignore[assignment]
        # The FP8 method only creates/loads the block-FP8 parameters; build it
        # before the MXFP4 backend override below (it resolves moe_backend too).
        self.fp8_method = Fp8MoEMethod(quant_config, layer)
        assert self.fp8_method.block_quant, "FP8 re-quant needs block-scaled experts"
        assert self.fp8_method.weight_block_size is not None
        self.fp8_block_size = list(self.fp8_method.weight_block_size)
        if act == "a16w4" and moe_config.moe_backend == "auto":
            moe_config.moe_backend = "aiter_triton_mxfp4_bf16"
        super().__init__(moe=moe_config)
        self.w13_precision_config = None
        self.w2_precision_config = None
        self.quant_config = quant_config
        logger.info_once(
            "FP8 block experts re-quantized to MXFP4 at load (%s, %s)",
            act,
            self.mxfp4_backend.value,
        )

    def get_fused_moe_quant_config(self, layer):
        # Triton-format backends keep the scales as PrecisionConfigs (set by
        # _setup_kernel), not as layer parameters.
        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
        ):
            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=self.w13_precision_config,
                w2_scale=self.w2_precision_config,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                layer=layer,
            )
        return super().get_fused_moe_quant_config(layer)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size != layer.moe_config.hidden_dim_unpadded or (
            intermediate_size_per_partition
            != layer.moe_config.intermediate_size_per_partition_unpadded
        ):
            raise ValueError(
                "FP8 -> MXFP4 re-quant does not pad expert dims (the FP8 "
                f"loader writes the unpadded checkpoint shapes): got "
                f"{hidden_size}x{intermediate_size_per_partition}"
            )
        device = torch.get_default_device()
        # The FP8 method names the parameters the checkpoint loads into. Built
        # on meta, its weights then become MXFP4; its block scales stay there,
        # never written: their loads only pair with the weights.
        with torch.device("meta"):
            self.fp8_method.create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
        for name, param in list(layer.named_parameters(recurse=False)):
            if name.endswith("weight_scale_inv"):
                continue
            if name in ("w13_weight", "w2_weight"):
                e, n, k = param.shape
                data = torch.zeros(e, n, k // 2, dtype=torch.uint8, device=device)
                scale = torch.nn.Parameter(
                    torch.zeros(
                        e, n, k // MXFP4_BLOCK_SIZE, dtype=torch.uint8, device=device
                    ),
                    requires_grad=False,
                )
                layer.register_parameter(f"{name}_scale", scale)
                set_weight_attrs(scale, extra_weight_attrs)
            else:
                data = torch.zeros_like(param, device=device)
            real = torch.nn.Parameter(data, requires_grad=False)
            real.__dict__.update(param.__dict__)
            layer.register_parameter(name, real)
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.weight_block_size = [1, MXFP4_BLOCK_SIZE]
        self._pending: dict[tuple, dict[str, tuple]] = {}
        for prefix in ("w13_", "w2_"):
            self._requant_on_load(layer, prefix)

    def _requant_on_load(self, layer: RoutedExperts, prefix: str) -> None:
        load = _load_quantized(layer, f"{prefix}weight", f"{prefix}weight_scale")
        pending = self._pending

        def make_loader(kind: str):
            def loader(param, loaded_weight, weight_name, shard_id, expert_id, **kw):
                if layer._map_global_expert_id_to_local_expert_id(expert_id) == -1:
                    return False
                weight_param = getattr(layer, f"{prefix}weight")
                copied = torch.Event()
                tensor = loaded_weight.to(weight_param.device, copy=True)
                copied.record()
                key = (prefix, shard_id, expert_id)
                got = pending.setdefault(key, {})
                got[kind] = (tensor, copied, weight_name, kw)
                if len(got) < 2:
                    return True
                del pending[key]
                for _, event, _, _ in got.values():
                    # held from another load, which may have run on another stream
                    event.wait()
                weight, _, name, kwargs = got["weight"]
                bf16 = dequant_block_fp8_to_bf16(
                    weight, got["scale"][0], self.fp8_block_size
                )
                q, s = mxfp4_quantize(bf16)
                quantized = (q.view(torch.uint8), s.view(torch.uint8))
                return load(
                    weight_param, quantized, name, shard_id, expert_id, **kwargs
                )

            return loader

        getattr(layer, f"{prefix}weight").weight_loader = make_loader("weight")
        getattr(layer, f"{prefix}weight_scale_inv").weight_loader = make_loader("scale")

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            raise ValueError(
                "VLLM_FP8_MOE_REQUANT=mxfp4 needs an MXFP4 MoE backend, and none "
                "is available on this platform."
            )
        if self._pending:
            unpaired = sorted(map(str, self._pending))
            self._pending.clear()
            raise RuntimeError(
                f"{layer.layer_name}: FP8 experts {unpaired} got only one of their "
                "weight and weight_scale_inv; both must be loaded in the same update"
            )
        delattr(layer, "w13_weight_scale_inv")
        delattr(layer, "w2_weight_scale_inv")
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        # The kernel-format conversion replaces or deletes them per backend.
        self._setup_kernel(
            layer,
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_scale.data,
            None,
            None,
        )
        layer._already_called_process_weights_after_loading = True
