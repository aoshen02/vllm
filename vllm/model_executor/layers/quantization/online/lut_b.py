# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online LUT-B quantization of routed MoE experts (MXFP8 activations)."""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lut_b_moe import (
    LutBEmulationTritonExperts,
    LutBSm107Experts,
    make_lut_b_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    LUT_B_CODEBOOK_SIZE,
    LUT_B_PACKED_TILE_BYTES,
    LUT_B_SCALE_BLOCK_K,
    LUT_B_SM107_TILE_K,
    LUT_B_SM107_TILE_N,
    dequantize_lut_b,
    lut_b_to_sm107_tiles,
    quantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kLutBStatic,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import (
    replace_parameter,
    retire_parameter,
    set_weight_attrs,
)

logger = init_logger(__name__)


class LutBOnlineMoEMethod(OnlineMoEMethodBase):
    """Quantize BF16/FP16 routed experts to LUT-B while loading.

    Weights use 3-bit indices into one E4M3 codebook (with a UE8M0 scale) per
    8x64 tile and activations are MXFP8. Rubin runs the native LUT-B tensor
    core kernel; other GPUs emulate the same numerics.

    Each expert tensor is quantized in its weight loader into the LUT-B
    layout (`<w>_weight` packed indices, `<w>_codebooks`, `<w>_scales`), which
    TP shards like the weight since tiles never straddle a shard.
    """

    supports_direct_reload = True  # unless set_requantization_source

    def __init__(self, *, moe: FusedMoEConfig):
        super().__init__(moe)
        if moe.has_bias:
            raise ValueError("LUT-B online MoE quantization does not support bias.")

        if moe.moe_backend not in ("auto", "emulation"):
            raise ValueError(
                f"moe_backend={moe.moe_backend!r} is not supported for LUT-B "
                "MoE; expected 'auto' or 'emulation'."
            )
        native_shape = (
            moe.hidden_dim % LUT_B_SM107_TILE_K == 0
            and moe.intermediate_size_per_partition % LUT_B_SM107_TILE_N == 0
        )
        supported, reason = LutBSm107Experts.is_supported_config(
            LutBSm107Experts,
            moe,
            kLutBStatic,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
        self.experts_cls: type[LutBSm107Experts | LutBEmulationTritonExperts]
        if moe.moe_backend == "emulation":
            self.experts_cls = LutBEmulationTritonExperts
        elif supported and native_shape:
            self.experts_cls = LutBSm107Experts
        else:
            self.experts_cls = LutBEmulationTritonExperts
            if reason is None:
                reason = "hidden and intermediate sizes must be multiples of 128"
            logger.info_once("LUT-B MoE falls back to emulation: %s", reason)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.requantization_source is not None:
            return super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
        if intermediate_size_per_partition % LUT_B_BLOCK_K or hidden_size % (
            LUT_B_BLOCK_K
        ):
            raise ValueError(
                "LUT-B MoE needs hidden and per-partition intermediate sizes "
                f"divisible by {LUT_B_BLOCK_K}, got {hidden_size} and "
                f"{intermediate_size_per_partition}."
            )
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        w13_rows = self.moe.w13_num_shards * intermediate_size_per_partition
        for prefix, (n, k) in (
            ("w13_", (w13_rows, hidden_size)),
            ("w2_", (hidden_size, intermediate_size_per_partition)),
        ):
            tiles = (num_experts, n // LUT_B_BLOCK_N, k // LUT_B_BLOCK_K)
            for name, shape, dtype in (
                ("weight", (*tiles, LUT_B_PACKED_TILE_BYTES), torch.uint8),
                ("codebooks", (*tiles, LUT_B_CODEBOOK_SIZE), torch.float8_e4m3fn),
                ("scales", (num_experts, n, k // LUT_B_SCALE_BLOCK_K), torch.uint8),
            ):
                # zeros: padding tiles dequantize to zero
                param = torch.nn.Parameter(
                    torch.zeros(shape, dtype=dtype), requires_grad=False
                )
                layer.register_parameter(prefix + name, param)
                set_weight_attrs(param, extra_weight_attrs)
            _quantize_lut_b_on_load(layer, prefix)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # Both experts quantize activations themselves, and the native weights
        # carry their codebooks and scales inline.
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        for prefix, weight in self._lut_b_weights(layer):
            packed, codebooks, scales = weight
            name = prefix + "weight"
            if self.experts_cls is LutBSm107Experts:
                weight = lut_b_to_sm107_tiles(packed, codebooks, scales)
            else:
                weight = dequantize_lut_b(packed, codebooks, scales, layer.orig_dtype)
            replace_parameter(layer, name, weight)
        self.release_requantization_source_weights(layer)

        quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant_config
        self.moe_kernel = make_lut_b_moe_kernel(
            self.experts_cls,
            self.moe,
            quant_config,
            layer._expert_routing_tables(),
        )
        layer._already_called_process_weights_after_loading = True

    def _lut_b_weights(self, layer: RoutedExperts) -> list[tuple[str, tuple]]:
        """`(prefix, (packed, codebooks, scales))` of w13 and w2."""
        if self.requantization_source is not None:
            w13, w2 = self.get_weights_for_quantization(layer)
            self._zero_padding(layer, w13, w2)
            return [("w13_", quantize_lut_b(w13)), ("w2_", quantize_lut_b(w2))]
        weights = []
        for prefix in ("w13_", "w2_"):
            weights.append(
                (
                    prefix,
                    tuple(
                        getattr(layer, prefix + name)
                        for name in ("weight", "codebooks", "scales")
                    ),
                )
            )
            # folded into the kernel weight below; kept for a reload to load
            retire_parameter(layer, prefix + "codebooks")
            retire_parameter(layer, prefix + "scales")
        return weights


def _quantize_lut_b_on_load(layer: torch.nn.Module, prefix: str) -> None:
    """Make `layer.<prefix>weight` load an fp16/bf16 expert tensor as LUT-B.

    The tiled parameters are 4D, which the MoE loader does not shard: this
    loader takes the rank's shard (gate/up rows, down columns) itself and
    writes it at its offset, leaving the zero padding tiles untouched."""
    names = [prefix + name for name in ("weight", "codebooks", "scales")]
    layer.quantizes_on_load = True

    def loader(param, loaded_weight, weight_name, shard_id, expert_id, **kwargs):
        expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if expert == -1:
            return False  # skip the costly fit for another rank's expert
        moe = layer.moe_config
        inter = moe.intermediate_size_per_partition_unpadded
        weight = loaded_weight.to(param.device, layer.orig_dtype)
        dim = 1 if shard_id == "w2" else 0  # the TP-sharded dim
        if weight.shape[dim] != inter:
            weight = weight.narrow(dim, moe.tp_rank * inter, inter)
        row = moe.intermediate_size_per_partition if shard_id == "w3" else 0
        n, k = weight.shape
        # whole tiles: zero rows/columns, as the padding tiles hold
        pad_k = -k % LUT_B_BLOCK_K
        pad_n = -n % LUT_B_BLOCK_N
        weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))
        packed, codebooks, scales = quantize_lut_b(weight.contiguous())
        tile_rows = slice(row // LUT_B_BLOCK_N, row // LUT_B_BLOCK_N + packed.shape[0])
        tile_cols = slice(0, packed.shape[1])
        for name, value in zip(names[:2], (packed, codebooks)):
            getattr(layer, name).data[expert, tile_rows, tile_cols].copy_(value)
        getattr(layer, names[2]).data[
            expert, row : row + scales.shape[0], : scales.shape[1]
        ].copy_(scales)
        return True if kwargs.get("return_success") else None

    getattr(layer, names[0]).weight_loader = loader
