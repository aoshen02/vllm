# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import ClassVar, cast

import torch

import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

# DeepseekV4 decode layer types, keyed by compress_ratio. Each type has a distinct
# (topk, extra_topk, extra_page_block_size) config, so they cannot share a
# FlashMLA tile-scheduler plan. Within a type, all ~60 DeepseekV4 layers share one
# plan per step because b / s_q / h_q / page_block_sizes / topks are identical.
_LAYER_TYPE_SWAONLY = "swaonly"
_LAYER_TYPE_C4A = "c4a"
_LAYER_TYPE_C128A = "c128a"

# FlashMLA's decode planner works in units of block_size_topk over the top-k
# list (get_decoding_sched_meta.cu). Measured: topk_length=256 yields 4 blocks.
_BLOCK_SIZE_TOPK = 64
# csrc/params.h DecodingSchedMeta: begin_req_idx, end_req_idx (both inclusive),
# begin_block_idx, end_block_idx (inclusive, exclusive), begin_split_idx,
# is_first_req_splitted, is_last_req_splitted, _pad.
_SCHED_META_INTS = 8

# num_sm_parts depends only on the arch and (h_q, s_q) — never on the batch —
# so learning it once from the kernel's own planner is safe. Doing that beats
# duplicating the four get_meta() formulas in sparse_decode.h.
_num_sm_parts_cache: dict[tuple[int, int], int] = {}


# Which partition owns which request depends only on (b, num_sm_parts) — never
# on any request's length — so it is identical for every layer in a step and for
# every step with the same batch size. Cache it: this is a per-layer hot path,
# and building it from a boolean mask costs a device sync (measured 610 us,
# ~37 ms per step over 61 layers).
_plan_cache: dict[
    tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def record_num_sm_parts(h_q: int, s_q: int, sched: FlashMLASchedMeta) -> None:
    """Remember the partition count the kernel chose, so later steps can pin."""
    tsm = sched.tile_scheduler_metadata
    if tsm is not None:
        _num_sm_parts_cache.setdefault((h_q, s_q), int(tsm.shape[0]))


def _split_budget(
    num_sm_parts: int,
    cfg: tuple[int, "CUDAGraphMode"] | None = None,
) -> tuple[int, int]:
    """``(max_splits_per_request, max_num_seqs)``, fixed for the server's life.

    Giving each request one whole partition is invariant but leaves most SMs
    idle: measured 1.19-1.28x the shipped planner's GPU time for b=4..17. A
    request can instead be cut at multiples of its *own* block count, which is
    still invariant, but only if the per-request split count is a static bound —
    derive it from the current batch and a row's reduction order would move
    again. The plan has exactly ``num_sm_parts`` rows (sparse_decode.h:457) and a
    split simply *is* a partition boundary landing inside a request, so the
    bound is ``num_sm_parts // max_num_seqs``.
    """
    if cfg is not None:
        max_num_seqs, cudagraph_mode = cfg
    else:
        # Direct kernel tests, which run inside a config context; engine
        # forward passes pass cfg captured at layer init instead, since
        # warmup has no active context.
        config = get_current_vllm_config()
        max_num_seqs = config.scheduler_config.max_num_seqs
        cudagraph_mode = config.compilation_config.cudagraph_mode
    if max_num_seqs <= 0 or max_num_seqs > num_sm_parts:
        # More requests than partitions: not even one partition each.
        return 1, max_num_seqs
    if cudagraph_mode == CUDAGraphMode.NONE:
        # The split plan cannot be cached — its layout follows this step's
        # lengths — so building it costs ~670 us of kernel launches against
        # ~86 us for the whole-request plan. Under cudagraphs those launches are
        # captured and replayed for free, and the GPU saving (0.2 ms/step at
        # b=8) is real; in eager the build dominates and it is a net loss.
        return 1, max_num_seqs
    return max(num_sm_parts // max_num_seqs, 1), max_num_seqs


def _blocks_per_split(
    topk_cap: int, extra_topk_cap: int | None, has_extra: bool, max_splits: int
) -> int:
    """Static block count per split, sized so the worst case still fits.

    ``topk_cap`` / ``extra_topk_cap`` are the layer's configured top-k widths,
    not this step's lengths — the divisor has to be a constant.
    """
    cap = topk_cap
    if has_extra and extra_topk_cap:
        cap = cdiv(cap, _BLOCK_SIZE_TOPK) * _BLOCK_SIZE_TOPK + extra_topk_cap
    max_blocks = max(cdiv(cap, _BLOCK_SIZE_TOPK), 1)
    return max(cdiv(max_blocks, min(max_splits, max_blocks)), 1)


def _split_plan(
    num_blocks: torch.Tensor,
    b: int,
    num_sm_parts: int,
    blocks_per_split: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan that cuts each request every ``blocks_per_split`` of its own blocks.

    Request j claims ``ceil(num_blocks[j] / blocks_per_split)`` consecutive
    partitions; partition p resolves to its request by a searchsorted over the
    running total. All device work, no host sync — the shapes are static, only
    the values depend on this step's lengths.
    """
    ks = cdiv(num_blocks.to(torch.int64), blocks_per_split)
    bounds = ks.cumsum(0)
    part = torch.arange(num_sm_parts, device=device, dtype=torch.int64)
    # right=True so a partition landing exactly on a boundary belongs to the
    # next request; idle partitions resolve to j == b.
    j = torch.searchsorted(bounds, part, right=True)
    busy = j < b
    jc = j.clamp(max=b - 1)
    k = part - (bounds[jc] - ks[jc])
    lo = k * blocks_per_split
    hi = torch.minimum(lo + blocks_per_split, num_blocks[jc].to(torch.int64))

    meta = torch.zeros(
        (num_sm_parts, _SCHED_META_INTS), dtype=torch.int32, device=device
    )
    zero = torch.zeros((), dtype=torch.int64, device=device)
    # Idle partitions are encoded begin_req_idx > end_req_idx, which is what the
    # planner itself leaves behind once it runs out of requests.
    meta[:, 0] = torch.where(busy, jc, b).to(torch.int32)
    meta[:, 1] = torch.where(busy, jc, b - 1).to(torch.int32)
    meta[:, 2] = torch.where(busy, lo, zero).to(torch.int32)
    meta[:, 3] = torch.where(busy, hi, zero).to(torch.int32)
    meta[:, 4] = torch.where(busy, k, zero).to(torch.int32)
    # get_decoding_sched_meta.cu collapses both flags to the OR of its two
    # conditions when a partition covers exactly one request, which here reduces
    # to "this request spans more than one partition".
    flag = (busy & (ks[jc] > 1)).to(torch.int32)
    meta[:, 5] = flag
    meta[:, 6] = flag

    num_splits = torch.cat(
        [torch.zeros(1, dtype=torch.int64, device=device), bounds]
    ).to(torch.int32)
    return meta, num_splits


def _plan_skeleton(
    b: int, num_sm_parts: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The batch-length-independent part of the plan: who owns what.

    Partition ``p`` owns requests ``[floor(p*b/P), floor((p+1)*b/P))``. Spans are
    contiguous, disjoint and cover every request exactly once, for any ``b`` --
    including ``b > P``, where a partition simply holds several whole requests.
    No request is ever split across partitions, so a row's reduction order still
    depends only on its own length.

    Returns ``(meta, last_owned, num_splits)``. ``last_owned`` indexes, for each
    busy partition, the last request it owns; that is the only place the caller
    has to write a per-call block count.
    """
    key = (b, num_sm_parts, device)
    cached = _plan_cache.get(key)
    if cached is not None:
        return cached

    # Plain host integer arithmetic -- no device work and no sync.
    spans = [
        ((p * b) // num_sm_parts, ((p + 1) * b) // num_sm_parts)
        for p in range(num_sm_parts)
    ]
    busy = [p for p, (lo, hi) in enumerate(spans) if lo < hi]
    meta = torch.zeros((num_sm_parts, _SCHED_META_INTS), dtype=torch.int32)
    # Idle partitions are encoded begin_req_idx > end_req_idx, which is what the
    # planner itself leaves behind once it runs out of requests.
    meta[:, 0] = b
    meta[:, 1] = b - 1
    busy_idx = torch.tensor(busy, dtype=torch.int64)
    meta[busy_idx, 0] = torch.tensor([spans[p][0] for p in busy], dtype=torch.int32)
    meta[busy_idx, 1] = torch.tensor([spans[p][1] - 1 for p in busy], dtype=torch.int32)
    # first_block_idx (column 2), begin_split_idx and the two is_splitted flags
    # (4..6) all stay 0: nothing is ever split.
    #
    # end_block_idx belongs to the partition's *last* request, so that is the
    # row the caller fills in.
    last_owned = torch.tensor([spans[p][1] - 1 for p in busy], dtype=torch.int64)

    logger.info_once(
        "Batch-invariant sparse MLA decode: pinning the tile-scheduler plan "
        "(num_sm_parts=%d), whole requests grouped per partition, no request "
        "split.",
        num_sm_parts,
        scope="local",
    )
    skeleton = (
        meta.to(device),
        (busy_idx.to(device), last_owned.to(device)),
        # Cumulative split count; exactly one split per request.
        torch.arange(b + 1, dtype=torch.int32, device=device),
    )
    _plan_cache[key] = skeleton
    return skeleton


def build_pinned_sched_meta(
    h_q: int,
    s_q: int,
    topk_length: torch.Tensor,
    extra_topk_length: torch.Tensor | None,
    has_extra: bool,
    device: torch.device,
    topk_cap: int | None = None,
    extra_topk_cap: int | None = None,
    split_budget_cfg: tuple[int, "CUDAGraphMode"] | None = None,
) -> FlashMLASchedMeta | None:
    """A tile-scheduler plan whose reduction order does not depend on the batch.

    The shipped planner sizes each partition's payload as
    ``ceil_div(total_num_blocks, num_sm_parts)``, so a given request's KV gets
    cut at different block boundaries depending on how much work the *other*
    requests brought. Measured effect: a pinned row's decode output changes at
    b>=17 on GB200.

    This plan cuts a request only at multiples of ``blocks_per_split`` of its
    *own* block count, so one row's reduction order is fixed by its own length
    and nothing else. ``topk_cap`` / ``extra_topk_cap`` are the layer's
    configured top-k widths; without them the plan degrades to one whole request
    per partition, which is equally invariant but leaves most SMs idle.

    Returns None until ``record_num_sm_parts`` has seen a real plan for this
    (h_q, s_q); the caller then falls back to the kernel planner for that one
    step (warmup covers this in practice).
    """
    num_sm_parts = _num_sm_parts_cache.get((h_q, s_q))
    if num_sm_parts is None:
        return None

    b = int(topk_length.shape[0])
    if b == 0:
        return None

    # Mirror get_decoding_sched_meta.cu's per-request block count. The extra
    # top-k branch rounds the main length up to a whole block *before* adding
    # the extra length, so the two cannot simply be summed.
    lens = topk_length.clamp(min=1)
    if has_extra and extra_topk_length is not None:
        lens = cdiv(lens, _BLOCK_SIZE_TOPK) * _BLOCK_SIZE_TOPK + extra_topk_length
    num_blocks = ((lens - 1) // _BLOCK_SIZE_TOPK + 1).to(torch.int32)

    max_splits, max_num_seqs = (
        (1, 0) if topk_cap is None else _split_budget(num_sm_parts, split_budget_cfg)
    )
    if max_splits > 1 and b <= max_num_seqs:
        f = _blocks_per_split(topk_cap, extra_topk_cap, has_extra, max_splits)
        logger.info_once(
            "Batch-invariant sparse MLA decode: pinning the tile-scheduler plan "
            "(num_sm_parts=%d), each request cut every %d of its own KV blocks, "
            "at most %d partitions per request.",
            num_sm_parts,
            f,
            max_splits,
            scope="local",
        )
        meta, num_splits = _split_plan(num_blocks, b, num_sm_parts, f, device)
    else:
        # Either splitting is off, or the batch is larger than the split budget
        # was sized for. Group whole requests per partition instead: still one
        # split per request, no request across partitions, and no fall back to
        # the shipped planner's batch-dependent layout.
        skeleton, (busy, last_owned), num_splits = _plan_skeleton(
            b, num_sm_parts, device
        )
        # Cloned rather than written in place: the kernel reads this buffer
        # asynchronously, and the next layer would otherwise overwrite it.
        meta = skeleton.clone()
        # end_block_idx (exclusive) is the block count of the partition's last
        # request. Integer indices, so no host sync.
        meta[busy, 3] = num_blocks[last_owned]

    sched = get_mla_metadata()[0]
    sched.tile_scheduler_metadata = meta
    sched.num_splits = num_splits
    return sched


def _layer_type_for(compress_ratio: int) -> str:
    if compress_ratio <= 1:
        return _LAYER_TYPE_SWAONLY
    if compress_ratio == 4:
        return _LAYER_TYPE_C4A
    if compress_ratio == 128:
        return _LAYER_TYPE_C128A
    raise ValueError(
        f"Unsupported DeepseekV4 compress_ratio={compress_ratio}; "
        "expected 1, 4, or 128."
    )


class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        backend_cls: "type[AttentionBackend] | None" = None,
    ):
        super().__init__()
        self.backend_cls = backend_cls or DeepseekSparseSWABackend
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # Captured here because forward runs (cudagraph warmup included)
        # have no active config context.
        self.split_budget_cfg = (
            vllm_config.scheduler_config.max_num_seqs,
            compilation_config.cudagraph_mode,
        )

        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(yifan): make SWA block size automatically determined and configurable.
        self.block_size = 64
        # uint8: fp8_ds_mla UE8M0 paged layout. bfloat16 / float8_e4m3fn:
        # contiguous full-cache layout.
        assert self.dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla's UE8M0 paged layout needs 576B alignment; contiguous
        # bf16/fp8 cache uses the natural element-size page.
        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # DeepseekV4 fp8_ds_mla: 584B per token (448B NoPE + 128B RoPE + 8B scales)
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls


class DeepseekSparseSWABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 256

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_builder_cls() -> type["DeepseekSparseSWAMetadataBuilder"]:
        if current_platform.is_rocm():
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
            )

            return DeepseekV4ROCMAiterSparseSWAMetadataBuilder
        return DeepseekSparseSWAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class DeepseekSparseSWAMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs]
    query_start_loc: torch.Tensor | None = None  # [num_seqs + 1]
    query_start_loc_cpu: torch.Tensor | None = None  # [num_seqs + 1]

    is_valid_token: torch.Tensor | None = None  # [num_tokens]
    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, width]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]
    # window_size (causal) or noncausal_index_width (DSpark non-causal).
    decode_swa_width: int = 0
    # Paged-coordinate prefill SWA indices/lens (FP8 paged-direct prefill).
    prefill_swa_indices: torch.Tensor | None = (
        None  # [num_prefill_tokens, 1, window_size]
    )
    prefill_swa_lens: torch.Tensor | None = None  # [num_prefill_tokens]

    # Number of decode/prefill requests/tokens (batch is reordered: decodes first)
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0
    max_decode_query_len: int = 1

    # Pre-computed prefill metadata shared across all DeepseekV4 attention layers.
    prefill_seq_lens: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_window_size: int = 0
    prefill_max_model_len: int = 0
    prefill_max_num_batched_tokens: int = 0

    # Per-layer-type FlashMLA tile-scheduler metadata. One FlashMLASchedMeta
    # per present DeepseekV4 layer type, shared across all ~60 layers of that type
    # within a decode step. The first forward call of a given type triggers
    # the in-kernel planner (which also allocates tile_scheduler_metadata and
    # num_splits via PyTorch's graph-aware allocator); subsequent same-type
    # calls skip planning and reuse the plan. Fresh instance per build(), so
    # have_initialized is always False at the start of a step and the plan
    # is re-derived from current seq_lens / topk_length on replay.
    # None for layer types the model does not use (or when num_decode_tokens
    # is zero).
    tile_sched_swaonly: "FlashMLASchedMeta | None" = None
    tile_sched_c4a: "FlashMLASchedMeta | None" = None
    tile_sched_c128a: "FlashMLASchedMeta | None" = None
    flashinfer_sparse_index_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    def get_prefill_chunk_plan(
        self, compress_ratio: int, prefill_chunk_size: int
    ) -> list[tuple[int, int, int, int]]:
        if self.num_prefills == 0:
            return []

        assert self.prefill_seq_lens_cpu is not None
        assert self.prefill_query_lens_cpu is not None

        # query_len <= max_num_batched_tokens and
        # gather_len = query_len + min(prefix_len, window_size - 1), so the
        # worst-case gathered width is bounded by
        # max_num_batched_tokens + window_size - 1. The compressed prefix pool
        # is bounded by ceil(max_model_len / compress_ratio).
        max_workspace_area = prefill_chunk_size * (
            (
                0
                if compress_ratio <= 1
                else cdiv(self.prefill_max_model_len, compress_ratio)
            )
            + self.prefill_window_size
            + self.prefill_max_num_batched_tokens
        )
        prefix_lens_cpu = self.prefill_seq_lens_cpu - self.prefill_query_lens_cpu
        gather_lens_cpu = self.prefill_query_lens_cpu + torch.clamp(
            prefix_lens_cpu, min=0, max=self.prefill_window_size - 1
        )
        compressed_lens_cpu = (
            torch.zeros_like(self.prefill_seq_lens_cpu)
            if compress_ratio <= 1
            else torch.div(
                self.prefill_seq_lens_cpu,
                compress_ratio,
                rounding_mode="floor",
            )
        )

        if envs.VLLM_BATCH_INVARIANT:
            # One request per chunk. The sparse-prefill kernel takes no
            # schedule metadata and plans its internal splits from the call
            # geometry (query rows, gathered-KV width), so packing neighbors
            # into a chunk changes a row's reduction plan with the batch —
            # the prefill twin of the decode tile-scheduler pin above. With
            # per-request calls the geometry is a function of the request
            # alone; a lone request is a single-request chunk by definition,
            # so both sides of the comparison run identical calls.
            compressed = compressed_lens_cpu.tolist()
            gather = gather_lens_cpu.tolist()
            return [
                (i, i + 1, compressed[i], compressed[i] + gather[i])
                for i in range(self.num_prefills)
            ]

        chunk_plan: list[tuple[int, int, int, int]] = []
        chunk_start = 0
        while chunk_start < self.num_prefills:
            chunk_max_compressed = int(compressed_lens_cpu[chunk_start].item())
            chunk_max_gather = int(gather_lens_cpu[chunk_start].item())
            chunk_end = chunk_start + 1

            while chunk_end < self.num_prefills:
                candidate_max_compressed = max(
                    chunk_max_compressed,
                    int(compressed_lens_cpu[chunk_end].item()),
                )
                candidate_max_gather = max(
                    chunk_max_gather,
                    int(gather_lens_cpu[chunk_end].item()),
                )
                candidate_width = candidate_max_compressed + candidate_max_gather
                candidate_area = (chunk_end - chunk_start + 1) * candidate_width
                if candidate_area > max_workspace_area:
                    break
                chunk_max_compressed = candidate_max_compressed
                chunk_max_gather = candidate_max_gather
                chunk_end += 1

            chunk_plan.append(
                (
                    chunk_start,
                    chunk_end,
                    chunk_max_compressed,
                    chunk_max_compressed + chunk_max_gather,
                )
            )
            chunk_start = chunk_end

        return chunk_plan


class ComputePrefillMetadataKernel(
    VllmJitKernel["ComputePrefillMetadataKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        BLOCK_SIZE: int

    @staticmethod
    @triton.jit(do_not_specialize=["num_prefills", "num_decodes", "window_size"])
    def kernel(
        # Outputs
        prefill_gather_lens_ptr,
        # Inputs
        seq_lens_ptr,
        query_start_loc_ptr,
        num_prefills,
        num_decodes,
        window_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute prefill gather_lens in a single pass."""
        offset = tl.arange(0, BLOCK_SIZE)
        mask = offset < num_prefills
        # SM12x + Triton 3.6 raises IMA on out-of-bounds address arithmetic for
        # masked-off lanes even though the load mask gates the actual read, so
        # clamp the offset. Caller guarantees num_prefills > 0.
        safe_offset = tl.minimum(offset, num_prefills - 1)

        seq_len = tl.load(seq_lens_ptr + num_decodes + safe_offset, mask=mask)
        qsl_start = tl.load(query_start_loc_ptr + num_decodes + safe_offset, mask=mask)
        qsl_end = tl.load(
            query_start_loc_ptr + num_decodes + safe_offset + 1, mask=mask
        )

        query_len = qsl_end - qsl_start
        prefix_len = seq_len - query_len
        gather_len = query_len + tl.minimum(prefix_len, window_size - 1)

        tl.store(prefill_gather_lens_ptr + offset, gather_len, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        num_prefills: int,
    ) -> CompileKey:
        return self.CompileKey(
            BLOCK_SIZE=next_power_of_2(num_prefills),
        )

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        scheduler_config = vllm_config.scheduler_config
        max_prefills = max(
            1,
            min(
                scheduler_config.max_num_seqs,
                scheduler_config.max_num_batched_tokens,
            ),
        )
        return self._trace_dispatch(self.dispatch)(
            num_prefills=WarmupIntRange(1, max_prefills + 1),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            int32_ptr,
            int32_ptr,
            int32_ptr,
            compile_key.BLOCK_SIZE,
            0,
            1,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            grid=(1,),
        )

    def __call__(
        self,
        prefill_gather_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_prefills: int,
        num_decodes: int,
        window_size: int,
    ) -> None:
        compile_key = self.dispatch(num_prefills=num_prefills)
        self.kernel[(1,)](
            prefill_gather_lens,
            seq_lens,
            query_start_loc,
            num_prefills,
            num_decodes,
            window_size,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
        )


_COMPUTE_PREFILL_METADATA_KERNEL = ComputePrefillMetadataKernel()


class DeepseekSparseSWAMetadataBuilder(AttentionMetadataBuilder):
    """Builds metadata for DeepseekV4 SWA cache.

    Similar to the indexer, this handles mixed batches by:
    1. Using split_decodes_and_prefills() to determine the boundary
    2. Building separate metadata for decode and prefill portions

    Supports:
    - Mixed decode/prefill batches
    - MTP (Multi-Token Prediction) where decode has query_len > 1
    - Chunked prefill (aligns with the indexer's chunking)
    """

    reorder_batch_threshold: int | None = None
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_draft_decode_metadata_update = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.head_size = mla_spec.head_size  # Already considered quantization.
        self.compress_ratio = mla_spec.compress_ratio
        self.block_size = mla_spec.block_size
        self.max_model_len = self.vllm_config.model_config.max_model_len
        self.max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )

        # Handle MTP: adjust decode_threshold like the indexer does
        spec_config = self.vllm_config.speculative_config
        self.num_speculative_tokens = (
            spec_config.num_speculative_tokens if spec_config else 0
        )
        # Decode can have query_len up to
        #   1 + (2 if parallel drafting else 1) * num_speculative_tokens.
        # sparse_swa has no MQA-vs-dense-MHA routing, so multi-token queries take
        # the prefill path and the decode/prefill split stays at that width.
        spec_mult = (
            2 if (spec_config is not None and spec_config.parallel_drafting) else 1
        )
        self.decode_threshold = 1 + spec_mult * self.num_speculative_tokens
        self.reorder_batch_threshold = None

        hf_config = self.vllm_config.model_config.hf_config
        assert hasattr(hf_config, "sliding_window")
        self.window_size = hf_config.sliding_window

        # Detect which DeepseekV4 layer types this model uses so we only build a
        # FlashMLA tile-scheduler plan for types that will actually be called.
        # Models without compress_ratios (pure SWA) fall back to swaonly.
        compress_ratios = getattr(hf_config, "compress_ratios", None) or [1]
        self._layer_types: set[str] = set()
        for ratio in compress_ratios:
            self._layer_types.add(_layer_type_for(int(ratio)))

        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_indices = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_indices = torch.zeros(
            max_tokens,
            1,
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # Allocated unconditionally — consumer picks paged-direct vs dequant
        # at call time.
        self.prefill_swa_indices = torch.zeros(
            max_tokens,
            1,
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.prefill_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.is_valid_token = torch.zeros(
            max_tokens,
            dtype=torch.bool,
            device=self.device,
        )

        # DSpark draft: the block is non-causal (every query attends to the
        # trailing window of context PLUS all query tokens, including future ones),
        # so its per-token index list is wider than `window_size`. The kernel pads
        # the q-head count to B_TOPK. Pad to a kernel-supported width; the logical
        # SWA window remains unchanged when the padded matrix is built.
        self.is_dspark = spec_config is not None and spec_config.use_dspark()
        self.noncausal_index_width = (
            get_dspark_swa_index_width(
                self.window_size,
                self.num_speculative_tokens,
            )
            if self.is_dspark
            else 0
        )
        self.decode_swa_indices_noncausal: torch.Tensor | None = None
        self._max_tokens = max_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekSparseSWAMetadata:
        """Build SWA metadata for mixed decode/prefill batches.

        The batch is assumed to be reordered with decodes first (by vLLM scheduler).
        We use split_decodes_and_prefills() to find the boundary, then build
        separate window_topk_idxs for each portion.

        For prefill, we use chunked prefill to align with the indexer's chunking.
        """
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Split into decode and prefill portions using configurable threshold
        (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens) = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.decode_threshold
            )
        )

        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility.
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)

        non_causal = not common_attn_metadata.causal
        decode_swa_width = (
            self.noncausal_index_width if non_causal else self.window_size
        )
        decode_swa_indices = self.decode_swa_indices
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            if non_causal:
                assert self.is_dspark, (
                    "Non-causal DeepseekV4 SWA is only supported for the DSpark "
                    "speculation mode, but causal=False was set without DSpark."
                )
                if self.decode_swa_indices_noncausal is None:
                    self.decode_swa_indices_noncausal = torch.zeros(
                        self._max_tokens,
                        1,
                        self.noncausal_index_width,
                        dtype=torch.int32,
                        device=self.device,
                    )
                decode_swa_indices = self.decode_swa_indices_noncausal
                _compute_dspark_noncausal_swa_indices_kernel[(num_decode_tokens,)](
                    decode_swa_indices,
                    decode_swa_indices.stride(0),
                    self.decode_swa_lens,
                    self.window_size,
                    self.noncausal_index_width,
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    block_table.stride(0),
                    self.block_size,
                    token_offset=0,
                    TRITON_BLOCK_SIZE=1024,
                )
            else:
                _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
                    decode_swa_indices,
                    decode_swa_indices.stride(0),
                    self.decode_swa_lens,
                    self.window_size,
                    query_start_loc,
                    seq_lens,
                    token_to_req_indices,
                    is_valid_token,
                    block_table,
                    block_table.stride(0),
                    self.block_size,
                    token_offset=0,
                    TRITON_BLOCK_SIZE=1024,
                )

        # Prefill SWA indices live in paged coordinates. `token_offset` lets
        # the kernel read is_valid_token / token_to_req_indices at absolute
        # prefill positions while writing output starting at index 0.
        if num_prefill_tokens > 0:
            prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
            prefill_swa_lens = self.prefill_swa_lens[:num_prefill_tokens]
            _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
                prefill_swa_indices,
                prefill_swa_indices.stride(0),
                prefill_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                token_offset=num_decode_tokens,
                TRITON_BLOCK_SIZE=1024,
            )

        # Pre-compute DeepseekV4 prefill metadata shared across all attention layers.
        deepseek_v4_fields = self._build_deepseek_v4_metadata(
            num_decodes,
            num_prefills,
            seq_lens,
            seq_lens_cpu,
            query_start_loc,
            query_start_loc_cpu,
        )

        # Per-layer-type tile-scheduler plan holders. Empty FlashMLASchedMeta
        # per present DeepseekV4 layer type; the first flash_mla_with_kvcache call of
        # each type triggers the planner and all same-type layers reuse the
        # resulting plan for the rest of the step.
        tile_sched = self.build_tile_scheduler(num_decode_tokens)

        return DeepseekSparseSWAMetadata(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_valid_token=is_valid_token,
            token_to_req_indices=token_to_req_indices,
            decode_swa_indices=decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            decode_swa_width=decode_swa_width,
            prefill_swa_indices=(
                self.prefill_swa_indices[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            prefill_swa_lens=(
                self.prefill_swa_lens[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            block_size=self.block_size,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            # Upper bound on decode-split rows for the kernel's max_q_len
            # hint. common max_query_len bounds every row (scheduled max under
            # adaptive verification), clamped to what the split can admit so a
            # mixed batch's prefill max does not inflate decode scheduling.
            max_decode_query_len=min(
                common_attn_metadata.max_query_len, self.decode_threshold
            ),
            tile_sched_swaonly=tile_sched[_LAYER_TYPE_SWAONLY],
            tile_sched_c4a=tile_sched[_LAYER_TYPE_C4A],
            tile_sched_c128a=tile_sched[_LAYER_TYPE_C128A],
            **deepseek_v4_fields,  # type: ignore[arg-type]
        )

    def update_draft_decode_metadata(
        self,
        metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        if metadata.num_decode_tokens == 0:
            return
        assert metadata.query_start_loc is not None
        assert metadata.seq_lens is not None
        assert metadata.token_to_req_indices is not None
        assert metadata.is_valid_token is not None
        assert metadata.decode_swa_indices is not None
        assert metadata.decode_swa_lens is not None

        _compute_swa_indices_and_lens_kernel[(metadata.num_decode_tokens,)](
            metadata.decode_swa_indices,
            metadata.decode_swa_indices.stride(0),
            metadata.decode_swa_lens,
            metadata.decode_swa_indices.shape[-1],
            metadata.query_start_loc,
            metadata.seq_lens,
            metadata.token_to_req_indices,
            metadata.is_valid_token,
            metadata.block_table,
            metadata.block_table.stride(0),
            self.block_size,
            token_offset=0,
            TRITON_BLOCK_SIZE=1024,
        )
        tile_sched = self.build_tile_scheduler(metadata.num_decode_tokens)
        metadata.tile_sched_swaonly = tile_sched[_LAYER_TYPE_SWAONLY]
        metadata.tile_sched_c4a = tile_sched[_LAYER_TYPE_C4A]
        metadata.tile_sched_c128a = tile_sched[_LAYER_TYPE_C128A]
        metadata.flashinfer_sparse_index_cache.clear()

    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        """Allocate one empty ``FlashMLASchedMeta`` per present DeepseekV4 layer type.

        Returned instances have ``tile_scheduler_metadata`` / ``num_splits``
        set to ``None``; the FlashMLA C++ decode path will allocate them and
        run the tile-scheduler planner on the first ``flash_mla_with_kvcache``
        call of each type. Subsequent same-type calls reuse the plan because
        the tensors (and ``have_initialized``) are populated on the struct.

        Returns all-``None`` when there are no decode tokens this step, so
        ``_forward_decode`` sees a clean sentinel.
        """
        out: dict[str, FlashMLASchedMeta | None] = {
            _LAYER_TYPE_SWAONLY: None,
            _LAYER_TYPE_C4A: None,
            _LAYER_TYPE_C128A: None,
        }
        if (
            num_decode_tokens == 0
            or current_platform.is_rocm()
            or current_platform.is_xpu()
            or current_platform.is_device_capability_family(120)
        ):
            return out
        for layer_type in self._layer_types:
            # get_mla_metadata() is the official FlashMLA entry point that
            # returns a fresh empty FlashMLASchedMeta; using it keeps this
            # call site aligned with the rest of the vLLM FlashMLA backends
            # that already go through the same stub.
            out[layer_type] = get_mla_metadata()[0]
        return out

    def _build_deepseek_v4_metadata(
        self,
        num_decodes: int,
        num_prefills: int,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
    ) -> dict[str, torch.Tensor | int | None]:
        """Pre-compute DeepseekV4 prefill metadata during the metadata build phase.

        Returns a dict of keyword arguments to pass to the
        DeepseekSparseSWAMetadata constructor.

        Note: C128A sparse metadata is computed by the FlashMLASparse builder
        (which owns the C128A block_table), not here.
        """
        result: dict[str, torch.Tensor | int | None] = {}

        # --- Prefill query metadata (single Triton kernel + CPU slicing) ---
        if num_prefills > 0:
            assert seq_lens_cpu is not None
            pfx_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=seq_lens.device
            )
            _COMPUTE_PREFILL_METADATA_KERNEL(
                pfx_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                self.window_size,
            )

            result["prefill_seq_lens"] = seq_lens[num_decodes:]
            result["prefill_seq_lens_cpu"] = seq_lens_cpu[num_decodes:]
            result["prefill_gather_lens"] = pfx_gather_lens
            result["prefill_query_lens_cpu"] = (
                query_start_loc_cpu[num_decodes + 1 : num_decodes + num_prefills + 1]
                - query_start_loc_cpu[num_decodes : num_decodes + num_prefills]
            ).to(dtype=torch.int32)
            result["prefill_window_size"] = self.window_size
            result["prefill_max_model_len"] = self.max_model_len
            result["prefill_max_num_batched_tokens"] = self.max_num_batched_tokens

        return result


@triton.jit(do_not_specialize=["token_offset"])
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, window_size, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < window_size,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )


# TODO(ben): unify this kernel to reduce duplication
@triton.jit(do_not_specialize=["token_offset"])
def _compute_dspark_noncausal_swa_indices_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Non-causal per-token indices for the DSpark draft block.

    Here, we populate the topk indices with the trailing window of context tokens,
    plus all query tokens (including future ones).
    """
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < index_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    # Block-anchored window (shared by every token in the block) + full block.
    start_pos = tl.maximum(prefix_len - window_size, 0)
    end_pos = seq_len

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )
