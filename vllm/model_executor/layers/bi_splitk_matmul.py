# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Batch-invariant split-K matmul for skinny-N / long-K unquantized linears under
# VLLM_BATCH_INVARIANT (e.g. DSv4 indexer weights_proj: M x 4096 @ 4096 x 64).
# The persistent kernel maps such a GEMM onto a single CTA (128x128 tile) that
# walks all of K serially (~40 us). Here K is cut into SPLIT_K chunks chosen
# from K only (never from M), each chunk accumulates in fp32 in fixed order,
# and a second kernel sums the chunks in fixed order. Row m's result depends
# only on row m of A, on B, and on (K, SPLIT_K) -> batch-invariant and
# deterministic. Numerics differ from the single-pass kernel (different
# reduction order); training must use the same kernel for train/infer parity.
import torch

try:
    from vllm.triton_utils import tl, triton
except ImportError:  # Megatron side: same kernel file, plain triton
    import triton
    import triton.language as tl


@triton.jit
def _splitk_partial_kernel(
    a_ptr,
    b_ptr,
    ws_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ws,
    stride_wm,
    stride_wn,
    K_CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_am = tl.where(offs_m < M, offs_m, 0)
    offs_bn = tl.where(offs_n < N, offs_n, 0)
    k_begin = pid_s * K_CHUNK
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, K_CHUNK, BLOCK_K):
        offs_k = k_begin + kt + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    ws_ptrs = (
        ws_ptr
        + pid_s * stride_ws
        + offs_m[:, None] * stride_wm
        + offs_n[None, :] * stride_wn
    )
    tl.store(ws_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _splitk_reduce_kernel(
    ws_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    stride_ws,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(
            ws_ptr
            + s * stride_ws
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn,
            mask=mask,
            other=0.0,
        )
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[
            None, :
        ]
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask,
    )


def splitk_plan(K: int, N: int):
    """Split-K factor from (K, N) only. None -> not eligible."""
    if N > 128 or K < 2048 or K % 64 != 0:
        return None
    return min(16, K // 256)


def matmul_splitk(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    split_k: int | None = None,
    block_m: int = 64,
):
    assert a.shape[1] == b.shape[0] and a.dtype == b.dtype
    M, K = a.shape
    _, N = b.shape
    if split_k is None:
        split_k = splitk_plan(K, N)
    assert split_k is not None
    BLOCK_K = 64
    k_chunk = triton.cdiv(triton.cdiv(K, split_k), BLOCK_K) * BLOCK_K
    split_k = triton.cdiv(K, k_chunk)
    BLOCK_N = max(16, triton.next_power_of_2(N))
    ws = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid1 = (triton.cdiv(M, block_m) * triton.cdiv(N, BLOCK_N), split_k)
    _splitk_partial_kernel[grid1](
        a,
        b,
        ws,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        ws.stride(0),
        ws.stride(1),
        ws.stride(2),
        K_CHUNK=k_chunk,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    RB_M = 32
    grid2 = (triton.cdiv(M, RB_M), triton.cdiv(N, BLOCK_N))
    _splitk_reduce_kernel[grid2](
        ws,
        c,
        bias if bias is not None else c,
        M,
        N,
        ws.stride(0),
        ws.stride(1),
        ws.stride(2),
        c.stride(0),
        c.stride(1),
        SPLIT_K=split_k,
        HAS_BIAS=bias is not None,
        BLOCK_M=RB_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return c
