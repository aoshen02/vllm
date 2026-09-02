// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Batch-invariant GEMM: C[M,N] = A[M,K] @ B[K,N] (+ bias[N]).
//
// Every output element is an fp32 accumulation over K in ascending order,
// built from identical mma.sync m16n8k16 instructions (bf16/fp16) or fp32 FMAs,
// so row m of C depends only on row m of A, on B, and on the split-K schedule.
// The schedule is a function of (K, N) only -- never of M: K is cut into S
// chunks of k_chunk, chunk partials are added to 0 in order, then the bias,
// then one rounding to the output dtype. How that sum is executed may depend
// on M (one CTA per chunk plus a reduce kernel, or one CTA walking every
// chunk), because both perform the same fp32 additions in the same order.
// Tile sizes may depend on M because they only regroup rows; they do not
// change any row's arithmetic.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "bi_gemm_sm100.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace vllm::batch_invariant {

namespace {

constexpr int kSplitKChunkAlign = 64;  // Kc is a multiple of this (>= any BK)

struct SplitPlan {
  int S;        // chunks along K
  int k_chunk;  // K per chunk (multiple of kSplitKChunkAlign when S > 1)
};

// Split-K plan from (K, N) only. Skinny-N GEMMs (e.g. DSv4 indexer
// weights_proj, N = 64) would otherwise map onto one CTA walking all of K.
SplitPlan split_k_plan(int64_t K, int64_t N) {
  SplitPlan pl{1, static_cast<int>(K)};
  if (N > 1024 || K < 1024) return pl;
  const int64_t want = std::min<int64_t>(16, K / 256);
  const int64_t per = (K + want - 1) / want;
  pl.k_chunk = static_cast<int>((per + kSplitKChunkAlign - 1) /
                                kSplitKChunkAlign * kSplitKChunkAlign);
  pl.S = static_cast<int>((K + pl.k_chunk - 1) / pl.k_chunk);
  return pl;
}

// How many chunks one CTA walks: 1 (partial per chunk, reduce kernel sums
// them) or S (direct output). Execution only; the sum is fixed by the plan.
enum class Exec { kSplit, kLocal };

int num_sms() {
  static int cached[64] = {};
  const int dev = at::cuda::current_device();
  if (cached[dev] == 0)
    cached[dev] = at::cuda::getDeviceProperties(dev)->multiProcessorCount;
  return cached[dev];
}

Exec pick_exec(int ctas_mn, const SplitPlan& pl) {
  if (pl.S == 1) return Exec::kSplit;  // one chunk, direct output
  static const char* env = std::getenv("VLLM_BI_GEMM_EXEC");
  if (env != nullptr) {
    if (env[0] == 's') return Exec::kSplit;
    if (env[0] == 'l') return Exec::kLocal;
  }
  return ctas_mn >= 2 * num_sms() ? Exec::kLocal : Exec::kSplit;
}

int chunks_per_cta(Exec ex, const SplitPlan& pl) {
  return ex == Exec::kSplit ? 1 : pl.S;
}

// Mirrors dispatch_mma's tile choice.
int mma_tile_count(int M, int N) {
  const int BM = M <= 16 ? 16 : (M <= 32 ? 32 : (M <= 64 ? 64 : 128));
  const int BN = N <= 64 ? 64 : 128;
  return ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src,
                                            bool valid) {
  int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(size));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1,
                                                  uint32_t& r2, uint32_t& r3,
                                                  uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

template <typename T>
struct MmaTraits;

template <>
struct MmaTraits<__nv_bfloat16> {
  __device__ __forceinline__ static void mma(float* c, const uint32_t* a,
                                             const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ __forceinline__ static float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ __forceinline__ static __nv_bfloat16 from_float(float x) {
    return __float2bfloat16_rn(x);
  }
};

template <>
struct MmaTraits<__half> {
  __device__ __forceinline__ static void mma(float* c, const uint32_t* a,
                                             const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ __forceinline__ static float to_float(__half x) {
    return __half2float(x);
  }
  __device__ __forceinline__ static __half from_float(float x) {
    return __float2half_rn(x);
  }
};

struct GemmParams {
  const void* a;     // [M, K], K contiguous
  const void* b;     // [K, N]
  const void* bias;  // [N] or nullptr
  void* c;           // [M, N] contiguous (dtype T), used when ws == nullptr
  float* ws;         // [S, M, N] fp32 partials, or nullptr for direct output
  int M, N, K;
  int64_t lda;  // elements between rows of A
  int64_t
      ldb;  // B_KMAJOR: elements between columns of B (B[k, n] = b[n*ldb + k])
            // else:     elements between rows of B    (B[k, n] = b[k*ldb + n])
  int k_chunk;         // K per chunk
  int chunks_per_cta;  // 1 or every chunk (see Exec)
  int m_tiles;
};

constexpr int kPad = 8;  // elements; keeps ldmatrix rows on distinct banks

// A: smem [BM][BK + kPad] (K-major). B: smem [BN][BK + kPad] when K-major,
// [BK][BN + kPad] when N-major. grid.x = m_tiles * n_tiles (M innermost so
// CTAs sharing a B tile run together), grid.y = split index.
template <typename T, int BM, int BN, int BK, int STAGES, int WARPS_M,
          int WARPS_N, bool B_KMAJOR, bool MULTI>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
    bi_gemm_kernel(const GemmParams p) {
  constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
  constexpr int WTM = BM / WARPS_M;
  constexpr int WTN = BN / WARPS_N;
  constexpr int MT = WTM / 16;
  constexpr int NT = WTN / 8;
  static_assert(MT >= 1 && NT >= 2 && NT % 2 == 0, "warp tile");
  constexpr int A_PITCH = BK + kPad;
  constexpr int B_PITCH = B_KMAJOR ? (BK + kPad) : (BN + kPad);
  constexpr int A_STAGE = BM * A_PITCH;
  constexpr int B_STAGE = B_KMAJOR ? BN * B_PITCH : BK * B_PITCH;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* As = reinterpret_cast<T*>(smem_raw);
  T* Bs = As + STAGES * A_STAGE;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;

  const int m_tile = blockIdx.x % p.m_tiles;
  const int n_tile = blockIdx.x / p.m_tiles;
  const int m0 = m_tile * BM;
  const int n0 = n_tile * BN;
  const int split = blockIdx.y;
  const int k_begin = split * p.chunks_per_cta * p.k_chunk;
  const int k_end = min(k_begin + p.chunks_per_cta * p.k_chunk, p.K);
  const int num_k_tiles = (k_end - k_begin + BK - 1) / BK;
  // Chunk boundaries fall on k-tile boundaries (k_chunk % BK == 0 when the
  // CTA walks more than one chunk), so chunk partials are the same mma
  // sequences a one-chunk CTA would produce.
  const int tiles_per_chunk =
      p.chunks_per_cta > 1 ? p.k_chunk / BK : num_k_tiles;

  const T* __restrict__ A = reinterpret_cast<const T*>(p.a);
  const T* __restrict__ B = reinterpret_cast<const T*>(p.b);

  auto load_stage = [&](int stage, int kt) {
    const int kbase = k_begin + kt * BK;
    T* as = As + stage * A_STAGE;
    T* bs = Bs + stage * B_STAGE;
    // A tile: BM rows x BK/8 chunks of 16 B.
    constexpr int A_CHUNKS = BM * (BK / 8);
    for (int c = tid; c < A_CHUNKS; c += NTHREADS) {
      const int r = c / (BK / 8);
      const int kc = (c % (BK / 8)) * 8;
      const int m = m0 + r;
      const int k = kbase + kc;
      const bool valid = (m < p.M) && (k < k_end);
      const T* src = valid ? (A + static_cast<int64_t>(m) * p.lda + k) : A;
      cp_async_16(smem_u32(as + r * A_PITCH + kc), src, valid);
    }
    if constexpr (B_KMAJOR) {
      constexpr int B_CHUNKS = BN * (BK / 8);
      for (int c = tid; c < B_CHUNKS; c += NTHREADS) {
        const int r = c / (BK / 8);
        const int kc = (c % (BK / 8)) * 8;
        const int n = n0 + r;
        const int k = kbase + kc;
        const bool valid = (n < p.N) && (k < k_end);
        const T* src = valid ? (B + static_cast<int64_t>(n) * p.ldb + k) : B;
        cp_async_16(smem_u32(bs + r * B_PITCH + kc), src, valid);
      }
    } else {
      constexpr int B_CHUNKS = BK * (BN / 8);
      for (int c = tid; c < B_CHUNKS; c += NTHREADS) {
        const int r = c / (BN / 8);
        const int nc = (c % (BN / 8)) * 8;
        const int k = kbase + r;
        const int n = n0 + nc;
        const bool valid = (k < k_end) && (n < p.N);
        const T* src = valid ? (B + static_cast<int64_t>(k) * p.ldb + n) : B;
        cp_async_16(smem_u32(bs + r * B_PITCH + nc), src, valid);
      }
    }
  };

  // acc: current chunk; tot: 0 + chunk partials in order. tot is only used
  // when the CTA walks more than one chunk (MULTI); the additions mirror
  // reduce_partials exactly.
  float acc[MT][NT][4], tot[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = tot[i][j][r] = 0.f;
  auto flush_chunk = [&]() {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          tot[i][j][r] += acc[i][j][r];
          acc[i][j][r] = 0.f;
        }
  };

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_k_tiles) load_stage(s, s);
    cp_async_commit();
  }

  for (int kt = 0; kt < num_k_tiles; ++kt) {
    if (MULTI && kt > 0 && kt % tiles_per_chunk == 0) flush_chunk();
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + STAGES - 1;
      if (nk < num_k_tiles) load_stage(nk % STAGES, nk);
      cp_async_commit();
    }
    const T* as = As + (kt % STAGES) * A_STAGE;
    const T* bs = Bs + (kt % STAGES) * B_STAGE;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t afrag[MT][4];
      uint32_t bfrag[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int row = wm * WTM + i * 16 + (lane % 16);
        const int col = kk + (lane / 16) * 8;
        ldmatrix_x4(afrag[i][0], afrag[i][1], afrag[i][2], afrag[i][3],
                    smem_u32(as + row * A_PITCH + col));
      }
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        if constexpr (B_KMAJOR) {
          const int n = wn * WTN + j * 8 + (lane / 16) * 8 + (lane % 8);
          const int k = kk + ((lane / 8) % 2) * 8;
          ldmatrix_x4(bfrag[j][0], bfrag[j][1], bfrag[j + 1][0],
                      bfrag[j + 1][1], smem_u32(bs + n * B_PITCH + k));
        } else {
          const int k = kk + ((lane / 8) % 2) * 8 + (lane % 8);
          const int n = wn * WTN + j * 8 + (lane / 16) * 8;
          ldmatrix_x4_trans(bfrag[j][0], bfrag[j][1], bfrag[j + 1][0],
                            bfrag[j + 1][1], smem_u32(bs + k * B_PITCH + n));
        }
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
          MmaTraits<T>::mma(acc[i][j], afrag[i], bfrag[j]);
    }
  }
  cp_async_wait<0>();
  if (MULTI) flush_chunk();

  // Epilogue. Fragment element r of tile (i, j): row = lane/4 (+8 for r >= 2),
  // col = (lane % 4) * 2 + (r & 1).
  const T* __restrict__ bias = reinterpret_cast<const T*>(p.bias);
  T* __restrict__ C = reinterpret_cast<T*>(p.c);
  float* __restrict__ ws = p.ws == nullptr
                               ? nullptr
                               : p.ws + static_cast<int64_t>(split) * p.M * p.N;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = m0 + wm * WTM + i * 16 + (lane / 4) + (r >= 2 ? 8 : 0);
        const int col = n0 + wn * WTN + j * 8 + (lane % 4) * 2 + (r & 1);
        if (row >= p.M || col >= p.N) continue;
        const int64_t idx = static_cast<int64_t>(row) * p.N + col;
        float v = MULTI ? tot[i][j][r] : acc[i][j][r];
        if (ws != nullptr) {
          ws[idx] = v;
        } else {
          if (bias != nullptr) v += MmaTraits<T>::to_float(bias[col]);
          C[idx] = MmaTraits<T>::from_float(v);
        }
      }
    }
  }
}

// Fixed-order reduction of P partials: c = cast(0 + ws[0] + ws[1] .. + bias).
__device__ __forceinline__ float reduce_partials(const float* __restrict__ ws,
                                                 int64_t MN, int64_t idx,
                                                 int P) {
  float v = 0.f;
  for (int s = 0; s < P; ++s) v += ws[static_cast<int64_t>(s) * MN + idx];
  return v;
}

template <typename T>
__global__ void bi_gemm_reduce_kernel(const float* __restrict__ ws,
                                      const T* __restrict__ bias,
                                      T* __restrict__ c, int64_t MN, int N,
                                      int P) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= MN) return;
  float v = reduce_partials(ws, MN, idx, P);
  if (bias != nullptr) v += MmaTraits<T>::to_float(bias[idx % N]);
  c[idx] = MmaTraits<T>::from_float(v);
}

// fp32: plain FMA accumulation, BMxBN tile, each thread owns a 4x4 block.
template <int BM, int BN, int BK, bool B_KMAJOR>
__global__ void __launch_bounds__(256) bi_gemm_fp32_kernel(const GemmParams p) {
  __shared__ float As[BK][BM + 4];
  __shared__ float Bs[BK][BN + 4];
  const int tid = threadIdx.x;
  const int m_tile = blockIdx.x % p.m_tiles;
  const int n_tile = blockIdx.x / p.m_tiles;
  const int m0 = m_tile * BM;
  const int n0 = n_tile * BN;
  const int split = blockIdx.y;
  const int k_begin = split * p.k_chunk;
  const int k_end = min(k_begin + p.k_chunk, p.K);
  const float* __restrict__ A = reinterpret_cast<const float*>(p.a);
  const float* __restrict__ B = reinterpret_cast<const float*>(p.b);
  const int tm = (tid / (BN / 4)) * 4;
  const int tn = (tid % (BN / 4)) * 4;
  float acc[4][4] = {};
  for (int kb = k_begin; kb < k_end; kb += BK) {
    for (int c = tid; c < BM * BK; c += 256) {
      const int r = c / BK, k = c % BK;
      const int m = m0 + r, kg = kb + k;
      As[k][r] = (m < p.M && kg < k_end)
                     ? A[static_cast<int64_t>(m) * p.lda + kg]
                     : 0.f;
    }
    for (int c = tid; c < BK * BN; c += 256) {
      const int k = c / BN, r = c % BN;
      const int n = n0 + r, kg = kb + k;
      float v = 0.f;
      if (n < p.N && kg < k_end)
        v = B_KMAJOR ? B[static_cast<int64_t>(n) * p.ldb + kg]
                     : B[static_cast<int64_t>(kg) * p.ldb + n];
      Bs[k][r] = v;
    }
    __syncthreads();
#pragma unroll 4
    for (int k = 0; k < BK; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[k][tm + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[k][tn + j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  const float* __restrict__ bias = reinterpret_cast<const float*>(p.bias);
  float* __restrict__ C = reinterpret_cast<float*>(p.c);
  float* __restrict__ ws = p.ws == nullptr
                               ? nullptr
                               : p.ws + static_cast<int64_t>(split) * p.M * p.N;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = m0 + tm + i, col = n0 + tn + j;
      if (row >= p.M || col >= p.N) continue;
      const int64_t idx = static_cast<int64_t>(row) * p.N + col;
      float v = acc[i][j];
      if (ws != nullptr) {
        ws[idx] = v;
      } else {
        if (bias != nullptr) v += bias[col];
        C[idx] = v;
      }
    }
  }
}

__global__ void bi_gemm_reduce_fp32_kernel(const float* __restrict__ ws,
                                           const float* __restrict__ bias,
                                           float* __restrict__ c, int64_t MN,
                                           int N, int P) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= MN) return;
  float v = reduce_partials(ws, MN, idx, P);
  if (bias != nullptr) v += bias[idx % N];
  c[idx] = v;
}

template <typename T, int BM, int BN, int BK, int STAGES, int WARPS_M,
          int WARPS_N, bool B_KMAJOR, bool MULTI>
void launch_mma(const GemmParams& p, int n_tiles, int grid_y,
                cudaStream_t stream) {
  constexpr int A_STAGE = BM * (BK + kPad);
  constexpr int B_STAGE = B_KMAJOR ? BN * (BK + kPad) : BK * (BN + kPad);
  constexpr size_t smem =
      static_cast<size_t>(STAGES) * (A_STAGE + B_STAGE) * sizeof(T);
  auto kernel =
      bi_gemm_kernel<T, BM, BN, BK, STAGES, WARPS_M, WARPS_N, B_KMAJOR, MULTI>;
  static bool attr_set[64] = {};  // per (instantiation, device)
  const int dev = at::cuda::current_device();
  if (!attr_set[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    attr_set[dev] = true;
  }
  dim3 grid(static_cast<unsigned>(p.m_tiles) * n_tiles, grid_y);
  kernel<<<grid, WARPS_M * WARPS_N * 32, smem, stream>>>(p);
}

size_t max_dynamic_smem() {
  static int cached[64] = {};
  const int dev = at::cuda::current_device();
  if (cached[dev] == 0) {
    int v = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    cached[dev] = v;
  }
  return static_cast<size_t>(cached[dev]);
}

#ifndef VLLM_BI_GEMM_SMEM_BUDGET_KB
  #define VLLM_BI_GEMM_SMEM_BUDGET_KB 100
#endif

// BK=64 with as deep a pipeline as fits VLLM_BI_GEMM_SMEM_BUDGET_KB (leaves
// two CTAs per SM on 227 KB parts) where the device allows it (>= 128 KB
// dynamic smem: SM80+ datacenter parts), else BK=32 / 3 stages (<= 61 KB).
template <typename T, int BM, int BN, int WARPS_M, int WARPS_N, bool B_KMAJOR,
          bool MULTI>
void launch_cfg(const GemmParams& p, int n_tiles, int grid_y,
                cudaStream_t stream) {
  constexpr size_t stage_bytes =
      (BM * (64 + kPad) + (B_KMAJOR ? BN * (64 + kPad) : 64 * (BN + kPad))) *
      sizeof(T);
  constexpr int fit =
      static_cast<int>(VLLM_BI_GEMM_SMEM_BUDGET_KB * 1024 / stage_bytes);
  constexpr int STAGES = fit < 3 ? 3 : (fit > 8 ? 8 : fit);
  if (STAGES * stage_bytes <= max_dynamic_smem())
    launch_mma<T, BM, BN, 64, STAGES, WARPS_M, WARPS_N, B_KMAJOR, MULTI>(
        p, n_tiles, grid_y, stream);
  else
    launch_mma<T, BM, BN, 32, 3, WARPS_M, WARPS_N, B_KMAJOR, MULTI>(
        p, n_tiles, grid_y, stream);
}

template <typename T, int BM, int BN, int WARPS_M, int WARPS_N, bool B_KMAJOR>
void launch_tile(GemmParams& p, const SplitPlan& pl, cudaStream_t stream) {
  p.m_tiles = (p.M + BM - 1) / BM;
  const int n_tiles = (p.N + BN - 1) / BN;
  const int grid_y = (pl.S + p.chunks_per_cta - 1) / p.chunks_per_cta;
  if (p.chunks_per_cta == 1)
    launch_cfg<T, BM, BN, WARPS_M, WARPS_N, B_KMAJOR, false>(p, n_tiles, grid_y,
                                                             stream);
  else
    launch_cfg<T, BM, BN, WARPS_M, WARPS_N, B_KMAJOR, true>(p, n_tiles, grid_y,
                                                            stream);
}

// Tile config chosen from M (row grouping only) and N (BN).
template <typename T, bool B_KMAJOR>
void dispatch_mma(GemmParams& p, const SplitPlan& pl, cudaStream_t stream) {
  const int N = p.N, M = p.M;
  if (N <= 64) {
    if (M <= 16)
      launch_tile<T, 16, 64, 1, 4, B_KMAJOR>(p, pl, stream);
    else if (M <= 32)
      launch_tile<T, 32, 64, 1, 4, B_KMAJOR>(p, pl, stream);
    else if (M <= 64)
      launch_tile<T, 64, 64, 2, 2, B_KMAJOR>(p, pl, stream);
    else
      launch_tile<T, 128, 64, 4, 2, B_KMAJOR>(p, pl, stream);
  } else {
    if (M <= 16)
      launch_tile<T, 16, 128, 1, 4, B_KMAJOR>(p, pl, stream);
    else if (M <= 32)
      launch_tile<T, 32, 128, 1, 4, B_KMAJOR>(p, pl, stream);
    else if (M <= 64)
      launch_tile<T, 64, 128, 2, 2, B_KMAJOR>(p, pl, stream);
    else
      launch_tile<T, 128, 128, 2, 4, B_KMAJOR>(p, pl, stream);
  }
}

template <bool B_KMAJOR>
void dispatch_fp32(GemmParams& p, const SplitPlan& pl, cudaStream_t stream) {
  constexpr int BM = 64, BN = 64, BK = 16;
  p.m_tiles = (p.M + BM - 1) / BM;
  const int n_tiles = (p.N + BN - 1) / BN;
  dim3 grid(static_cast<unsigned>(p.m_tiles) * n_tiles, pl.S);
  bi_gemm_fp32_kernel<BM, BN, BK, B_KMAJOR><<<grid, 256, 0, stream>>>(p);
}

}  // namespace

// SM100 runs every bf16/fp16 shape whose TMA strides are 16 B aligned on
// tcgen05: skinny N (<= kSm100SplitKMaxN) through the chunked split-K kernel,
// wider N as one CUTLASS pass. The choice depends on (dtype, N), never on M.
bool use_sm100(at::ScalarType dtype, int64_t N, int device) {
  return dtype != at::kFloat && N % 8 == 0 && sm100::available(device);
}

int64_t sm100_splitk_max_n() {
  static const int64_t v = [] {
    const char* e = std::getenv("VLLM_BI_GEMM_SM100_MAX_N");
    return e != nullptr ? std::atoll(e) : int64_t{2048};
  }();
  return v;
}

bool use_sm100_splitk(int64_t N) { return N <= sm100_splitk_max_n(); }

// Full mode (one CTA per output tile walking every chunk) once the tile
// count alone fills the GPU; scheduling only, the chunk sum is unchanged.
bool sm100_splitk_full(int tiles) {
  static const int thr = [] {
    const char* e = std::getenv("VLLM_BI_GEMM_SM100_FULL_TILES");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return tiles >= (thr > 0 ? thr : num_sms());
}

int64_t bi_gemm_split_k(int64_t K, int64_t N) {
  if (use_sm100(at::kBFloat16, N, at::cuda::current_device())) {
    if (use_sm100_splitk(N))
      return sm100::splitk_plan(static_cast<int>(K), static_cast<int>(N)).S;
    return 1;
  }
  return split_k_plan(K, N).S;
}

torch::Tensor bi_gemm(torch::Tensor a, torch::Tensor b,
                      const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "bi_gemm: CUDA tensors required");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "bi_gemm: 2D operands required");
  TORCH_CHECK(a.size(1) == b.size(0), "bi_gemm: incompatible dimensions");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "bi_gemm: dtype mismatch");
  const auto dtype = a.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat,
      "bi_gemm: unsupported dtype ", dtype);
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX);
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == N && bias->is_cuda() &&
            bias->scalar_type() == dtype && bias->is_contiguous(),
        "bi_gemm: bias must be a contiguous [N] tensor of the input dtype");
  }
  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  if (M == 0 || N == 0) return c;
  if (K == 0) {
    if (bias.has_value())
      c.copy_(bias->unsqueeze(0).expand({M, N}));
    else
      c.zero_();
    return c;
  }

  // Operand layouts the kernels read directly; anything else is copied once.
  const bool is_half = dtype != at::kFloat;
  const int64_t align = is_half ? 8 : 1;  // 16 B cp.async chunks
  auto a_ok = [&](const torch::Tensor& t) {
    return t.stride(1) == 1 && t.stride(0) % align == 0 &&
           reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  torch::Tensor a2 = a;
  if (!a_ok(a2)) a2 = a.contiguous();
  if (is_half && K % align != 0) {
    // K tail below the 16 B chunk: zero-pad K (x + 0*0 == x exactly).
    const int64_t Kp = (K + align - 1) / align * align;
    a2 = torch::constant_pad_nd(a.contiguous(), {0, Kp - K}, 0);
  }
  torch::Tensor b2 = b;
  bool b_kmajor;
  auto b_aligned = [&](const torch::Tensor& t, int64_t ld) {
    return ld % align == 0 &&
           reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  if (b.stride(0) == 1 && b_aligned(b, b.stride(1)) && K % align == 0) {
    b_kmajor = true;  // weight.t(): B[k, n] at n * K + k
  } else if (b.stride(1) == 1 && b_aligned(b, b.stride(0)) && N % align == 0 &&
             K % align == 0) {
    b_kmajor = false;
  } else {
    // Rare: pack to K-major (n, k) so the K tail can be zero-padded too.
    const int64_t Kp = a2.size(1);
    auto bt = b.t().contiguous();
    if (Kp != K) bt = torch::constant_pad_nd(bt, {0, Kp - K}, 0);
    b2 = bt.t();
    b_kmajor = true;
  }
  const int64_t Kp = a2.size(1);
  TORCH_CHECK(!b_kmajor || b2.stride(0) == 1);
  auto stream = at::cuda::getCurrentCUDAStream(a.device().index()).stream();

  const bool bias_aligned =
      !bias.has_value() ||
      reinterpret_cast<uintptr_t>(bias->data_ptr()) % 16 == 0;
  if (bias_aligned && use_sm100(dtype, N, a.device().index())) {
    sm100::Args sp{a2.data_ptr(),
                   b2.data_ptr(),
                   bias.has_value() ? bias->data_ptr() : nullptr,
                   c.data_ptr(),
                   a2.stride(0),
                   b_kmajor ? b2.stride(1) : b2.stride(0),
                   N,
                   static_cast<int>(M),
                   static_cast<int>(N),
                   static_cast<int>(Kp)};
    const char* err = nullptr;
    if (use_sm100_splitk(N)) {
      const sm100::SplitPlan pl =
          sm100::splitk_plan(static_cast<int>(Kp), sp.N);
      const int tiles = sm100::splitk_tiles(sp.M, sp.N);
      const bool full = pl.S == 1 || sm100_splitk_full(tiles);
      err = dtype == at::kBFloat16
                ? sm100::gemm_splitk_bf16(sp, b_kmajor, pl, full, stream)
                : sm100::gemm_splitk_fp16(sp, b_kmajor, pl, full, stream);
    } else {
      err = dtype == at::kBFloat16 ? sm100::gemm_bf16(sp, b_kmajor, stream)
                                   : sm100::gemm_fp16(sp, b_kmajor, stream);
    }
    TORCH_CHECK(err == nullptr, err);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
  }

  const SplitPlan pl = split_k_plan(Kp, N);
  const int cpc =
      is_half ? chunks_per_cta(pick_exec(mma_tile_count(M, N), pl), pl) : 1;
  const int partials = (pl.S + cpc - 1) / cpc;

  torch::Tensor ws;
  if (partials > 1)
    ws = torch::empty({partials, M, N}, a.options().dtype(at::kFloat));

  GemmParams p;
  p.a = a2.data_ptr();
  p.b = b2.data_ptr();
  p.bias = bias.has_value() ? bias->data_ptr() : nullptr;
  p.c = c.data_ptr();
  p.ws = partials > 1 ? ws.data_ptr<float>() : nullptr;
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(Kp);
  p.lda = a2.stride(0);
  p.ldb = b_kmajor ? b2.stride(1) : b2.stride(0);
  p.k_chunk = pl.k_chunk;
  p.chunks_per_cta = cpc;
  p.m_tiles = 0;

  if (dtype == at::kBFloat16) {
    if (b_kmajor)
      dispatch_mma<__nv_bfloat16, true>(p, pl, stream);
    else
      dispatch_mma<__nv_bfloat16, false>(p, pl, stream);
  } else if (dtype == at::kHalf) {
    if (b_kmajor)
      dispatch_mma<__half, true>(p, pl, stream);
    else
      dispatch_mma<__half, false>(p, pl, stream);
  } else {
    if (b_kmajor)
      dispatch_fp32<true>(p, pl, stream);
    else
      dispatch_fp32<false>(p, pl, stream);
  }
  if (partials > 1) {
    const int P = partials;
    const int64_t MN = M * N;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((MN + threads - 1) / threads);
    if (dtype == at::kBFloat16) {
      bi_gemm_reduce_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
          ws.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(p.bias),
          reinterpret_cast<__nv_bfloat16*>(p.c), MN, p.N, P);
    } else if (dtype == at::kHalf) {
      bi_gemm_reduce_kernel<__half><<<blocks, threads, 0, stream>>>(
          ws.data_ptr<float>(), reinterpret_cast<const __half*>(p.bias),
          reinterpret_cast<__half*>(p.c), MN, p.N, P);
    } else {
      bi_gemm_reduce_fp32_kernel<<<blocks, threads, 0, stream>>>(
          ws.data_ptr<float>(), reinterpret_cast<const float*>(p.bias),
          reinterpret_cast<float*>(p.c), MN, p.N, P);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace vllm::batch_invariant
