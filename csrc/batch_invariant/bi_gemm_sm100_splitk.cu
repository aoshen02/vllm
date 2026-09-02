// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Chunked split-K tcgen05 GEMM for skinny-N shapes (DSv4 decode linears:
// K = 4096, N = 64 .. 2048 at small M), where one CTA walking all of K is
// latency-bound and the CUTLASS single-pass kernel leaves the GPU idle.
//
// Numerics are fixed by (K, N): K is cut into S chunks of k_chunk (a
// multiple of the k-tile); chunk j is accumulated from zero in TMEM by the
// same UMMA sequence whatever M is, and the chunk partials are added in fp32
// by a fixed tree: pairs (q_0 + q_1), then pairs of pairs, up to kMaxCpc
// aligned chunks per node (a missing right sibling is skipped), and the
// nodes are added left to right; bias is added and the sum rounded once.
// Two launch modes reproduce that sum exactly:
//   cluster: a cluster of G = ceil(S / cpc) CTAs per output tile, cpc (a
//            power of two) aligned chunks each, so a CTA's partial is one
//            tree node; the fp32 partials are pushed over DSMEM to the CTA
//            owning each row slice, which finishes the tree;
//   full:    one CTA walks every chunk, keeping the tree in registers.
// The mode and cpc are chosen from the tile count (scheduling only).
//
// CTA = 6 warps: 0-3 epilogue (one TMEM lane quadrant each), 4 TMA producer,
// 5 MMA issuer. Smem holds kStages (A 64xTileK, B TileNxTileK) SW128 stages
// and the DSMEM receive slots; TMEM holds two TileN-column accumulators in
// full mode (chunk j+1 overlaps the epilogue of chunk j), one per chunk in
// cluster mode.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_types.h"

#include "bi_gemm_sm100.h"

namespace vllm::batch_invariant::sm100 {

namespace {

using namespace cute;

// UMMA M: 64 halves the A operand the tensor core reads from smem per
// k-tile (the small-N tiles are bound by that, not by FLOPs), so a 64-row
// tile costs ~2/3 of a 128-row one and needs half the TMEM lanes.
constexpr int kTileM = 64;
// k-tile: 128 for the 64-wide N tile (a 64x64x64 stage is too little work
// per pipeline step), 64 for the 128-wide one (keeps enough stages).
constexpr int kSplitKTileK(int tile_n) { return tile_n == 64 ? 128 : 64; }
constexpr int kThreads = 192;
// Chunks per CTA in cluster mode is capped by the receive area carved out
// of smem next to the pipeline stages.
constexpr int kSplitKMaxCpc(int tile_n) { return tile_n == 64 ? 4 : 2; }
constexpr int kEpiThreads = 128;

struct KParams {
  int M, N, K;
  int64_t ldc;
  int k_chunk, S, cpc;  // chunks per CTA; cpc == S is full mode
  int rp;               // cluster mode: tile rows owned by each rank
  int n_tiles, n_pad;
  const void* bias;
  void* c;
};

template <typename T, bool kBKMajor, int kTileN, int kStages>
struct Cfg {
  static constexpr int kTileK = kSplitKTileK(kTileN);
  static constexpr UMMA::Major kMajorB =
      kBKMajor ? UMMA::Major::K : UMMA::Major::MN;
  using TiledMma =
      decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<T, T, float, kTileM, kTileN,
                                                   UMMA::Major::K, kMajorB>{}));
  using MmaTiler = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using MmaShapeA = decltype(partition_shape_A(
      TiledMma{}, Shape<Int<kTileM>, Int<kTileK>>{}));
  using MmaShapeB = decltype(partition_shape_B(
      TiledMma{}, Shape<Int<kTileN>, Int<kTileK>>{}));
  using AtomA = UMMA::Layout_K_SW128_Atom<T>;
  using AtomB = std::conditional_t<kBKMajor, UMMA::Layout_K_SW128_Atom<T>,
                                   UMMA::Layout_MN_SW128_Atom<T>>;
  using StepB =
      std::conditional_t<kBKMajor, Step<_1, _2, _3>, Step<_2, _1, _3>>;
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      AtomA{}, append(MmaShapeA{}, Int<kStages>{})));
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      AtomB{}, append(MmaShapeB{}, Int<kStages>{}), StepB{}));
  static constexpr uint32_t kTmaBytes =
      (kTileM * kTileK + kTileN * kTileK) * sizeof(T);
  // Full mode ping-pongs two TMEM accumulators; cluster mode gives every
  // chunk of the CTA its own, so the chunk partials stay separate until
  // the cluster-wide in-order sum.
  static constexpr int kTmemBufs = (512 / kTileN) * (128 / kTileM);
  // Cluster mode: G receive slots of rp rows, one per sending CTA, that
  // every CTA streams its fp32 partial rows into (st.async over DSMEM,
  // counted on recv_full); rows are padded so the row-per-lane stores hit
  // distinct banks. rp = ceil(kTileM / G), so G * rp can exceed kTileM by
  // up to G - 1 rows.
  static constexpr int kPartStride = kTileN + 4;
  static constexpr int kMaxCpc = kSplitKMaxCpc(kTileN);
  static constexpr int kLevels = kMaxCpc == 4 ? 2 : 1;
  static constexpr int kRecvRows = kTileM + kSplitKMaxS - 1;
  static_assert(kMaxCpc <= kTmemBufs, "one TMEM accumulator per chunk");
  struct SharedStorage {
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutA>> A;
    alignas(1024) ArrayEngine<T, cosize_v<SmemLayoutB>> B;
    alignas(16) float recv[kRecvRows * kPartStride];
    alignas(8) uint64_t full[kStages];
    alignas(8) uint64_t empty[kStages];
    alignas(8) uint64_t acc_full[kTmemBufs];
    alignas(8) uint64_t acc_empty[kTmemBufs];
    alignas(8) uint64_t recv_full;
    uint32_t tmem_base;
  };
  static constexpr int kSmemBytes = sizeof(SharedStorage) + 1024;
};

// 16 B register -> (remote) cluster smem store, counted on that CTA's
// mbarrier tx-count.
__device__ __forceinline__ void dsmem_st_async(uint32_t dst, float4 v,
                                               uint32_t bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32"
      " [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(dst),
      "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)),
      "r"(__float_as_uint(v.z)), "r"(__float_as_uint(v.w)), "r"(bar)
      : "memory");
}

// The chunk-sum tree over n leaves, streamed in leaf order: load(a, i,
// add) sets or adds leaf i into a. subtree<L> builds the level-L node
// rooted at leaf g0 (a missing right sibling is skipped); tree_sum<L> adds
// the level-L nodes left to right into run. Level 1 streams straight into
// its left leaf, so a level-2 node needs one extra register slab.
template <int NF, int L, class Load>
__device__ __forceinline__ void subtree(float (&a)[NF], int g0, int n,
                                        Load& load) {
  if constexpr (L == 0) {
    load(a, g0, false);
  } else if constexpr (L == 1) {
    load(a, g0, false);
    if (g0 + 1 < n) load(a, g0 + 1, true);
  } else {
    subtree<NF, L - 1>(a, g0, n, load);
    constexpr int kHalf = 1 << (L - 1);
    if (g0 + kHalf < n) {
      float b[NF];
      subtree<NF, L - 1>(b, g0 + kHalf, n, load);
      CUTE_UNROLL
      for (int e = 0; e < NF; ++e) a[e] += b[e];
    }
  }
}

template <int NF, int L, class Load>
__device__ __forceinline__ void tree_sum(float (&run)[NF], int n, Load& load) {
  subtree<NF, L>(run, 0, n, load);
  for (int g0 = 1 << L; g0 < n; g0 += 1 << L) {
    float a[NF];
    subtree<NF, L>(a, g0, n, load);
    CUTE_UNROLL
    for (int e = 0; e < NF; ++e) run[e] += a[e];
  }
}

// Runtime level l <= L (l = log2 of the chunks per CTA).
template <int NF, int L, class Load>
__device__ __forceinline__ void subtree_at(float (&a)[NF], int l, int n,
                                           Load& load) {
  if constexpr (L == 0) {
    subtree<NF, 0>(a, 0, n, load);
  } else {
    if (l == L)
      subtree<NF, L>(a, 0, n, load);
    else
      subtree_at<NF, L - 1>(a, l, n, load);
  }
}

template <int NF, int L, class Load>
__device__ __forceinline__ void tree_sum_from(float (&run)[NF], int l, int n,
                                              Load& load) {
  if constexpr (L == 0) {
    tree_sum<NF, 0>(run, n, load);
  } else {
    if (l == 0)
      tree_sum<NF, L>(run, n, load);
    else
      tree_sum_from<NF, L - 1>(run, l - 1, n, load);
  }
}

template <typename T>
__device__ __forceinline__ uint32_t pack2(float a, float b);
template <>
__device__ __forceinline__ uint32_t pack2<cutlass::bfloat16_t>(float a,
                                                               float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
template <>
__device__ __forceinline__ uint32_t pack2<cutlass::half_t>(float a, float b) {
  __half2 v = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <typename T>
__device__ __forceinline__ float bias_at(const void* bias, int n) {
  return static_cast<float>(static_cast<const T*>(bias)[n]);
}

template <class Cfg, class TmaA, class TmaB>
__global__ void __launch_bounds__(kThreads, 1)
    bi_gemm_sm100_splitk_kernel(CUTE_GRID_CONSTANT TmaA const tma_a,
                                CUTE_GRID_CONSTANT TmaB const tma_b,
                                KParams p) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  using T = typename Cfg::T_;
  constexpr int kTileN = Cfg::kTileN_;
  constexpr int kStages = Cfg::kStages_;
  using SharedStorage = typename Cfg::SharedStorage;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SharedStorage*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));

  // shfl makes the warp index provably uniform: without it ptxas wraps every
  // tcgen05.mma in a WARPSYNC.COLLECTIVE/ELECT block (~2x per-MMA cost).
  const int warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
  const int m_tile = blockIdx.x, n_tile = blockIdx.y, z = blockIdx.z;
  const int chunk0 = z * p.cpc;
  const int chunk1 = min(chunk0 + p.cpc, p.S);
  constexpr int kTileK = Cfg::kTileK;
  const int kt_per_chunk = p.k_chunk / kTileK;
  const int kt_total = (p.K + kTileK - 1) / kTileK;

  Tensor mA = tma_a.get_tma_tensor(make_shape(p.M, p.K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(p.N, p.K));
  auto coord = make_coord(m_tile, n_tile, _);
  Tensor gA =
      local_tile(mA, typename Cfg::MmaTiler{}, coord, Step<_1, X, _1>{});
  Tensor gB =
      local_tile(mB, typename Cfg::MmaTiler{}, coord, Step<X, _1, _1>{});
  Tensor sA =
      make_tensor(make_smem_ptr(smem.A.begin()), typename Cfg::SmemLayoutA{});
  Tensor sB =
      make_tensor(make_smem_ptr(smem.B.begin()), typename Cfg::SmemLayoutB{});

  typename Cfg::TiledMma tiled_mma;
  ThrMMA cta_mma = tiled_mma.get_slice(0);
  Tensor tCgA = cta_mma.partition_A(gA);      // (MmaA, MMA_M, MMA_K, k_tiles)
  Tensor tCgB = cta_mma.partition_B(gB);      // (MmaB, MMA_N, MMA_K, k_tiles)
  Tensor tCrA = cta_mma.make_fragment_A(sA);  // (MmaA, MMA_M, MMA_K, stages)
  Tensor tCrB = cta_mma.make_fragment_B(sB);
  Tensor tCtAcc = tiled_mma.make_fragment_C(
      append(partition_shape_C(tiled_mma, Shape<Int<kTileM>, Int<kTileN>>{}),
             Int<Cfg::kTmemBufs>{}));  // ((128, TileN), 1, 1, bufs) in TMEM
  const bool full = p.cpc == p.S;
  const int nbuf = full ? 2 : p.cpc;
  // Cluster mode: rank r owns tile rows [r*rp, (r+1)*rp) of the m_valid.
  const int m_valid = min(kTileM, p.M - m_tile * kTileM);
  const int rank = full ? 0 : static_cast<int>(block_rank_in_cluster());
  const int rows_me = full ? 0 : max(0, min(p.rp, m_valid - rank * p.rp));
  // tcgen05.alloc wants a power of two >= 32 columns; take only what the
  // resident buffers need (64-row tiles pair up in the lanes of a column).
  constexpr int kBufsPerCol = 128 / kTileM;
  int tmem_cols = 32;
  while (tmem_cols < (nbuf + kBufsPerCol - 1) / kBufsPerCol * kTileN)
    tmem_cols *= 2;

  auto [tAgA, tAsA] =
      tma_partition(tma_a, _0{}, Layout<_1>{}, group_modes<0, 3>(sA),
                    group_modes<0, 3>(tCgA));
  auto [tBgB, tBsB] =
      tma_partition(tma_b, _0{}, Layout<_1>{}, group_modes<0, 3>(sB),
                    group_modes<0, 3>(tCgB));

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp == 4) {
    tmem_allocator.allocate(tmem_cols, &smem.tmem_base);
  }
  if (threadIdx.x == 0) {
    prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    for (int s = 0; s < kStages; ++s) {
      initialize_barrier(smem.full[s], 1);
      initialize_barrier(smem.empty[s], 1);
    }
    for (int b = 0; b < Cfg::kTmemBufs; ++b) {
      initialize_barrier(smem.acc_full[b], 1);
      initialize_barrier(smem.acc_empty[b], kEpiThreads);
    }
    initialize_barrier(smem.recv_full, 1);
    cutlass::arch::fence_barrier_init();
    if (!full) {
      // Every row this CTA owns arrives once per CTA of the cluster (this
      // one included).
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          &smem.recv_full, gridDim.z * rows_me * kTileN * sizeof(float));
    }
  }
  __syncthreads();
  // Cluster-wide: barriers are initialised before anyone stores into them.
  // The wait is deferred to just before the first remote store.
  if (!full) cluster_arrive();
  tCtAcc.data() = smem.tmem_base;

  if (warp == 4) {
    if (elect_one_sync()) {
      int it = 0;
      for (int c = chunk0; c < chunk1; ++c) {
        const int kt0 = c * kt_per_chunk;
        const int kt1 = min(kt0 + kt_per_chunk, kt_total);
        for (int kt = kt0; kt < kt1; ++kt, ++it) {
          const int stage = it % kStages;
          const int ph = (it / kStages) & 1;
          wait_barrier(smem.empty[stage], ph ^ 1);
          set_barrier_transaction_bytes(smem.full[stage], Cfg::kTmaBytes);
          copy(tma_a.with(smem.full[stage]), tAgA(_, kt), tAsA(_, stage));
          copy(tma_b.with(smem.full[stage]), tBgB(_, kt), tBsB(_, stage));
        }
      }
    }
    __syncwarp();
    if (!full) cluster_wait();
  } else if (warp == 5) {
    {
      int it = 0;
      int buf = 0, aph = 0;
      for (int c = chunk0; c < chunk1; ++c) {
        wait_barrier(smem.acc_empty[buf], aph ^ 1);
        Tensor acc = tCtAcc(_, _, _, buf);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        const int kt0 = c * kt_per_chunk;
        const int kt1 = min(kt0 + kt_per_chunk, kt_total);
        for (int kt = kt0; kt < kt1; ++kt, ++it) {
          const int stage = it % kStages;
          const int ph = (it / kStages) & 1;
          wait_barrier(smem.full[stage], ph);
          CUTE_UNROLL
          for (int kb = 0; kb < size<2>(tCrA); ++kb) {
            gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage), acc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive(&smem.empty[stage]);
        }
        cutlass::arch::umma_arrive(&smem.acc_full[buf]);
        if (++buf == nbuf) {
          buf = 0;
          aph ^= 1;
        }
      }
    }
    __syncwarp();
    if (!full) cluster_wait();
  } else {
    // Epilogue warps 0-3: each TMEM load hands every thread 32 consecutive
    // columns of one row (128-row tiles: 32-column slabs, one row per
    // thread; 64-row tiles: 64-column slabs, two threads per row).
    using TmemLoad =
        std::conditional_t<kTileM == 128, SM100_TMEM_LOAD_32dp32b32x,
                           SM100_TMEM_LOAD_16dp32b32x>;
    TiledCopy t2r = make_tmem_copy(TmemLoad{}, tCtAcc(_, _, _, _0{}));
    ThrCopy thr = t2r.get_slice(threadIdx.x);
    Tensor cC = make_identity_tensor(Shape<Int<kTileM>, Int<kTileN>>{});
    // (CPY, rest...) -> (CPY, slabs)
    Tensor tDcC = group_modes<1, 4>(thr.partition_D(cta_mma.partition_C(cC)));
    constexpr int kSlabCols = 32 * (128 / kTileM);
    constexpr int kSlabs = kTileN / kSlabCols;
    static_assert(size<1>(decltype(tDcC){}) == kSlabs, "slab partition");
    static_assert(size<0>(decltype(tDcC){}) == 32, "32 values per load");
    constexpr int NF = kSlabs * 32;
    const int row_in_tile = get<0>(tDcC(_0{}, _0{}));
    const int row = m_tile * kTileM + row_in_tile;
    int scol[kSlabs];  // first tile column of this thread in each slab
    CUTE_UNROLL
    for (int s = 0; s < kSlabs; ++s) scol[s] = get<1>(tDcC(_0{}, s));

    Tensor rSlab = make_tensor<float>(shape(tDcC(_, _0{})));
    const int col0 = n_tile * kTileN;
    const int n_chunks = chunk1 - chunk0;
    // Leaf i = this CTA's i-th chunk: full mode ping-pongs two TMEM
    // buffers, cluster mode gives each chunk its own.
    auto load = [&](float (&a)[NF], int i, bool add) {
      const int buf = full ? (i & 1) : i;
      wait_barrier(smem.acc_full[buf], full ? ((i >> 1) & 1) : 0);
      Tensor tDtAcc = group_modes<1, 4>(thr.partition_S(tCtAcc(_, _, _, buf)));
      CUTE_UNROLL
      for (int s = 0; s < kSlabs; ++s) {
        copy(t2r, tDtAcc(_, s), rSlab);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int e = 0; e < 32; ++e) {
          if (add)
            a[s * 32 + e] += rSlab(e);
          else
            a[s * 32 + e] = rSlab(e);
        }
      }
      if (full) arrive_barrier(smem.acc_empty[buf]);
    };
    float run[NF];
    if (full) {
      tree_sum<NF, Cfg::kLevels>(run, n_chunks, load);
      if (row < p.M) {
        T* crow = static_cast<T*>(p.c) + static_cast<int64_t>(row) * p.ldc;
        CUTE_UNROLL
        for (int i = 0; i < NF; i += 8) {
          const int n = col0 + scol[i / 32] + i % 32;
          if (n < p.N) {
            float v[8];
            CUTE_UNROLL
            for (int u = 0; u < 8; ++u) {
              v[u] = run[i + u];
              if (p.bias != nullptr) v[u] += bias_at<T>(p.bias, n + u);
            }
            uint4 o;
            o.x = pack2<T>(v[0], v[1]);
            o.y = pack2<T>(v[2], v[3]);
            o.z = pack2<T>(v[4], v[5]);
            o.w = pack2<T>(v[6], v[7]);
            *reinterpret_cast<uint4*>(crow + n) = o;
          }
        }
      }
    } else {
      // This CTA's cpc aligned chunks form one level-log2(cpc) node.
      subtree_at<NF, Cfg::kLevels>(run, __ffs(p.cpc) - 1, n_chunks, load);
      // Each thread streams its own row slice straight into the owner's
      // slot for this rank (rows past the tile's valid rows are dropped).
      cluster_wait();
      if (row_in_tile < m_valid) {
        const int owner = row_in_tile / p.rp;
        const int rr = row_in_tile - owner * p.rp;
        const uint32_t dst0 = set_block_rank(
            cast_smem_ptr_to_uint(smem.recv +
                                  (rank * p.rp + rr) * Cfg::kPartStride),
            owner);
        const uint32_t bar =
            set_block_rank(cast_smem_ptr_to_uint(&smem.recv_full), owner);
        CUTE_UNROLL
        for (int s = 0; s < kSlabs; ++s) {
          CUTE_UNROLL
          for (int i = 0; i < 32; i += 4)
            dsmem_st_async(
                dst0 + (scol[s] + i) * 4,
                make_float4(run[s * 32 + i], run[s * 32 + i + 1],
                            run[s * 32 + i + 2], run[s * 32 + i + 3]),
                bar);
        }
      }
    }
  }

  if (!full) {
    // Cluster split-K: rank r now holds every CTA's node for its rows;
    // finish the tree from level log2(cpc).
    const int slot_floats = p.rp * Cfg::kPartStride;
    wait_barrier(smem.recv_full, 0);
    constexpr int kGroups = kTileN / 8;
    const int col0 = n_tile * kTileN;
    const int G = gridDim.z;
    const int lvl = __ffs(p.cpc) - 1;
    for (int idx = threadIdx.x; idx < rows_me * kGroups; idx += kThreads) {
      const int rr = idx / kGroups;
      const int c = (idx % kGroups) * 8;
      const int row = rank * p.rp + rr;
      const int n = col0 + c;
      if (n >= p.N) continue;
      auto loadp = [&](float (&v)[8], int r, bool add) {
        const float* src =
            smem.recv + r * slot_floats + rr * Cfg::kPartStride + c;
        const float4 a = *reinterpret_cast<const float4*>(src);
        const float4 b = *reinterpret_cast<const float4*>(src + 4);
        const float w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        CUTE_UNROLL
        for (int u = 0; u < 8; ++u) {
          if (add)
            v[u] += w[u];
          else
            v[u] = w[u];
        }
      };
      float v[8];
      tree_sum_from<8, Cfg::kLevels>(v, lvl, G, loadp);
      if (p.bias != nullptr) {
        CUTE_UNROLL
        for (int u = 0; u < 8; ++u) v[u] += bias_at<T>(p.bias, n + u);
      }
      uint4 o;
      o.x = pack2<T>(v[0], v[1]);
      o.y = pack2<T>(v[2], v[3]);
      o.z = pack2<T>(v[4], v[5]);
      o.w = pack2<T>(v[6], v[7]);
      *reinterpret_cast<uint4*>(
          static_cast<T*>(p.c) +
          static_cast<int64_t>(m_tile * kTileM + row) * p.ldc + n) = o;
    }
  }
  __syncthreads();
  if (warp == 4) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base, tmem_cols);
  }
#else
  asm volatile("trap;");
#endif
}

template <typename T, bool kBKMajor, int kTileN, int kStages>
struct CfgT : Cfg<T, kBKMajor, kTileN, kStages> {
  using T_ = T;
  static constexpr int kTileN_ = kTileN;
  static constexpr int kStages_ = kStages;
};

int num_sms() {
  static int cached = 0;
  if (cached == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
  }
  return cached;
}

template <typename T, bool kBKMajor, int kTileN>
const char* launch(const Args& p, const SplitPlan& pl, bool full,
                   cudaStream_t stream) {
  // Fill the 227 KB opt-in smem around the receive area (64-row tiles: 6
  // stages for TileN=64, 7 for TileN=128).
  constexpr int kTileK = kSplitKTileK(kTileN);
  constexpr int kStageBytes = (kTileM * kTileK + kTileN * kTileK) * sizeof(T);
  constexpr int kRecvBytes =
      (kTileM + kSplitKMaxS - 1) * (kTileN + 4) * sizeof(float);
  constexpr int kStages = (232448 - 3072 - kRecvBytes) / kStageBytes;
  using C = CfgT<T, kBKMajor, kTileN, kStages>;
  Tensor mA = make_tensor(make_gmem_ptr(static_cast<const T*>(p.a)),
                          make_shape(p.M, p.K), make_stride(p.lda, _1{}));
  auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA,
                             typename C::SmemLayoutA{}(_, _, _, _0{}),
                             Shape<Int<kTileM>, Int<kTileK>>{});
  auto b_stride = [&] {
    if constexpr (kBKMajor)
      return make_stride(p.ldb, _1{});
    else
      return make_stride(_1{}, p.ldb);
  }();
  Tensor mB = make_tensor(make_gmem_ptr(static_cast<const T*>(p.b)),
                          make_shape(p.N, p.K), b_stride);
  auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB,
                             typename C::SmemLayoutB{}(_, _, _, _0{}),
                             Shape<Int<kTileN>, Int<kTileK>>{});
  auto* kernel =
      &bi_gemm_sm100_splitk_kernel<C, decltype(tma_a), decltype(tma_b)>;
  static bool attr_set = false;
  if (!attr_set) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             C::kSmemBytes) != cudaSuccess ||
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeNonPortableClusterSizeAllowed,
                             1) != cudaSuccess)
      return "bi_gemm_sm100_splitk: cudaFuncSetAttribute failed";
    attr_set = true;
  }
  const int m_tiles = (p.M + kTileM - 1) / kTileM;
  const int n_tiles = (p.N + kTileN - 1) / kTileN;
  const int tiles = m_tiles * n_tiles;
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = C::kSmemBytes;
  cfg.stream = stream;
  // Scheduling only (the chunk sum is fixed): pick chunks-per-CTA by a wave
  // cost model, k-tiles per CTA plus a fixed launch/epilogue cost, against
  // how many G-CTA clusters the device co-schedules.
  static int cap[kSplitKMaxS + 1] = {};
  const int kt_chunk = pl.k_chunk / kTileK;
  constexpr int kFixed = 28;  // ~3 us in k-tiles
  constexpr int kReduce = 5;
  int cpc = pl.S;
  static const int force_cpc = [] {
    const char* e = std::getenv("VLLM_BI_GEMM_SM100_CPC");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (!full && force_cpc > 0) {
    cpc = std::min(force_cpc, C::kMaxCpc);
    cpc = 1 << (31 - __builtin_clz(cpc));  // aligned tree nodes
    if (force_cpc >= pl.S) cpc = pl.S;
  } else if (!full) {
    const int waves_full = (tiles + num_sms() - 1) / num_sms();
    int64_t best =
        static_cast<int64_t>(waves_full) * (pl.S * kt_chunk + kFixed);
    for (int c = 1; c <= C::kMaxCpc && c < pl.S; c *= 2) {
      const int G = (pl.S + c - 1) / c;
      if (cap[G] == 0) {
        cfg.gridDim = dim3(1, 1, G);
        attr[0].val.clusterDim = {1, 1, static_cast<unsigned>(G)};
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        int n = 0;
        cap[G] =
            cudaOccupancyMaxActiveClusters(&n, kernel, &cfg) == cudaSuccess &&
                    n > 0
                ? n
                : -1;
        cudaGetLastError();
      }
      if (cap[G] < 0) continue;
      const int waves = (tiles + cap[G] - 1) / cap[G];
      const int64_t cost =
          static_cast<int64_t>(waves) * (c * kt_chunk + kFixed + kReduce);
      if (cost < best) {
        best = cost;
        cpc = c;
      }
    }
  }
  const int G = (pl.S + cpc - 1) / cpc;
  static const bool dbg = std::getenv("VLLM_BI_GEMM_SM100_DEBUG") != nullptr;
  if (dbg) {
    fprintf(stderr,
            "bi_gemm_sm100_splitk: M=%d N=%d K=%d tiles=%d S=%d cpc=%d G=%d",
            p.M, p.N, p.K, tiles, pl.S, cpc, G);
    for (int g = 1; g <= kSplitKMaxS; ++g)
      if (cap[g] != 0) fprintf(stderr, " cap[%d]=%d", g, cap[g]);
    fprintf(stderr, "\n");
  }
  KParams kp{p.M,        p.N,
             p.K,        p.ldc,
             pl.k_chunk, pl.S,
             cpc,        G == 1 ? 0 : (kTileM + G - 1) / G,
             n_tiles,    n_tiles * kTileN,
             p.bias,     p.c};
  cfg.gridDim = dim3(m_tiles, n_tiles, G);
  attr[0].val.clusterDim = {1, 1, static_cast<unsigned>(G)};
  cfg.attrs = attr;
  cfg.numAttrs = G > 1 ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, tma_a, tma_b, kp) == cudaSuccess
             ? nullptr
             : "bi_gemm_sm100_splitk: launch failed";
}

template <typename T>
const char* launch_t(const Args& p, bool b_kmajor, const SplitPlan& pl,
                     bool full, cudaStream_t stream) {
  if (splitk_tile_n(p.N) == 64) {
    return b_kmajor ? launch<T, true, 64>(p, pl, full, stream)
                    : launch<T, false, 64>(p, pl, full, stream);
  }
  return b_kmajor ? launch<T, true, 128>(p, pl, full, stream)
                  : launch<T, false, 128>(p, pl, full, stream);
}

}  // namespace

int splitk_tile_n(int N) { return N <= 64 ? 64 : 128; }

SplitPlan splitk_plan(int K, int N) {
  const int kTileK = kSplitKTileK(splitk_tile_n(N));
  SplitPlan pl{1, (K + kTileK - 1) / kTileK * kTileK};
  if (K < kSplitKMinK) return pl;
  const int want = std::min(kSplitKMaxS, K / (kSplitKMinK / 2));
  const int per = (K + want - 1) / want;
  pl.k_chunk = (per + kTileK - 1) / kTileK * kTileK;
  pl.S = (K + pl.k_chunk - 1) / pl.k_chunk;
  return pl;
}

int splitk_tiles(int M, int N) {
  return ((M + kTileM - 1) / kTileM) *
         ((N + splitk_tile_n(N) - 1) / splitk_tile_n(N));
}

const char* gemm_splitk_bf16(const Args& p, bool b_kmajor, const SplitPlan& pl,
                             bool full, cudaStream_t stream) {
  return launch_t<cutlass::bfloat16_t>(p, b_kmajor, pl, full, stream);
}

const char* gemm_splitk_fp16(const Args& p, bool b_kmajor, const SplitPlan& pl,
                             bool full, cudaStream_t stream) {
  return launch_t<cutlass::half_t>(p, b_kmajor, pl, full, stream);
}

}  // namespace vllm::batch_invariant::sm100
