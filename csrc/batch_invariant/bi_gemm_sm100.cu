// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// tcgen05 (Blackwell SM100) path of the batch-invariant GEMM, built on the
// CUTLASS 3.x TMA warp-specialized collectives.
//
// Every output row is one fp32 accumulation over K in ascending 64-wide
// k-tiles, executed by the same UMMA instruction sequence whatever M is: the
// tile shape and cluster are compile-time, TMA zero-fills M and K tails, and
// nothing in the schedule depends on M. The kernel variant is selected from
// (dtype, B layout, N) only.

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "bi_gemm_sm100.h"

namespace vllm::batch_invariant::sm100 {

namespace {

using namespace cute;

// Kernel bodies only exist in the sm_100a/sm_103a device passes; other
// architectures get a trap so a multi-arch build still links.
template <typename Kernel>
struct Sm100Only : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
  #if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
    Kernel::operator()(static_cast<Args&&>(args)...);
  #else
    asm volatile("trap;");
  #endif
#endif
  }
};

__global__ void probe_kernel(int* ok) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  *ok = 1;
#else
  *ok = 0;
#endif
}

template <typename T, bool kBKMajor, int kTileN>
struct Gemm {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = std::conditional_t<kBKMajor, cutlass::layout::ColumnMajor,
                                     cutlass::layout::RowMajor>;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int kAlign = 128 / cutlass::sizeof_bits<T>::value;
  using TileShape = Shape<_128, Int<kTileN>, _64>;
  using ClusterShape = Shape<_1, _1, _1>;
  // D = 1 * acc + bias, one fp32 add then one rounding to T (C is unused).
  using Fusion = cutlass::epilogue::fusion::LinCombPerColBias<T, float, T, void,
                                                              float, kAlign>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, float,
          float, void, LayoutD, kAlign, T, LayoutD, kAlign,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          Fusion>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, T, LayoutA,
          kAlign, T, LayoutB, kAlign, float, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = Sm100Only<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static const char* run(const Args& p, cudaStream_t stream) {
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;
    StrideA sa{};
    StrideB sb{};
    StrideD sd{};
    get<0>(sa) = p.lda;
    if constexpr (kBKMajor)
      get<0>(sb) = p.ldb;
    else
      get<1>(sb) = p.ldb;
    get<0>(sd) = p.ldc;
    typename Op::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {p.M, p.N, p.K, 1},
        {static_cast<const T*>(p.a), sa, static_cast<const T*>(p.b), sb},
        {{}, nullptr, StrideD{}, static_cast<T*>(p.c), sd}};
    auto& fusion = args.epilogue.thread;
    fusion.alpha = 1.0f;
    fusion.beta = 0.0f;
    fusion.bias_ptr = static_cast<const T*>(p.bias);
    Op op;
    if (op.can_implement(args) != cutlass::Status::kSuccess)
      return "bi_gemm_sm100: can_implement failed";
    if (op.get_workspace_size(args) != 0)
      return "bi_gemm_sm100: kernel unexpectedly needs workspace";
    if (op.run(args, nullptr, stream) != cutlass::Status::kSuccess)
      return "bi_gemm_sm100: launch failed";
    return nullptr;
  }
};

template <typename T, bool kBKMajor>
const char* run_tile(const Args& p, cudaStream_t stream) {
  if (p.N <= 64) return Gemm<T, kBKMajor, 64>::run(p, stream);
  if (p.N <= 128) return Gemm<T, kBKMajor, 128>::run(p, stream);
  return Gemm<T, kBKMajor, 256>::run(p, stream);
}

}  // namespace

bool available(int device) {
  static std::mutex mu;
  static std::unordered_map<int, bool> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device);
  if (it != cache.end()) return it->second;
  bool ok = false;
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) == cudaSuccess &&
      major == 10) {
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(device);
    int* d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) == cudaSuccess) {
      probe_kernel<<<1, 1>>>(d);
      if (cudaDeviceSynchronize() == cudaSuccess &&
          cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess)
        ok = h == 1;
      cudaGetLastError();
      cudaFree(d);
    }
    cudaSetDevice(cur);
  }
  cache[device] = ok;
  return ok;
}

const char* gemm_bf16(const Args& p, bool b_kmajor, cudaStream_t stream) {
  return b_kmajor ? run_tile<cutlass::bfloat16_t, true>(p, stream)
                  : run_tile<cutlass::bfloat16_t, false>(p, stream);
}

const char* gemm_fp16(const Args& p, bool b_kmajor, cudaStream_t stream) {
  return b_kmajor ? run_tile<cutlass::half_t, true>(p, stream)
                  : run_tile<cutlass::half_t, false>(p, stream);
}

}  // namespace vllm::batch_invariant::sm100
