// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm::batch_invariant::sm100 {

// C[M,N] = A[M,K] @ B[K,N] (+ bias[N]); A row-major (lda), B K-major (ldb =
// stride between n) or N-major (ldb = stride between k), C row-major (ldc).
// Requires lda, ldb, ldc multiples of 8 elements and 16 B aligned pointers.
struct Args {
  const void* a;
  const void* b;
  const void* bias;  // nullable
  void* c;
  int64_t lda, ldb, ldc;
  int M, N, K;
};

// True when `device` is SM100-class and this build carries sm_100a code.
bool available(int device);

// Return nullptr on success, else a static error message.
const char* gemm_bf16(const Args& p, bool b_kmajor, cudaStream_t stream);
const char* gemm_fp16(const Args& p, bool b_kmajor, cudaStream_t stream);

// Chunked split-K path (bi_gemm_sm100_splitk.cu) for skinny N: K is cut into
// S chunks of k_chunk fixed by K alone; the fp32 sum over chunks is in index
// order. `full` runs every chunk in one CTA; otherwise an S-CTA cluster
// computes one chunk each and reduces the partials through DSMEM.
struct SplitPlan {
  int S;
  int k_chunk;  // multiple of the k-tile
};
constexpr int kSplitKMinK = 1024;  // below this a single chunk
constexpr int kSplitKMaxS = 16;
SplitPlan splitk_plan(int K, int N);
int splitk_tile_n(int N);  // 64 or 128
int splitk_tiles(int M, int N);
const char* gemm_splitk_bf16(const Args& p, bool b_kmajor, const SplitPlan& pl,
                             bool full, cudaStream_t stream);
const char* gemm_splitk_fp16(const Args& p, bool b_kmajor, const SplitPlan& pl,
                             bool full, cudaStream_t stream);

}  // namespace vllm::batch_invariant::sm100
