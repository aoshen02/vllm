// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cstdint>

namespace vllm::deterministic_topk {

template <int Threads, int Capacity>
using StableSort = cub::BlockRadixSort<
    uint32_t, Threads, Capacity / Threads, int,
    (Threads >= 512 || Capacity <= 1024 ? 4 : (Capacity <= 2048 ? 5 : 6))>;

template <int Threads, int Capacity>
__device__ void sort_row(const float* logits, int64_t stride, int length,
                         int* output, int top_k, void* storage) {
  using Sort = StableSort<Threads, Capacity>;
  constexpr int items = Capacity / Threads;
  uint32_t keys[items];
  int indices[items];
#pragma unroll
  for (int i = 0; i < items; ++i) {
    // Stable score sorting preserves the descending input index tie order.
    const int index = length - 1 - (threadIdx.x * items + i);
    if (index >= 0) {
      const uint32_t bits = __float_as_uint(logits[index * stride]);
      keys[i] = (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
      indices[i] = index;
    } else {
      keys[i] = 0;
      indices[i] = -1;
    }
  }
  Sort(*static_cast<typename Sort::TempStorage*>(storage))
      .SortDescendingBlockedToStriped(keys, indices);
#pragma unroll
  for (int i = 0; i < items; ++i) {
    const int column = i * Threads + threadIdx.x;
    if (column < top_k) output[column] = column < length ? indices[i] : -1;
  }
}

__device__ inline uint64_t score_index_key(const float* logits, int index) {
  const uint32_t bits = __float_as_uint(logits[index]);
  const uint32_t ordered = (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
  return (static_cast<uint64_t>(ordered) << 32) | static_cast<uint32_t>(index);
}

// Rare coarse-bin overflow: retain every candidate until the exact Kth key is
// known. The caller canonicalizes the selected output after the persistent
// pass.
template <int Threads, int TopK>
__device__ void select_overflow(const float* logits, int length, int* output,
                                void* storage) {
  struct State {
    uint32_t histogram[256];
    uint64_t prefix;
    int rank;
    int count;
  };
  auto& state = *static_cast<State*>(storage);
  __syncthreads();
  if (threadIdx.x == 0) {
    state.prefix = 0;
    state.rank = TopK;
    state.count = 0;
  }
  __syncthreads();
  for (int shift = 56; shift >= 0; shift -= 8) {
    if (threadIdx.x < 256) state.histogram[threadIdx.x] = 0;
    __syncthreads();
    for (int index = threadIdx.x; index < length; index += Threads) {
      const uint64_t key = score_index_key(logits, index);
      const uint64_t prefix = shift == 56 ? 0 : key >> (shift + 8);
      if (prefix == state.prefix)
        atomicAdd(&state.histogram[(key >> shift) & 255], 1U);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int bucket = 255;
      for (; bucket > 0; --bucket) {
        if (state.rank <= state.histogram[bucket]) break;
        state.rank -= state.histogram[bucket];
      }
      state.prefix = (state.prefix << 8) | bucket;
    }
    __syncthreads();
  }
  for (int index = threadIdx.x; index < length; index += Threads) {
    if (score_index_key(logits, index) >= state.prefix) {
      const int slot = atomicAdd(&state.count, 1);
      if (slot < TopK) output[slot] = index;
    }
  }
  __syncthreads();
}

}  // namespace vllm::deterministic_topk
