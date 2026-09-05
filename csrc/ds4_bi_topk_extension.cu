// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>
#include <torch/library.h>

#include "libtorch_stable/persistent_topk.cuh"

namespace {

constexpr int kThreads = 512;
constexpr int kChunkColumns = 8192;
constexpr int kItemsPerThread = kChunkColumns / kThreads;
constexpr int kMaxTopK = 2048;

std::mutex workspace_mutex;
using WorkspacePool = std::unordered_map<uint64_t, std::vector<at::Tensor>>;
WorkspacePool workspace_by_stream;
WorkspacePool persistent_workspace_by_stream;
WorkspacePool key_workspace_by_stream;

at::Tensor get_cached_workspace(WorkspacePool& cache, const at::Tensor& logits,
                                int64_t needed, at::ScalarType dtype) {
  const int device = logits.get_device();
  const auto stream = at::cuda::getCurrentCUDAStream(device);
  const uint64_t key = (static_cast<uint64_t>(device) << 32) ^
                       reinterpret_cast<uintptr_t>(stream.stream());
  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto& allocations = cache[key];
  if (allocations.empty() || allocations.back().numel() < needed ||
      allocations.back().device() != logits.device()) {
    allocations.push_back(at::empty({needed}, logits.options().dtype(dtype)));
  }
  return allocations.back().narrow(0, 0, needed);
}

at::Tensor get_workspace(const at::Tensor& logits, int64_t num_rows,
                         int num_chunks, int64_t top_k) {
  const auto needed = num_rows * num_chunks * top_k;
  return get_cached_workspace(workspace_by_stream, logits, needed, at::kLong);
}

at::Tensor get_persistent_workspace(const at::Tensor& logits,
                                    int64_t num_groups) {
  const auto needed = num_groups * static_cast<int64_t>(
                                       sizeof(vllm::persistent::RadixRowState));
  return get_cached_workspace(persistent_workspace_by_stream, logits, needed,
                              at::kByte);
}

at::Tensor get_key_workspace(const at::Tensor& logits, int64_t num_rows,
                             int64_t top_k) {
  const auto needed = num_rows * top_k;
  return get_cached_workspace(key_workspace_by_stream, logits, needed,
                              at::kLong);
}

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__global__ __launch_bounds__(kThreads) void deterministic_top_k_chunk(
    const float* logits, const int* row_starts, const int* row_ends,
    uint64_t* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  extern __shared__ unsigned char chunk_smem[];
  auto& chunk_temp =
      *reinterpret_cast<typename ScoreSort::TempStorage*>(chunk_smem);

  const int row = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_offset = chunk * kChunkColumns;
  const int row_start = row_starts[row] + chunk_offset;
  const int row_end =
      min(row_starts[row] + chunk_offset + kChunkColumns, row_ends[row]);
  uint64_t score_keys[kItemsPerThread];

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int chunk_index = item * kThreads + threadIdx.x;
    const int local_index = chunk_offset + chunk_index;
    if (row_start + chunk_index < row_end) {
      const int absolute_index = row_start + chunk_index;
      const float score =
          logits[static_cast<int64_t>(row) * stride0 +
                 static_cast<int64_t>(absolute_index) * stride1];
      // Descending score, then descending request-local source index,
      // matching vLLM's insertion-sort tie and output semantics. The key is
      // unique, so neither candidate selection nor output depends on warp
      // scheduling or the request's offset in a packed buffer.
      score_keys[item] =
          (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
          static_cast<uint32_t>(local_index);
    } else {
      score_keys[item] = 0;
    }
  }
  ScoreSort(chunk_temp).SortDescendingBlockedToStriped(score_keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int output_column = item * kThreads + threadIdx.x;
    if (output_column < top_k) {
      output[(static_cast<int64_t>(row) * gridDim.y + chunk) * top_k +
             output_column] = score_keys[item];
    }
  }
}

template <int Threads, int ItemsPerThread>
__global__ __launch_bounds__(Threads) void deterministic_top_k_single_small(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, Threads, ItemsPerThread>;
  extern __shared__ unsigned char small_smem[];
  auto& small_temp =
      *reinterpret_cast<typename ScoreSort::TempStorage*>(small_smem);
  const int row = blockIdx.x;
  const int row_start = row_starts[row];
  const int row_end = row_ends[row];
  uint64_t keys[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = item * Threads + threadIdx.x;
    if (row_start + index < row_end) {
      const float score =
          logits[static_cast<int64_t>(row) * stride0 +
                 static_cast<int64_t>(row_start + index) * stride1];
      keys[item] = (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
                   static_cast<uint32_t>(index);
    } else {
      keys[item] = 0;
    }
  }
  ScoreSort(small_temp).SortDescendingBlockedToStriped(keys);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = item * Threads + threadIdx.x;
    if (index < top_k)
      output[static_cast<int64_t>(row) * top_k + index] =
          keys[item] == 0 ? -1
                          : static_cast<int>(static_cast<uint32_t>(keys[item]));
  }
}

template <int MergeThreads, int MergeItems>
__global__ __launch_bounds__(MergeThreads) void merge_top_k_chunks(
    const uint64_t* candidates, int* output, int num_chunks, int top_k) {
  __shared__ uint64_t current[kMaxTopK];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  for (int i = tid; i < top_k; i += MergeThreads) current[i] = 0;
  __syncthreads();
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const uint64_t* tile =
        candidates + (static_cast<int64_t>(row) * num_chunks + chunk) * top_k;
    uint64_t keys[MergeItems];
#pragma unroll
    for (int item = 0; item < MergeItems; ++item) {
      const int rank = item * MergeThreads + tid;
      if (rank >= top_k) continue;
      // Find the rank-th element of two descending sorted lists.  This is a
      // merge-path partition: O(log(top_k)) comparisons per output, versus a
      // full 64-bit radix sort of 2*top_k keys for every chunk.
      int lo = max(0, rank - top_k);
      int hi = min(rank, top_k);
      while (lo < hi) {
        const int i = (lo + hi) >> 1;
        const int j = rank - i;
        if (i > 0 && j < top_k && current[i - 1] < tile[j]) {
          hi = i;
        } else if (j > 0 && i < top_k && tile[j - 1] < current[i]) {
          lo = i + 1;
        } else {
          lo = i;
          break;
        }
      }
      const int i = lo;
      const int j = rank - i;
      const uint64_t a = i < top_k ? current[i] : 0;
      const uint64_t b = j < top_k ? tile[j] : 0;
      keys[item] = a >= b ? a : b;
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < MergeItems; ++item) {
      const int i = item * MergeThreads + tid;
      if (i < top_k) current[i] = keys[item];
    }
    __syncthreads();
  }
  for (int i = tid; i < top_k; i += MergeThreads)
    output[static_cast<int64_t>(row) * top_k + i] =
        current[i] == 0 ? -1
                        : static_cast<int>(static_cast<uint32_t>(current[i]));
}

template <int Threads, int ItemsPerThread>
void launch_single_small(const at::Tensor& logits, const at::Tensor& row_starts,
                         const at::Tensor& row_ends, at::Tensor& indices,
                         int64_t num_rows, int64_t stride0, int64_t stride1,
                         int top_k, cudaStream_t stream) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, Threads, ItemsPerThread>;
  constexpr size_t smem_size = sizeof(typename ScoreSort::TempStorage);
  auto kernel = &deterministic_top_k_single_small<Threads, ItemsPerThread>;
  cudaError_t attr_err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  TORCH_CHECK(attr_err == cudaSuccess,
              "generic Top-K shared memory setup failed: ",
              cudaGetErrorString(attr_err));
  kernel<<<num_rows, Threads, smem_size, stream>>>(
      logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
      row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(), stride0,
      stride1, top_k);
}

__global__ __launch_bounds__(256) void prepare_deterministic_top_k(
    const float* logits, const int* starts, const int* ends, int* output,
    int64_t stride0, int top_k, vllm::persistent::RadixRowState* states,
    int num_groups) {
  namespace D = vllm::deterministic_topk;
  __shared__ D::StableSort<256, 8192>::TempStorage storage;
  const int row = blockIdx.x;
  if (row < num_groups) {
    auto* words = reinterpret_cast<uint32_t*>(states + row);
    for (int i = threadIdx.x;
         i < sizeof(vllm::persistent::RadixRowState) / sizeof(uint32_t);
         i += 256)
      words[i] = 0;
  }
  const int length = max(0, ends[row] - starts[row]);
  if (length >= vllm::persistent::HIST2048_THRESHOLD) return;
  int* row_output = output + static_cast<int64_t>(row) * top_k;
  if (length == 0) {
    for (int i = threadIdx.x; i < top_k; i += 256) row_output[i] = -1;
    return;
  }
  const float* input = logits + row * stride0 + starts[row];
  const int capacity = max(length, top_k);
  if (capacity <= 512)
    D::sort_row<256, 512>(input, 1, length, row_output, top_k, &storage);
  else if (capacity <= 1024)
    D::sort_row<256, 1024>(input, 1, length, row_output, top_k, &storage);
  else if (capacity <= 2048)
    D::sort_row<256, 2048>(input, 1, length, row_output, top_k, &storage);
  else if (capacity <= 4096)
    D::sort_row<256, 4096>(input, 1, length, row_output, top_k, &storage);
  else
    D::sort_row<256, 8192>(input, 1, length, row_output, top_k, &storage);
}

template <int TopK, int VecSize = 1>
void launch_deterministic_persistent(const at::Tensor& logits,
                                     const at::Tensor& row_starts,
                                     const at::Tensor& row_ends,
                                     at::Tensor& indices, int64_t num_rows,
                                     int64_t stride0, int64_t stride1,
                                     at::Tensor& key_output) {
  namespace P = vllm::persistent;
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const uint32_t max_len = static_cast<uint32_t>(logits.size(1));
  const uint32_t kChunk =
      max_len == P::RADIX_THRESHOLD ? 16384 : (num_rows > 8 ? 24576 : 8192);
  const uint32_t ctas_per_group =
      max_len <= P::RADIX_THRESHOLD ? 1 : (max_len + kChunk - 1) / kChunk;
  const size_t smem_size = std::max(
      static_cast<size_t>(P::kSmemMedium),
      P::kFixedSmemLarge + static_cast<size_t>(kChunk) * sizeof(uint32_t));
  int occupancy = 1;
  const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
  cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, P::persistent_topk_kernel<TopK, VecSize, true>,
      P::kThreadsPerBlock, smem_size);
  TORCH_CHECK(
      occ_err == cudaSuccess,
      "persistent Top-K occupancy query failed: ", cudaGetErrorString(occ_err));
  if (occupancy < 1) occupancy = 1;
  const uint32_t resident_ctas =
      static_cast<uint32_t>(props->multiProcessorCount) * occupancy;
  uint32_t num_groups =
      std::min(static_cast<uint32_t>(num_rows), resident_ctas / ctas_per_group);
  if (num_groups == 0) num_groups = 1;
  TORCH_CHECK(num_groups * ctas_per_group <= resident_ctas,
              "persistent Top-K grid does not fit resident capacity");
  auto state = get_persistent_workspace(logits, num_groups);
  prepare_deterministic_top_k<<<num_rows, 256, 0, stream>>>(
      logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
      row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(), stride0,
      TopK,
      reinterpret_cast<P::RadixRowState*>(state.mutable_data_ptr<uint8_t>()),
      num_groups);

  P::PersistentTopKParams params;
  params.input = logits.const_data_ptr<float>();
  params.output = indices.mutable_data_ptr<int>();
  params.output_keys =
      reinterpret_cast<uint64_t*>(key_output.mutable_data_ptr<int64_t>());
  params.lengths = nullptr;
  params.row_starts = row_starts.const_data_ptr<int>();
  params.row_ends = row_ends.const_data_ptr<int>();
  params.row_states =
      reinterpret_cast<P::RadixRowState*>(state.mutable_data_ptr<uint8_t>());
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(stride0);
  params.top_k = static_cast<uint32_t>(TopK);
  params.chunk_size = kChunk;
  params.ctas_per_group = ctas_per_group;
  params.max_seq_len = max_len;

  TORCH_CHECK(stride1 == 1,
              "persistent deterministic Top-K requires contiguous columns");
  auto kernel = &P::persistent_topk_kernel<TopK, VecSize, true>;
  cudaError_t attr_err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  TORCH_CHECK(attr_err == cudaSuccess,
              "persistent Top-K shared memory setup failed: ",
              cudaGetErrorString(attr_err));
  kernel<<<num_groups * ctas_per_group, P::kThreadsPerBlock, smem_size,
           stream>>>(params);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "deterministic persistent Top-K launch failed");
}

template <int TopK, int Threads, int ItemsPerThread>
__global__ __launch_bounds__(Threads) void canonicalize_top_k(
    const uint64_t* input_keys, int* indices, const int* row_starts,
    const int* row_ends) {
  using ScoreSort = cub::BlockRadixSort<uint32_t, Threads, ItemsPerThread, int>;
  __shared__ typename ScoreSort::TempStorage temp;
  __shared__ uint64_t original[TopK];
  __shared__ uint32_t sorted_scores[TopK];
  __shared__ int has_tie;
  const int row = blockIdx.x;
  if (row_ends[row] - row_starts[row] < 8192) return;
  uint32_t scores[ItemsPerThread];
  int values[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int pos = item * Threads + threadIdx.x;
    if (pos < TopK) {
      const uint64_t key = input_keys[static_cast<int64_t>(row) * TopK + pos];
      original[pos] = key;
      scores[item] = static_cast<uint32_t>(key >> 32);
      values[item] = static_cast<int>(key);
    } else {
      scores[item] = 0;
      values[item] = -1;
    }
  }
  ScoreSort(temp).SortDescendingBlockedToStriped(scores, values);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int pos = item * Threads + threadIdx.x;
    if (pos < TopK) {
      sorted_scores[pos] = scores[item];
    }
  }
  if (threadIdx.x == 0) has_tie = 0;
  __syncthreads();
  for (int pos = threadIdx.x; pos + 1 < TopK; pos += Threads) {
    if (sorted_scores[pos] == sorted_scores[pos + 1] && sorted_scores[pos] != 0)
      atomicExch(&has_tie, 1);
  }
  __syncthreads();
  if (has_tie == 0) {
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
      const int pos = item * Threads + threadIdx.x;
      if (pos < TopK)
        indices[static_cast<int64_t>(row) * TopK + pos] =
            sorted_scores[pos] == 0 ? -1 : values[item];
    }
    return;
  }

  // Rare tie path: place each selected key by (score descending, index
  // descending). This is O(K^2) only for rows that actually contain a tie.
  for (int pos = threadIdx.x; pos < TopK; pos += Threads) {
    const uint64_t key = original[pos];
    if (key == 0) continue;
    const uint32_t score = static_cast<uint32_t>(key >> 32);
    const uint32_t index = static_cast<uint32_t>(key);
    int rank = 0;
    for (int j = 0; j < TopK; ++j) {
      const uint64_t other = original[j];
      if (other == 0) continue;
      const uint32_t other_score = static_cast<uint32_t>(other >> 32);
      const uint32_t other_index = static_cast<uint32_t>(other);
      rank += (other_score > score) ||
              (other_score == score && other_index > index);
    }
    indices[static_cast<int64_t>(row) * TopK + rank] = static_cast<int>(index);
  }
}

template <int TopK, int Threads, int ItemsPerThread>
__global__ __launch_bounds__(Threads) void canonicalize_top_k_64(
    const uint64_t* input_keys, int* indices, const int* row_starts,
    const int* row_ends) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, Threads, ItemsPerThread>;
  __shared__ typename ScoreSort::TempStorage temp;
  const int row = blockIdx.x;
  if (row_ends[row] - row_starts[row] < 8192) return;
  uint64_t keys[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int pos = item * Threads + threadIdx.x;
    keys[item] =
        pos < TopK ? input_keys[static_cast<int64_t>(row) * TopK + pos] : 0;
  }
  ScoreSort(temp).SortDescendingBlockedToStriped(keys);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int pos = item * Threads + threadIdx.x;
    if (pos < TopK)
      indices[static_cast<int64_t>(row) * TopK + pos] =
          keys[item] == 0 ? -1 : static_cast<int>(keys[item]);
  }
}

template <int TopK, int Threads, int ItemsPerThread>
__global__ __launch_bounds__(Threads) void canonicalize_top_k_logits(
    const float* logits, const int* row_starts, const int* row_ends,
    int* indices, int64_t stride0) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, Threads, ItemsPerThread>;
  __shared__ typename ScoreSort::TempStorage temp;
  const int row = blockIdx.x;
  if (row_ends[row] - row_starts[row] < 8192) return;
  uint64_t keys[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int pos = item * Threads + threadIdx.x;
    if (pos < TopK) {
      const int index = indices[static_cast<int64_t>(row) * TopK + pos];
      if (index < 0) {
        keys[item] = 0;
      } else {
        const float score = logits[static_cast<int64_t>(row) * stride0 +
                                   row_starts[row] + index];
        keys[item] = (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
                     static_cast<uint32_t>(index);
      }
    } else {
      keys[item] = 0;
    }
  }
  ScoreSort(temp).SortDescendingBlockedToStriped(keys);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int pos = item * Threads + threadIdx.x;
    if (pos < TopK)
      indices[static_cast<int64_t>(row) * TopK + pos] =
          keys[item] == 0 ? -1 : static_cast<int>(keys[item]);
  }
}

template <int TopK>
__global__ void build_top_k_keys(const float* logits, const int* row_starts,
                                 const int* row_ends, const int* indices,
                                 uint64_t* keys, int64_t stride0) {
  const int row = blockIdx.x;
  if (row_ends[row] - row_starts[row] < 8192) return;
  for (int pos = threadIdx.x; pos < TopK; pos += blockDim.x) {
    const int index = indices[static_cast<int64_t>(row) * TopK + pos];
    if (index < 0) {
      keys[static_cast<int64_t>(row) * TopK + pos] = 0;
      continue;
    }
    const float score =
        logits[static_cast<int64_t>(row) * stride0 + row_starts[row] + index];
    keys[static_cast<int64_t>(row) * TopK + pos] =
        (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
        static_cast<uint32_t>(index);
  }
}

__global__ __launch_bounds__(256) void deterministic_top_k_small(
    const float* logits, const int* starts, const int* ends, int* indices,
    int64_t stride0, int64_t stride1, int top_k) {
  namespace D = vllm::deterministic_topk;
  __shared__ union {
    D::StableSort<256, 512>::TempStorage s512;
    D::StableSort<256, 1024>::TempStorage s1024;
    D::StableSort<256, 2048>::TempStorage s2048;
    D::StableSort<256, 4096>::TempStorage s4096;
  } temp;
  const int row = blockIdx.x;
  const int length = max(0, ends[row] - starts[row]);
  const int capacity = max(length, top_k);
  const float* input = logits + row * stride0 + starts[row] * stride1;
  int* output = indices + static_cast<int64_t>(row) * top_k;
  if (capacity <= 512)
    D::sort_row<256, 512>(input, stride1, length, output, top_k, &temp.s512);
  else if (capacity <= 1024)
    D::sort_row<256, 1024>(input, stride1, length, output, top_k, &temp.s1024);
  else if (capacity <= 2048)
    D::sort_row<256, 2048>(input, stride1, length, output, top_k, &temp.s2048);
  else
    D::sort_row<256, 4096>(input, stride1, length, output, top_k, &temp.s4096);
}

__global__ __launch_bounds__(kThreads) void deterministic_top_k_generic(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  constexpr int kItemsPerThread = kMaxTopK / kThreads;
  constexpr int kRadixBits = 8;
  constexpr int kRadixBuckets = 256;
  using ScoreSort = cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  union SharedStorage {
    uint32_t histogram[kRadixBuckets];
    uint64_t selected[kMaxTopK];
    typename ScoreSort::TempStorage sort;
    vllm::deterministic_topk::StableSort<512, 8192>::TempStorage fast;
  };
  __shared__ SharedStorage shared;
  __shared__ uint64_t selected_prefix;
  __shared__ int rank;
  __shared__ int selected_count;

  const int row = blockIdx.x;
  const int row_start = row_starts[row];
  const int row_end = row_ends[row];
  const int row_length = max(0, row_end - row_start);
  if (max(row_length, top_k) <= 8192) {
    namespace D = vllm::deterministic_topk;
    const float* input = logits + row * stride0 + row_start * stride1;
    int* dest = output + static_cast<int64_t>(row) * top_k;
    const int capacity = max(row_length, top_k);
    if (capacity <= 1024)
      D::sort_row<512, 1024>(input, stride1, row_length, dest, top_k,
                             &shared.fast);
    else if (capacity <= 2048)
      D::sort_row<512, 2048>(input, stride1, row_length, dest, top_k,
                             &shared.fast);
    else if (capacity <= 4096)
      D::sort_row<512, 4096>(input, stride1, row_length, dest, top_k,
                             &shared.fast);
    else
      D::sort_row<512, 8192>(input, stride1, row_length, dest, top_k,
                             &shared.fast);
    return;
  }

  const int effective_top_k = min(top_k, row_length);
  uint64_t score_keys[kItemsPerThread];

  if (effective_top_k == 0) {
    for (int output_column = threadIdx.x; output_column < top_k;
         output_column += kThreads) {
      output[static_cast<int64_t>(row) * top_k + output_column] = -1;
    }
    return;
  }

  if (threadIdx.x == 0) {
    selected_prefix = 0;
    // One-indexed rank of the key that bounds the final Top-K set.
    rank = effective_top_k;
  }
  __syncthreads();

  // Select the exact Kth-largest 64-bit (score, local-index) key eight radix
  // bits at a time.  Unlike the old 4096-element register sort, each pass can
  // scan a row of any length while retaining the same deterministic tie break.
#pragma unroll
  for (int shift = 64 - kRadixBits; shift >= 0; shift -= kRadixBits) {
    if (threadIdx.x < kRadixBuckets) {
      shared.histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int local_index = threadIdx.x; local_index < row_length;
         local_index += kThreads) {
      const int absolute_index = row_start + local_index;
      const float score =
          logits[static_cast<int64_t>(row) * stride0 +
                 static_cast<int64_t>(absolute_index) * stride1];
      // Descending score, then descending request-local source index,
      // matching vLLM's insertion-sort tie and output semantics. The key is
      // unique, so neither candidate selection nor output depends on warp
      // scheduling or the request's offset in a packed buffer.
      const uint64_t key =
          (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
          static_cast<uint32_t>(local_index);
      const uint64_t upper_prefix =
          shift == 64 - kRadixBits ? 0 : key >> (shift + kRadixBits);
      if (upper_prefix == selected_prefix) {
        atomicAdd(&shared.histogram[(key >> shift) & (kRadixBuckets - 1)], 1U);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int remaining_rank = rank;
      int selected_bucket = 0;
      for (int bucket = kRadixBuckets - 1; bucket >= 0; --bucket) {
        const int bucket_count = static_cast<int>(shared.histogram[bucket]);
        if (remaining_rank <= bucket_count) {
          selected_bucket = bucket;
          break;
        }
        remaining_rank -= bucket_count;
      }
      selected_prefix = (selected_prefix << kRadixBits) | selected_bucket;
      rank = remaining_rank;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    selected_count = 0;
  }
  __syncthreads();

  // Composite keys are unique because their low bits contain the row-local
  // index, so exactly effective_top_k keys are >= the selected threshold.
  for (int local_index = threadIdx.x; local_index < row_length;
       local_index += kThreads) {
    const int absolute_index = row_start + local_index;
    const float score = logits[static_cast<int64_t>(row) * stride0 +
                               static_cast<int64_t>(absolute_index) * stride1];
    const uint64_t key =
        (static_cast<uint64_t>(ordered_float_bits(score)) << 32) |
        static_cast<uint32_t>(local_index);
    if (key >= selected_prefix) {
      const int slot = atomicAdd(&selected_count, 1);
      if (slot < effective_top_k) {
        shared.selected[slot] = key;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int slot = item * kThreads + threadIdx.x;
    score_keys[item] =
        slot < effective_top_k ? shared.selected[slot] : uint64_t{0};
  }
  // All reads from the selected-key view must finish before CUB reuses the
  // same union storage for its sort scratch space.
  __syncthreads();
  ScoreSort(shared.sort).SortDescendingBlockedToStriped(score_keys);
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int output_column = item * kThreads + threadIdx.x;
    if (output_column < top_k) {
      output[static_cast<int64_t>(row) * top_k + output_column] =
          output_column < effective_top_k
              ? static_cast<int>(static_cast<uint32_t>(score_keys[item]))
              : -1;
    }
  }
}

void ds4_top_k_per_row_prefill(const at::Tensor& logits,
                               const at::Tensor& row_starts,
                               const at::Tensor& row_ends, at::Tensor& indices,
                               int64_t num_rows, int64_t stride0,
                               int64_t stride1, int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits must be float32");
  TORCH_CHECK(row_starts.scalar_type() == at::kInt, "row_starts must be int32");
  TORCH_CHECK(row_ends.scalar_type() == at::kInt, "row_ends must be int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be rank 2");
  TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK,
              "DS4 deterministic Top-K requires 0 < top_k <= ", kMaxTopK,
              ", got ", top_k);
  TORCH_CHECK(indices.size(0) >= num_rows && indices.size(1) >= top_k,
              "indices output is too small");
  if (num_rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (logits.size(1) <= 4096) {
    deterministic_top_k_small<<<num_rows, 256, 0, stream>>>(
        logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
        row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
        stride0, stride1, static_cast<int>(top_k));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "deterministic small Top-K launch failed");
    return;
  }
  if (stride1 == 1 && (top_k == 512 || top_k == 1024 || top_k == 2048) &&
      logits.size(1) > vllm::persistent::RADIX_THRESHOLD) {
    if (top_k == 512) {
      auto keys = get_key_workspace(logits, num_rows, 512);
      launch_deterministic_persistent<512, 4>(logits, row_starts, row_ends,
                                              indices, num_rows, stride0,
                                              stride1, keys);
      canonicalize_top_k_logits<512, 128, 4><<<num_rows, 128, 0, stream>>>(
          logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
          row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
          stride0);
    } else if (top_k == 1024) {
      auto keys = get_key_workspace(logits, num_rows, 1024);
      launch_deterministic_persistent<1024, 4>(logits, row_starts, row_ends,
                                               indices, num_rows, stride0,
                                               stride1, keys);
      canonicalize_top_k_logits<1024, 256, 4><<<num_rows, 256, 0, stream>>>(
          logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
          row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
          stride0);
    } else {
      auto keys = get_key_workspace(logits, num_rows, 2048);
      launch_deterministic_persistent<2048, 4>(logits, row_starts, row_ends,
                                               indices, num_rows, stride0,
                                               stride1, keys);
      canonicalize_top_k_logits<2048, 512, 4><<<num_rows, 512, 0, stream>>>(
          logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
          row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
          stride0);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "deterministic persistent canonicalization launch failed");
    return;
  }
  // The persistent selector chooses an exact set. Canonicalization below
  // orders that set; it cannot repair omitted candidates.
  if (stride1 == 1 && (top_k == 512 || top_k == 1024 || top_k == 2048) &&
      logits.size(1) >= 8192) {
    const bool persistent_keys =
        logits.size(1) > vllm::persistent::RADIX_THRESHOLD;
    if (top_k == 512) {
      auto keys = get_key_workspace(logits, num_rows, 512);
      launch_deterministic_persistent<512, 4>(logits, row_starts, row_ends,
                                              indices, num_rows, stride0,
                                              stride1, keys);
      if (!persistent_keys)
        build_top_k_keys<512><<<num_rows, 128, 0, stream>>>(
            logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
            row_ends.const_data_ptr<int>(), indices.const_data_ptr<int>(),
            reinterpret_cast<uint64_t*>(keys.mutable_data_ptr<int64_t>()),
            stride0);
      if (num_rows <= 8) {
        canonicalize_top_k<512, 128, 4><<<num_rows, 128, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
            indices.mutable_data_ptr<int>(), row_starts.const_data_ptr<int>(),
            row_ends.const_data_ptr<int>());
      } else {
        canonicalize_top_k_64<512, 512, 1><<<num_rows, 512, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
            indices.mutable_data_ptr<int>(), row_starts.const_data_ptr<int>(),
            row_ends.const_data_ptr<int>());
      }
    } else if (top_k == 1024) {
      auto keys = get_key_workspace(logits, num_rows, 1024);
      launch_deterministic_persistent<1024, 4>(logits, row_starts, row_ends,
                                               indices, num_rows, stride0,
                                               stride1, keys);
      if (!persistent_keys)
        build_top_k_keys<1024><<<num_rows, 256, 0, stream>>>(
            logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
            row_ends.const_data_ptr<int>(), indices.const_data_ptr<int>(),
            reinterpret_cast<uint64_t*>(keys.mutable_data_ptr<int64_t>()),
            stride0);
      canonicalize_top_k_64<1024, 256, 4><<<num_rows, 256, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
          indices.mutable_data_ptr<int>(), row_starts.const_data_ptr<int>(),
          row_ends.const_data_ptr<int>());
    } else {
      auto keys = get_key_workspace(logits, num_rows, 2048);
      launch_deterministic_persistent<2048, 4>(logits, row_starts, row_ends,
                                               indices, num_rows, stride0,
                                               stride1, keys);
      if (!persistent_keys)
        build_top_k_keys<2048><<<num_rows, 512, 0, stream>>>(
            logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
            row_ends.const_data_ptr<int>(), indices.const_data_ptr<int>(),
            reinterpret_cast<uint64_t*>(keys.mutable_data_ptr<int64_t>()),
            stride0);
      canonicalize_top_k_64<2048, 512, 4><<<num_rows, 512, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
          indices.mutable_data_ptr<int>(), row_starts.const_data_ptr<int>(),
          row_ends.const_data_ptr<int>());
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "deterministic Top-K canonicalization launch failed");
    return;
  }
  const int num_chunks = (logits.size(1) + kChunkColumns - 1) / kChunkColumns;
  if (num_rows > 8 && top_k != 512 && top_k != 1024 && top_k != 2048) {
    deterministic_top_k_generic<<<num_rows, kThreads, 0, stream>>>(
        logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
        row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
        stride0, stride1, static_cast<int>(top_k));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "deterministic generic Top-K launch failed");
    return;
  }
  if (num_chunks == 1) {
    if (logits.size(1) <= 1024) {
      launch_single_small<64, 16>(logits, row_starts, row_ends, indices,
                                  num_rows, stride0, stride1,
                                  static_cast<int>(top_k), stream);
    } else if (logits.size(1) <= 2048) {
      launch_single_small<128, 16>(logits, row_starts, row_ends, indices,
                                   num_rows, stride0, stride1,
                                   static_cast<int>(top_k), stream);
    } else if (logits.size(1) <= 4096) {
      launch_single_small<256, 16>(logits, row_starts, row_ends, indices,
                                   num_rows, stride0, stride1,
                                   static_cast<int>(top_k), stream);
    } else {
      launch_single_small<kThreads, kItemsPerThread>(
          logits, row_starts, row_ends, indices, num_rows, stride0, stride1,
          static_cast<int>(top_k), stream);
    }
  } else {
    auto keys = get_workspace(logits, num_rows, num_chunks, top_k);
    using ChunkScoreSort =
        cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
    constexpr size_t chunk_smem_size =
        sizeof(typename ChunkScoreSort::TempStorage);
    cudaError_t attr_err = cudaFuncSetAttribute(
        deterministic_top_k_chunk, cudaFuncAttributeMaxDynamicSharedMemorySize,
        chunk_smem_size);
    TORCH_CHECK(attr_err == cudaSuccess,
                "chunk Top-K shared memory setup failed: ",
                cudaGetErrorString(attr_err));
    deterministic_top_k_chunk<<<dim3(num_rows, num_chunks), kThreads,
                                chunk_smem_size, stream>>>(
        logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
        row_ends.const_data_ptr<int>(),
        reinterpret_cast<uint64_t*>(keys.mutable_data_ptr<int64_t>()), stride0,
        stride1, static_cast<int>(top_k));
    if (top_k == 512) {
      merge_top_k_chunks<128, 8><<<num_rows, 128, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
          indices.mutable_data_ptr<int>(), num_chunks, static_cast<int>(top_k));
    } else if (top_k == 1024) {
      merge_top_k_chunks<256, 8><<<num_rows, 256, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
          indices.mutable_data_ptr<int>(), num_chunks, static_cast<int>(top_k));
    } else {
      merge_top_k_chunks<512, 8><<<num_rows, 512, 0, stream>>>(
          reinterpret_cast<const uint64_t*>(keys.const_data_ptr<int64_t>()),
          indices.mutable_data_ptr<int>(), num_chunks, static_cast<int>(top_k));
    }
  }
}

}  // namespace

TORCH_LIBRARY(ds4_bi, m) {
  m.def(
      "top_k_per_row_prefill(Tensor logits, Tensor row_starts, "
      "Tensor row_ends, Tensor(a!) indices, int num_rows, int stride0, "
      "int stride1, int top_k) -> ()");
}

TORCH_LIBRARY_IMPL(ds4_bi, CUDA, m) {
  m.impl("top_k_per_row_prefill", &ds4_top_k_per_row_prefill);
}
