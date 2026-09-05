// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>
#include <torch/library.h>

namespace {

constexpr int kThreads = 512;
constexpr int kMaxTopK = 2048;
constexpr int kItemsPerThread = kMaxTopK / kThreads;
constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__global__ __launch_bounds__(kThreads) void deterministic_top_k_per_row_prefill(
    const float* logits, const int* row_starts, const int* row_ends,
    int* output, int64_t stride0, int64_t stride1, int top_k) {
  using ScoreSort = cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;
  union SharedStorage {
    uint32_t histogram[kRadixBuckets];
    uint64_t selected[kMaxTopK];
    typename ScoreSort::TempStorage sort;
  };
  __shared__ SharedStorage shared;
  __shared__ uint64_t selected_prefix;
  __shared__ int rank;
  __shared__ int selected_count;

  const int row = blockIdx.x;
  const int row_start = row_starts[row];
  const int row_end = row_ends[row];
  const int row_length = max(0, row_end - row_start);
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
  deterministic_top_k_per_row_prefill<<<num_rows, kThreads, 0, stream>>>(
      logits.const_data_ptr<float>(), row_starts.const_data_ptr<int>(),
      row_ends.const_data_ptr<int>(), indices.mutable_data_ptr<int>(), stride0,
      stride1, static_cast<int>(top_k));
}

constexpr int kCombineThreads = 256;
constexpr int kCombineItemsPerThread = 2;

__global__
__launch_bounds__(kCombineThreads) void combine_topk_swa_decode_kernel(
    int* combined_indices, int* combined_lens, const int* topk_indices,
    const int* seq_lens, const bool* is_valid, int64_t output_stride,
    int64_t topk_stride, int output_width, int M, int N, int top_k,
    int compress_ratio, int window_size) {
  using Sort =
      cub::BlockRadixSort<int, kCombineThreads, kCombineItemsPerThread>;
  __shared__ typename Sort::TempStorage sort_storage;
  const int row = blockIdx.x;
  const int seq_len = seq_lens[row];
  const int topk_len = min(seq_len / compress_ratio, top_k);
  const int swa_len = min(seq_len, window_size);
  const int row_base = M * row;
  int sorted_indices[kCombineItemsPerThread];

#pragma unroll
  for (int item = 0; item < kCombineItemsPerThread; ++item) {
    const int column = threadIdx.x * kCombineItemsPerThread + item;
    sorted_indices[item] =
        column < topk_len
            ? topk_indices[static_cast<int64_t>(row) * topk_stride + column]
            : -1;
  }
  Sort(sort_storage).SortDescendingBlockedToStriped(sorted_indices);

  for (int column = threadIdx.x; column < output_width; column += blockDim.x) {
    int value = -1;
    if (column < topk_len) {
      const int item = column / kCombineThreads;
      value = sorted_indices[item] + row_base;
    } else if (column < topk_len + swa_len) {
      value = row_base + N + column - topk_len;
    }
    combined_indices[static_cast<int64_t>(row) * output_stride + column] =
        value;
  }
  if (threadIdx.x == 0) {
    combined_lens[row] = is_valid[row] ? topk_len + swa_len : 0;
  }
}

void ds4_combine_topk_swa_decode(at::Tensor& combined_indices,
                                 at::Tensor& combined_lens,
                                 const at::Tensor& topk_indices,
                                 const at::Tensor& seq_lens,
                                 const at::Tensor& is_valid, int64_t M,
                                 int64_t N, int64_t top_k,
                                 int64_t compress_ratio, int64_t window_size) {
  TORCH_CHECK(combined_indices.is_cuda() && combined_lens.is_cuda() &&
                  topk_indices.is_cuda() && seq_lens.is_cuda() &&
                  is_valid.is_cuda(),
              "decode tensors must be CUDA");
  TORCH_CHECK(combined_indices.scalar_type() == at::kInt &&
                  combined_lens.scalar_type() == at::kInt &&
                  topk_indices.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt,
              "decode index tensors must be int32");
  TORCH_CHECK(is_valid.scalar_type() == at::kBool, "is_valid must be bool");
  TORCH_CHECK(combined_indices.dim() == 2 && topk_indices.dim() == 2,
              "index tensors must be rank 2");
  const int64_t num_rows = seq_lens.numel();
  TORCH_CHECK(combined_indices.size(0) == num_rows &&
                  combined_lens.numel() == num_rows &&
                  topk_indices.size(0) == num_rows &&
                  is_valid.numel() == num_rows,
              "decode tensors must have the same row count");
  TORCH_CHECK(combined_indices.stride(1) == 1 && topk_indices.stride(1) == 1,
              "index rows must be contiguous");
  TORCH_CHECK(top_k >= 0 && top_k <= topk_indices.size(1),
              "top_k exceeds the input width");
  TORCH_CHECK(top_k <= kCombineThreads * kCombineItemsPerThread,
              "fused decode combine supports top_k <= 512");
  TORCH_CHECK(compress_ratio > 0 && window_size >= 0,
              "invalid sparse attention dimensions");
  if (num_rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(combined_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  combine_topk_swa_decode_kernel<<<num_rows, kCombineThreads, 0, stream>>>(
      combined_indices.mutable_data_ptr<int>(),
      combined_lens.mutable_data_ptr<int>(), topk_indices.const_data_ptr<int>(),
      seq_lens.const_data_ptr<int>(), is_valid.const_data_ptr<bool>(),
      combined_indices.stride(0), topk_indices.stride(0),
      combined_indices.size(1), static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(top_k), static_cast<int>(compress_ratio),
      static_cast<int>(window_size));
}

__global__
__launch_bounds__(kCombineThreads) void combine_c128_swa_decode_kernel(
    int* combined_indices, int* combined_lens, const int* seq_lens,
    const bool* is_valid, int64_t output_stride, int output_width, int M, int N,
    int top_k, int compress_ratio, int window_size) {
  const int row = blockIdx.x;
  const int seq_len = seq_lens[row];
  const int topk_len = min(seq_len / compress_ratio, top_k);
  const int swa_len = min(seq_len, window_size);
  const int row_base = M * row;
  for (int column = threadIdx.x; column < output_width; column += blockDim.x) {
    int value = -1;
    if (column < topk_len) {
      value = row_base + topk_len - 1 - column;
    } else if (column < topk_len + swa_len) {
      value = row_base + N + column - topk_len;
    }
    combined_indices[static_cast<int64_t>(row) * output_stride + column] =
        value;
  }
  if (threadIdx.x == 0) {
    combined_lens[row] = is_valid[row] ? topk_len + swa_len : 0;
  }
}

void ds4_combine_c128_swa_decode(at::Tensor& combined_indices,
                                 at::Tensor& combined_lens,
                                 const at::Tensor& seq_lens,
                                 const at::Tensor& is_valid, int64_t M,
                                 int64_t N, int64_t top_k,
                                 int64_t compress_ratio, int64_t window_size) {
  TORCH_CHECK(combined_indices.is_cuda() && combined_lens.is_cuda() &&
                  seq_lens.is_cuda() && is_valid.is_cuda(),
              "decode tensors must be CUDA");
  TORCH_CHECK(combined_indices.scalar_type() == at::kInt &&
                  combined_lens.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt,
              "decode index tensors must be int32");
  TORCH_CHECK(is_valid.scalar_type() == at::kBool, "is_valid must be bool");
  const int64_t num_rows = seq_lens.numel();
  TORCH_CHECK(
      combined_indices.dim() == 2 && combined_indices.size(0) == num_rows &&
          combined_lens.numel() == num_rows && is_valid.numel() == num_rows,
      "decode tensors must have the same row count");
  TORCH_CHECK(combined_indices.stride(1) == 1, "index rows must be contiguous");
  TORCH_CHECK(top_k >= 0 && top_k <= combined_indices.size(1),
              "top_k exceeds the output width");
  TORCH_CHECK(compress_ratio > 0 && window_size >= 0,
              "invalid sparse attention dimensions");
  if (num_rows == 0) {
    return;
  }

  const c10::cuda::CUDAGuard device_guard(combined_indices.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  combine_c128_swa_decode_kernel<<<num_rows, kCombineThreads, 0, stream>>>(
      combined_indices.mutable_data_ptr<int>(),
      combined_lens.mutable_data_ptr<int>(), seq_lens.const_data_ptr<int>(),
      is_valid.const_data_ptr<bool>(), combined_indices.stride(0),
      combined_indices.size(1), static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(top_k), static_cast<int>(compress_ratio),
      static_cast<int>(window_size));
}

}  // namespace

TORCH_LIBRARY(ds4_bi, m) {
  m.def(
      "top_k_per_row_prefill(Tensor logits, Tensor row_starts, "
      "Tensor row_ends, Tensor(a!) indices, int num_rows, int stride0, "
      "int stride1, int top_k) -> ()");
  m.def(
      "combine_topk_swa_decode(Tensor(a!) combined_indices, "
      "Tensor(b!) combined_lens, Tensor topk_indices, Tensor seq_lens, "
      "Tensor is_valid, int M, int N, int top_k, int compress_ratio, "
      "int window_size) -> ()");
  m.def(
      "combine_c128_swa_decode(Tensor(a!) combined_indices, "
      "Tensor(b!) combined_lens, Tensor seq_lens, Tensor is_valid, int M, "
      "int N, int top_k, int compress_ratio, int window_size) -> ()");
}

TORCH_LIBRARY_IMPL(ds4_bi, CUDA, m) {
  m.impl("top_k_per_row_prefill", &ds4_top_k_per_row_prefill);
  m.impl("combine_topk_swa_decode", &ds4_combine_topk_swa_decode);
  m.impl("combine_c128_swa_decode", &ds4_combine_c128_swa_decode);
}
