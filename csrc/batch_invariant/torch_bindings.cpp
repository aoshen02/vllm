#include <torch/extension.h>

#include <optional>

namespace vllm::batch_invariant {

void fused_silu_mul_per_token_group_quant(
    torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s,
    int64_t group_size, double eps, double min_8bit, double max_8bit,
    double clamp_limit, bool round_scale, bool scale_ue8m0,
    bool fuse_silu_and_mul, const std::optional<torch::Tensor>& masked_m);

torch::Tensor bi_gemm(torch::Tensor a, torch::Tensor b,
                      const std::optional<torch::Tensor>& bias);

int64_t bi_gemm_split_k(int64_t K, int64_t N);

}  // namespace vllm::batch_invariant

TORCH_LIBRARY(vllm_batch_invariant, ops) {
  ops.def(
      "fused_silu_mul_per_token_group_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float min_8bit, "
      "float max_8bit, float clamp_limit, bool round_scale, bool scale_ue8m0, "
      "bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
  ops.def("bi_gemm(Tensor a, Tensor b, Tensor? bias) -> Tensor");
  ops.def("bi_gemm_split_k(int K, int N) -> int",
          &vllm::batch_invariant::bi_gemm_split_k);
}

TORCH_LIBRARY_IMPL(vllm_batch_invariant, CUDA, ops) {
  ops.impl("fused_silu_mul_per_token_group_quant",
           &vllm::batch_invariant::fused_silu_mul_per_token_group_quant);
  ops.impl("bi_gemm", &vllm::batch_invariant::bi_gemm);
}
