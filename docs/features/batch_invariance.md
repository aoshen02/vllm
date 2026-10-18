# Batch Invariance

!!! note
    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm/issues/27433>

This document shows how to enable batch invariance in vLLM. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training. For bitwise train-inference consistency with an NCCL weight-sync group, also set up the trainer; see [RL: Bitwise Train-Inference Consistency](#rl-bitwise-train-inference-consistency-trainer-side).
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance is supported on the following platforms:

- NVIDIA GPUs with compute capability 8.0 or higher. On SM8x (Ampere, Ada), vLLM replaces `mm`/`addmm`/`matmul`/`linear` with Triton persistent kernels because cuBLASLt is not batch invariant there. On SM90 and newer (Hopper, Blackwell), it keeps cuBLASLt and disables split-K through `CUBLAS_WORKSPACE_CONFIG`.
- AMD Instinct GPUs (gfx90a, gfx942, gfx950). These use the same Triton matmul overrides as SM8x.
- Intel XPUs with Triton support.

### Attention Backend Selection for XPU

On XPU, Triton Attention backend is required for batch invariance. If no backend is set, vLLM selects `TRITON_ATTN` automatically. An explicit `FLASH_ATTN` is honored with a warning, but it is not validated for batch invariance.
Select this backend using Qwen/Qwen3-1.7B model as an example:

```python
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    attention_config={"backend": "TRITON_ATTN"},
)
```

Or via the CLI:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-1.7B \
    --attention-config.backend TRITON_ATTN
```

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## RL: Bitwise Train-Inference Consistency (Trainer Side)

For RL rollouts that must be bitwise reproducible (train-inference consistency, on-policy RL), set `VLLM_BATCH_INVARIANT=1` for vLLM **and** for every trainer process that joins an NCCL weight-sync group, and call `init_batch_invariance()` in the trainer before its first NCCL call:

```python
# Trainer process. VLLM_BATCH_INVARIANT=1 must be in this process's environment;
# without it, init_batch_invariance() does nothing.
from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

init_batch_invariance()  # before creating any NCCL communicator / weight-transfer engine

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
# ... then WeightTransferTrainerFactory.trainer_init(...) as usual
```

vLLM workers call `init_batch_invariance()` themselves at startup (`init_worker_distributed_environment` in `vllm/v1/worker/gpu_worker.py`). With Ray, put the variable in `runtime_env={"env_vars": {"VLLM_BATCH_INVARIANT": "1"}}` so trainer actors and vLLM workers both see it. See [`rlhf_async_new_apis.py`](../../examples/rl/rlhf_async_new_apis.py) for a complete trainer + `AsyncLLMEngine` example.

`init_batch_invariance()` sets these environment variables in the calling process (`override_envs_for_invariance()` in `vllm/model_executor/determinism/batch_invariant.py`):

| Variables | Value | Applies to |
| --- | --- | --- |
| `NCCL_LAUNCH_MODE` | `GROUP` | Every communicator in the process, including the weight-sync communicator shared with the trainer. |
| `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`, `NCCL_NTHREADS`, `NCCL_SOCKET_NTHREADS`, `NCCL_COLLNET_ENABLE`, `NCCL_NVLS_ENABLE`, `NCCL_P2P_NET_DISABLE` | `ring,tree;allreduce:tree`, `Simple`, `1`, `1`, `1`, `1`, `0`, `0`, `1` | The process's own communicators only, if NCCL >= 2.29.7 (vLLM lists them in `NCCL_NO_CACHE` and creates weight-sync communicators without them). With older NCCL they also apply to the weight-sync communicator. |
| `VLLM_CUSTOM_ALLREDUCE_ALGO`, `VLLM_ALLREDUCE_USE_SYMM_MEM` | `1stage`, `0` | vLLM tensor-parallel all-reduce. |
| `CUBLAS_WORKSPACE_CONFIG` (+ `CUBLASLT_WORKSPACE_SIZE` on SM90+) | `:4096:8`, then `:16:8` and `1` on SM90+ | cuBLAS / cuBLASLt (disables split-K). |
| `VLLM_USE_AOT_COMPILE` | `0` | vLLM `torch.compile`. |
| `VLLM_ROCM_USE_SKINNY_GEMM` (ROCm only, instead of the `NCCL_*` variables) | `0` | ROCm skinny GEMMs. |

It also changes the trainer's own math: it installs batch-invariant `aten` overrides (`softmax`, `_log_softmax`, `bmm`, `mean.dim`; plus `mm`, `addmm`, `matmul` and `linear` on SM8x, ROCm and XPU), disables TF32 and reduced-precision fp16/bf16 reductions, and sets `torch._inductor.config.batch_invariant = True`. Expect the trainer to run slower.

Failure behavior: on NCCL < 2.29.7, vLLM logs `NCCL_NO_CACHE needs CUDA NCCL >= 2.29.7, ...` when it creates the weight-sync communicator. A trainer that has not set the same `NCCL_*` variables (for example, because it skipped `init_batch_invariance()`) hangs on that communicator.

Scope: batch invariance makes vLLM's outputs independent of batching. It does not make vLLM logprobs equal to logprobs from a different model implementation (for example, Hugging Face Transformers). A bitwise match with the trainer also needs the trainer's forward pass to use the same kernels, dtype and reduction order as vLLM. The `rlhf_async_new_apis.py` example only checks the weight-synced engine against a fresh vLLM instance loaded with the trained weights. To check against your trainer, recompute logprobs of the sampled token ids in the trainer and compare them to vLLM's returned `logprobs` with `torch.equal`. These are `raw_logprobs` by default (before temperature and top-k/top-p); set `logprobs_mode="processed_logprobs"` if the trainer applies temperature.

## Tested Models

Batch invariance has been tested and verified on the following models:

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ`
- **Qwen3-VL (Vision-Language)**: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` (single image and video inputs)
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Llama 3**: Llama3.1 and 3.2 series, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` for example
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`
- **GLM-4 series**: `zai-org/glm-4-9b-chat`
- **Phi series**: `microsoft/Phi-3.5-mini-instruct`, `microsoft/phi-4`
- **Granite 3.1 (MoE)**: `ibm-granite/granite-3.1-1b-a400m-instruct`, `ibm-granite/granite-3.1-3b-a800m-instruct`
- **Granite 3.1 (Dense)**: `ibm-granite/granite-3.1-2b-instruct`, `ibm-granite/granite-3.1-8b-instruct`
- **EXAONE 4.0 series**: `LGAI-EXAONE/EXAONE-4.0-1.2B`, `LGAI-EXAONE/EXAONE-4.0.1-32B`, `LGAI-EXAONE/EXAONE-4.0-32B`
- **OLMo 2**: `allenai/OLMo-2-0425-1B-Instruct`
- **ERNIE 4.5**: `baidu/ERNIE-4.5-0.3B-PT`
- **SmolLM2**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **PLaMo3**: `pfnet/plamo-3-nict-2b-base`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm/issues/new/choose).

## Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism
4. Under tensor parallelism, keeps custom all-reduce on with a fixed reduction order (the 1-stage kernel is pinned, and large inputs are reduced in fixed-size chunks), and disables FlashInfer, AITER and QuickReduce all-reduce
5. On CUDA devices with tuned matmul table entries for the model's bf16 unquantized forward linear layers (Ada, Hopper, Blackwell),
   runs without `torch.compile` using breakable CUDA graphs so tile configs follow the runtime batch size; set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to opt out.
6. Splits the LoRA shrink reduction into 8 fixed K chunks that are summed in a fixed order instead of with atomics

!!! warning
    Batch invariance under tensor parallelism is not yet supported for all-reduces whose size isn't a multiple of 16 bytes, for example a hidden size that isn't a multiple of 8 in fp16/bf16. Such tensors can switch between NCCL and custom all-reduce depending on batch size. All validated models meet this requirement.

!!! note
    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional GPU architectures
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm/issues/27433).
