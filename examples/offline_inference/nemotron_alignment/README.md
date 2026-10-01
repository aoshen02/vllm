# Nemotron Lightning NVFP4 alignment

This change integrates the selected inference implementation into normal vLLM
model construction. It requires runner V2, TP1, EP4, BF16 activations, FP8 E4M3
KV cache, FP32 Mamba state, and matching FlashInfer/Humming dependencies on GB200.
There is no custom worker, monkey patch, external script path, or provenance
environment required to select the implementation.

## Implementation

- Mamba2 exact replay is in the dependency branch, with its kernel and
  batch-invariance tests. This reuses the existing Mamba work, not a competing PR.
- Nemotron BI shared W4A16 experts select FlashInfer during MLP construction.
  The kernel retains Humming's FP32 reciprocal-roundtrip for the global scale.
- BI Lightning TP1/EP4 indexed routed experts receive the selected Humming
  schedule during kernel construction: K32 and FP32 accumulation are unchanged;
  N256 uses four stages and small-token intervals use warp-N32.
- The one-sided communicator uses an instance-local workspace subclass for BI
  Lightning. DLPack metadata is owned by the tensor storage through a normal
  CMake-built `vllm._mnnvl_C` extension; FlashInfer functions are not patched.
- Mixed checkpoint loading, model-local norms/router arithmetic, FP8 attention,
  and layerwise reload changes remain in their normal vLLM entry points.

## Launch

Build the C++ extension with the normal source-install/incremental-build workflow
in `docs/contributing/incremental_build.md`; a precompiled older wheel does not
contain the new extension. Then use the standard worker:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_BATCH_INVARIANT=1 \
VLLM_HUMMING_MOE_GEMM_TYPE=indexed \
vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend humming \
  --all2all-backend flashinfer_nvlink_one_sided \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 9216 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --block-size 6768 \
  --mamba-block-size 6768
```

EP4 is implemented by the existing vLLM DP4 ranks plus expert parallelism;
the experts are sharded across four ranks rather than replicated DP-only models.
Compilation and prefix caching use the target version's defaults.

For the explicit BI0 routed-expert baseline, use the same launch settings with
`VLLM_BATCH_INVARIANT=0` and `--moe-backend flashinfer_cutedsl`; the indexed
Humming environment variable is unnecessary. This selects the native FlashInfer
CuTeDSL W4A16 adapter, not Humming or an experimental worker. BI0 does not select
the aligned shared-expert kernel, fixed FA4 backend, or indexed BI schedule.
This is a supported launch recipe, **not a fresh performance acceptance**.

## Workspace ownership review item

The native DLPack owner is a separately reviewable commit. It preserves the
previous experiment's workspace wrapper without replacing FlashInfer functions.
Its tests check metadata destruction, aliases, and borrowed-memory behavior;
they do **not** establish that the original FlashInfer wrapper is faulty.
No original-versus-native failure reproduction currently justifies requiring
this extension. Before merging, reproduce the original failure or validate
removing this change with the same EP4 lifecycle tests. It is neither an
alignment algorithm nor a demonstrated performance optimization.

The workspace cache retains `MnnvlMemory`; the tensor's deleter owns metadata
only. Holding a tensor after clearing that cache is unsupported because its
underlying device allocation is borrowed.

## Validation boundaries

The base remains tested nightly `7f1a5398`; rebasing onto main `87a4bf664f`
conflicted in `mamba_mixer2.py` and was aborted. This is a fork-only Draft for
human review, not an upstream-ready submission.

Prior experimental overlays were tested in separate training/serving images:
five updates compared 883998 valid selected-token FP32 logprobs with zero byte
mismatches. GSM8K on the same theta5 checkpoint scored 1237/1319 for BI0 and
1242/1319 for BI1, a 0.379075 percentage-point difference. Earlier 8K/1K
performance medians at c=1/8/32/64 lost 12.367/11.578/11.858/19.285%.
These results are historical context, **not acceptance of this integration**.

The integrated implementation needs fresh proxy PD/BI checks, real actor-forward
replay, resident weight reload, full-model evaluation, and performance reruns.
Unified-image resident RL is not delivered by this vLLM-only change.
AI assistance was used; human line-by-line review and test reruns are required.
