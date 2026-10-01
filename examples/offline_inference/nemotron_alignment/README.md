# Nemotron-H NVFP4/FP8: exact-source review snapshot

Fork-only draft for human review of the inference-side code used in our GB200
experiments. This is **not** a turnkey verl integration, unified-image RL
delivery, or a competing upstream Mamba implementation.

## Dependency and base

The tested nightly is `7f1a5398e9610d96c473931a26c0e12bbe0d0423`, from
`vllm/vllm-openai@sha256:13ea3b228a97a12482302341942fd1e5694395ad2c46f940de15fa8f4b4a7a1d`.
The dependency branch preserves its port of the existing Mamba replay stack,
with the emit, metadata and invariance tests restored. Credit: NolenLiang
(bcsdhjew), upstream #55905/#57539/#55627/#55895, via ISEEKYAN/vllm#25.
The main PR is stacked on that dependency, not an independent replacement.

The owner's existing PR55 is an older BF16 alignment snapshot; this delta adds
the NVFP4 W4A16 path, direct FP8 fixed-schedule FA4, device FP8 quantization,
shared-expert FlashInfer selection and hybrid Humming tuning. Do not submit
this complete stack against upstream main while its dependencies are open.

Keep the exact tested revision separate from a future current-main rebase:
upstream model, Mamba and quantization APIs have changed. A rebased candidate
requires fresh correctness tests and model evaluation, not borrowed evidence.

A disposable rebase onto upstream `87a4bf664f` was attempted and conflicted
in `vllm/model_executor/layers/mamba/mamba_mixer2.py` while applying the
dependency. It was aborted without changing the tested review branch.

## What is included

- Nemotron-specific BI norms and FP8 quantizer/gate dispatch, with TP1 guards.
- Fixed direct-FP8 CuTe FA4 on Blackwell: 32 query heads, 2 KV heads, head
  size 128, FP8 KV, 9216-token bound and the frozen split/reduction schedule.
- Explicit FlashInfer CuTeDSL W4A16 routed-expert support for the tested
  geometry; automatic/default selection is not silently replaced by it.
- Mixed per-prefix ModelOpt loading, stacked expert expansion and layerwise
  reload handling.
- `overlays/`: the actual experimental worker, shared-expert FlashInfer
  selection, indexed Humming tuning, and native uint8 DLPack metadata owner.
  The C++ source is included; compiled `.so` files are not committed.
- Nearby layout, kernel-selection, model-loading and reload tests, plus the
  existing Mamba dependency tests. No performance benchmarks live in tests.

`source-manifest.json` binds all 33 included runtime source files to the
actual experiment files by SHA256. Relocating the overlays does not prove
their new package layout is production-ready. Their historical module names,
dependency hash guards and worker provenance requirements remain intact.
Some wrappers are global under BI; only the stated Nemotron configuration was
evaluated. Other models, TP>1 and changing dependency versions need review.

One incidental GDN `kernel_o.py` change was deliberately excluded: Nemotron's
`MambaMixer2` does not import it. The included-file hashes therefore do not
assert identity of the complete previous image/package closure.

## Runtime contract

The measured model is full 52-layer Lightning-NVFP4, GB200, runner V2,
TP1+EP4, BF16 activations, FP8 E4M3 KV, FP32 SSM state, indexed Humming routed
experts, FlashInfer shared experts and `flashinfer_nvlink_one_sided`.
Scheduler budget is 16384, max sequences 64, max model length 9216, and BI1
block/Mamba block size 6768. No explicit compilation-config or prefix-cache
argument was added; the target defaults remain enabled.

The worker overlays require the matching FlashInfer and Humming installations,
an ABI-compatible build of `fi_w4a16_native_dlpack.cpp`, and their explicit
experiment/provenance environment. Copying only Python files or using stock
`vllm serve` without the worker will **not** reproduce the measured hybrid
recipe. An in-tree worker/kernel integration and unified RL image are pending.

## Existing experiment evidence and boundaries

The final serving image was
`sha256:0689da16e5878bb6b783bdbd55824020908cedd568f7f00b9377bd7152037beb`.
The trainer was a **different** immutable image with Megatron PP4. Five DAPO
Math17K updates used saved rollout replay and separately cold-started serving
versions: 640 samples, 883998 valid raw FP32 selected-token logprobs with zero
byte mismatches. The final theta5 actor/autograd probe compared 2048 valid
tokens, zero byte mismatches, one backward and no extra optimizer update.
This does not verify verl's resident rollout/online synchronization path.

The current theta5 GSM8K protocol used all 1319 same questions, c=8, identical
generation settings and no retries/deletions: BI0 1237/1319 (93.7832%), BI1
1242/1319 (94.1622%), absolute gap 0.379075 percentage points. Both correct:
1215; only BI0: 22; only BI1: 27; both wrong: 55. Both services exited normally.
The current user gate is <=1 percentage point, not the withdrawn 96% gate.

The user accepted the performance item separately. Earlier same-node
8192/1024 ignore-EOS Rust Bench, c=1/8/32/64, three AB/BA/AB pairs on serving
image `608f6004...` yielded median losses 12.367/11.578/11.858/19.285%.
Some c64 individual pairs exceeded 20%. Those runs were **not** rerun on the
new `0689da16...` control image or this relocated checkout.

The following sealed receipts locate the original evidence; they are not new
checkout test results:

| Receipt | SHA256 |
|---|---|
| Five updates + final reload probe | `e55b1c25348ff6b5d6e9fc66f6cfe52a9d95726334a537b5535808ab15a6e559` |
| Current GSM8K pair | `71fb878d1b600bbd1280cc5dcea414809f038ea6362ed6513771dd8f1606ad9e` |
| Actual four-worker quality runtime | `65277505b3f1c6a1d409cdae0873efe882b9720400ad5ce47890ff49fc4366a0` |

No SFT, long-term convergence, all-vocabulary logits equality or original
optimizer-checkpoint resume is claimed. The prior overall completion claim
does not cover the now-required unified-image, resident RL deployment.

## Review and checks

AI assistance (OpenAI Codex) was used. Human line-by-line review and personal
test reruns remain required before merge/upstream submission. Source hashes
and Python AST checks pass; `git diff --check` passes. Full CI/pre-commit is
not claimed. Non-mutating Ruff found three import-order issues and one long
line in preserved overlays, plus five UP038 reports in `modelopt.py`; no
hook-driven source rewrite was used to preserve the tested bytes.

New checkout test attempt:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/test_nemotron_h_quantization.py \
  tests/kernels/moe/test_flashinfer_cutedsl_layout.py \
  tests/kernels/quantization/test_nvfp4_kernel_selection.py -q
```

Collection failed with two ImportErrors: this host's checkout lacks the CUDA
FlashAttention extensions `_vllm_fa2_C` / `_vllm_fa3_C`. This is **not** a
passing pytest result, and no new GPU suite or model evaluation is claimed
for the relocated checkout. The immutable-image evidence above is separate.
