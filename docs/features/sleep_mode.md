# Sleep Mode

vLLM's Sleep Mode allows you to temporarily release most GPU memory used by a model, including model weights and KV cache, without stopping the server or unloading the Docker container. This is especially useful for RLHF, training, or cost-saving scenarios where GPU resources need to be freed between inference workloads.

Key benefits:

- **Frees GPU memory**: Offloads model weights to CPU RAM and discards KV cache, releasing up to 90%+ of GPU memory for other tasks.
- **Fast resume**: Quickly wake up the engine and resume inference without full model reload.
- **API endpoints**: Control sleep/wake_up state via HTTP endpoints or Python API.
- **Supports distributed workloads**: Works with tensor parallelism, pipeline parallelism, etc.
- **Fine-grained control**: Optionally wake up only model weights or KV cache to avoid OOM during weight updates.

!!! note
    This feature is now supported on CUDA and ROCm platform.

!!! note
    For more information, see this [Blog Post](https://blog.vllm.ai/2025/10/26/sleep-mode.html).

## Sleep levels

Level 1 sleep will offload the model weights and discard the KV cache. The content of KV cache is forgotten. Level 1 sleep is good for sleeping and waking up the engine to run the same model again. The model weights are backed up in CPU memory. Please make sure there's enough CPU memory to store the model weights. Level 2 sleep will discard both the model weights and the KV cache (while the model's buffers are kept in CPU, like rope scaling tensors). The content of both the model weights and KV cache is forgotten. Level 2 sleep is good for sleeping and waking up the engine to run a different model or update the model, where previous model weights are not needed, e.g. RLHF weight update. Level 2 sleep is also useful when there is not enough CPU memory to back up the model weights, e.g. when a colocated trainer already uses CPU memory for offloading its own state; since nothing is backed up, restore the weights after waking up with `collective_rpc("reload_weights")`.

## Usage

### Offline inference

Enable sleep mode by passing `enable_sleep_mode=True` to the `LLM` class.

```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

#### Python API

```python
# Sleep level 1
# Put the engine to sleep (level=1: offload weights to CPU RAM, discard KV cache)
llm.sleep(level=1)

# Wake up the engine (restore weights)
llm.wake_up()
```

```python
# Sleep level 2
# Put the engine to sleep (level=2: discard both weights and KV cache)
llm.sleep(level=2)

# Reallocate weights memory only
llm.wake_up(tags=["weights"])

# Load weights in-place
llm.collective_rpc("reload_weights")

# Reallocate KV cache
llm.wake_up(tags=["kv_cache"])
```

#### RLHF weight updates

During RLHF training, vLLM allows you to selectively wake up only the model weights or the KV cache using the tags argument in wake_up(). This fine-grained control is especially useful when updating model weights: by waking up just the weights (e.g., llm.wake_up(tags=["weights"])), you avoid allocating memory for the KV cache until after the weight update is complete. This approach helps prevent GPU out-of-memory (OOM) errors, particularly with large models, by minimizing peak memory usage during weight synchronization and update operations.

Use `tags=["weights"]` or `tags=["kv_cache"]` to control which resources are restored, useful for RLHF and weight updates. **Note** that `is_sleeping` will report `true` until all components are awake.

```python
# Put engine to deep sleep (level=2)
llm.sleep(level=2)
# ... Get the new weights
# Wake up only weights to avoid OOM
llm.wake_up(tags=["weights"])
# ... Update the weights
# wake up KV cache after weights are updated
llm.wake_up(tags=["kv_cache"])
```

#### Retaining frozen weights during RLHF updates

Set `sleep_preserve_parameter_names` (CLI: `--sleep-preserve-parameter-names`)
to runtime parameter-name glob patterns for weights that stay frozen during
training. It covers two use cases:

- **Frozen Engram tables (original use case):** retain table weights and scales
  omitted by the trainer, without retransmitting them after every sleep.
- **Frozen draft weights:** retain draft-private parameters when the trainer
  updates only the target model.

Target patterns match
`model.named_parameters()`; checkpoint names and `requires_grad` are not used.

Level-2 sleep backs up selected GPU parameters to pageable CPU memory; weights
wake-up restores them in place and releases the backups. CPU parameters need no
copy. Allow enough host memory for backups. Level-1 behavior is unchanged.

The trainer must omit these parameters from updates, and reload/post-processing
must preserve their values and storage (including avoiding replacement with meta
tensors). This option does not filter updates.

For frozen Engram tables omitted by the trainer, select the target table weights
and scales. For example, DeepSeek V4.1 uses these runtime parameter names:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve deepseek-ai/DeepSeek-V4.1-Flash --enable-sleep-mode \
    --sleep-preserve-parameter-names '*.engram.embed_tokens.weight*'
```

CPU-offloaded tables remain on CPU without an extra backup copy. Engram's
[inference-time CPU offload](engram.md#cpu-offload) is configured separately;
this option preserves the frozen tables across level-2 sleep and weight updates.

To retain a frozen drafter while updating only target weights, use `draft.*`:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.8-27B --enable-sleep-mode \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --sleep-preserve-parameter-names 'draft.*'
```

vLLM selects all draft-private parameters internally; you do not need to list
them individually. Use `draft.<name>` to select specific draft parameters.
Parameters shared with the target are excluded from the draft selection, so
target updates remain authoritative. An empty `draft.*` selection is valid.
The Engram and draft patterns can be supplied together.

#### Release only KV cache memory

`LLM.release_kv_cache_memory()` discards KV cache physical memory while keeping model weights resident. It requires a completed pause and all executor memory to be resident: full sleep, partial wake-up, and repeated release without restoring memory are rejected. Requests retained with `mode="keep"` are recomputed after wake-up.

Use the default `cumem` backend with managed allocations: `enable_cumem_allocator=True` on CUDA/ROCm (also enabled by `enable_sleep_mode=True`), or `enable_sleep_mode=True` on XPU. Other backends must implement selective discard; CPU and unmanaged allocations are unsupported.

```python
llm.sleep(level=0, mode="keep")  # Wait for the pause to complete.
llm.release_kv_cache_memory()
llm.wake_up(tags=["kv_cache"])  # Reallocate KV cache and resume scheduling.
```

#### Offloading CUDA graph memory

By default, CUDA graph memory stays on the GPU while asleep. With
`sleep_mode_offload_cudagraph=True` (off by default), CUDA graphs are captured
into a cuMem pool that sleep backs up to CPU memory at both levels and any wake
restores in place, so graphs are replayed, not recaptured. It needs the default
`cumem` backend, CUDA or ROCm, and CUDA graphs; otherwise it has no effect.

```python
llm = LLM("Qwen/Qwen3-8B", enable_sleep_mode=True, sleep_mode_offload_cudagraph=True)
```

or `vllm serve <model> --enable-sleep-mode --sleep-mode-offload-cudagraph`.

The cost is pinned host memory for the pool's backup, also at level 2, and one
extra copy per captured custom allreduce (about 3% decode latency at batch
size 1). `NCCL_GRAPH_REGISTER` defaults to `0`, since NCCL graph registration
would pin the pool; an explicit value is kept with a warning. Graph executables
outside PyTorch pools stay resident.

#### Releasing NCCL communicator memory

By default, NCCL communicators keep their GPU buffers while asleep. With
`enable_nccl_comm_suspend=True` (off by default, experimental), sleep releases
them with `ncclCommSuspend` and wake restores them with `ncclCommResume`. The
communicators keep their topology, so they are not re-created. It needs NCCL
2.29.7 or newer; with an older library, a warning is logged and the memory stays
on the GPU.

```python
llm = LLM("Qwen/Qwen3-8B", enable_sleep_mode=True, enable_nccl_comm_suspend=True)
```

or `vllm serve <model> --enable-sleep-mode --enable-nccl-comm-suspend`.

#### Choosing a sleep-mode backend

`sleep_mode_backend` selects how GPU memory is freed and restored. The default
is the built-in `cumem` backend. Plugins can register other backends with
`SleepModeBackendFactory.register_backend` from a `vllm.general_plugins` entry
point, and you select one by name:

```python
llm = LLM("Qwen/Qwen3-8B", enable_sleep_mode=True, sleep_mode_backend="my_backend")
```

or `vllm serve <model> --enable-sleep-mode --sleep-mode-backend my_backend`.

An unregistered name is not caught at startup; it fails the first time the
backend is used, such as on the first sleep. CUDA graph offloading and
`release_kv_cache_memory()` have their own backend requirements, described
above.

### Online Serving

To enable sleep mode in a vLLM server you need to initialize it with the flag `VLLM_SERVER_DEV_MODE=1` and pass `--enable-sleep-mode` to the vLLM server.

#### Server in development mode

When using the flag `VLLM_SERVER_DEV_MODE=1` you enable development endpoints, and these endpoints should not be exposed to users.

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

Below is an example of how to sleep and wake up a model in level 1.

```bash
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

And this is an example of how to sleep and wake up a model in level 2.

```bash
curl -X POST 'http://localhost:8000/sleep?level=2'
# Reallocate weights memory only
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
# Load weights in-place
curl -X POST 'http://localhost:8000/collective_rpc' -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
# Reallocate KV cache
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

To release only KV cache memory, wait for the level 0 pause to complete before calling the release endpoint. The same resident-memory and backend requirements as the Python API apply.

```bash
curl -X POST 'http://localhost:8000/sleep?level=0&mode=keep'
curl -X POST 'http://localhost:8000/release_kv_cache_memory'
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

#### HTTP endpoints

- `POST /sleep?level=1` — Put the model to sleep (`level=1`).
- `POST /release_kv_cache_memory` — Discard KV cache memory after a completed pause, while all executor memory is resident.
- `POST /wake_up` — Wake up the model. Supports optional `tags` query parameters for partial wake-up (e.g., `?tags=weights`).
- `POST /collective_rpc` — Perform a collective remote procedure call (RPC).
- `GET /is_sleeping` — Check if the model is sleeping.

!!! note
    These endpoints are only available when passing `VLLM_SERVER_DEV_MODE=1`.

## Custom sleep backends

Sleep and wake-up (`LLM.sleep()` / `LLM.wake_up()`, `/sleep` / `/wake_up`) call a
pluggable sleep-mode backend (`SleepModeBackend`) in each GPU worker. The backend is
selected by name with `ModelConfig.sleep_mode_backend`, which defaults to `"cumem"`
(the built-in `CuMemBackend`). Use this extension point to replace the allocator
offload with your own mechanism, such as CUDA checkpoint or CRIU. vLLM ships only
`cumem`. The interface is early-stage (RFC #34303) and may change.

!!! warning
    There is no `--sleep-mode-backend` CLI flag and no `EngineArgs` field yet.
    `vllm serve` and `LLM(...)` always use `cumem`. To select another backend, set
    `model_config.sleep_mode_backend` on a `VllmConfig` (see step 3 below).

### 1. Implement the backend

Subclass `SleepModeBackend` and implement `suspend` and `resume`:

```python
# my_pkg/backend.py
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend


class MyCkptBackend(SleepModeBackend):
    def suspend(self, level: int = 1) -> None:
        ...  # free GPU state (level 1 or 2, same meaning as sleep levels)
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        ...  # restore GPU state; tags may be ["weights"] or ["kv_cache"]
        self._state = "RUNNING"

    @classmethod
    def is_supported(cls) -> bool:
        return True  # probe the driver or tool; False fails backend creation
```

- Required methods: `suspend(level)` and `resume(tags)`.
- Optional methods:
    - `discard(tags)` is needed for `release_kv_cache_memory`. The default raises `RuntimeError`.
    - `state()` returns `self._state`, which is one of `"RUNNING"`, `"SUSPENDED"` or `"RESUMING"`.
- Capability classmethods:
    - `is_supported()` defaults to `True`.
    - `preserves_communicators()`, `preserves_compiled_artifacts()`, `preserves_graphs_with_communicators()` and `supports_durable_storage()` all default to `False`.
- vLLM checks only `is_supported()` today. It does not act on the other capability flags. For example, it does not re-create NCCL communicators, recompile kernels or recapture CUDA graphs after `resume`. Your backend must restore anything it destroys.
- The backend is constructed with no arguments. It is created lazily, once per worker process (each TP/PP rank), on the first sleep, wake-up or discard. If several ranks must act in a fixed order, the backend has to coordinate them.

### 2. Register it from a plugin

Register the backend under a name with `SleepModeBackendFactory.register_backend(name, module_path, class_name)`. Registration is lazy: the module is imported only when the backend is created. The backend is resolved inside each worker, so register it from a `vllm.general_plugins` entry point. vLLM loads general plugins in every process it starts (see [Plugin System](../design/plugin_system.md)).

```python
# my_pkg/__init__.py
def register():
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    try:
        SleepModeBackendFactory.register_backend(
            "my_ckpt", "my_pkg.backend", "MyCkptBackend"
        )
    except ValueError:
        pass  # already registered; plugins must be re-entrant
```

```python
# setup.py
setup(
    name="my_pkg",
    entry_points={"vllm.general_plugins": ["my_ckpt = my_pkg:register"]},
)
```

### 3. Select it

```python
from vllm import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine

vllm_config = EngineArgs(
    model="Qwen/Qwen3-0.6B", enable_sleep_mode=True
).create_engine_config()
vllm_config.model_config.sleep_mode_backend = "my_ckpt"
engine = LLMEngine.from_vllm_config(vllm_config)

engine.sleep(level=1)
engine.wake_up()
```

Use `AsyncLLM.from_vllm_config(vllm_config)` for the async engine.

### Behavior and limitations

- **`--enable-sleep-mode` is still required.** On CUDA/ROCm it still turns on the cumem allocator (`enable_cumem_allocator`), so weights and KV cache are allocated from the CuMem pool whichever backend you select.
- **The worker still runs its own steps around your backend.** These steps include:
    - saving and restoring buffers and `sleep_preserve_parameter_names` at level 2;
    - unregistering and restoring KV-connector caches;
    - the `enable_nccl_comm_suspend` suspend and resume.
- **Freed memory is checked.** After `suspend`, the worker asserts that free GPU memory did not decrease (`Memory usage increased after sleeping.`).
- **CUDA graph offload needs `cumem`.** `sleep_mode_offload_cudagraph` takes effect only with the `cumem` backend.
- **Errors appear on the first sleep, wake-up or discard, not at startup:**
    - An unregistered name raises `ValueError: Unsupported sleep-mode backend '<name>'. Registered backends: ...`.
    - `is_supported()` returning `False` raises `ValueError: Sleep-mode backend '<name>' is not supported on this platform.`
    - Registering a name twice, including `cumem`, raises `ValueError: Sleep-mode backend '<name>' is already registered.`

## Limitation

On ROCm, the virtual memory allocation on ROCm is done through chunked memory allocation. You can control the chunk size through `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` (in MB). The default value is set at 256MB. The larger the chunk size the faster the performance. However, setting it too large will cause OOM. So if you encounter OOM when using sleep mode. Try reducing the chunk size. It is recommended to define the chunk size as a power of 2.

## Experimental CUDA checkpoint executor

This capability is experimental and disabled by default. Selecting the executor
below checkpoints each CUDA worker after ordinary level-1 sleep,
then restores all workers before waking allocator-managed memory. This preserves
the remaining CUDA state, including captured graph executable objects, without
recapturing graphs. Ordinary level-1 semantics still apply: KV cache contents are
discarded. This is not a durable application checkpoint or model switching API.

Set `VLLM_USE_V2_MODEL_RUNNER=1` before starting the engine:

```python
from vllm import LLM

llm = LLM(
    "Qwen/Qwen3-0.6B",
    enable_sleep_mode=True,
    distributed_executor_backend=(
        "vllm.v1.executor.cuda_checkpoint_executor.CUDACheckpointExecutor"
    ),
)
llm.sleep(level=1)
llm.wake_up()
```

This experimental implementation requires Linux, NVIDIA driver checkpoint APIs,
and dedicated worker processes. It uses an external helper to call
`cuCheckpointProcessLock`, `cuCheckpointProcessCheckpoint`,
`cuCheckpointProcessRestore`, and `cuCheckpointProcessUnlock`. CPU threads remain
alive, but worker CUDA calls cannot execute while checkpointed. The helper uses
the same process namespace and user as the executor. Driver permission and
resource restrictions still apply; see the
[CUDA checkpoint API documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__CHECKPOINT.html).

The initial validation used NVIDIA H20 with driver 580.126.09, NCCL 2.29.7,
and FlashInfer 0.6.18.post1. See [PR #58183](https://github.com/vllm-project/vllm/pull/58183)
for versioned test results and communication performance comparisons. Support
for other GPU, driver, and communication-library combinations is not established.

Only single-node tensor parallelism is accepted. Tensor parallel sizes 1, 2,
and 4 have been GPU-tested; other sizes remain unvalidated. Use
`VLLM_USE_V2_MODEL_RUNNER=1`, the standard GPU worker and the `cumem` sleep
backend. Pipeline, data, context and expert parallelism, external cache or
weight transfer, custom worker classes, and sleep level 2 are rejected before
worker startup (sleep level 2 is rejected when requested). Worker extensions
that allocate additional CUDA IPC resources are outside the validated scope.
Level 0 remains
a scheduler pause and does not perform a CUDA checkpoint.

For tensor parallel sizes greater than 1, use NCCL 2.29.7 or later with memory
suspension support, set `enable_nccl_comm_suspend=True`, and pass
`disable_custom_all_reduce=True`. The following explicit configuration retains
NCCL GPU peer-to-peer or legacy shared-memory communication, selected by NCCL.
`NCCL_P2P_DISABLE` and `NCCL_SHM_DISABLE` need not be set when communicator
suspension is enabled. NCCL CUDA host-memory allocation remains disabled; it
is not equivalent to legacy shared-memory transport. Before driver checkpoint,
ordinary worker
sleep suspends NCCL mappings; worker wake restores them at their reserved
addresses. The executor checks each rank's actual suspension state.

```bash
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_NVLS_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
```

FlashInfer must provide stable-address workspace `checkpoint_prepare` and
`checkpoint_restore` methods. The executor calls existing worker hooks around
the driver checkpoint. Disabling standalone FlashInfer all-reduce with
`VLLM_ALLREDUCE_USE_FLASHINFER=0` does not disable the compiler
`fuse_allreduce_rms` pass, which can allocate FlashInfer workspaces independently.
When that pass is enabled, the same backend restriction and workspace hooks
apply. A configuration using only NCCL must also disable that compiler pass.
The FlashInfer `mnnvl` backend is not accepted, and the NCCL suspend configuration
requires NVLS to be disabled. Live multicast
allocations can block the combined sleep/checkpoint flow even on newer
user-mode drivers. Keep these restrictions when upgrading the driver. The
helper must use the same user-mode driver as the workers; mixed versions are
unsupported. Performance costs of the restrictions must be measured against
the workload's default communication
configuration; retaining GPU peer-to-peer communication alone does not establish
unchanged inference performance.

A more restrictive Socket-only configuration is also accepted without
`enable_nccl_comm_suspend`: set `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`,
`NCCL_CUMEM_ENABLE=0`, and
`VLLM_ALLREDUCE_USE_FLASHINFER=0`, retaining the other communication restrictions
above. This configuration is primarily useful for compatibility comparisons.
Do not create additional CUDA IPC resources or fall back to an independently
created PyTorch NCCL process group while using the checkpoint executor.

Provide host memory for both the ordinary weight backup and the driver's
remaining CUDA state backup. Restore needs sufficient free GPU memory; release
any temporary GPU workload before waking. Checkpoint and restore add latency to
ordinary sleep/wake. The helper itself can retain a small driver allocation, so
worker offload does not imply a zero device-wide memory reading.

All ranks are locked before any is checkpointed. All ranks are restored before
any is unlocked. Operations within each stage run concurrently across workers.
The whole stage has a 30-second response deadline (each lock also has a
10-second driver timeout). Worker sleep/wake and FlashInfer workspace
hooks have a 120-second response deadline. Failure terminates the workers rather than
serving with a partially restored rank set; restart the engine after failure.
Shutdown terminates the dedicated worker and helper processes without restoring
their CUDA state or requiring free GPU memory for the model.

Repeated level-1 sleep is idempotent. A first partial wake, such as
`wake_up(tags=["weights"])`, restores the **whole CUDA process** before restoring
the selected allocator tags. It cannot selectively restore graph or driver
state. Ordinary worker RPCs are rejected while checkpointed. A checkpointed
worker cannot execute another model until restored.

Serialize sleep, wake, pause, and resume calls across all clients, including
while level-1 sleep is draining requests. Wait for each call to complete before
issuing the next. A partial wake does not automatically resume scheduling; wake
all allocator tags before explicitly resuming or submitting new inference.
Do not use an explicit resume call to bypass sleeping memory.

If EngineCore exits during a management call, the client may remain waiting even
after the workers terminate. Use an external timeout and restart the failed
engine; a timeout does not establish that the operation completed. Do not retry
control operations on an engine whose checkpoint outcome is unknown.

As with the ordinary executor, `sleep()` is a no-op while weights remain asleep.
For example, after `wake_up(tags=["kv_cache"])`, another `sleep()` does not
checkpoint the process again. Wake the remaining weights before starting a new
full sleep/checkpoint cycle.
