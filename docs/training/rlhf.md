# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is a technique that fine-tunes language models using human-generated preference data to align model outputs with desired behaviors. vLLM can be used to generate the completions for RLHF.

The following open-source RL libraries use vLLM for fast rollouts (sorted alphabetically and non-exhaustive):

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)

For weight synchronization between training and inference, see the [Weight Transfer](weight_transfer/README.md) documentation, which covers the pluggable backend system with [NCCL](weight_transfer/nccl.md) (multi-GPU) and [IPC](weight_transfer/ipc.md) (same-GPU) engines. To verify that an update landed on every rank, see the [Weight Checker](weight_checker.md).

For MoE policies, see [Routed Experts Return](routed_experts.md) (`--enable-return-routed-experts`) to get the per-token expert routing of each rollout for routing replay (R3) in the trainer.

For pipelining generation and training to improve GPU utilization and throughput, see the [Async Reinforcement Learning](async_rl.md) guide, which covers the pause/resume API for safely updating weights mid-flight.

See the following notebooks showing how to use vLLM for GRPO:

- [Efficient Online Training with GRPO and vLLM in TRL](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)

## Worker extensions (`worker_extension_cls`)

RL frameworks that need custom methods on vLLM's GPU workers (for example, to set up their own communicator) pass a worker extension class with `--worker-extension-cls pkg.module.Class` (CLI) or `LLM(worker_extension_cls="pkg.module.Class")`. The value must be a fully qualified name string that is importable in every worker process. Passing a class object is not supported.

- vLLM mixes the class into the worker class by appending it to `worker_class.__bases__`. Inside its methods, `self` is the worker, which provides `self.device`, `self.rank`, `self.model_runner`, and so on.
- No non-dunder attribute of the extension may already exist on the worker class. A name collision fails worker startup with `AssertionError: Worker class ... already has an attribute ...`.
- Optional lifecycle hooks:
    - `init_worker_extension(self)` runs once per worker, right after `init_device()`. At that point the device and `torch.distributed` are initialized, but the model weights are not loaded yet.
    - `destroy_worker_extension(self)` runs at shutdown, before the worker's own `shutdown()`. This hook is best-effort. Exceptions are logged and not raised. With the default `mp` backend it runs when workers exit gracefully or on SIGTERM, but not after SIGKILL. With `--distributed-executor-backend ray`, the executor stops workers with `ray.kill`, so this hook does not run.
- Call extension methods with `collective_rpc`, which returns a list with one result per worker. It is available as `LLM.collective_rpc(method, args=..., kwargs=...)` offline and as `await AsyncLLM.collective_rpc(...)` in async engines. Over HTTP, use `POST /collective_rpc` with `{"method": ..., "args": [...], "kwargs": {...}}`; it requires `VLLM_SERVER_DEV_MODE=1` and accepts string arguments only.

```python
# my_rl/worker_ext.py
class MyWorkerExtension:
    def init_worker_extension(self):
        # Device and distributed state are ready; the model is not loaded yet.
        self.my_comm = f"comm-on-{self.device}"

    def destroy_worker_extension(self):
        self.my_comm = None

    def my_comm_info(self) -> str:
        return f"rank={self.rank} {self.my_comm}"
```

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tensor_parallel_size=2,
    worker_extension_cls="my_rl.worker_ext.MyWorkerExtension",
)
print(llm.collective_rpc("my_comm_info"))  # one entry per TP worker
```

For weight synchronization, prefer the built-in [Weight Transfer](weight_transfer/README.md) API over a custom extension. The older `examples/offline_inference/rlhf*.py` examples, which used `worker_extension_cls` for this, were removed in favor of that API.
