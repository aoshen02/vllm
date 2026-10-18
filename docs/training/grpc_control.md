# gRPC Control API for RL (Rust Frontend)

The Rust frontend's `vllm.Control` gRPC service lets an RL orchestrator pause and resume generation, sleep and wake the engine, drive weight updates (`InitWeightTransferEngine`, `StartWeightUpdate`, `UpdateWeights`, `FinishWeightUpdate`), and query weight versions and RL capabilities. It covers the same lifecycle as the HTTP endpoints in [Weight Transfer](weight_transfer/README.md#api-endpoints) and [Async RL](async_rl.md). The schema is [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/main/rust/proto/control.proto).

## Enabling gRPC with `--grpc-port`

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve my-model \
    --host 127.0.0.1 --port 8000 --grpc-port 50051 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --enable-sleep-mode
```

- `--grpc-port` starts the `vllm.Control`, `vllm.Inference` and `grpc.health.v1.Health` services on a separate port. HTTP stays on `--port`. Without `--grpc-port`, no gRPC server runs.
- Requires `VLLM_USE_RUST_FRONTEND=1` with `vllm serve`. The gRPC listener binds to `--host`, or to `127.0.0.1` if HTTP uses `--uds`. `0` picks a free port.
- Startup fails with `ValueError` if any of these hold:
    - `--grpc` is also set. `--grpc` is a different server, the Python SMG gRPC API.
    - `--headless` is set, or `--api-server-count <= 0`.
    - `--data-parallel-multi-port-external-lb` is set.
    - `--grpc-port` equals `--port` and HTTP uses TCP.
- `VLLM_SERVER_DEV_MODE` is **not** required for gRPC; it only gates the HTTP RL routes.
- The weight RPCs need `--weight-transfer-config`, and the sleep RPCs need `--enable-sleep-mode`. Without them, these RPCs fail with `FAILED_PRECONDITION`.
- There is no authentication: `--api-key` does not apply to gRPC. When `--ssl-keyfile`/`--ssl-certfile` are set, gRPC uses the same certificate. Only expose the port on a trusted network (see [Security](../usage/security.md#grpc-interface)).

## RL RPCs

All RPCs are in the `vllm.Control` service. Fields not listed below take no input.

| RPC | Request fields | Response | Notes |
| --- | -------------- | -------- | ----- |
| `GetServerInfo` | none | `ServerInfo` | Contains `rl_capabilities`, `parallelism.world_size` and `proto_version` |
| `PauseGeneration` | `mode`: `PAUSE_MODE_ABORT` \| `PAUSE_MODE_WAIT` \| `PAUSE_MODE_KEEP`; `clear_cache` (optional bool) | empty | Unset `mode` means `ABORT`. Unset `clear_cache` means `true`. Modes are the same as [`pause_generation`](async_rl.md#pause_generation) |
| `ResumeGeneration` | none | empty | |
| `IsPaused` | none | `paused` | |
| `Sleep` | `level` (optional uint32); `mode` (`PauseMode`) | empty | Unset `level` means `1`. Needs sleep mode |
| `WakeUp` | `tags` (repeated string) | empty | Empty `tags` wakes everything. Needs sleep mode |
| `IsSleeping` | none | `sleeping` | |
| `InitWeightTransferEngine` | `init_info_json` (bytes) | empty | Needs weight transfer |
| `StartWeightUpdate` | none | empty | Needs weight transfer |
| `StartDraftWeightUpdate` | none | empty | Needs weight transfer and `draft_weight_updates_enabled` |
| `UpdateWeights` | `update_info_json` (bytes) | empty | Needs weight transfer. Call it between Start and Finish |
| `FinishWeightUpdate` | `weight_version` (optional string) | empty | Needs weight transfer. Commits the version if one is given |
| `UpdateWeightVersion` | `weight_version` (string) | empty | Sets the label without changing weights |
| `GetWeightVersion` | none | `weight_version` | |

The service also has `Shutdown`, `Abort`, the LoRA RPCs (`LoadLora`, `UnloadLora`, `ListLoras`), `GetModelInfo` and `GetKvEventSources`.

### JSON payloads (`init_info_json`, `update_info_json`)

- Each field holds the UTF-8 bytes of a JSON **object**: the dict the trainer engine passes to `VLLMWeightSyncClient`.
- The HTTP routes wrap the same dict as `{"init_info": ...}` or `{"update_info": ...}`. gRPC sends the inner object with no wrapper.
- Tensors never cross gRPC. They move over the configured backend (NCCL, IPC, sparse-NCCL, sharded-RDT and so on).

| Backend | `init_info_json` | `update_info_json` |
| ------- | ---------------- | ------------------ |
| `nccl` | `{"master_address", "master_port", "rank_offset", "world_size", "packed", "packed_buffer_size_bytes", "packed_num_buffers"}`, or `nccl_unique_id_b64` in place of address and port | `{"names", "dtype_names", "shapes"}` |
| `ipc` | `{"packed"}` | IPC handles must be made JSON-safe as `ipc_handles_pickled`, as `HTTPVLLMWeightSyncClient` does |

For the other backends, see their pages under [Weight Transfer](weight_transfer/README.md).

## Capability discovery: `GetServerInfo.rl_capabilities`

| Field | True when |
| ----- | --------- |
| `weight_transfer_enabled` | Every engine reports the same `--weight-transfer-config` backend |
| `weight_transfer_backend` | That backend name, for example `"nccl"`. Empty when disabled |
| `sleep_mode_enabled` | Every engine was started with `--enable-sleep-mode` |
| `draft_weight_updates_enabled` | Every engine has a speculative draft model and a backend that supports draft updates |

Use `parallelism.world_size` from the same response to size the NCCL group. It replaces HTTP `GET /get_world_size`.

## Errors

| Code | When |
| ---- | ---- |
| `FAILED_PRECONDITION` | A weight RPC without `--weight-transfer-config`, a sleep RPC without `--enable-sleep-mode`, `StartDraftWeightUpdate` without a draft model, or `Shutdown` when the frontend does not manage the engine |
| `INVALID_ARGUMENT` | `*_json` is not valid JSON or not an object, `weight_version` is empty, or the pause mode is invalid |
| `INTERNAL` | The engine call failed, for example `UpdateWeights` before `StartWeightUpdate`. The message is `"<method> failed: ..."` |

## Concurrency

- Every RPC that changes state (pause, resume, sleep, wake, the weight RPCs, `UpdateWeightVersion`) runs under one lock, so these calls are serialized.
- `IsPaused`, `IsSleeping`, `GetWeightVersion` and `GetServerInfo` do not take the lock.
- With NCCL, `InitWeightTransferEngine` and `UpdateWeights` block until the trainer joins the rendezvous or broadcast. They hold the lock while they block. Issue them concurrently with the trainer-side send, as `HTTPVLLMWeightSyncClient` usage does, and set a long gRPC deadline.

## Minimal clients

### grpcurl

The server does not enable gRPC reflection, so pass the proto file:

```bash
P="-plaintext -import-path rust/proto -proto control.proto"
grpcurl $P localhost:50051 vllm.Control/GetServerInfo
grpcurl $P -d '{"mode": "PAUSE_MODE_KEEP", "clear_cache": false}' \
    localhost:50051 vllm.Control/PauseGeneration
grpcurl $P -d '{"weight_version": "step-42"}' localhost:50051 vllm.Control/UpdateWeightVersion
grpcurl $P localhost:50051 vllm.Control/ResumeGeneration
```

grpcurl prints camelCase names (for example `rlCapabilities.weightTransferEnabled`) and leaves out fields that are `false` or empty. Add `-emit-defaults` to show them. In grpcurl JSON, `bytes` fields such as `init_info_json` must be base64-encoded.

### Python (`grpcio`) as a `VLLMWeightSyncClient`

Generate stubs with `python -m grpc_tools.protoc -I rust/proto --python_out=. --grpc_python_out=. rust/proto/control.proto`. Then plug this client into `WeightTransferTrainerFactory.trainer_init(..., client=...)` in place of `HTTPVLLMWeightSyncClient`:

```python
import json

import grpc

import control_pb2 as pb
import control_pb2_grpc


class GrpcVLLMWeightSyncClient:
    def __init__(self, target: str, timeout: float = 300):
        self.stub = control_pb2_grpc.ControlStub(grpc.insecure_channel(target))
        self.timeout = timeout

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        req = pb.InitWeightTransferEngineRequest(init_info_json=json.dumps(init_info).encode())
        self.stub.InitWeightTransferEngine(req, timeout=self.timeout)

    def start_weight_update(self) -> None:
        self.stub.StartWeightUpdate(pb.StartWeightUpdateRequest(), timeout=self.timeout)

    def update_weights(self, update_info: dict) -> None:
        req = pb.UpdateWeightsRequest(update_info_json=json.dumps(update_info).encode())
        self.stub.UpdateWeights(req, timeout=self.timeout)

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        req = pb.FinishWeightUpdateRequest()
        if weight_version is not None:
            req.weight_version = weight_version
        self.stub.FinishWeightUpdate(req, timeout=self.timeout)


client = GrpcVLLMWeightSyncClient("localhost:50051")
caps = client.stub.GetServerInfo(pb.GetServerInfoRequest()).rl_capabilities
assert caps.weight_transfer_enabled, "start vLLM with --weight-transfer-config"
client.stub.PauseGeneration(pb.PauseGenerationRequest(mode=pb.PAUSE_MODE_KEEP))
# ... engine.send_weights() via WeightTransferTrainerFactory(client=client) ...
client.stub.ResumeGeneration(pb.ResumeGenerationRequest())
```

## Limitations

- **No `ShutdownWeightTransferEngine` RPC.** To release the trainer connection without stopping inference, use HTTP `POST /shutdown_weight_transfer_engine`, which needs `VLLM_SERVER_DEV_MODE=1`. `Control.Shutdown` stops the whole engine.
- **`update_info_json` must be a JSON object.** A per-worker list payload, which HTTP and the Python API accept, is rejected with `INVALID_ARGUMENT`.
- **Rust frontend only.** The Python frontend and the `--grpc` server do not have the `vllm.Control` service.
