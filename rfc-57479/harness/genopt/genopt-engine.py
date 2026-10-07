"""Experiment-only engine protocol peer; no model, scheduler or coordinator."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

spec = importlib.util.spec_from_file_location(
    "payload", Path(__file__).with_name("frontend-mock-payload.py")
)
payload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payload)


async def peer(args, group):
    pool = payload.load_token_pool(args.token_pool)
    ctx = zmq.asyncio.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, bytes.fromhex(args.engine_identity))
    dealer.connect(args.input_address)
    push = ctx.socket(zmq.PUSH)
    push.connect(args.output_address)
    ready = EngineCoreReadyResponse(
        max_model_len=1048576,
        num_gpu_blocks=0,
        block_size=16,
        dp_stats_address=None,
        dtype="bfloat16",
        vllm_version="mock-0fd2e8d",
        world_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_rank=0,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        instance_id="frontend-mock",
        supports_lora=False,
        max_loras=0,
    )
    await dealer.send(msgspec.msgpack.encode(ready))
    encoder = MsgpackEncoder()
    request_decoder = MsgpackDecoder(EngineCoreRequest)
    utility_decoder = MsgpackDecoder()
    requests = {}
    tasks = {}
    failed = asyncio.get_running_loop().create_future()

    def produced(task):
        if not task.cancelled() and task.exception() and not failed.done():
            failed.set_exception(task.exception())

    async def send_own(batch):
        await push.send_multipart(encoder.encode(batch), copy=False)

    group["senders"][args.peer_index] = send_own

    def route(client_index):
        """--route-by-client-index: like EngineCoreProc.process_output_sockets,
        send to output socket number client_index (endpoint order) instead of
        the connection the request/utility arrived on (default, unchanged)."""
        if not args.route_by_client_index:
            return send_own
        if not 0 <= client_index < len(group["senders"]):
            raise IndexError(f"client_index {client_index} has no output socket")
        return group["senders"][client_index]

    send = send_own

    async def produce(request):
        await group["admitted"].wait()
        for start in range(0, args.output_tokens, args.chunk_size):
            count = min(args.chunk_size, args.output_tokens - start)
            output = payload.make_engine_output(
                request.request_id, pool, start, count,
                routed_layers=args.routed_experts_layers, prompt_tokens=args.input_tokens,
            )
            await route(request.client_index)(EngineCoreOutputs(outputs=[output]))
            await asyncio.sleep(0)
        print(
            json.dumps(
                {
                    "event": "sent",
                    "request": request.request_id,
                    "tokens": args.output_tokens,
                }
            ),
            flush=True,
        )
        # Remain in requests until abort; never emit a length/stop finish.

    async def abort(ids):
        by_client = {}
        for request_id in ids:
            task = tasks.pop(request_id, None)
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            request = requests.pop(request_id, None)
            if request is not None:
                by_client.setdefault(request.client_index, []).append(
                    EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ABORT,
                    )
                )
        for client_index, outputs in by_client.items():
            await route(client_index)(
                EngineCoreOutputs(
                    outputs=outputs,
                    finished_requests={output.request_id for output in outputs},
                )
            )

    group["abort"].append(lambda: abort(list(requests)))

    try:
        while True:
            received = asyncio.ensure_future(dealer.recv_multipart())
            done, _ = await asyncio.wait(
                (received, failed), return_when=asyncio.FIRST_COMPLETED
            )
            if failed in done:
                received.cancel()
                await asyncio.gather(received, return_exceptions=True)
                await failed
            frames = received.result()
            kind, data = frames[0], frames[1:]
            if kind == EngineCoreRequestType.ADD.value:
                request = request_decoder.decode(data)
                if group["paused"]:
                    raise RuntimeError("Received a request while mock is paused")
                if request.request_id in requests:
                    raise RuntimeError("Duplicate request ID")
                if len(request.prompt_token_ids or []) != args.input_tokens:
                    raise RuntimeError("Input token count differs from experiment")
                sp = request.sampling_params
                eos_disabled = sp is not None and (
                    sp.ignore_eos
                    or (sp._eos_token_id is None and not sp.stop_token_ids)
                )
                if (
                    sp is None
                    or sp.logprobs != 128
                    or sp.prompt_logprobs is not None
                    or not eos_disabled
                    or sp.stop
                    or sp.stop_token_ids
                    or sp.max_tokens <= args.output_tokens
                ):
                    raise RuntimeError(
                        f"Sampling parameters differ from experiment: {sp!r}"
                    )
                requests[request.request_id] = request
                group["count"] += 1
                print(
                    json.dumps(
                        {
                            "event": "admitted",
                            "request": request.request_id,
                            "input": args.input_address,
                            "total": group["count"],
                        }
                    ),
                    flush=True,
                )
                if group["count"] >= args.expected_requests:
                    group["admitted"].set()
                task = asyncio.create_task(produce(request))
                task.add_done_callback(produced)
                tasks[request.request_id] = task
            elif kind == EngineCoreRequestType.ABORT.value:
                await abort(utility_decoder.decode(data))
            elif kind == EngineCoreRequestType.UTILITY.value:
                client_index, call_id, method, method_args = utility_decoder.decode(
                    data
                )
                result = None
                failure = None
                if method == "pause_scheduler":
                    if method_args[0] != "abort":
                        raise RuntimeError("Only abort pause is implemented")
                    group["paused"] = True
                    await asyncio.gather(*(abort_all() for abort_all in group["abort"]))
                elif method == "resume_scheduler":
                    group["paused"] = False
                elif method == "is_scheduler_paused":
                    result = group["paused"]
                elif method == "get_supported_tasks":
                    result = ("generate",)
                elif method not in ("reset_mm_cache", "reset_prefix_cache"):
                    failure = f"Unsupported mock utility: {method}"
                await route(client_index)(
                    EngineCoreOutputs(
                        utility_output=UtilityOutput(
                            call_id=call_id,
                            failure_message=failure,
                            result=UtilityResult(result) if failure is None else None,
                        )
                    )
                )
                print(
                    json.dumps(
                        {"event": "utility", "method": method, "client": client_index}
                    ),
                    flush=True,
                )
            else:
                raise RuntimeError(f"Unsupported mock request type: {kind!r}")
    finally:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        dealer.close(linger=0)
        push.close(linger=0)
        ctx.term()


async def run(args):
    group = {"paused": False, "count": 0, "abort": [], "admitted": asyncio.Event()}
    endpoints = (
        json.loads(args.endpoints_json.read_text())
        if args.endpoints_json
        else [{"input": args.input_address, "output": args.output_address}]
    )
    group["senders"] = [None] * len(endpoints)
    peers = []
    for endpoint in endpoints:
        config = argparse.Namespace(**vars(args))
        config.input_address = endpoint["input"]
        config.output_address = endpoint["output"]
        config.peer_index = len(peers)
        peers.append(asyncio.create_task(peer(config, group)))
    try:
        await asyncio.gather(*peers)
    finally:
        for task in peers:
            task.cancel()
        await asyncio.gather(*peers, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-address")
    parser.add_argument("--output-address")
    parser.add_argument("--endpoints-json", type=Path)
    parser.add_argument("--expected-requests", type=int, default=1)
    parser.add_argument("--engine-identity", default="0000")
    parser.add_argument("--token-pool", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=245760)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--routed-experts-layers", type=int, default=0)
    parser.add_argument(
        "--route-by-client-index",
        action="store_true",
        help="opt-in: route outputs/utility replies by request client_index "
        "(real EngineCoreProc behavior); default: by arriving connection",
    )
    asyncio.run(run(parser.parse_args()))
