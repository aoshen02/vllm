"""Micro-benchmark: one /inference/v1/generate request through the real
OutputProcessor and ServingTokens.serve_tokens_full_generator.

Engine rows come from frozen-r1 frontend-mock-payload.py make_chunk (1024
positions x 129 slots, float32), fed as EngineCoreOutput.new_logprobs; then
the request is aborted (like pause(mode=abort)) and the final response is
built and rendered exactly as the api_router would.

Works against the base code (no logprobs_format: legacy path) and the
optimized worktree (select with PYTHONPATH). Excluded: msgpack decode, HTTP.
"""

import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import argparse
import asyncio
import gc
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

FROZEN = Path(
    GENOPT_ROOT + "/agent_run/results/"
    "frontend-mock-256k-20261004/frozen-r1"
)


def status_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise KeyError(field)


def load_payload():
    spec = importlib.util.spec_from_file_location(
        "payload", FROZEN / "scripts" / "frontend-mock-payload.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format", choices=["openai", "compact", "none"], default="openai"
    )
    parser.add_argument(
        "--routed-layers",
        type=int,
        default=0,
        help="R3: routed_experts rows of (layers, topk) uint8; 0 = off. The "
        "first output carries the prompt rows (input tokens) plus count-1 "
        "rows, later outputs one row per token: P+N-1 rows in total.",
    )
    parser.add_argument("--routed-topk", type=int, default=8)
    parser.add_argument(
        "--compact-sampled", choices=["true", "false"], default="true"
    )
    parser.add_argument("--compact-ranks", choices=["true", "false"], default="true")
    parser.add_argument("--positions", type=int, default=245760)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--input-tokens", type=int, default=16384)
    parser.add_argument("--no-tokenizer", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-body", type=Path)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument(
        "--engine-int32",
        action="store_true",
        help="cast token ids to int32 like the real sampler (mock gives int64)",
    )
    args = parser.parse_args()

    import numpy as np
    from fastapi.responses import JSONResponse

    import vllm
    from vllm.entrypoints.scale_out.token_in_token_out import serving as serving_mod
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
        GenerateRequest,
    )
    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
    from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
    from vllm.sampling_params import RequestOutputKind
    from vllm.tokenizers import get_tokenizer
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
    from vllm.v1.engine.output_processor import (
        OutputProcessor,
        RequestOutputCollector,
    )
    from vllm.v1.outputs import LogprobsLists

    payload = load_payload()
    pool = payload.load_token_pool(FROZEN / "model-assets" / "token-pool-v2.json")
    tokenizer = (
        None
        if args.no_tokenizer
        else get_tokenizer(str(FROZEN / "model-assets"), tokenizer_mode="auto")
    )
    supports_format = "logprobs_format" in GenerateRequest.model_fields
    if args.format == "compact" and not supports_format:
        raise SystemExit("compact needs the optimized code")

    body = {
        "request_id": "bench",
        "token_ids": list(range(1, args.input_tokens + 1)),
        "sampling_params": {
            "logprobs": None if args.format == "none" else args.top_k,
            "max_tokens": args.positions + 1,
            "ignore_eos": True,
        },
    }
    if supports_format:
        body["logprobs_format"] = "openai" if args.format == "none" else args.format
        if args.format == "compact":
            if args.compact_sampled == "false":
                body["compact_include_sampled"] = False
            if args.compact_ranks == "false":
                body["compact_include_ranks"] = False
    request = GenerateRequest.model_validate(body)

    # Same sampling-param mutations as ServingTokens.serve_tokens.
    sp = request.sampling_params
    sp.output_kind = RequestOutputKind.FINAL_ONLY
    if hasattr(ServingTokens, "_configure_logprobs"):
        ServingTokens._configure_logprobs(request, sp)
    elif hasattr(ServingTokens, "_use_array_logprobs"):  # round-1 commits
        sp.array_logprobs = ServingTokens._use_array_logprobs(request)
        if sp.array_logprobs and not sp.stop:
            sp.detokenize = False
    elif hasattr(serving_mod, "ARRAY_LOGPROBS_CONTAINER"):  # PR3 (r23): as serve_tokens
        import vllm.envs as envs
        from vllm.logprobs import set_sample_logprobs_container

        if envs.VLLM_GENERATE_ARRAY_LOGPROBS and sp.logprobs is not None:
            set_sample_logprobs_container(sp, serving_mod.ARRAY_LOGPROBS_CONTAINER)

    processor = OutputProcessor(tokenizer, log_stats=False)
    engine_request = EngineCoreRequest(
        request_id="bench-int",
        external_req_id="bench",
        prompt_token_ids=body["token_ids"],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=sp,
        pooling_params=None,
    )
    queue = RequestOutputCollector(sp.output_kind, "bench")
    processor.add_request(engine_request, None, queue=queue)
    state = processor.request_states["bench-int"]

    rss_start = status_kb("VmRSS")
    consume_s = 0.0
    samples = []
    t_wall = time.perf_counter()
    for start in range(0, args.positions, args.chunk):
        count = min(args.chunk, args.positions - start)
        sampled, token_ids, logprobs, ranks = payload.make_chunk(
            pool, start, count, args.top_k
        )
        if args.engine_int32:
            token_ids = token_ids.astype(np.int32)
        routed = None
        if args.routed_layers:
            rows = count + (args.input_tokens - 1 if start == 0 else 0)
            routed = np.random.default_rng(start).integers(
                0, 256, size=(rows, args.routed_layers, args.routed_topk), dtype=np.uint8
            )
        output = EngineCoreOutput(
            routed_experts=routed,
            request_id="bench-int",
            new_token_ids=sampled,
            new_logprobs=(
                None
                if args.format == "none"
                else LogprobsLists(token_ids, logprobs, ranks)
            ),
        )
        t0 = time.perf_counter()
        processor.process_outputs([output])
        consume_s += time.perf_counter() - t0
        done = start + count
        if (done // args.chunk) % 16 == 0 or done == args.positions:
            samples.append(
                {
                    "positions": done,
                    "consume_s": round(consume_s, 4),
                    "rss_kb": status_kb("VmRSS"),
                    # Same expressions as the orchestrator's observer.
                    "detok_tokens": state.detokenizer.num_output_tokens(),
                    "logprob_positions": (
                        args.positions
                        if args.format == "none"
                        else len(state.logprobs_processor.logprobs)
                    ),
                }
            )
    wall_consume = time.perf_counter() - t_wall
    rss_after_consume = status_kb("VmRSS")
    assert samples[-1]["logprob_positions"] == args.positions

    # pause(mode=abort): the engine aborts; the frontend emits the final output.
    t0 = time.perf_counter()
    processor.abort_requests(["bench-int"], internal=True)
    final_res = queue.get_nowait()
    abort_s = time.perf_counter() - t0

    # Stage timers around the logprob builders/renderers.
    stage = {}

    def timed(name, fn):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                stage[name] = stage.get(name, 0.0) + time.perf_counter() - t

        return wrapper

    ServingTokens._create_tokens_logprobs = timed(
        "legacy_create_tokens_logprobs", ServingTokens._create_tokens_logprobs
    )
    for name in (
        "render_openai_logprobs",
        "render_openai_logprobs_parts",
        "render_compact_logprobs",
        "render_compact_logprobs_parts",
        "render_json_with_fragments",
        "numpy2base64",
        "routed_experts_parts",
    ):
        if hasattr(serving_mod, name):
            setattr(serving_mod, name, timed(name, getattr(serving_mod, name)))

    handler = object.__new__(ServingTokens)
    handler.enable_log_outputs = False
    handler.request_logger = None
    handler.enable_prompt_tokens_details = False

    async def results():
        yield final_res

    time_time = time.time
    time.time = lambda: 1700000000.0  # deterministic "created"
    t0 = time.perf_counter()
    response = asyncio.run(
        handler.serve_tokens_full_generator(
            request,
            results(),
            "generate-tokens-bench",
            "mock-model",
            RequestResponseMetadata(request_id="generate-tokens-bench"),
        )
    )
    build_s = time.perf_counter() - t0
    time.time = time_time
    rss_after_build = status_kb("VmRSS")

    # Final render as in api_router.generate.
    t0 = time.perf_counter()
    if hasattr(response, "parts"):
        # Router sends the parts as-is (no join); nothing left to render.
        out = None
        render_kind = "RenderedGenerateResponse.parts (sent unjoined)"
        render_s_parts = time.perf_counter() - t0
        digest = hashlib.sha256()
        for part in response.parts:
            digest.update(part)
        body_sha = digest.hexdigest()
        body_len = response.content_length
    elif hasattr(response, "body") and not hasattr(response, "model_dump"):
        out = response.body
        render_kind = "RenderedGenerateResponse.body"
    else:
        dumped = response.model_dump()
        stage["model_dump"] = time.perf_counter() - t0
        out = JSONResponse(content=dumped).body
        del dumped
        render_kind = "JSONResponse(model_dump())"
    render_s = time.perf_counter() - t0
    if out is not None:
        body_sha = hashlib.sha256(out).hexdigest()
        body_len = len(out)
    else:
        render_s = render_s_parts
    del response
    gc.collect()

    result = {
        "agent": "claude-python-subagent",
        "vllm_file": vllm.__file__,
        "git_commit": args.git_commit,
        "routed_layers": args.routed_layers,
        "routed_topk": args.routed_topk,
        "compact_include_sampled": args.compact_sampled,
        "compact_include_ranks": args.compact_ranks,
        "engine_int32": args.engine_int32,
        "format": args.format,
        "supports_logprobs_format": supports_format,
        "array_logprobs": getattr(sp, "array_logprobs", None),
        "detokenize": sp.detokenize,
        "tokenizer": None if tokenizer is None else type(tokenizer).__name__,
        "positions": args.positions,
        "chunk": args.chunk,
        "slots": args.top_k + 1,
        "consume_s": round(consume_s, 3),
        "consume_us_per_position": round(consume_s / args.positions * 1e6, 3),
        "consume_wall_incl_chunk_generation_s": round(wall_consume, 3),
        "abort_s": round(abort_s, 4),
        "build_s": round(build_s, 3),
        "render_s": round(render_s, 3),
        "build_plus_render_s": round(build_s + render_s, 3),
        "render_kind": render_kind,
        "stage_s": {k: round(v, 3) for k, v in stage.items()},
        "body_bytes": body_len,
        "body_sha256": body_sha,
        "rss_start_kb": rss_start,
        "rss_after_consume_kb": rss_after_consume,
        "rss_after_build_kb": rss_after_build,
        "vm_hwm_kb": status_kb("VmHWM"),
        "samples": samples,
        "pid": os.getpid(),
    }
    if args.keep_body and out is not None:
        args.keep_body.write_bytes(out)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))


if __name__ == "__main__":
    main()
