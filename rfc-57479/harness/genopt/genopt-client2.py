"""Client v2 for /inference/v1/generate: multi-process, parse-on-arrival.

Why v2: client v1 (one asyncio receiver, all bodies buffered, then 4 fork
parsers) cannot hold the default-format 256-request cohort (563 GB Python,
873 GB Rust) and its single receiver caps throughput at ~0.8 GB/s. v2 keeps the
same request bodies, barrier, pause and validation logic (imported from
genopt-client.py) but:
  * P worker processes (default 8, all within the client's 8 cores); request i
    belongs to worker i % P; every request is in flight before the barrier;
  * each worker reads at most --max-reading bodies at once (bounded memory);
    a body is parsed and validated as soon as it is fully read, then dropped;
  * per request it reports: received (body fully read) and parsed (parse +
    validation done), both CLOCK_MONOTONIC on the client node (system-wide, so
    comparable across the worker processes and the coordinating main process).
Reported cohort metrics: all_bodies_received_s = max(received) - pause start;
all_parsed_s = max(parsed) - pause start. Because parsing overlaps receiving,
these overlap and must not be added.
"""

import argparse
import asyncio
import importlib.util
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

import httpx
import numpy as np

spec = importlib.util.spec_from_file_location(
    "client1", Path(__file__).with_name("genopt-client.py")
)
client1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client1)


class Reply:
    """Minimal stand-in for an httpx response, for client1.parse_and_validate."""

    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}: {self.content[:300]!r}")


def worker(index, args, indices, out, go):
    client1.CONFIG = args
    pool = json.loads(args.token_pool.read_text())
    client1.POOL_IDS = np.asarray([e["id"] for e in pool], dtype=np.int64)
    prompt = [int(client1.POOL_IDS[i % len(pool)]) for i in range(args.input_tokens)]
    body = {
        "model": "hy4-mock",
        "token_ids": prompt,
        "stream": False,
        "sampling_params": {
            "max_tokens": args.output_tokens + 1,
            "logprobs": client1.TOP_K,
            "ignore_eos": True,
        },
    }
    if args.logprobs_format:
        body["logprobs_format"] = args.logprobs_format
    if not args.compact_include_sampled:
        body["compact_include_sampled"] = False
    if not args.compact_include_ranks:
        body["compact_include_ranks"] = False
    payload = json.dumps(body)[:-1]

    async def run():
        limits = httpx.Limits(
            max_connections=len(indices) + 2,
            max_keepalive_connections=len(indices) + 2,
        )
        reading = asyncio.Semaphore(args.max_reading)
        async with httpx.AsyncClient(
            timeout=args.http_timeout, limits=limits, trust_env=False
        ) as client:

            async def one(i):
                content = payload + f', "request_id": "mock-{i:04d}"}}'
                url = args.urls[i % len(args.urls)] + "/inference/v1/generate"
                async with client.stream(
                    "POST",
                    url,
                    content=content,
                    headers={"content-type": "application/json"},
                ) as response:
                    headers_t = time.monotonic()
                    async with reading:
                        data = await response.aread()
                    received_t = time.monotonic()
                    if not go.is_set():
                        out.put({"index": i, "early": True, "status": response.status_code,
                                 "body": data[:500].decode(errors="replace")})
                        return
                    record = client1.parse_and_validate_reply(
                        i, Reply(response.status_code, data)
                    )
                    del data
                    record.update(
                        index=i,
                        headers_t=headers_t,
                        received_t=received_t,
                        parsed_t=time.monotonic(),
                        worker=index,
                    )
                    out.put(record)

            async def guarded(i):
                try:
                    await one(i)
                except Exception as error:  # noqa: BLE001
                    out.put({"index": i, "error": f"{type(error).__name__}: {error}"})

            await asyncio.gather(*(guarded(i) for i in indices))

    asyncio.run(run())


def main(args):
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    go = ctx.Event()
    workers = [
        ctx.Process(
            target=worker,
            args=(w, args, list(range(w, args.requests, args.workers)), out, go),
            daemon=True,
        )
        for w in range(args.workers)
    ]
    for p in workers:
        p.start()
    records = {}

    def drain(block_s=0.0):
        try:
            while True:
                r = out.get(timeout=block_s) if block_s else out.get_nowait()
                block_s = 0.0
                if r.get("early") or r.get("error"):
                    raise RuntimeError(f"Request failed or returned early: {r}")
                records[r["index"]] = r
        except queue.Empty:
            pass

    try:
        deadline = time.monotonic() + args.barrier_timeout
        while True:
            drain()
            if records:
                raise RuntimeError("A request returned before the consumption barrier")
            for p in workers:
                if not p.is_alive():
                    raise RuntimeError(f"client worker exited early: {p.exitcode}")
            consumed = []
            for path in args.consumption_log.rglob("consumed-*.jsonl"):
                for line in path.read_text().splitlines(keepends=True):
                    if line.endswith("\n"):
                        rec = json.loads(line)
                        if rec.get("event", "consumed") == "consumed":
                            consumed.append(rec)
            if len(consumed) == args.requests:
                if len({r["request_id"] for r in consumed}) != args.requests:
                    raise RuntimeError("Duplicate barrier request IDs")
                if any(
                    r["tokens"] != args.output_tokens
                    or r["logprob_positions"] != args.output_tokens
                    for r in consumed
                ):
                    raise RuntimeError("Barrier counts differ from target")
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Frontend consumption barrier not reached: {len(consumed)}")
            time.sleep(0.05)
        barrier_wall = time.time()
        go.set()
        started = time.monotonic()
        started_wall = time.time()
        args.result.with_name("pause-start.json").write_text(
            json.dumps({"monotonic": started, "wall": started_wall, "barrier_wall": barrier_wall})
        )
        pause = httpx.post(args.urls[0] + "/pause?mode=abort", timeout=args.http_timeout,
                           trust_env=False)
        pause_elapsed = time.monotonic() - started
        pause.raise_for_status()
        while len(records) < args.requests:
            drain(block_s=1.0)
            if len(records) < args.requests and not any(p.is_alive() for p in workers):
                drain()
                if len(records) < args.requests:
                    raise RuntimeError(f"workers exited with {len(records)} records")
        rows = [records[i] for i in range(args.requests)]
        report = {
            "client": "v2-multiprocess-parse-on-arrival",
            "workers": args.workers,
            "max_reading_per_worker": args.max_reading,
            "endpoint": "/inference/v1/generate",
            "logprobs_format": args.logprobs_format or "openai",
            "requests": args.requests,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "top_logprobs": client1.TOP_K,
            "pause_s": pause_elapsed,
            "all_headers_s": max(r["headers_t"] for r in rows) - started,
            "all_bodies_received_s": max(r["received_t"] for r in rows) - started,
            "all_parsed_s": max(r["parsed_t"] for r in rows) - started,
            "post_receive_parse_cpu_s_sum": sum(r["parse_cpu_s"] for r in rows),
            "post_receive_parse_s_sum": sum(r["parse_s"] for r in rows),
            "post_receive_validate_s_sum": sum(r["validate_s"] for r in rows),
            "semantically_validated_requests": sum(r["semantically_validated"] for r in rows),
            "top_entries_per_position": sorted(
                {v for r in rows for v in r["top_entries_per_position"]}
            ),
            "response_bytes": [r["response_bytes"] for r in rows],
            "response_sha256": [r["sha256"] for r in rows],
            "per_request": [
                {k: r[k] for k in ("index", "worker", "headers_t", "received_t", "parsed_t")}
                for r in rows
            ],
            "pause_monotonic": started,
            "status": "PASS",
            "note": (
                "Parse-on-arrival: receive and parse overlap; all_bodies_received_s and "
                "all_parsed_s are separate cohort end points, never add them. Full "
                "semantic validation for indices < validate_requests; structural checks "
                "for the rest."
            ),
        }
        args.result.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("response_sha256", "response_bytes", "per_request")}),
              flush=True)
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-reading", type=int, default=4)
    parser.add_argument("--barrier-timeout", type=float, default=3600)
    parser.add_argument("--http-timeout", type=float, default=7200)
    parser.add_argument("--validate-requests", type=int, default=1 << 30)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--logprobs-format", choices=("openai", "compact"))
    parser.add_argument("--compact-no-sampled", dest="compact_include_sampled", action="store_false")
    parser.add_argument("--compact-no-ranks", dest="compact_include_ranks", action="store_false")
    parser.add_argument("--routed-experts-layers", type=int, default=0)
    parser.add_argument("--token-pool", type=Path, required=True)
    parser.add_argument("--consumption-log", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    main(parser.parse_args())
