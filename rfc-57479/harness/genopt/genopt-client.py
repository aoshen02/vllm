"""Timing client for /inference/v1/generate under one global pause(mode=abort).

Derived from the frozen Codex frontend-mock-client.py. Changes: generate
endpoint request/response shapes, optional logprobs_format, vectorized
validation of the compact format, and an explicit validation subset knob.
Timing of pause / receive is identical in structure to the frozen client.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import numpy as np

REPLIES = None
CONFIG = None
POOL_IDS = None
TOP_K = 128


def expected_scores():
    scores = -np.arange(TOP_K, dtype=np.float64) * 0.037
    return (scores - np.log(np.exp(scores).sum())).astype(np.float32)


def expected_arrays(n):
    """Engine-slot layout produced by frontend-mock-payload.make_chunk."""
    pool = POOL_IDS
    positions = np.arange(n, dtype=np.int64)
    cand = pool[(positions[:, None] + np.arange(TOP_K)) % len(pool)]
    sampled_idx = positions % TOP_K
    sampled = cand[positions, sampled_idx]
    scores = expected_scores()
    ids = np.column_stack((sampled, cand)).astype(np.int32)
    lps = np.column_stack(
        (scores[sampled_idx], np.broadcast_to(scores, (n, TOP_K)))
    ).astype(np.float32)
    ranks = (sampled_idx + 1).astype(np.int32)
    return ids, lps, ranks


def b64_array(text, dtype, count):
    raw = base64.b64decode(text, validate=True)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
    if arr.size != count:
        raise AssertionError(f"compact array size {arr.size} != {count}")
    return arr


def include_sampled():
    return getattr(CONFIG, "compact_include_sampled", True)


def include_ranks():
    return getattr(CONFIG, "compact_include_ranks", True)


def compact_arrays(block, n):
    """Decode a compact block (shared by parse and validation)."""
    slots = block["num_slots"]
    ids = b64_array(block["token_ids"], block["dtype_token_ids"], n * slots)
    lps = b64_array(block["logprobs"], block["dtype_logprobs"], n * slots)
    ranks = b64_array(block["ranks"], "int32", n) if "ranks" in block else None
    return slots, ids, lps, ranks


def validate_compact(choice, n):
    block = choice["compact_logprobs"]
    assert choice.get("logprobs") is None
    assert block["num_positions"] == n
    assert block["byteorder"] == "little"
    slots, ids, lps, ranks = compact_arrays(block, n)
    exp_ids, exp_lps, exp_ranks = expected_arrays(n)
    if include_sampled():
        assert slots == TOP_K + 1, slots
        assert "sampled_slot" not in block
    else:
        assert slots == TOP_K, slots
        assert block["sampled_slot"] is False
        exp_ids, exp_lps = exp_ids[:, 1:], exp_lps[:, 1:]
    assert np.array_equal(ids.reshape(n, slots), exp_ids)
    # Raw engine float32 must be bit-preserved in compact mode.
    assert np.array_equal(
        lps.reshape(n, slots).view(np.uint32),
        np.ascontiguousarray(exp_lps).view(np.uint32),
    )
    if include_ranks():
        assert np.array_equal(ranks, exp_ranks)
    else:
        assert ranks is None and "ranks" not in block
    positions = np.arange(n)
    return POOL_IDS[(positions + positions % TOP_K) % len(POOL_IDS)]


def expected_routed(rows, layers, topk=8):
    """Must match frontend-mock-payload.make_routed."""
    t = np.arange(rows, dtype=np.int64)[:, None, None]
    lyr = np.arange(layers, dtype=np.int64)[None, :, None]
    j = np.arange(topk, dtype=np.int64)[None, None, :]
    return ((t * 7 + lyr * 13 + j * 31) % 256).astype(np.uint8)


def decode_routed(text, rows, layers):
    import io

    arr = np.load(io.BytesIO(base64.b64decode(text, validate=True)), allow_pickle=False)
    assert arr.dtype == np.uint8, arr.dtype
    assert arr.shape == (rows, layers, 8), arr.shape
    return arr


def validate_openai(choice, n, stats):
    content = choice["logprobs"]["content"]
    assert len(content) == n
    scores = expected_scores()
    pool = POOL_IDS
    period = len(pool)
    templates = []
    for position in range(min(period, n)):
        row = content[position]
        sampled = int(pool[(position + position % TOP_K) % period])
        assert row["token"] == f"token_id:{sampled}"
        assert math.isclose(
            row["logprob"], float(scores[position % TOP_K]), rel_tol=1e-6, abs_tol=1e-7
        )
        expected = {
            f"token_id:{int(pool[(position + r) % period])}": float(scores[r])
            for r in range(TOP_K)
        }
        tops = row["top_logprobs"]
        stats.setdefault("top_entries_per_position", set()).add(len(tops))
        got = {}
        for entry in tops:
            got.setdefault(entry["token"], entry["logprob"])
        assert got.keys() == expected.keys(), (position, len(got))
        assert all(
            math.isclose(v, expected[t], rel_tol=1e-6, abs_tol=1e-7)
            for t, v in got.items()
        )
        templates.append(row)
    for position in range(period, n):
        assert content[position] == templates[position % period], position
    positions = np.arange(n)
    return pool[(positions + positions % TOP_K) % period]


def parse_and_validate(index):
    _, response = REPLIES[index]
    return parse_and_validate_reply(index, response)


def parse_and_validate_reply(index, response):
    response.raise_for_status()
    body = response.content
    started = time.monotonic()
    cpu_started = time.process_time()
    data = json.loads(body)
    choice = data["choices"][0]
    if CONFIG.logprobs_format == "compact":
        block = choice["compact_logprobs"]
        arrays = compact_arrays(block, block["num_positions"])
        del arrays
    routed_layers = getattr(CONFIG, "routed_experts_layers", 0)
    routed_rows = CONFIG.input_tokens + CONFIG.output_tokens - 1
    routed = (
        decode_routed(choice["routed_experts"], routed_rows, routed_layers)
        if routed_layers
        else None
    )
    parse_s = time.monotonic() - started
    parse_cpu_s = time.process_time() - cpu_started
    validated = index < CONFIG.validate_requests
    started = time.monotonic()
    stats = {}
    n = CONFIG.output_tokens
    assert data["request_id"] in (f"mock-{index:04d}", f"generate-tokens-mock-{index:04d}"), data["request_id"]
    assert len(data["choices"]) == 1
    assert choice["finish_reason"] == "abort", choice["finish_reason"]
    assert len(choice["token_ids"]) == n
    if routed_layers:
        if validated:
            assert np.array_equal(routed, expected_routed(routed_rows, routed_layers))
    else:
        assert choice.get("routed_experts") is None
    del routed
    if validated:
        if CONFIG.logprobs_format == "compact":
            sampled = validate_compact(choice, n)
        else:
            sampled = validate_openai(choice, n, stats)
        assert np.array_equal(np.asarray(choice["token_ids"]), sampled)
    validate_s = time.monotonic() - started
    return {
        "parse_s": parse_s,
        "parse_cpu_s": parse_cpu_s,
        "validate_s": validate_s,
        "semantically_validated": validated,
        "top_entries_per_position": sorted(stats.get("top_entries_per_position", [])),
        "response_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


async def run(args):
    global REPLIES, CONFIG, POOL_IDS
    pool = json.loads(args.token_pool.read_text())
    POOL_IDS = np.asarray([entry["id"] for entry in pool], dtype=np.int64)
    prompt = [int(POOL_IDS[i % len(POOL_IDS)]) for i in range(args.input_tokens)]
    body = {
        "model": "hy4-mock",
        "token_ids": prompt,
        "stream": False,
        "sampling_params": {
            "max_tokens": args.output_tokens + 1,
            "logprobs": TOP_K,
            "ignore_eos": True,
        },
    }
    if args.logprobs_format:
        body["logprobs_format"] = args.logprobs_format
    if not args.compact_include_sampled:
        body["compact_include_sampled"] = False
    if not args.compact_include_ranks:
        body["compact_include_ranks"] = False
    payload = json.dumps(body)[:-1]  # reuse the encoded prompt; append id below
    limits = httpx.Limits(
        max_connections=args.requests + 2, max_keepalive_connections=args.requests + 2
    )
    async with httpx.AsyncClient(
        timeout=args.http_timeout, limits=limits, trust_env=False
    ) as client:
        urls = args.urls

        async def request(index):
            content = payload + f', "request_id": "mock-{index:04d}"}}'
            response = await client.post(
                urls[index % len(urls)] + "/inference/v1/generate",
                content=content,
                headers={"content-type": "application/json"},
            )
            return time.monotonic(), response

        tasks = [asyncio.create_task(request(i)) for i in range(args.requests)]
        try:
            deadline = time.monotonic() + args.barrier_timeout
            while True:
                done = [task for task in tasks if task.done()]
                if done:
                    detail = ""
                    try:
                        _, early = done[0].result()
                        detail = f" status={early.status_code} body={early.text[:500]}"
                    except Exception as error:  # noqa: BLE001
                        detail = f" error={error!r}"
                    raise RuntimeError(
                        "A request returned before the consumption barrier" + detail
                    )
                records = []
                for path in args.consumption_log.rglob("consumed-*.jsonl"):
                    for line in path.read_text().splitlines(keepends=True):
                        if not line.endswith("\n"):
                            continue
                        record = json.loads(line)
                        if record.get("event", "consumed") == "consumed":
                            records.append(record)
                if len(records) == args.requests:
                    if len({r["request_id"] for r in records}) != args.requests:
                        raise RuntimeError("Duplicate barrier request IDs")
                    if any(
                        r["tokens"] != args.output_tokens
                        or r["logprob_positions"] != args.output_tokens
                        for r in records
                    ):
                        raise RuntimeError("Barrier counts differ from target")
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Frontend consumption barrier not reached: {len(records)}"
                    )
                await asyncio.sleep(0.05)
            barrier_wall = time.time()
            started = time.monotonic()
            started_wall = time.time()
            args.result.with_name("pause-start.json").write_text(
                json.dumps(
                    {"monotonic": started, "wall": started_wall, "barrier_wall": barrier_wall}
                )
            )
            pause = await client.post(urls[0] + "/pause?mode=abort")
            pause_elapsed = time.monotonic() - started
            pause.raise_for_status()
            replies = await asyncio.gather(*tasks)
            received_elapsed = max(t for t, _ in replies) - started
            received = {
                "status": "RECEIVED_UNVALIDATED",
                "pause_s": pause_elapsed,
                "all_bodies_received_s": received_elapsed,
                "requests": args.requests,
                "response_bytes": [len(r.content) for _, r in replies],
                "http_statuses": [r.status_code for _, r in replies],
                "body_received_s": [t - started for t, _ in replies],
            }
            args.result.with_name("received.json").write_text(
                json.dumps(received, indent=2)
            )
            REPLIES, CONFIG = replies, args
            parse_started = time.monotonic()
            with ProcessPoolExecutor(
                max_workers=args.parse_workers,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                checks = list(executor.map(parse_and_validate, range(args.requests)))
            parse_wall = time.monotonic() - parse_started
            report = {
                "endpoint": "/inference/v1/generate",
                "logprobs_format": args.logprobs_format or "openai",
                "requests": args.requests,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "top_logprobs": TOP_K,
                "pause_s": pause_elapsed,
                "all_bodies_received_s": received_elapsed,
                "post_receive_parse_validate_wall_s": parse_wall,
                "receive_plus_parse_validate_s": received_elapsed + parse_wall,
                "post_receive_parse_s_sum": sum(r["parse_s"] for r in checks),
                "post_receive_parse_cpu_s_sum": sum(r["parse_cpu_s"] for r in checks),
                "post_receive_validate_s_sum": sum(r["validate_s"] for r in checks),
                "semantically_validated_requests": sum(
                    r["semantically_validated"] for r in checks
                ),
                "top_entries_per_position": sorted(
                    {v for r in checks for v in r["top_entries_per_position"]}
                ),
                "parse_workers": args.parse_workers,
                "response_bytes": [r["response_bytes"] for r in checks],
                "response_sha256": [r["sha256"] for r in checks],
                "status": "PASS",
                "note": (
                    "All bodies buffered before parsing; every body is JSON-parsed "
                    "(and compact arrays base64-decoded). Full per-position semantic "
                    "validation only for the first validate_requests requests; others "
                    "get structural checks (id, finish=abort, token count). Parse/"
                    "validate per-request durations are sums, not cohort elapsed."
                ),
            }
            args.result.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items() if "sha" not in k and k != "response_bytes"}), flush=True)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--barrier-timeout", type=float, default=3600)
    parser.add_argument("--http-timeout", type=float, default=7200)
    parser.add_argument("--parse-workers", type=int, default=4)
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
    asyncio.run(run(parser.parse_args()))
