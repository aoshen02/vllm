"""RL-style pause / weight-sync / resume / resubmit driver for the DP harness.

Phase 1: submit --requests non-streaming /inference/v1/generate requests
(round-robin over --urls) with per-request max_tokens drawn from the length
distribution; at --pause-at seconds after the last submission, POST
/pause?mode=abort to urls[0]. Requests that hit max_tokens before the pause
return finish=length; the rest return partial results (finish=abort).
Phase 2 (--resume): sleep --sleep-s (weight sync stand-in), POST /resume,
then resubmit every aborted request with prompt = prompt + partial tokens
and max_tokens = remaining (request_id "<id>-r1"), and wait for all.

Every response is read fully, json-parsed and validated (status, request_id,
finish reason, token count vs max_tokens, token values = the mock engine's
deterministic sequence at the right offset, compact num_positions / array
sizes or openai len(content)). Times are CLOCK_MONOTONIC on the client node;
wall times are recorded too for joining with engine/coordinator logs.
Measurement tool only (Python receive/parse is slower than client v3).
"""

import argparse
import base64
import http.client
import json
import math
import multiprocessing as mp
import queue
import random
import threading
import time
import urllib.parse
from pathlib import Path

TOP_K = 128


def expected_tokens(pool, offset, n):
    pos = range(offset, offset + n)
    return [pool[(p + p % TOP_K) % len(pool)] for p in pos]


def post(url, path, body: bytes | None, timeout):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    conn.request("POST", path, body=body or b"", headers={"content-type": "application/json"})
    resp = conn.getresponse()
    t_headers = time.monotonic()
    data = resp.read()
    t_received = time.monotonic()
    conn.close()
    return resp.status, data, t_headers, t_received


def post_control(urls, path, timeout):
    """POST a control call (/pause, /resume) to every URL concurrently;
    returns (all statuses, monotonic time the last one returned, bodies)."""
    out = [None] * len(urls)

    def one(i, url):
        status, data, _, t_r = post(url, path, None, timeout)
        out[i] = (status, t_r, data[:200].decode(errors="replace"))

    threads = [threading.Thread(target=one, args=(i, u)) for i, u in enumerate(urls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [o[0] for o in out], max(o[1] for o in out), [o[2] for o in out]


def run_job(job, pool, cfg, results):
    out = {k: job[k] for k in ("phase", "index", "request_id", "url", "max_tokens", "offset")}
    out["t_submit"] = time.monotonic()
    out["wall_submit"] = time.time()
    try:
        status, data, t_h, t_r = post(job["url"], "/inference/v1/generate", job["body"], cfg["timeout"])
        out.update(status=status, bytes=len(data), t_headers=t_h, t_received=t_r)
        if status != 200:
            out["error"] = data[:300].decode(errors="replace")
        else:
            doc = json.loads(data)
            choice = doc["choices"][0]
            tokens = choice["token_ids"]
            n = len(tokens)
            out.update(finish=choice["finish_reason"], n_tokens=n)
            assert doc["request_id"] in (job["request_id"], "generate-tokens-" + job["request_id"]), doc["request_id"]
            if choice["finish_reason"] == "length":
                assert n == job["max_tokens"], (n, job["max_tokens"])
            else:
                assert choice["finish_reason"] == "abort", choice["finish_reason"]
                assert n < job["max_tokens"], (n, job["max_tokens"])
            assert tokens == expected_tokens(pool, job["offset"], n), "token values"
            fmt = cfg["logprobs_format"]
            if fmt == "compact":
                block = choice["compact_logprobs"]
                assert block["num_positions"] == n
                slots = block["num_slots"]
                assert len(base64.b64decode(block["token_ids"], validate=True)) == 4 * n * slots
                assert len(base64.b64decode(block["logprobs"], validate=True)) == 4 * n * slots
            elif fmt == "openai":
                assert len(choice["logprobs"]["content"]) == n
            out["tokens"] = tokens if out["finish"] == "abort" else None
        out["t_parsed"] = time.monotonic()
    except Exception as e:  # recorded, the run is marked FAIL
        out["error"] = f"{type(e).__name__}: {e}"
        out["t_parsed"] = time.monotonic()
    results.put(out)


def worker_main(jobs, results, pool, cfg):
    threads = []
    while True:
        job = jobs.get()
        if job is None:
            break
        t = threading.Thread(target=run_job, args=(job, pool, cfg, results), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def sample_lengths(args, rng):
    cap, lo = args.length_cap, args.length_min
    out = []
    for _ in range(args.requests):
        if args.lengths == "fixed":
            n = cap
        elif args.lengths == "uniform":
            n = rng.randint(lo, cap)
        else:  # lognormal, median = --lognormal-median
            n = int(round(math.exp(rng.gauss(math.log(args.lognormal_median), args.lognormal_sigma))))
        out.append(max(lo, min(cap, n)))
    return out


def main(args):
    pool_entries = json.loads(args.token_pool.read_text())
    pool = [e["id"] for e in pool_entries]
    prompt = [pool[i % len(pool)] for i in range(args.input_tokens)]
    rng = random.Random(args.seed)
    lengths = sample_lengths(args, rng)
    cfg = {"timeout": args.http_timeout, "logprobs_format": args.logprobs_format or "absent"}

    def body(rid, tokens, max_tokens):
        b = {"model": "hy4-mock", "token_ids": tokens, "stream": False, "request_id": rid,
             "sampling_params": {"max_tokens": max_tokens, "logprobs": TOP_K, "ignore_eos": True}}
        if args.logprobs_format:
            b["logprobs_format"] = args.logprobs_format
        if not args.compact_include_sampled:
            b["compact_include_sampled"] = False
        if not args.compact_include_ranks:
            b["compact_include_ranks"] = False
        return json.dumps(b).encode()

    ctx = mp.get_context("fork")
    jobs, results = ctx.Queue(), ctx.Queue()
    workers = [ctx.Process(target=worker_main, args=(jobs, results, pool, cfg), daemon=True)
               for _ in range(args.workers)]
    for w in workers:
        w.start()
    events = {}
    t0 = time.monotonic()
    events["start"] = {"mono": t0, "wall": time.time()}
    phase1 = []
    for i, length in enumerate(lengths):
        rid = f"mock-{i:04d}"
        phase1.append({"phase": 1, "index": i, "request_id": rid, "url": args.urls[i % len(args.urls)],
                       "max_tokens": length, "offset": 0, "body": body(rid, prompt, length)})
    for i, job in enumerate(phase1):
        jobs.put(job)
    events["submitted"] = {"mono": time.monotonic(), "wall": time.time()}
    done = {}

    def collect(expected, deadline):
        while len([k for k in done if k[0] == expected[0]]) < expected[1]:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"phase {expected[0]}: {len([k for k in done if k[0] == expected[0]])}/{expected[1]} finished")
            try:
                r = results.get(timeout=min(left, 5))
            except queue.Empty:
                continue
            done[(r["phase"], r["index"])] = r

    if args.pause_at is not None:
        target = events["submitted"]["mono"] + args.pause_at
        while time.monotonic() < target:
            try:
                r = results.get(timeout=max(0.001, min(0.05, target - time.monotonic())))
                done[(r["phase"], r["index"])] = r
            except queue.Empty:
                pass
        events["pause_start"] = {"mono": time.monotonic(), "wall": time.time()}
        control = args.urls if args.pause_all else args.urls[:1]
        statuses, t_r, bodies = post_control(control, "/pause?mode=abort", args.http_timeout)
        events["pause_return"] = {"mono": t_r, "wall": time.time(),
                                  "status": max(statuses) if all(s == 200 for s in statuses) else min(statuses),
                                  "statuses": statuses, "body": bodies[0]}
    collect((1, len(phase1)), time.monotonic() + args.completion_timeout)
    events["phase1_done"] = {"mono": time.monotonic(), "wall": time.time()}

    phase2 = []
    if args.resume:
        time.sleep(args.sleep_s)
        events["resume_start"] = {"mono": time.monotonic(), "wall": time.time()}
        control = args.urls if args.pause_all else args.urls[:1]
        statuses, t_r, bodies = post_control(control, "/resume", args.http_timeout)
        events["resume_return"] = {"mono": t_r, "wall": time.time(),
                                   "status": max(statuses) if all(s == 200 for s in statuses) else min(statuses),
                                   "statuses": statuses, "body": bodies[0]}
        if args.resubmit:
            for i in range(len(phase1)):
                r = done[(1, i)]
                if r.get("finish") != "abort":
                    continue
                partial = r["tokens"]
                remaining = lengths[i] - len(partial)
                rid = f"mock-{i:04d}-r1"
                phase2.append({"phase": 2, "index": i, "request_id": rid,
                               "url": args.urls[i % len(args.urls)], "max_tokens": remaining,
                               "offset": len(partial), "body": body(rid, prompt + partial, remaining)})
            events["resubmit_start"] = {"mono": time.monotonic(), "wall": time.time()}
            for job in phase2:
                jobs.put(job)
            collect((2, len(phase2)), time.monotonic() + args.completion_timeout)
            events["phase2_done"] = {"mono": time.monotonic(), "wall": time.time()}
    for _ in workers:
        jobs.put(None)
    for w in workers:
        w.join(timeout=30)

    rows = sorted(done.values(), key=lambda r: (r["phase"], r["index"]))
    for r in rows:
        r.pop("tokens", None)
    errors = [r for r in rows if r.get("error")]

    def rel(key, phase, origin, finish=None):
        vals = [r[key] - origin for r in rows if r["phase"] == phase and key in r
                and (finish is None or r.get("finish") == finish)]
        return [min(vals), sorted(vals)[len(vals) // 2], max(vals)] if vals else None

    summary = {"status": "PASS" if not errors and all(
        e.get("status", 200) == 200 for e in events.values()) else "FAIL",
               "errors": errors[:10], "events": events, "lengths": lengths,
               "phase1_finish_counts": {f: sum(1 for r in rows if r["phase"] == 1 and r.get("finish") == f)
                                        for f in ("length", "abort")},
               "phase1_tokens_total": sum(r.get("n_tokens", 0) for r in rows if r["phase"] == 1),
               "phase1_bytes_total": sum(r.get("bytes", 0) for r in rows if r["phase"] == 1)}
    if "pause_start" in events:
        p = events["pause_start"]["mono"]
        summary["pause_return_s"] = events["pause_return"]["mono"] - p
        summary["aborted_received_after_pause_s_min_med_max"] = rel("t_received", 1, p, "abort")
        summary["aborted_parsed_after_pause_s_min_med_max"] = rel("t_parsed", 1, p, "abort")
    if "resume_start" in events:
        r0 = events["resume_start"]["mono"]
        summary["resume_return_s"] = events["resume_return"]["mono"] - r0
        if phase2:
            summary["resubmitted"] = len(phase2)
            summary["phase2_headers_after_resume_s_min_med_max"] = rel("t_headers", 2, r0)
            summary["phase2_parsed_after_resume_s_min_med_max"] = rel("t_parsed", 2, r0)
            summary["phase2_finish_counts"] = {f: sum(1 for r in rows if r["phase"] == 2 and r.get("finish") == f)
                                               for f in ("length", "abort")}
    args.result.write_text(json.dumps({**summary, "requests": rows,
                                       "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("lengths", "errors")}))
    if summary["status"] != "PASS":
        print(json.dumps(summary["errors"]))
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--urls", nargs="+", required=True)
    p.add_argument("--token-pool", type=Path, required=True)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--input-tokens", type=int, default=16384)
    p.add_argument("--lengths", choices=("fixed", "uniform", "lognormal"), default="fixed")
    p.add_argument("--length-cap", type=int, default=245760)
    p.add_argument("--length-min", type=int, default=1)
    p.add_argument("--lognormal-median", type=float, default=16384)
    p.add_argument("--lognormal-sigma", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pause-at", type=float, help="seconds after the last submission")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pause-all", action="store_true",
                   help="send /pause and /resume to every URL (needed for per-rank frontends, external LB)")
    p.add_argument("--resubmit", action="store_true")
    p.add_argument("--sleep-s", type=float, default=5.0)
    p.add_argument("--logprobs-format", choices=("openai", "compact"))
    p.add_argument("--compact-no-sampled", dest="compact_include_sampled", action="store_false")
    p.add_argument("--compact-no-ranks", dest="compact_include_ranks", action="store_false")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--http-timeout", type=float, default=7200)
    p.add_argument("--completion-timeout", type=float, default=1800)
    p.add_argument("--result", type=Path, required=True)
    main(p.parse_args())
