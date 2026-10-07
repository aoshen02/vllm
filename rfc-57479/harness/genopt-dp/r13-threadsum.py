"""Summarize r13-threadprof logs over the ingest phase of a run.

Window: from 5 s after all frontends were ready (run.json all_ready_wall) to
the consumption barrier (pause-start.json barrier_wall), i.e. the pre-pause
phase in which the frontend ingests engine output. Per-thread CPU
utilisation is grouped by thread name (vllm-zmq-*, vllm-request,
tokio-rt-worker, ...).
"""
import collections
import json
import sys
from pathlib import Path

for run in map(Path, sys.argv[1:]):
    rows = [json.loads(l) for l in (run / "threadprof.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "run.json").read_text())
    pause = run / "pause-start.json"
    t0 = meta["all_ready_wall"] + 5
    t1 = json.loads(pause.read_text())["barrier_wall"] if pause.exists() else rows[-1]["wall"]
    inside = [r for r in rows if t0 <= r["wall"] <= t1]
    a, b = inside[0], inside[-1]
    dt = b["wall"] - a["wall"]
    groups = collections.defaultdict(list)
    for tid, (comm, cpu) in b["threads"].items():
        before = a["threads"].get(tid, [comm, 0.0])[1]
        groups[comm.rstrip("0123456789-")].append((cpu - before) / dt)
    print(json.dumps({
        "run": run.name, "ingest_window_s": round(dt, 1),
        "process_cores": round(sum(sum(v) for v in groups.values()), 2),
        "groups": {g: {"threads": len(v), "cores": round(sum(v), 2),
                       "busiest": sorted((round(x, 2) for x in v), reverse=True)[:4]}
                   for g, v in sorted(groups.items(), key=lambda kv: -sum(kv[1]))},
    }))
