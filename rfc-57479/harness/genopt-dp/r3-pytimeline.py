"""dp-round3: per-server timeline for Python single-node default cells (read-only).

Per frontend pid: requests owned (consumption log), serialization_ready /
body_sent times after the pause (http observer, wall clock of the frontend
node), CPU seconds from the pause to the last body (monitor ticks), and the
engine-side request distribution (DP ranks) when present.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

def jl(p):
    out = []
    for line in Path(p).read_text().splitlines():
        if line.strip():
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out

def cell(run):
    run = Path(run)
    pause = json.loads((run / "pause-start.json").read_text())["wall"]
    owners = Counter(); req_pid = {}
    for p in run.glob("observe/*/consumed-*.jsonl"):
        for r in jl(p):
            if r.get("event") == "consumed":
                owners[r["pid"]] += 1; req_pid[r["request_id"]] = r["pid"]
    ready = defaultdict(list); sent = defaultdict(list)
    for p in run.glob("observe/*/http-*.jsonl"):
        for r in jl(p):
            if r["path"] != "/inference/v1/generate": continue
            (ready if r["event"] == "serialization_ready" else sent)[r["pid"]].append(round(r["wall"] - pause, 2))
    # CPU per pid from the server monitor between pause and pause+40s
    cpu = {}
    samples = jl(next(run.glob("metrics-server-*.jsonl")))
    def at(t):
        best = None
        for s in samples:
            if s["wall"] <= t: best = s
        return best
    a, b = at(pause), at(pause + 45)
    hz = samples[0]["clock_ticks"]
    ta = {p["pid"]: p["cpu_ticks"] for p in a["processes"]}
    tb = {p["pid"]: p["cpu_ticks"] for p in b["processes"]}
    for pid in owners:
        cpu[pid] = round((tb.get(pid, 0) - ta.get(pid, 0)) / hz, 1)
    rows = []
    for pid, n in sorted(owners.items(), key=lambda kv: -max(sent.get(kv[0], [0]))):
        rows.append({"pid": pid, "requests": n, "ready_s": sorted(ready[pid]), "sent_s": sorted(sent[pid]), "cpu_s_after_pause": cpu.get(pid)})
    ranks = {}
    for p in sorted(run.glob("engine-r*.jsonl")):
        ranks[p.stem] = sum(1 for e in jl(p) if e["event"] == "admitted")
    dist = sorted(owners.values(), reverse=True)
    return {"run": run.name, "owner_distribution": dist, "servers_with_requests": len(owners),
            "last_ready_s": max(max(v) for v in ready.values()), "last_sent_s": max(max(v) for v in sent.values()),
            "engine_admitted": ranks, "slowest_servers": rows[:6]}

if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(cell(r)))
