"""Average cores busy per launched process group during the ingest window
(all_ready+5 s .. consumption barrier), from the genopt monitor metrics."""
import json
import sys
from pathlib import Path

for run in map(Path, sys.argv[1:]):
    meta = json.loads((run / "run.json").read_text())
    t0 = meta["all_ready_wall"] + 5
    t1 = json.loads((run / "pause-start.json").read_text())["barrier_wall"]
    out = {}
    for m in sorted(run.glob("metrics-*.jsonl")):
        rows = [json.loads(l) for l in m.read_text().splitlines() if l.strip()]
        inside = [r for r in rows if t0 <= r["wall"] <= t1]
        if len(inside) < 2:
            continue
        a, b = inside[0], inside[-1]
        tot = lambda r: sum(p["cpu_ticks"] for p in r["processes"])
        out[m.stem.removeprefix("metrics-")] = round((tot(b) - tot(a)) / a["clock_ticks"] / (b["wall"] - a["wall"]), 2)
    print(json.dumps({"run": run.name, "ingest_s": round(t1 - meta["all_ready_wall"], 1), "cores": out}))
