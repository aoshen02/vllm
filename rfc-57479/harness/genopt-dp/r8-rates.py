"""dp-round8: client-side receive rates per cohort run (client v3 result.json).

Per run: all bodies received / parsed after the pause, aggregate receive rate
(total bytes / (last received - first headers)), per-connection rate
(bytes / (received - headers)) min/median/max, receive window start/end,
time between last body received and last parsed, client CPU, server stage
timing (Rust observer: body_wall_s / poll_busy_s), and the overlapping
concurrent cells noted from the final-validation logs is left to the report.
"""
import json
import statistics
import sys
from pathlib import Path


def stages(run):
    walls, busy = [], []
    for p in run.glob("observe/*/stages-*.jsonl"):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "body_done":
                walls.append(r.get("body_wall_s", 0))
                busy.append(r.get("poll_busy_s", 0))
    return walls, busy


for path in sys.argv[1:]:
    run = Path(path)
    res = json.loads((run / "result.json").read_text())
    meta = json.loads((run / "run.json").read_text())
    t0 = res["pause_monotonic"]
    per = res["per_request"]
    sizes = res["response_bytes"]
    hdr = [p["headers_t"] - t0 for p in per]
    rcv = [p["received_t"] - t0 for p in per]
    rates = sorted(s / (p["received_t"] - p["headers_t"]) / 1e9 for s, p in zip(sizes, per))
    walls, busy = stages(run)
    out = {
        "run": f"{run.parent.name}/{run.name}",
        "binary": Path(meta.get("binary") or "").name,
        "started": meta.get("started_wall"),
        "all_headers_s": round(res["all_headers_s"], 3),
        "first_received_s": round(min(rcv), 2),
        "all_bodies_received_s": round(res["all_bodies_received_s"], 2),
        "all_parsed_s": round(res["all_parsed_s"], 2),
        "parse_tail_s": round(res["all_parsed_s"] - res["all_bodies_received_s"], 2),
        "aggregate_GBps": round(sum(sizes) / (max(rcv) - min(hdr)) / 1e9, 2),
        "per_conn_GBps_min_med_max": [round(rates[0], 3), round(statistics.median(rates), 3), round(rates[-1], 3)],
        "client_cpu_s": round(res.get("client_process_cpu_s", 0), 1),
        "sha_parse_validate_cpu_s": round(res.get("v3_sha_parse_validate_cpu_s_sum", 0), 1),
        "server_body_wall_s_med_max": [round(statistics.median(walls), 2), round(max(walls), 2)] if walls else None,
        "server_poll_busy_s_sum": round(sum(busy), 2) if busy else None,
    }
    print(json.dumps(out))
