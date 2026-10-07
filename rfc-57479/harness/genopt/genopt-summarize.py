"""Summarize genopt run directories into one JSON row each (read-only).

Times relative to pause start (client wall clock, node B). Server markers use
the frontend nodes' wall clocks: cross-node NTP uncertainty applies. Overlapping
intervals are reported separately and must never be added.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def jsonl(path):
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def summarize(run: Path):
    row = {"run": run.name}
    meta = json.loads((run / "run.json").read_text())
    for key in ("label", "impl", "logprobs_format", "independent_listeners", "nodes",
                "requests", "output_tokens", "status", "error", "binary_sha256", "source"):
        row[key] = meta.get(key)
    pause_path = run / "pause-start.json"
    pause = json.loads(pause_path.read_text()) if pause_path.exists() else None
    if meta.get("all_ready_wall") and pause:
        row["consumption_phase_s"] = pause["barrier_wall"] - meta["all_ready_wall"] if "barrier_wall" in pause else pause["wall"] - meta["all_ready_wall"]
    result_path = run / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        for key in ("pause_s", "all_bodies_received_s", "post_receive_parse_validate_wall_s",
                    "receive_plus_parse_validate_s", "post_receive_parse_cpu_s_sum",
                    "semantically_validated_requests", "top_entries_per_position",
                    "all_parsed_s", "all_headers_s", "client",
                    # client v3 only (sha256+parse+validate CPU; not comparable to v1/v2 parse CPU)
                    "v3_sha_parse_validate_cpu_s_sum", "receive_policy"):
            row[key] = result.get(key)
        sizes = result.get("response_bytes", [])
        row["bytes_per_request"] = sorted(set(sizes))
        row["bytes_total"] = sum(sizes)
    else:
        received = run / "received.json"
        if received.exists():
            r = json.loads(received.read_text())
            row["pause_s"] = r.get("pause_s")
            row["all_bodies_received_s"] = r.get("all_bodies_received_s")
    observe = run / "observe"
    ready, sent = [], []
    owners = Counter()
    progress = {}
    for path in observe.rglob("http-*.jsonl"):
        for rec in jsonl(path):
            if rec["path"] != "/inference/v1/generate":
                continue
            (ready if rec["event"] == "serialization_ready" else sent).append(rec["wall"])
    for path in observe.rglob("consumed-*.jsonl"):
        node = path.parent.name
        for rec in jsonl(path):
            progress[rec["request_id"]] = max(progress.get(rec["request_id"], 0), rec["tokens"])
            if rec.get("event", "consumed") == "consumed":
                owners[(node, rec.get("pid", path.name))] += 1
    row["requests_with_progress"] = len(progress)
    row["requests_consumed_target"] = sum(owners.values())
    row["owner_distribution"] = sorted(owners.values(), reverse=True)
    if pause:
        if ready:
            row["last_serialization_ready_s"] = max(ready) - pause["wall"]
            row["first_serialization_ready_s"] = min(ready) - pause["wall"]
        if sent:
            row["last_body_handed_to_transport_s"] = max(sent) - pause["wall"]
    stages = Counter()
    stage_max = {}
    stage_last_wall = {}
    for path in observe.rglob("stages-*.jsonl"):
        for rec in jsonl(path):
            stages[rec["event"]] += rec["elapsed_s"]
            stage_max[rec["event"]] = max(stage_max.get(rec["event"], 0), rec["elapsed_s"])
            stage_last_wall[rec["event"]] = max(stage_last_wall.get(rec["event"], 0), rec["wall"])
    if pause:
        # Rust observer: wall at end of each stage event. json_render end on the
        # streamed openai path includes network backpressure (body_wall_s).
        row["stage_last_end_s"] = {k: v - pause["wall"] for k, v in stage_last_wall.items()}
    row["stage_sum_s"] = dict(stages)
    row["stage_max_s"] = stage_max
    # resources
    hwm = 0.0
    min_avail = None
    cpu = {}
    for path in run.glob("metrics-*.jsonl"):
        label = path.stem.removeprefix("metrics-")
        samples = list(jsonl(path))
        for s in samples:
            avail = s["node_mem_available_kib"] / 1024**2
            min_avail = avail if min_avail is None else min(min_avail, avail)
            if label.startswith("server"):
                for p in s["processes"]:
                    if "VmHWM" in p:
                        hwm = max(hwm, int(p["VmHWM"].split()[0]) / 1024**2)
        if pause and samples:
            end = pause["wall"] + (row.get("all_bodies_received_s") or 0)

            def ticks_at(t):
                best = None
                for s in samples:
                    if s["wall"] <= t:
                        best = s
                if best is None:
                    return 0, 100
                return sum(p["cpu_ticks"] for p in best["processes"]), best["clock_ticks"]

            a, hz = ticks_at(pause["wall"])
            b, _ = ticks_at(end + 1)
            cpu[label] = (b - a) / hz
    row["max_frontend_process_hwm_gib"] = round(hwm, 3)
    row["min_node_available_gib"] = round(min_avail, 3) if min_avail is not None else None
    row["cpu_s_pause_to_last_body"] = cpu
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = [summarize(run) for run in args.runs]
    text = json.dumps(rows, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
