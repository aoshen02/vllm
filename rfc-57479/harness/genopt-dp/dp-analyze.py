"""Summarize DP harness run directories (read-only), one JSON row per run.

Client metrics: client v3 result.json (pause cohort) or dp-client result.json
(pause / resume / resubmit). Engine metrics from engine-r*.jsonl: per-rank
admitted requests and tokens (load balance), resume -> first token of each
resubmitted request (engine wall clock vs client wall clock: cross-node NTP
uncertainty applies, typically ~ms). Frontend memory: per-process VmHWM
from metrics-server-*.jsonl. Wave-state checks:
  * no collective_mismatch, every rank ends idle with the same current wave;
  * per (rank, wave) at most one START_DP_WAVE that actually woke the rank
    (acted=true), i.e. no duplicate wakeups;
  * every pause_scheduler call answered (pause_complete) on every rank, one
    pause_consensus per pause per rank;
  * the coordinator's last publish says engines_running=false with the same
    wave as the engines, and its wave moves = wave_complete messages.
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def mmm(values):
    values = sorted(values)
    return [values[0], values[len(values) // 2], values[-1]] if values else None


def analyze(run: Path):
    meta = json.loads((run / "run.json").read_text())
    row = {k: meta.get(k) for k in ("label", "impl", "dp", "logprobs_format", "compact_include_sampled",
                                     "compact_include_ranks", "requests", "output_tokens",
                                     "workers_per_node", "client", "status", "error", "binary_sha256", "source")}
    row["run"] = run.name
    res_path = run / "result.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    pause_wall = resume_wall = None
    if meta.get("client") == "v3" and res:
        for k in ("pause_s", "all_headers_s", "all_bodies_received_s", "all_parsed_s",
                  "semantically_validated_requests", "v3_sha_parse_validate_cpu_s_sum"):
            row[k] = res.get(k)
        row["bytes_total"] = sum(res.get("response_bytes", []))
        pause = json.loads((run / "pause-start.json").read_text())
        row["consumption_phase_s"] = pause["barrier_wall"] - meta["all_ready_wall"]
        pause_wall = pause["wall"]
    elif res:
        for k in ("pause_return_s", "aborted_received_after_pause_s_min_med_max",
                  "aborted_parsed_after_pause_s_min_med_max", "resume_return_s", "resubmitted",
                  "phase2_headers_after_resume_s_min_med_max", "phase2_parsed_after_resume_s_min_med_max",
                  "phase1_finish_counts", "phase2_finish_counts", "phase1_tokens_total", "phase1_bytes_total"):
            row[k] = res.get(k)
        pause_wall = res["events"].get("pause_start", {}).get("wall")
        resume_wall = res["events"].get("resume_start", {}).get("wall")
    else:
        pause_wall = resume_wall = None
    # engines
    ranks = sorted(run.glob("engine-r*.jsonl"), key=lambda p: int(p.stem[8:]))
    per_rank = {}
    checks = []
    final_waves = {}
    first_tokens = []
    collective_mismatch = 0
    for path in ranks:
        rank = int(path.stem[8:])
        ev = jsonl(path)
        kinds = Counter(e["event"] for e in ev)
        admitted = [e for e in ev if e["event"] == "admitted"]
        phase2 = [e for e in admitted if "-r1" in e["request"]]
        stats = [e["stats"] for e in ev if e["event"] == "wave_end"]
        acted = Counter(e["wave"] for e in ev if e["event"] == "start_dp_wave_recv" and e["acted"])
        dup = {w: c for w, c in acted.items() if c > 1}
        pauses = kinds["utility"] and sum(1 for e in ev if e["event"] == "utility" and e["method"] == "pause_scheduler")
        per_rank[rank] = {
            "admitted_phase1": len(admitted) - len(phase2), "admitted_phase2": len(phase2),
            "tokens": stats[-1]["tokens"] if stats else None,
            "waves_ended": [e["wave"] for e in ev if e["event"] == "wave_end"],
            "start_dp_wave_recv": kinds["start_dp_wave_recv"],
            "start_dp_wave_acted": sum(acted.values()),
            "pause_calls": pauses or 0, "pause_complete": kinds["pause_complete"],
            "pause_consensus": kinds["pause_consensus"],
            "client_index_mismatch": stats[-1]["client_index_mismatch"] if stats else None,
        }
        collective_mismatch += kinds["collective_mismatch"]
        if dup:
            checks.append(f"rank {rank}: duplicate wakeups {dup}")
        if (pauses or 0) != kinds["pause_complete"]:
            checks.append(f"rank {rank}: {pauses} pause calls, {kinds['pause_complete']} completed")
        last_end = max((e["mono"] for e in ev if e["event"] == "wave_end"), default=None)
        last_any = max((e["mono"] for e in ev if e["event"] in ("admitted", "first_token", "start_dp_wave_recv")), default=None)
        if last_end is None or (last_any is not None and last_any > last_end):
            checks.append(f"rank {rank}: activity after its last wave_end (not idle at end)")
        final_waves[rank] = (per_rank[rank]["waves_ended"][-1] + 1) if per_rank[rank]["waves_ended"] else 0
        if resume_wall is not None:
            first_tokens += [e["wall"] - resume_wall for e in ev
                             if e["event"] == "first_token" and "-r1" in e["request"]]
    if collective_mismatch:
        checks.append(f"{collective_mismatch} collective mismatches")
    if len(set(final_waves.values())) > 1:
        checks.append(f"final waves differ across ranks: {final_waves}")
    pubs = jsonl(run / "coordinator-publish.jsonl")
    if pubs:
        last = pubs[-1]
        row["coordinator_last_publish"] = {"wave": last["wave"], "running": last["running"]}
        if last["running"] or (final_waves and last["wave"] != max(final_waves.values())):
            checks.append(f"coordinator final state {last['wave']}/{last['running']} vs engines {final_waves}")
    coord_log = (run / "coordinator.log").read_text(errors="replace") if (run / "coordinator.log").exists() else ""
    row["coordinator_wave_moves"] = coord_log.count("Moving DP wave")
    row["coordinator_stale_starts"] = coord_log.count("Starting wave")
    row["coordinator_out_of_order_warnings"] = coord_log.count("out-of-order")
    rank0 = jsonl(run / "engine-r0.jsonl")
    if row["coordinator_wave_moves"] != sum(1 for e in rank0 if e["event"] == "wave_complete_sent"):
        checks.append("coordinator wave moves != rank-0 wave_complete messages")
    row["per_rank"] = per_rank
    adm = [v["admitted_phase1"] for v in per_rank.values()]
    row["load_balance_phase1_admitted"] = adm
    tok = [v["tokens"] or 0 for v in per_rank.values()]
    row["load_balance_tokens_max_over_mean"] = (max(tok) / statistics.mean(tok)) if tok and statistics.mean(tok) else None
    if first_tokens:
        row["resume_to_first_token_s_min_med_max"] = mmm(first_tokens)
    if pause_wall is not None:
        wave_ends = [e["wall"] - pause_wall for p in ranks for e in jsonl(p) if e["event"] == "pause_complete"]
        row["pause_to_engine_pause_complete_s_max"] = max(wave_ends) if wave_ends else None
    # frontend memory
    hwm = {}
    for path in run.glob("metrics-server-*.jsonl"):
        for s in jsonl(path):
            for p in s["processes"]:
                if "VmHWM" in p:
                    hwm[(path.stem, p["pid"])] = max(hwm.get((path.stem, p["pid"]), 0), int(p["VmHWM"].split()[0]))
    if hwm:
        top = sorted(hwm.values(), reverse=True)
        row["frontend_hwm_gib_max_median"] = [round(top[0] / 2**20, 3), round(top[len(top) // 2] / 2**20, 3)]
    row["wave_checks"] = checks or ["PASS"]
    return row


if __name__ == "__main__":
    rows = [analyze(Path(p)) for p in sys.argv[1:]]
    print(json.dumps(rows, indent=1))
