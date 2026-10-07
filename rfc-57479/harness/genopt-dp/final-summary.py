"""Summarize a final-validation output directory (cells named <cell>-<arm>-<rep>).

Per (cell, arm): status counts, median and [min, max] of the key metrics,
busiest frontend process (generate responses served by one process; the
Python HTTP observer, or the single Rust process), wave checks, and sha
consistency (cohort cells: the full list of response sha256s must be equal
across all repeats and across arms). Writes final-summary.json and
final-summary.md into the directory and prints the markdown.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dp_analyze", HERE / "dp-analyze.py")
dp_analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp_analyze)

COHORT = {"cohort-rl-v3", "cohort-rl-wide", "cohort-def-wide", "n32-def", "n32-rl"}


def busiest(run: Path):
    per_pid = Counter()
    for path in run.glob("observe/*/http-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path") == "/inference/v1/generate" and rec.get("event") == "body_sent":
                per_pid[(path.parent.name, rec["pid"])] += 1
    if per_pid:
        return max(per_pid.values()), len(per_pid)
    return None, None  # Rust: one process serves every request


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": round(statistics.median(values), 3), "min": round(min(values), 3),
            "max": round(max(values), 3), "n": len(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--impl")
    p.add_argument("--candidate")
    p.add_argument("--baseline")
    a = p.parse_args()
    groups = defaultdict(list)
    for run in sorted(a.out.iterdir()):
        m = re.fullmatch(r"(.+)-([AB])-(\d+)", run.name)
        if not m or not run.is_dir() or not (run / "run.json").exists():
            continue
        groups[(m.group(1), m.group(2))].append(run)
    table = []
    sha_sets = defaultdict(set)
    len_sets = defaultdict(set)
    for (cell, arm), runs in sorted(groups.items()):
        rows, statuses, checks, busy, servers = [], Counter(), Counter(), [], []
        for run in runs:
            row = dp_analyze.analyze(run)
            rows.append(row)
            statuses[row["status"]] += 1
            checks["PASS" if row.get("wave_checks") == ["PASS"] else "FAIL"] += 1
            b, n = busiest(run)
            busy.append(b)
            servers.append(n)
            res = run / "result.json"
            if cell in COHORT and res.exists() and row["status"] == "PASS":
                result = json.loads(res.read_text())
                shas = result.get("response_sha256")
                if shas:
                    sha_sets[cell].add((arm, hashlib.sha256(json.dumps(shas).encode()).hexdigest()[:16]))
                if result.get("response_bytes"):
                    len_sets[cell].add((arm, hashlib.sha256(json.dumps(result["response_bytes"]).encode()).hexdigest()[:16]))
        entry = {"cell": cell, "arm": arm, "runs": [r["run"] for r in rows], "status": dict(statuses),
                 "wave_checks": dict(checks), "hwm_gib_max": stat([(r.get("frontend_hwm_gib_max_median") or [None])[0] for r in rows]),
                 "busiest_process_requests": stat(busy) if any(b is not None for b in busy) else "single process",
                 "frontend_processes_used": stat(servers) if any(s is not None for s in servers) else 1}
        ok = [r for r in rows if r["status"] == "PASS"]
        if cell in COHORT:
            reqs = ok[0]["requests"] if ok else 0
            entry.update({
                "consumption_s": stat([r.get("consumption_phase_s") for r in ok]),
                "ingest_Mpos_per_s": stat([reqs * r["output_tokens"] / r["consumption_phase_s"] / 1e6 for r in ok if r.get("consumption_phase_s")]),
                "pause_s": stat([r.get("pause_s") for r in ok]),
                "all_headers_s": stat([r.get("all_headers_s") for r in ok]),
                "all_bodies_received_s": stat([r.get("all_bodies_received_s") for r in ok]),
                "all_parsed_s": stat([r.get("all_parsed_s") for r in ok]),
                "bytes_total_GB": stat([(r.get("bytes_total") or 0) / 1e9 for r in ok]),
                "engine_balance_max_over_min": stat([max(r["load_balance_phase1_admitted"]) / max(1, min(r["load_balance_phase1_admitted"])) for r in ok if r.get("load_balance_phase1_admitted")]),
            })
        else:
            entry.update({
                "pause_return_s": stat([r.get("pause_return_s") for r in ok]),
                "partials_parsed_max_s": stat([(r.get("aborted_parsed_after_pause_s_min_med_max") or [None, None, None])[2] for r in ok]),
                "resume_first_token_median_s": stat([(r.get("resume_to_first_token_s_min_med_max") or [None, None, None])[1] for r in ok]),
                "aborted_at_pause": stat([(r.get("phase1_finish_counts") or {}).get("abort") for r in ok]),
                "resubmitted_finished": stat([(r.get("phase2_finish_counts") or {}).get("length") for r in ok]),
                "phase2_parsed_max_s": stat([(r.get("phase2_parsed_after_resume_s_min_med_max") or [None, None, None])[2] for r in ok]),
                "phase1_GB": stat([(r.get("phase1_bytes_total") or 0) / 1e9 for r in ok]),
            })
        table.append(entry)
    sha = {cell: {"distinct_sha_lists": len({h for _, h in s}), "by_arm": sorted(s),
                  "distinct_byte_length_lists": len({h for _, h in len_sets[cell]})}
           for cell, s in sha_sets.items()}
    summary = {"impl": a.impl, "candidate": a.candidate, "baseline": a.baseline or None,
               "cells": table, "sha_consistency": sha}
    (a.out / "final-summary.json").write_text(json.dumps(summary, indent=1))

    def fmt(s):
        if not isinstance(s, dict):
            return str(s)
        return f"{s['median']} [{s['min']}–{s['max']}] (n={s['n']})"
    lines = [f"# Final validation summary: {a.impl}", "",
             f"candidate (B): `{a.candidate}`; baseline (A): `{a.baseline or 'none'}`", "",
             "| cell | arm | status | key metrics (median [min–max]) | busiest process (requests) | HWM GiB | waves |",
             "|---|---|---|---|---|---|---|"]
    for e in table:
        if e["cell"] in COHORT:
            key = (f"all parsed {fmt(e['all_parsed_s'])}; received {fmt(e['all_bodies_received_s'])}; "
                   f"pause {fmt(e['pause_s'])}; consumption {fmt(e['consumption_s'])}; ingest Mpos/s {fmt(e['ingest_Mpos_per_s'])}; "
                   f"GB {fmt(e['bytes_total_GB'])}")
        else:
            key = (f"pause {fmt(e['pause_return_s'])}; partials parsed max {fmt(e['partials_parsed_max_s'])}; "
                   f"resume→first token {fmt(e['resume_first_token_median_s'])}; aborted {fmt(e['aborted_at_pause'])}; "
                   f"resubmitted finished {fmt(e['resubmitted_finished'])}")
        lines.append(f"| {e['cell']} | {e['arm']} | {e['status']} | {key} | {fmt(e['busiest_process_requests'])} | "
                     f"{fmt(e['hwm_gib_max'])} | {e['wave_checks']} |")
    lines += ["", "sha consistency (cohort cells; 1 = every repeat and arm returned identical bodies).",
              "The Python generate response carries `created` (unix seconds), so its body sha differs per run",
              "by construction; for Python the per-request byte-length lists are the consistency check.", ""]
    for cell, s in sha.items():
        lines.append(f"- {cell}: {s['distinct_sha_lists']} distinct body-sha list(s), "
                     f"{s['distinct_byte_length_lists']} distinct byte-length list(s) {s['by_arm']}")
    md = "\n".join(lines) + "\n"
    (a.out / "final-summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
