"""Round 13 profiling tool (perf is unavailable: no binary, perf_event_paranoid=4).

Waits for the Rust frontend pid (observer pids.json), then every --period
seconds reads /proc/<pid>/task/*/{comm,stat} and logs per-thread CPU seconds.
The summary groups threads by name (vllm-zmq-*, vllm-request, tokio-runtime-w,
...) and reports, per group, the busiest threads' CPU utilisation over the
window where the process was busiest (the ingest phase).
"""
import argparse
import json
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--pids-json", required=True)
p.add_argument("--log", required=True)
p.add_argument("--period", type=float, default=1.0)
p.add_argument("--duration", type=float, default=7200)
a = p.parse_args()
hz = os.sysconf("SC_CLK_TCK")
deadline = time.time() + a.duration
while not Path(a.pids_json).exists():
    if time.time() > deadline:
        raise SystemExit("no pids.json")
    time.sleep(0.5)
pid = int(next(iter(json.loads(Path(a.pids_json).read_text()).values())))
with open(a.log, "w", buffering=1) as out:
    while time.time() < deadline:
        row = {"wall": time.time(), "threads": {}}
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                try:
                    comm = Path(f"/proc/{pid}/task/{tid}/comm").read_text().strip()
                    st = Path(f"/proc/{pid}/task/{tid}/stat").read_text().rsplit(")", 1)[1].split()
                    row["threads"][tid] = [comm, (int(st[11]) + int(st[12])) / hz]
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except FileNotFoundError:
            pass  # process gone: idle until terminated (the runner treats an exit as a failure)
        out.write(json.dumps(row) + "\n")
        time.sleep(a.period)
