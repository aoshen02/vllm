"""Record the owned process tree and stop it if the node exhausts memory."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def process_tree(pid):
    pending = [pid]
    records = []
    while pending:
        current = pending.pop()
        directory = Path(f"/proc/{current}")
        try:
            pending.extend(
                map(int, (directory / f"task/{current}/children").read_text().split())
            )
            status = {}
            for line in (directory / "status").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in ("VmRSS", "VmHWM", "Cpus_allowed_list", "Name"):
                    status[key] = value.strip()
            stats = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            records.append(
                {"pid": current, **status, "cpu_ticks": int(stats[11]) + int(stats[12])}
            )
        except (FileNotFoundError, ProcessLookupError):
            pass
    return records


def main(args):
    process = subprocess.Popen(args.command, start_new_session=True)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    reason = None
    try:
        with args.log.open("w", buffering=1) as log:
            while True:
                memory = {
                    key: int(value.split()[0])
                    for key, value in (
                        line.split(":", 1)
                        for line in Path("/proc/meminfo").read_text().splitlines()
                    )
                }
                records = process_tree(process.pid)
                record = {
                    "wall": time.time(),
                    "monotonic": time.monotonic(),
                    "root_pid": process.pid,
                    "clock_ticks": os.sysconf("SC_CLK_TCK"),
                    "node_mem_available_kib": memory["MemAvailable"],
                    "processes": records,
                }
                if memory["MemAvailable"] < args.memory_floor_gib * 1024**2:
                    reason = "node_memory_floor"
                    record["failure"] = reason
                log.write(json.dumps(record) + "\n")
                if reason or process.poll() is not None:
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        reason = "terminated"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return 88 if reason == "node_memory_floor" else process.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--memory-floor-gib", type=int, default=64)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command.pop(0)
    raise SystemExit(main(args))
