"""Bootstrap Rust frontend processes (one node's share) against DP mock engines.

Same as genopt/genopt-rust-server.py (inherited listen socket, one process
per endpoint pair, observer env), plus the DP arguments that
vllm.v1.utils.RustFrontendProcessManager passes in a real DP launch:
--engine-start-index 0 --engine-count N --data-parallel-size N
--coordinator-address <coordinator front publish>.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--endpoints-json", type=Path, required=True)
    parser.add_argument("--observer-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=245760)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--coordinator-address", help="omit at DP=1 (no coordinator)")
    parser.add_argument("--engine-start-index", type=int, default=0)
    parser.add_argument("--engine-count", type=int, help="default: --dp-size (internal LB)")
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(4096)
    sock.set_inheritable(True)
    runtime = {
        "model_tag": args.model,
        "served_model_name": ["hy4-mock"],
        "max_logprobs": 128,
        "disable_log_stats": True,
    }
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.environ["VLLM_MOCK_OUTPUT_TOKENS"] = str(args.output_tokens)
    argv = [
        args.binary, "frontend", "--listen-fd", str(sock.fileno()),
        "--input-address", "", "--output-address", "",
        "--engine-start-index", str(args.engine_start_index),
        "--engine-count", str(args.engine_count or args.dp_size),
        "--data-parallel-size", str(args.dp_size),
        "--args-json", json.dumps(runtime),
    ]
    if args.coordinator_address:
        argv[-2:-2] = ["--coordinator-address", args.coordinator_address]
    processes, logs = [], []

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for index, endpoint in enumerate(json.loads(args.endpoints_json.read_text())):
            env = os.environ.copy()
            env["VLLM_MOCK_CONSUMPTION_LOG"] = str(args.observer_dir / f"consumed-{index}.jsonl")
            env["VLLM_MOCK_STAGE_LOG"] = str(args.observer_dir / f"stages-{index}.jsonl")
            command = argv.copy()
            command[command.index("--input-address") + 1] = endpoint["input"]
            command[command.index("--output-address") + 1] = endpoint["output"]
            log = (args.observer_dir / f"worker-{index}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, env=env, pass_fds=(sock.fileno(),),
                                              stdout=log, stderr=subprocess.STDOUT))
        (args.observer_dir / "pids.json").write_text(
            json.dumps({str(i): p.pid for i, p in enumerate(processes)}))
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        raise RuntimeError("Rust frontend worker exited")
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()
        sock.close()
