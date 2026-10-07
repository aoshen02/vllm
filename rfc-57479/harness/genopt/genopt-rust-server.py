"""Bootstrap the real Rust frontend using its existing inherited-socket CLI."""

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
    parser.add_argument("--input-address")
    parser.add_argument("--output-address")
    parser.add_argument("--consumption-log")
    parser.add_argument("--endpoints-json", type=Path)
    parser.add_argument("--observer-dir", type=Path)
    parser.add_argument("--output-tokens", type=int, default=245760)
    parser.add_argument("--enable-return-routed-experts", action="store_true")
    args = parser.parse_args()
    if args.consumption_log and os.path.exists(args.consumption_log):
        raise RuntimeError("Consumption log must be new for each service")
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
    if args.enable_return_routed_experts:
        runtime["enable_return_routed_experts"] = True
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    if args.consumption_log:
        os.environ["VLLM_MOCK_CONSUMPTION_LOG"] = args.consumption_log
    os.environ["VLLM_MOCK_OUTPUT_TOKENS"] = str(args.output_tokens)
    argv = [
        args.binary,
        "frontend",
        "--listen-fd",
        str(sock.fileno()),
        "--input-address",
        args.input_address,
        "--output-address",
        args.output_address,
        "--args-json",
        json.dumps(runtime),
    ]
    if not args.endpoints_json:
        os.execv(args.binary, argv)
    processes = []
    logs = []

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for index, endpoint in enumerate(json.loads(args.endpoints_json.read_text())):
            env = os.environ.copy()
            env["VLLM_MOCK_CONSUMPTION_LOG"] = str(
                args.observer_dir / f"consumed-{index}.jsonl"
            )
            env["VLLM_MOCK_STAGE_LOG"] = str(
                args.observer_dir / f"stages-{index}.jsonl"
            )
            command = argv.copy()
            command[command.index("--input-address") + 1] = endpoint["input"]
            command[command.index("--output-address") + 1] = endpoint["output"]
            log = (args.observer_dir / f"worker-{index}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command,
                    env=env,
                    pass_fds=(sock.fileno(),),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        (args.observer_dir / "pids.json").write_text(
            json.dumps(
                {str(index): process.pid for index, process in enumerate(processes)}
            )
        )
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError("Rust frontend worker exited")
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        sock.close()
