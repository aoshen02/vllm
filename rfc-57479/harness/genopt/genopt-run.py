"""Run one isolated frontend-mock cell for /inference/v1/generate.

Derived from the frozen Codex frontend-mock-run.py. Topology and resource
layout are unchanged (32 frontends/node on cores 0-127 of two nodes, engine
mock on node A cores 128-131, client on node B cores 128-135, monitors on
136-143). New: implementation/source/binary/listeners/format are explicit
arguments so every arm is described by its command line, and the node pair
is selectable.
"""

import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(GENOPT_ROOT + "")
PYTHON = GENOPT_PYTHON
# Frozen native measurement client v3 (agent_run/scripts/genopt/client3, see reports/.../client3.md)
CLIENT3_BINARY = ROOT / "agent_run/results/generate-opt-20261004/src/client3/genopt-client3-b0c7182e"
IPS = {
    "node01": _genopt_os.environ.get("GENOPT_IP_NODE01", "127.0.0.1"),
    "node02": _genopt_os.environ.get("GENOPT_IP_NODE02", "127.0.0.1"),
    "node03": _genopt_os.environ.get("GENOPT_IP_NODE03", "127.0.0.1"),
    "node04": _genopt_os.environ.get("GENOPT_IP_NODE04", "127.0.0.1"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    nodes = [(name, IPS[name]) for name in args.nodes]
    args.run_dir.mkdir(parents=True)
    observers = args.run_dir / "observe"
    observers.mkdir()
    scripts = args.scripts_dir
    processes = {}
    logs = []
    commands = {}
    started = time.time()

    def launch(label, node, command, cpus, pythonpath=None, extra_env=()):
        env = [
            "/usr/bin/env",
            # client v3 debug hooks must never leak into measurement runs (srun exports env)
            "-u",
            "GENOPT_CLIENT3_DUMP_DIR",
            "-u",
            "GENOPT_CLIENT3_CORRUPT_EXPECTED_POSITION",
            "VLLM_TARGET_DEVICE=cpu",
            "VLLM_CPU_KVCACHE_SPACE=0",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "VLLM_SERVER_DEV_MODE=1",
            "TOKENIZERS_PARALLELISM=false",
            "RAYON_NUM_THREADS=2",
            "TOKIO_WORKER_THREADS=2",
            "VLLM_RS_REQUEST_WORKER_THREADS=2",
            *extra_env,
        ]
        if pythonpath:
            env.append(f"PYTHONPATH={pythonpath}")
        invocation = [
            "srun",
            f"--jobid={args.job}",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={node}",
            "--cpu-bind=none",
            "taskset",
            "-c",
            "136-143",
            PYTHON,
            str(scripts / "genopt-monitor.py"),
            "--log",
            str(args.run_dir / f"metrics-{label}.jsonl"),
            "--memory-floor-gib",
            str(args.memory_floor_gib),
            "--",
            "taskset",
            "-c",
            cpus,
            *env,
            *command,
        ]
        commands[label] = invocation
        (args.run_dir / "commands.json").write_text(json.dumps(commands, indent=2))
        log = (args.run_dir / f"{label}.log").open("w")
        logs.append(log)
        processes[label] = subprocess.Popen(
            invocation, stdout=log, stderr=subprocess.STDOUT
        )

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    summary = {
        "label": args.label,
        "impl": args.impl,
        "source": str(args.source) if args.source else None,
        "binary": str(args.binary) if args.binary else None,
        "binary_sha256": sha256(args.binary) if args.binary else None,
        "independent_listeners": args.listeners,
        "logprobs_format": args.logprobs_format or "absent",
        "compact_include_sampled": args.compact_include_sampled,
        "compact_include_ranks": args.compact_include_ranks,
        "routed_experts_layers": args.routed_experts_layers,
        "nodes": args.nodes,
        "client_node": args.client_node or args.nodes[1],
        "job": args.job,
        "requests": args.requests,
        "input_tokens": 16384,
        "output_tokens": args.output_tokens,
        "top_logprobs": 128,
        "workers_per_node": args.workers_per_node,
        "chunk_size": args.chunk_size,
        "validate_requests": args.validate_requests,
        "client": args.client,
        "client_workers": args.client_workers if args.client in ("v2", "v3") else 1,
        "max_reading": args.max_reading if args.client in ("v2", "v3") else None,
        "client3_binary": str(args.client3_binary) if args.client == "v3" else None,
        "client3_binary_sha256": sha256(args.client3_binary) if args.client == "v3" else None,
        "started_wall": started,
        "status": "RUNNING",
    }
    (args.run_dir / "run.json").write_text(json.dumps(summary, indent=2))
    try:
        all_endpoints = []
        for node_index, (node, ip) in enumerate(nodes):
            directory = observers / node
            directory.mkdir()
            endpoints = [
                {
                    "input": f"tcp://{ip}:{21000 + node_index * 100 + index * 2}",
                    "output": f"tcp://{ip}:{21001 + node_index * 100 + index * 2}",
                }
                for index in range(args.workers_per_node)
            ]
            all_endpoints.extend(endpoints)
            endpoint_path = args.run_dir / f"endpoints-{node}.json"
            endpoint_path.write_text(json.dumps(endpoints, indent=2))
            if args.impl == "python":
                command = [
                    PYTHON,
                    str(scripts / "genopt-python-server.py"),
                    "--endpoints-json",
                    str(endpoint_path),
                    "--observer-dir",
                    str(directory),
                    "--output-tokens",
                    str(args.output_tokens),
                    *(f"--stage-timer={spec}" for spec in args.stage_timer),
                    "--model",
                    str(args.model),
                    "--served-model-name",
                    "hy4-mock",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8199",
                    "--max-logprobs",
                    "128",
                    "--disable-log-stats",
                    *args.server_arg,
                ]
                if args.listeners:
                    command.append("--independent-listeners")
                launch(f"server-{node}", node, command, "0-127", args.source)
            else:
                command = [
                    PYTHON,
                    str(scripts / "genopt-rust-server.py"),
                    "--binary",
                    str(args.binary),
                    "--model",
                    str(args.model),
                    "--endpoints-json",
                    str(endpoint_path),
                    "--observer-dir",
                    str(directory),
                    "--output-tokens",
                    str(args.output_tokens),
                    "--port",
                    "8199",
                ]
                if args.routed_experts_layers:
                    command.append("--enable-return-routed-experts")
                launch(f"server-{node}", node, command, "0-127", args.engine_source)

        engine_endpoints = args.run_dir / "endpoints-all.json"
        engine_endpoints.write_text(json.dumps(all_endpoints, indent=2))
        launch(
            "engine",
            nodes[0][0],
            [
                PYTHON,
                str(scripts / "genopt-engine.py"),
                "--endpoints-json",
                str(engine_endpoints),
                "--expected-requests",
                str(args.requests),
                "--output-tokens",
                str(args.output_tokens),
                "--chunk-size",
                str(args.chunk_size),
                "--routed-experts-layers",
                str(args.routed_experts_layers),
                "--token-pool",
                str(args.model / "token-pool-v2.json"),
            ],
            "128-131",
            args.engine_source,
        )
        deadline = time.monotonic() + 600
        while True:
            for label, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{label} exited before readiness: {process.returncode}"
                    )
            if args.impl == "python":
                ready = len(list(observers.rglob("ready-*.json")))
            else:
                ready = sum(
                    "ready to accept requests" in path.read_text()
                    for path in observers.rglob("worker-*.log")
                )
            if ready == len(nodes) * args.workers_per_node:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Frontend readiness: {ready} workers")
            time.sleep(1)
        summary["all_ready_wall"] = time.time()
        (args.run_dir / "run.json").write_text(json.dumps(summary, indent=2))
        if args.client == "v3":
            client_exe = [str(args.client3_binary)]
        else:
            client_exe = [
                PYTHON,
                str(scripts / ("genopt-client2.py" if args.client == "v2" else "genopt-client.py")),
            ]
        client = [
            *client_exe,
            "--urls",
            *(f"http://{ip}:8199" for _, ip in nodes),
            "--requests",
            str(args.requests),
            "--output-tokens",
            str(args.output_tokens),
            "--barrier-timeout",
            str(args.barrier_timeout),
            "--validate-requests",
            str(args.validate_requests),
            "--token-pool",
            str(args.model / "token-pool-v2.json"),
            "--consumption-log",
            str(observers),
            "--result",
            str(args.run_dir / "result.json"),
        ]
        if args.logprobs_format:
            client += ["--logprobs-format", args.logprobs_format]
        if not args.compact_include_sampled:
            client.append("--compact-no-sampled")
        if not args.compact_include_ranks:
            client.append("--compact-no-ranks")
        if args.routed_experts_layers:
            client += ["--routed-experts-layers", str(args.routed_experts_layers)]
        if args.client in ("v2", "v3"):
            client += ["--workers", str(args.client_workers), "--max-reading", str(args.max_reading)]
        launch("client", args.client_node or nodes[1][0], client, "128-135", args.engine_source)
        while processes["client"].poll() is None:
            for label, process in processes.items():
                if label != "client" and process.poll() is not None:
                    raise RuntimeError(
                        f"{label} exited during cohort: {process.returncode}"
                    )
            time.sleep(1)
        if processes["client"].returncode:
            raise RuntimeError(f"Client failed: {processes['client'].returncode}")
        summary["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for label, process in processes.items():
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            summary.setdefault("exit_codes", {})[label] = process.returncode
        for log in logs:
            log.close()
        summary["ended_wall"] = time.time()
        (args.run_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    base = ROOT / "agent_run/results/frontend-mock-256k-20261004/frozen-r1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--impl", choices=("python", "rust"), required=True)
    parser.add_argument("--source", type=Path, help="PYTHONPATH for python impl")
    parser.add_argument("--binary", type=Path, help="vllm-rs binary for rust impl")
    parser.add_argument("--listeners", action="store_true")
    parser.add_argument("--logprobs-format", choices=("openai", "compact"))
    parser.add_argument("--compact-no-sampled", dest="compact_include_sampled", action="store_false")
    parser.add_argument("--compact-no-ranks", dest="compact_include_ranks", action="store_false")
    parser.add_argument("--routed-experts-layers", type=int, default=0)
    parser.add_argument("--stage-timer", action="append", default=[])
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--engine-source",
        type=Path,
        default=base / "python-baseline",
        help="PYTHONPATH for mock engine/client (vLLM types + msgpack encoder)",
    )
    parser.add_argument("--model", type=Path, default=base / "model-assets")
    parser.add_argument(
        "--nodes", nargs=2, default=["node02", "node03"]
    )
    parser.add_argument("--client-node", choices=sorted(IPS), help="default: second frontend node")
    parser.add_argument("--client", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument(
        "--client3-binary",
        type=Path,
        default=CLIENT3_BINARY,
        help="native client v3 binary (used only with --client v3)",
    )
    parser.add_argument("--client-workers", type=int, default=8)
    parser.add_argument("--max-reading", type=int, default=4)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=245760)
    parser.add_argument("--workers-per-node", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--barrier-timeout", type=float, default=3600)
    parser.add_argument("--validate-requests", type=int, default=4)
    parser.add_argument("--memory-floor-gib", type=int, default=64)
    args = parser.parse_args()
    if (args.impl == "python") != (args.source is not None) or (
        args.impl == "rust"
    ) != (args.binary is not None):
        parser.error("python needs --source; rust needs --binary")
    raise SystemExit(main(args))
