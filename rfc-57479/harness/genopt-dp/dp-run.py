"""Run one data-parallel frontend-mock cell for /inference/v1/generate.

Topology (derived from genopt/genopt-run.py, which stays unchanged):
  * frontends: --workers-per-node API server processes on cores 0-127 of each
    --nodes node (Python: APIServerProcessManager per node with global
    client_index; Rust: one bootstrapped process per endpoint);
  * the REAL DPCoordinatorProc (dp-coordinator.py) on the first node, cores
    136-137, plus its publish spy;
  * --dp mock engine ranks (dp-engine.py) on cores 128-135 of the engine
    nodes (default: first frontend node, then the second for DP > 8); every
    rank connects to every frontend and to the coordinator; rank 0 hosts the
    DP-group reducer;
  * the client on --client-node cores 128-135: client v3 (official barrier
    + pause cohort, frozen binary) or dp-client.py (pause at a time, sleep,
    resume, resubmit);
  * a genopt-monitor.py per launched process group (cores 136-143).
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
HERE = Path(__file__).resolve().parent
GENOPT = HERE.parent / "genopt"
# Client v3.1 = v3 b0c7182e + rs-pr-stack client3-r15b.patch (accepts the compact block with or without
# "sampled_slot": false when include_sampled=false). v3 binary kept: src/client3/genopt-client3-b0c7182e.
CLIENT3_BINARY = ROOT / "agent_run/results/generate-opt-20261004/src/client3/genopt-client3-efa44ef2"
IPS = {
    "node01": _genopt_os.environ.get("GENOPT_IP_NODE01", "127.0.0.1"),
    "node02": _genopt_os.environ.get("GENOPT_IP_NODE02", "127.0.0.1"),
    "node03": _genopt_os.environ.get("GENOPT_IP_NODE03", "127.0.0.1"),
    "node04": _genopt_os.environ.get("GENOPT_IP_NODE04", "127.0.0.1"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_layout(dp, engine_nodes):
    """(node, cpus) per rank: cores 128-135 of each engine node, filled in order."""
    slots = [(n, c) for n in engine_nodes for c in range(128, 136)]
    if dp > len(slots):
        raise ValueError(f"DP={dp} needs more than {len(slots)} engine cores")
    per = max(1, min(2, len(slots) // dp))
    if dp * per > len(slots):
        per = 1
    return [(slots[r * per][0], ",".join(str(slots[r * per + k][1]) for k in range(per)))
            for r in range(dp)]


def main(args):
    nodes = [(name, IPS[name]) for name in args.nodes]
    engine_nodes = args.engine_nodes or ([args.nodes[0]] if args.dp <= 8 else args.nodes)
    layout = engine_layout(args.dp, engine_nodes)
    single = args.dp == 1  # real DP=1: no coordinator; genopt-engine.py is the engine
    coord_node = args.nodes[0]
    coord_ip = IPS[coord_node]
    coord = {
        "front_publish": f"tcp://{coord_ip}:22000",
        "back_output": f"tcp://{coord_ip}:22001",
        "back_publish": f"tcp://{coord_ip}:22002",
    }
    dp_group = f"tcp://{IPS[layout[0][0]]}:22010"
    args.run_dir.mkdir(parents=True)
    observers = args.run_dir / "observe"
    observers.mkdir()
    processes, logs, commands = {}, [], {}
    started = time.time()

    def launch(label, node, command, cpus, pythonpath=None, extra_env=()):
        env = ["/usr/bin/env", "-u", "GENOPT_CLIENT3_DUMP_DIR", "-u", "GENOPT_CLIENT3_CORRUPT_EXPECTED_POSITION",
               "VLLM_TARGET_DEVICE=cpu", "VLLM_CPU_KVCACHE_SPACE=0", "OMP_NUM_THREADS=1",
               "OPENBLAS_NUM_THREADS=1", "VLLM_SERVER_DEV_MODE=1", "TOKENIZERS_PARALLELISM=false",
               "RAYON_NUM_THREADS=2", "TOKIO_WORKER_THREADS=2", "VLLM_RS_REQUEST_WORKER_THREADS=2",
               *extra_env]
        if pythonpath:
            env.append(f"PYTHONPATH={pythonpath}")
        invocation = ["srun", f"--jobid={args.job}", "--overlap", "--nodes=1", "--ntasks=1",
                      f"--nodelist={node}", "--cpu-bind=none", "taskset", "-c", "136-143", PYTHON,
                      str(GENOPT / "genopt-monitor.py"), "--log", str(args.run_dir / f"metrics-{label}.jsonl"),
                      "--memory-floor-gib", str(args.memory_floor_gib), "--", "taskset", "-c", cpus,
                      *env, *command]
        commands[label] = invocation
        (args.run_dir / "commands.json").write_text(json.dumps(commands, indent=2))
        log = (args.run_dir / f"{label}.log").open("w")
        logs.append(log)
        processes[label] = subprocess.Popen(invocation, stdout=log, stderr=subprocess.STDOUT)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    summary = {
        "label": args.label, "impl": args.impl,
        "source": str(args.source) if args.source else None,
        "binary": str(args.binary) if args.binary else None,
        "binary_sha256": sha256(args.binary) if args.binary else None,
        "dp": args.dp, "engine_layout": layout, "coordinator": coord, "dp_group": dp_group,
        "coordinator_source": str(args.engine_source),
        "logprobs_format": args.logprobs_format or "absent",
        "compact_include_sampled": args.compact_include_sampled,
        "compact_include_ranks": args.compact_include_ranks,
        "nodes": args.nodes, "client_node": args.client_node, "job": args.job,
        "requests": args.requests, "input_tokens": 16384, "output_tokens": args.output_tokens,
        "top_logprobs": 128, "workers_per_node": args.workers_per_node,
        "tokens_per_step": args.tokens_per_step, "engine_args": args.engine_arg, "step_ms": args.step_ms, "lockstep": args.lockstep,
        "client": args.client, "client_args": args.client_arg,
        "rust_multiport": args.rust_multiport, "client_cpus": args.client_cpus,
        "route_by": args.route_by, "rust_threads": args.rust_threads, "server_env": args.server_env,
        "client3_binary_sha256": sha256(CLIENT3_BINARY) if args.client == "v3" else None,
        "started_wall": started, "status": "RUNNING",
    }
    (args.run_dir / "run.json").write_text(json.dumps(summary, indent=2))
    try:
        if not single:
          launch("coordinator", coord_node,
               [PYTHON, str(HERE / "dp-coordinator.py"), "--engine-count", str(args.dp),
                "--front-publish", coord["front_publish"], "--back-output", coord["back_output"],
                "--back-publish", coord["back_publish"],
                "--spy-log", str(args.run_dir / "coordinator-publish.jsonl"),
                "--source-log", str(args.run_dir / "coordinator-source.json")],
               "136-137", args.engine_source, extra_env=("VLLM_LOGGING_LEVEL=DEBUG",))
        all_endpoints = []
        total_clients = len(nodes) * args.workers_per_node
        multiport = args.impl == "rust" and args.rust_multiport and not single
        urls = [f"http://{ip}:8199" for _, ip in nodes]
        rank_endpoints = {}
        if multiport:
            # DP supervisor multi-port / external-LB layout (dp_supervisor.py): one
            # Rust frontend per DP rank, its own port, connected only to its engine
            # (engine_count=1), so client_index 0 is the correct index.
            urls = []
            for rank in range(args.dp):
                node_index = rank * len(nodes) // args.dp
                local = rank - (node_index * args.dp + len(nodes) - 1) // len(nodes)
                node, ip = nodes[node_index]
                directory = observers / f"{node}-r{rank}"
                directory.mkdir()
                endpoint = {"input": f"tcp://{ip}:{21000 + node_index * 100 + local * 2}",
                            "output": f"tcp://{ip}:{21001 + node_index * 100 + local * 2}"}
                rank_endpoints[rank] = args.run_dir / f"endpoints-r{rank}.json"
                rank_endpoints[rank].write_text(json.dumps([endpoint], indent=2))
                all_endpoints.append(endpoint)
                urls.append(f"http://{ip}:{8199 + local}")
                command = [PYTHON, str(HERE / "dp-rust-server.py"), "--binary", str(args.binary),
                           "--model", str(args.model), "--endpoints-json", str(rank_endpoints[rank]),
                           "--observer-dir", str(directory), "--output-tokens", str(args.output_tokens),
                           "--port", str(8199 + local), "--dp-size", str(args.dp),
                           "--engine-start-index", str(rank), "--engine-count", "1",
                           "--coordinator-address", coord["front_publish"]]
                threads = str(args.rust_threads)
                launch(f"server-{node}-r{rank}", node, command, "0-127", args.engine_source,
                       extra_env=(f"TOKIO_WORKER_THREADS={threads}",
                                  f"VLLM_RS_REQUEST_WORKER_THREADS={threads}", *args.server_env))
        for node_index, (node, ip) in enumerate([] if multiport else nodes):
            directory = observers / node
            directory.mkdir()
            endpoints = [{"input": f"tcp://{ip}:{21000 + node_index * 100 + i * 2}",
                          "output": f"tcp://{ip}:{21001 + node_index * 100 + i * 2}"}
                         for i in range(args.workers_per_node)]
            all_endpoints.extend(endpoints)
            endpoint_path = args.run_dir / f"endpoints-{node}.json"
            endpoint_path.write_text(json.dumps(endpoints, indent=2))
            if args.impl == "python":
                command = [PYTHON, str(HERE / "dp-python-server.py"),
                           "--endpoints-json", str(endpoint_path), "--observer-dir", str(directory),
                           "--output-tokens", str(args.output_tokens),
                           *([] if single else ["--stats-update-address", coord["front_publish"]]),
                           "--client-index-offset", str(node_index * args.workers_per_node),
                           "--client-count", str(total_clients),
                           "--model", str(args.model), "--served-model-name", "hy4-mock",
                           "--host", "0.0.0.0", "--port", "8199", "--max-logprobs", "128",
                           "--disable-log-stats", "--data-parallel-size", str(args.dp),
                           *args.server_arg]
                if args.listeners:
                    command.append("--independent-listeners")
                if args.round_robin_ports:
                    command.append("--per-server-ports")
                launch(f"server-{node}", node, command, "0-127", args.source)
            else:
                command = [PYTHON, str(HERE / "dp-rust-server.py"), "--binary", str(args.binary),
                           "--model", str(args.model), "--endpoints-json", str(endpoint_path),
                           "--observer-dir", str(directory), "--output-tokens", str(args.output_tokens),
                           "--port", "8199", "--dp-size", str(args.dp),
                           *([] if single else ["--coordinator-address", coord["front_publish"]])]
                threads = str(args.rust_threads)
                launch(f"server-{node}", node, command, "0-127", args.engine_source,
                       extra_env=(f"TOKIO_WORKER_THREADS={threads}",
                                  f"VLLM_RS_REQUEST_WORKER_THREADS={threads}", *args.server_env))
        if args.profile_threads:
            node0 = (f"{args.nodes[0]}-r0" if multiport else args.nodes[0])
            launch("threadprof", args.nodes[0],
                   [PYTHON, str(HERE / "r13-threadprof.py"), "--pids-json", str(observers / node0 / "pids.json"),
                    "--log", str(args.run_dir / "threadprof.jsonl")], "136-143")
        engine_endpoints = args.run_dir / "endpoints-all.json"
        engine_endpoints.write_text(json.dumps(all_endpoints, indent=2))
        if single:
            launch("engine", args.nodes[0],
                   [PYTHON, str(GENOPT / "genopt-engine.py"), "--endpoints-json", str(engine_endpoints),
                    "--expected-requests", str(args.requests), "--output-tokens", str(args.output_tokens),
                    "--chunk-size", str(args.tokens_per_step), "--routed-experts-layers", "0",
                    "--token-pool", str(args.model / "token-pool-v2.json"),
                    *(["--route-by-client-index"] if args.route_by == "client-index" else [])],
                   "128-131", args.engine_source)
        for rank, (node, cpus) in enumerate([] if single else layout):
            launch(f"engine-r{rank}", node,
                   [PYTHON, str(HERE / "dp-engine.py"), "--rank", str(rank), "--dp-size", str(args.dp),
                    "--endpoints-json", str(rank_endpoints.get(rank, engine_endpoints)),
                    "--coord-in", coord["back_publish"],
                    "--coord-out", coord["back_output"], "--dp-group-address", dp_group,
                    "--token-pool", str(args.model / "token-pool-v2.json"),
                    "--output-tokens", str(args.output_tokens),
                    "--tokens-per-step", str(args.tokens_per_step), "--step-ms", str(args.step_ms),
                    "--idle-step-ms", str(args.idle_step_ms), "--lockstep", str(args.lockstep),
                    *args.engine_arg,
                    "--log", str(args.run_dir / f"engine-r{rank}.jsonl"), "--route-by", args.route_by],
                   cpus, args.engine_source)
        deadline = time.monotonic() + args.ready_timeout
        while True:
            for label, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited before readiness: {process.returncode}")
            if args.impl == "python":
                ready = len(list(observers.rglob("ready-*.json")))
            else:
                ready = sum("ready to accept requests" in p.read_text() for p in observers.rglob("worker-*.log"))
            if ready == (args.dp if multiport else len(nodes) * args.workers_per_node):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Frontend readiness: {ready} workers")
            time.sleep(1)
        summary["all_ready_wall"] = time.time()
        if args.round_robin_ports:
            # dp-round12 round-robin arm: one URL per API server (port 8200 + i), ordered
            # node-interleaved, so client request i (urls[i % k]) goes to server i mod k.
            urls = [f"http://{ip}:{8200 + i}" for i in range(args.workers_per_node) for _, ip in nodes]
            summary["round_robin_urls"] = len(urls)
        (args.run_dir / "run.json").write_text(json.dumps(summary, indent=2))
        common = ["--urls", *urls, "--requests", str(args.requests),
                  "--token-pool", str(args.model / "token-pool-v2.json"),
                  "--result", str(args.run_dir / "result.json")]
        if args.logprobs_format:
            common += ["--logprobs-format", args.logprobs_format]
        if not args.compact_include_sampled:
            common.append("--compact-no-sampled")
        if not args.compact_include_ranks:
            common.append("--compact-no-ranks")
        if args.client == "v3":
            client = [str(CLIENT3_BINARY), *common, "--output-tokens", str(args.output_tokens),
                      "--barrier-timeout", "3600", "--validate-requests", str(args.validate_requests),
                      "--consumption-log", str(observers), "--workers", "8", "--max-reading", "4"]
            client_env = args.engine_source
        else:
            client = [PYTHON, str(HERE / "dp-client.py"), *common, "--length-cap", str(args.output_tokens)]
            client_env = None
        client += args.client_arg
        launch("client", args.client_node, client, args.client_cpus, client_env)
        while processes["client"].poll() is None:
            for label, process in processes.items():
                if label != "client" and process.poll() is not None:
                    raise RuntimeError(f"{label} exited during cohort: {process.returncode}")
            time.sleep(1)
        if processes["client"].returncode:
            raise RuntimeError(f"Client failed: {processes['client'].returncode}")
        summary["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        time.sleep(args.linger_s)  # let engines/coordinator log the final wave state
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
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--impl", choices=("python", "rust"), required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--listeners", action="store_true")
    p.add_argument("--round-robin-ports", action="store_true",
                   help="dp-round12 (opt-in, Python): each API server on its own port (8200+i), client "
                        "sends request i to server i mod k (deterministic round-robin)")
    p.add_argument("--dp", type=int, required=True)
    p.add_argument("--engine-nodes", nargs="+", choices=sorted(IPS))
    p.add_argument("--logprobs-format", choices=("openai", "compact"))
    p.add_argument("--compact-no-sampled", dest="compact_include_sampled", action="store_false")
    p.add_argument("--compact-no-ranks", dest="compact_include_ranks", action="store_false")
    p.add_argument("--server-arg", action="append", default=[])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--engine-source", type=Path, default=base / "python-baseline")
    p.add_argument("--model", type=Path, default=base / "model-assets")
    p.add_argument("--nodes", nargs="+", default=["node02", "node03"],
                   help="frontend nodes (1 or 2); engines/coordinator on the first")
    p.add_argument("--route-by", choices=("client-index", "connection"), default="client-index",
                   help="engine output routing; client-index = real engine (dp-round2 default), connection = dp-round1")
    p.add_argument("--rust-multiport", action="store_true",
                   help="Rust: one frontend per DP rank on its own port (DP supervisor multi-port/external-LB mode)")
    p.add_argument("--client-cpus", default="128-135", help="official: 128-135 (8 cores)")
    p.add_argument("--profile-threads", action="store_true",
                   help="sample per-thread CPU of the first Rust frontend process (r13-threadprof.py)")
    p.add_argument("--server-env", action="append", default=[],
                   help="extra KEY=VALUE for Rust frontend processes (e.g. ablation toggles)")
    p.add_argument("--rust-threads", type=int, default=2,
                   help="TOKIO_WORKER_THREADS and VLLM_RS_REQUEST_WORKER_THREADS for Rust frontends")
    p.add_argument("--client-node", choices=sorted(IPS), default="node04")
    p.add_argument("--client", choices=("v3", "dp"), default="v3")
    p.add_argument("--client-arg", action="append", default=[])
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--output-tokens", type=int, default=245760)
    p.add_argument("--workers-per-node", type=int, default=32)
    p.add_argument("--tokens-per-step", type=int, default=1024)
    p.add_argument("--step-ms", type=float, default=0.0)
    p.add_argument("--idle-step-ms", type=float, default=1.0)
    p.add_argument("--lockstep", type=int, default=1)
    p.add_argument("--engine-arg", action="append", default=[],
                   help="extra dp-engine.py argument (e.g. --engine-arg=--int32-ids)")
    p.add_argument("--validate-requests", type=int, default=4)
    p.add_argument("--memory-floor-gib", type=int, default=64)
    p.add_argument("--ready-timeout", type=float, default=600)
    p.add_argument("--linger-s", type=float, default=3.0)
    a = p.parse_args()
    if (a.impl == "python") != (a.source is not None) or (a.impl == "rust") != (a.binary is not None):
        p.error("python needs --source; rust needs --binary")
    raise SystemExit(main(a))
