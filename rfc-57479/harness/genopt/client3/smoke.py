"""Small single-node smoke cell for client v3 cross-checks (debug only, not a measurement).

Runs on ONE node inside one `srun ... taskset -c 128-143` allocation (node01):
frontends on 136-139, mock engine on 140-141, client on 128-135 (as the harness).
Same commands as genopt-run.py minus srun/monitor; fresh services per cell.

  python smoke.py --run-dir D --impl python|rust --client v2|v3 [--format compact]
                  [--requests 4] [--output-tokens 4096] [--workers 2]
"""

import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(GENOPT_ROOT + "")
PY = GENOPT_PYTHON
SCRIPTS = ROOT / "agent_run/scripts/genopt"
BASE = ROOT / "agent_run/results/frontend-mock-256k-20261004/frozen-r1"
SRC = ROOT / "agent_run/results/generate-opt-20261004/src"
ENV = [
    "VLLM_TARGET_DEVICE=cpu", "VLLM_CPU_KVCACHE_SPACE=0", "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1", "VLLM_SERVER_DEV_MODE=1", "TOKENIZERS_PARALLELISM=false",
    "RAYON_NUM_THREADS=2", "TOKIO_WORKER_THREADS=2", "VLLM_RS_REQUEST_WORKER_THREADS=2",
]


def main(a):
    a.run_dir.mkdir(parents=True)
    obs = a.run_dir / "observe" / "node"
    obs.mkdir(parents=True)
    ip = a.ip
    eps = [{"input": f"tcp://{ip}:{a.zmq_base + 2 * i}", "output": f"tcp://{ip}:{a.zmq_base + 2 * i + 1}"}
           for i in range(a.workers)]
    ep = a.run_dir / "endpoints.json"
    ep.write_text(json.dumps(eps))
    procs, logs = {}, []

    def launch(label, cpus, cmd, pythonpath=None, extra=()):
        env = ["/usr/bin/env", *ENV, *extra] + ([f"PYTHONPATH={pythonpath}"] if pythonpath else [])
        log = (a.run_dir / f"{label}.log").open("w")
        logs.append(log)
        procs[label] = subprocess.Popen(["taskset", "-c", cpus, *env, *cmd], stdout=log, stderr=subprocess.STDOUT)

    model = BASE / "model-assets"
    if a.impl == "python":
        cmd = [PY, str(SCRIPTS / "genopt-python-server.py"), "--endpoints-json", str(ep), "--observer-dir", str(obs),
               "--output-tokens", str(a.output_tokens), "--model", str(model), "--served-model-name", "hy4-mock",
               "--host", "0.0.0.0", "--port", str(a.port), "--max-logprobs", "128", "--disable-log-stats"]
        if a.listeners:
            cmd.append("--independent-listeners")
        launch("server", "136-139", cmd, a.source)
    else:
        cmd = [PY, str(SCRIPTS / "genopt-rust-server.py"), "--binary", str(a.binary), "--model", str(model),
               "--endpoints-json", str(ep), "--observer-dir", str(obs), "--output-tokens", str(a.output_tokens),
               "--port", str(a.port)]
        launch("server", "136-139", cmd, BASE / "python-baseline")
    launch("engine", "140-141", [PY, str(SCRIPTS / "genopt-engine.py"), "--endpoints-json", str(ep),
                                 "--expected-requests", str(a.requests), "--output-tokens", str(a.output_tokens),
                                 "--chunk-size", "1024", "--token-pool", str(model / "token-pool-v2.json"),
                                 "--routed-experts-layers", str(a.routed_experts_layers)],
           BASE / "python-baseline")
    status = "FAIL"
    try:
        deadline = time.monotonic() + 300
        while True:
            for k, p in procs.items():
                if p.poll() is not None:
                    raise RuntimeError(f"{k} exited early {p.returncode}")
            if a.impl == "python":
                ready = len(list(obs.glob("ready-*.json")))
            else:
                ready = sum("ready to accept requests" in f.read_text() for f in obs.glob("worker-*.log"))
            if ready == a.workers:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("readiness")
            time.sleep(0.5)
        url = f"http://{ip}:{a.port}"
        common = ["--urls", url, "--requests", str(a.requests), "--output-tokens", str(a.output_tokens),
                  "--barrier-timeout", "600", "--validate-requests", str(a.validate_requests),
                  "--token-pool", str(model / "token-pool-v2.json"), "--consumption-log", str(a.run_dir / "observe"),
                  "--result", str(a.run_dir / "result.json")]
        if a.format:
            common += ["--logprobs-format", a.format]
        if a.compact_no_sampled:
            common.append("--compact-no-sampled")
        if a.compact_no_ranks:
            common.append("--compact-no-ranks")
        if a.routed_experts_layers:
            common += ["--routed-experts-layers", str(a.routed_experts_layers)]
        # v2 needs workers <= requests (a worker with no request exits -> v2 aborts)
        common += ["--workers", str(min(8, a.requests)), "--max-reading", str(a.max_reading)]
        if a.client == "v3":
            client = [str(a.client3_binary), *common]
            if a.corrupt is not None or a.dump_dir:
                client.append("--debug-hooks")  # v3 refuses hook env vars without it
        else:
            client = [PY, str(SCRIPTS / "genopt-client2.py"), *common]
        extra = [f"GENOPT_CLIENT3_CORRUPT_EXPECTED_POSITION={a.corrupt}"] if a.corrupt is not None else []
        if a.dump_dir:
            a.dump_dir.mkdir(parents=True, exist_ok=True)
            extra.append(f"GENOPT_CLIENT3_DUMP_DIR={a.dump_dir}")
        launch("client", "128-135", client, BASE / "python-baseline", extra)
        while procs["client"].poll() is None:
            time.sleep(0.2)
        rc = procs["client"].returncode
        status = "PASS" if rc == 0 else ("DEBUG" if rc == 2 else f"CLIENT_EXIT_{rc}")  # v3 exits 2 with hooks
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        for log in logs:
            log.close()
        (a.run_dir / "smoke.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "status": status}))
        print(a.run_dir, status, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--impl", choices=("python", "rust"), required=True)
    p.add_argument("--client", choices=("v2", "v3"), required=True)
    p.add_argument("--format", choices=("openai", "compact"))
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--output-tokens", type=int, default=4096)
    p.add_argument("--validate-requests", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-reading", type=int, default=4)
    p.add_argument("--listeners", action="store_true")
    p.add_argument("--source", type=Path, default=SRC / "python-3e947ef")
    p.add_argument("--binary", type=Path, default=SRC / "rust/vllm-rs-c314734-obs")
    p.add_argument("--client3-binary", type=Path, default=Path("/tmp/claude-client3-target/release/genopt-client3"))
    p.add_argument("--corrupt", type=int, help="v3 negative control: corrupt expectation at this position")
    p.add_argument("--compact-no-sampled", action="store_true")
    p.add_argument("--compact-no-ranks", action="store_true")
    p.add_argument("--routed-experts-layers", type=int, default=0)
    p.add_argument("--dump-dir", type=Path, help="v3: record bodies here (debug)")
    p.add_argument("--ip", default=_genopt_os.environ.get("GENOPT_IP_NODE01", "127.0.0.1"))
    p.add_argument("--port", type=int, default=8299)
    p.add_argument("--zmq-base", type=int, default=23000)
    main(p.parse_args())
