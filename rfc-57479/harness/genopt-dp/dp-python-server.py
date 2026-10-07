"""Launch real Python API servers (one node's share) against DP mock engines.

Same observers as genopt/genopt-python-server.py (imported, unchanged). DP
additions, matching a real multi-API-server DP launch
(vllm serve --data-parallel-size N --api-server-count M):
  * server args carry --data-parallel-size N (DPLBAsyncMPClient, internal LB);
  * client_config carries stats_update_address = coordinator front publish;
  * client_index/client_count are global across nodes (node offset).
"""

import argparse
import importlib.util
import json
import signal
import sys
import time
from functools import partial
from pathlib import Path

GENOPT = Path(__file__).resolve().parent.parent / "genopt"
_spec = importlib.util.spec_from_file_location(
    "genopt_python_server", GENOPT / "genopt-python-server.py"
)
base = importlib.util.module_from_spec(_spec)
sys.modules["genopt_python_server"] = base
_spec.loader.exec_module(base)


def worker(listen_address, sock, server_args, client_config):
    client_config["client_index"] += server_args.mock_client_index_offset
    client_config["client_count"] = server_args.mock_client_count
    base.worker(listen_address, sock, server_args, client_config)


if __name__ == "__main__":
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.entrypoints.launchers.launcher import create_server_socket, setup_server
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.utils import APIServerProcessManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tokens", type=int, default=245760)
    parser.add_argument("--endpoints-json", type=Path, required=True)
    parser.add_argument("--observer-dir", type=Path, required=True)
    parser.add_argument("--independent-listeners", action="store_true")
    parser.add_argument(
        "--per-server-ports",
        action="store_true",
        help="dp-round12 round-robin arm (opt-in): worker i listens alone on port "
        "<--port> + 1 + i (no SO_REUSEPORT); needs a tree with socket_factory",
    )
    parser.add_argument("--stage-timer", action="append", default=[])
    parser.add_argument("--stats-update-address", help="omit at DP=1")
    parser.add_argument("--client-index-offset", type=int, required=True)
    parser.add_argument("--client-count", type=int, required=True)
    observer_args, server_argv = parser.parse_known_args()
    server_parser = make_arg_parser(FlexibleArgumentParser())
    server_args = server_parser.parse_args(server_argv)
    listen_address, sock = setup_server(
        server_args, reuse_port=observer_args.independent_listeners
    )
    endpoints = json.loads(observer_args.endpoints_json.read_text())
    server_args.mock_observer_dir = str(observer_args.observer_dir)
    server_args.mock_output_tokens = observer_args.output_tokens
    server_args.mock_stage_timers = observer_args.stage_timer
    server_args.mock_client_index_offset = observer_args.client_index_offset
    server_args.mock_client_count = observer_args.client_count
    manager_kwargs = {}
    if observer_args.independent_listeners:
        manager_kwargs["socket_factory"] = partial(
            create_server_socket, sock.getsockname(), reuse_port=True
        )
    elif observer_args.per_server_ports:
        host, base_port = sock.getsockname()[:2]
        next_port = iter(range(base_port + 1, base_port + 1 + len(endpoints)))
        manager_kwargs["socket_factory"] = lambda: create_server_socket(
            (host, next(next_port)), reuse_port=False
        )
    manager = APIServerProcessManager(
        listen_address,
        sock,
        server_args,
        len(endpoints),
        [e["input"] for e in endpoints],
        [e["output"] for e in endpoints],
        target_server_fn=worker,
        stats_update_address=observer_args.stats_update_address,
        **manager_kwargs,
    )

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        manager.gather_actual_addresses()
        while all(process.is_alive() for process in manager.processes):
            time.sleep(1)
        raise RuntimeError("Frontend worker exited")
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown(timeout=30)
        sock.close()
