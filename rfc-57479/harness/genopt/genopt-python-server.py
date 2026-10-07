"""Launch the real Python HTTP frontend against external mock sockets.

Derived from the frozen Codex frontend-mock-python-server.py for the
/inference/v1/generate endpoint. Observation is implementation-agnostic:
  * consumption receipts from OutputProcessor.process_outputs (positions read
    via len(state.logprobs_processor.logprobs));
  * ASGI markers for every /inference/v1/generate and /pause response:
    `serialization_ready` (http.response.start; Starlette renders the body
    before this) and `body_sent` (final http.response.body);
  * optional named stage timers given as module:Qualified.name specs.
"""

import argparse
import importlib
import inspect
import json
import os
import signal
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import uvloop
from vllm.entrypoints.launchers.api_server.entry import run_server_worker
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.entrypoints.launchers.launcher import create_server_socket, setup_server

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.utils import APIServerProcessManager

OBSERVED_PATHS = ("/inference/v1/generate", "/pause")


def install_consumption_observer(path: Path, target: int):
    original = OutputProcessor.process_outputs
    observed = set()
    milestones = {}

    def process_outputs(self, engine_core_outputs, *args, **kwargs):
        ids = [output.request_id for output in engine_core_outputs]
        result = original(self, engine_core_outputs, *args, **kwargs)
        for request_id in ids:
            state = self.request_states.get(request_id)
            if state is None or request_id in observed or state.detokenizer is None:
                continue
            count = state.detokenizer.num_output_tokens()
            if count == target or count // 16384 > milestones.get(request_id, 0):
                positions = len(state.logprobs_processor.logprobs)
                if positions != count:
                    raise RuntimeError("Consumed token/logprob position counts differ")
                if count == target:
                    observed.add(request_id)
                milestones[request_id] = count // 16384
                with path.open("a") as output:
                    output.write(
                        json.dumps(
                            {
                                "request_id": state.external_req_id or request_id,
                                "event": "consumed" if count == target else "progress",
                                "pid": os.getpid(),
                                "tokens": count,
                                "logprob_positions": positions,
                                "monotonic": time.monotonic(),
                                "wall": time.time(),
                            }
                        )
                        + "\n"
                    )
            elif count > target:
                raise RuntimeError("Mock output exceeds target")
        return result

    OutputProcessor.process_outputs = process_outputs


def install_stage_timers(directory: Path, specs: list[str]):
    """Wrap module:Qualified.name callables with wall-duration markers."""
    for spec in specs:
        module_name, qualname = spec.split(":", 1)
        owner = importlib.import_module(module_name)
        parts = qualname.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        original = getattr(owner, parts[-1], None)
        if original is None:
            # Absent in this source tree (e.g. baseline vs optimized); record it.
            with (directory / f"stages-missing-{os.getpid()}.txt").open("a") as log:
                log.write(spec + "\n")
            continue

        def wrap(original, name):
            if hasattr(original, "__func__") or isinstance(original, staticmethod):
                return None

            def record(started):
                ended = time.monotonic()
                with (directory / f"stages-{os.getpid()}.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "event": name,
                                "pid": os.getpid(),
                                "elapsed_s": ended - started,
                                "wall": time.time(),
                                "monotonic": ended,
                            }
                        )
                        + "\n"
                    )

            if inspect.iscoroutinefunction(original):

                async def measured_async(*args, **kwargs):
                    started = time.monotonic()
                    result = await original(*args, **kwargs)
                    record(started)
                    return result

                return measured_async

            def measured(*args, **kwargs):
                started = time.monotonic()
                result = original(*args, **kwargs)
                record(started)
                return result

            return measured

        wrapped = wrap(original, qualname)
        if wrapped is None:
            raise RuntimeError(f"Stage timer on static/class method unsupported: {spec}")
        setattr(owner, parts[-1], wrapped)


def install_http_observer(directory: Path):
    import vllm.entrypoints.launchers.api_server.entry as entry

    original = entry.build_app

    def build_app(*args, **kwargs):
        app = original(*args, **kwargs)
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with original_lifespan(app) as state:
                (directory / f"ready-{os.getpid()}.json").write_text(
                    json.dumps({"pid": os.getpid(), "wall": time.time()})
                )
                yield state

        app.router.lifespan_context = lifespan

        class Observe:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http" or scope["path"] not in OBSERVED_PATHS:
                    await self.app(scope, receive, send)
                    return
                state = {"bytes": 0}

                def log(event, **extra):
                    with (directory / f"http-{os.getpid()}.jsonl").open("a") as out:
                        out.write(
                            json.dumps(
                                {
                                    "event": event,
                                    "pid": os.getpid(),
                                    "path": scope["path"],
                                    "wall": time.time(),
                                    "monotonic": time.monotonic(),
                                    **extra,
                                }
                            )
                            + "\n"
                        )

                async def observed_send(message):
                    if message["type"] == "http.response.start":
                        state["status"] = message["status"]
                        log("serialization_ready", status=message["status"])
                    elif message["type"] == "http.response.body":
                        state["bytes"] += len(message.get("body", b""))
                        if not message.get("more_body", False):
                            await send(message)
                            log(
                                "body_sent",
                                status=state.get("status"),
                                bytes=state["bytes"],
                            )
                            return
                    await send(message)

                await self.app(scope, receive, observed_send)

        app.middleware_stack = Observe(
            app.middleware_stack or app.build_middleware_stack()
        )
        return app

    entry.build_app = build_app


def worker(listen_address, sock, server_args, client_config):
    import hashlib

    directory = Path(server_args.mock_observer_dir)
    modules = {}
    for name in (
        "vllm",
        "vllm.logprobs",
        "vllm.v1.utils",
        "vllm.v1.engine.logprobs",
        "vllm.v1.engine.output_processor",
        "vllm.entrypoints.scale_out.token_in_token_out.api_router",
        "vllm.entrypoints.scale_out.token_in_token_out.serving",
        "vllm.entrypoints.scale_out.token_in_token_out.protocol",
    ):
        path = Path(importlib.import_module(name).__file__)
        modules[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (directory / f"source-{os.getpid()}.json").write_text(json.dumps(modules, indent=2))
    install_consumption_observer(
        directory / f"consumed-{os.getpid()}.jsonl", server_args.mock_output_tokens
    )
    install_stage_timers(directory, server_args.mock_stage_timers)
    install_http_observer(directory)
    uvloop.run(
        run_server_worker(
            listen_address, sock, server_args, client_config=client_config
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tokens", type=int, default=245760)
    parser.add_argument("--endpoints-json", type=Path, required=True)
    parser.add_argument("--observer-dir", type=Path, required=True)
    parser.add_argument("--independent-listeners", action="store_true")
    parser.add_argument("--stage-timer", action="append", default=[])
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
    manager_kwargs = {}
    if observer_args.independent_listeners:
        manager_kwargs["socket_factory"] = partial(
            create_server_socket, sock.getsockname(), reuse_port=True
        )
    manager = APIServerProcessManager(
        listen_address,
        sock,
        server_args,
        len(endpoints),
        [e["input"] for e in endpoints],
        [e["output"] for e in endpoints],
        target_server_fn=worker,
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
