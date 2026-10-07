# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Golden harness (r15, claude-python-subagent): full-router bytes and framing
of /inference/v1/generate scenarios on any tree (base, in-tree compact, plugin
prototype with or without the plugin).

Engine rows go through the real OutputProcessor. Endpoint plugins are attached
through the real mechanism (attach_endpoint_plugins +
init_endpoint_plugins_state), so VLLM_PLUGINS / PYTHONPATH decide whether the
plugin loads. ``main`` prints one JSON line per scenario: status, number of
body messages, content-length, body sha256.
Usage: TREE=label python golden_harness.py [scenario-filter]

The plugin has one compact format, the former ``compact_include_sampled=false``
+ ``compact_include_ranks=false`` ("RL-lean"). With ``GOLDEN_LEAN_SWITCHES=1``
every compact request carries those two switches, and ``"sampled_slot":false,``
(which the plugin does not send) is removed from the body before it is summarized,
so a tree with the switches (the in-tree implementation 214f522fbf) gives the
reference bytes.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from argparse import Namespace
from copy import copy
from unittest.mock import MagicMock

import numpy as np
import torch
from fastapi import FastAPI

from tests.entrypoints.scale_out.token_in_token_out.test_generate_stream import (
    MODEL_NAME,
    _build_serving_tokens,
    _mock_engine,
)
from vllm.entrypoints.scale_out.token_in_token_out import api_router
from vllm.sampling_params import RequestOutputKind  # noqa: F401
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

_TOKENIZER = []


def _tokenizer():
    if not _TOKENIZER:
        from vllm.tokenizers import get_tokenizer

        _TOKENIZER.append(get_tokenizer(MODEL_NAME))
    return _TOKENIZER[0]


def _prompt_logprobs_tensors(k):
    """Prompt logprobs for prompt [1, 2, 3] (positions 1 and 2)."""
    ids = torch.tensor([[2, 1000 + k, 3000][: k + 1], [3, 5000, 7000][: k + 1]])
    values = torch.tensor([[-0.5, -1.5, -2.5][: k + 1], [-0.25, -3.0, -4.0][: k + 1]])
    return LogprobsTensors(ids, values, torch.tensor([4, 1]))


try:
    from vllm.plugins.endpoint_plugins.interface import (
        attach_endpoint_plugins,
        init_endpoint_plugins_state,
    )
except ImportError:  # trees without endpoint plugins
    attach_endpoint_plugins = init_endpoint_plugins_state = None


def freeze_time() -> None:
    time.time = lambda: 1700000000.0  # deterministic "created"


def rows(start, count, width, seed=0, narrow_at=None):
    rng = np.random.default_rng(seed + start)
    top = (
        np.stack(
            [rng.choice(50_000, size=width - 1, replace=False) for _ in range(count)]
        ).astype(np.int64)
        if width > 1
        else np.empty((count, 0), np.int64)
    )
    positions = np.arange(start, start + count)
    sampled = (
        np.where(positions % 2 == 0, top[:, 0], 50_000 + positions)
        if width > 1
        else 50_000 + positions
    )
    ids = np.column_stack((sampled, top))
    lps = (-rng.random((count, width), dtype=np.float32) * 20).astype(np.float32)
    if count:
        lps[0, -1] = -np.inf  # non-finite values must survive
    ranks = rng.integers(1, 1000, size=count).astype(np.int64)
    return ids, lps, ranks


class Feeder:
    """Engine rows through the real OutputProcessor; n > 1 fans out child
    requests like ``AsyncLLM.add_request`` (``ParentRequest``)."""

    def __init__(self, chunks, finish=FinishReason.ABORT):
        self.chunks, self.finish = chunks, finish

    def generate(self, engine_input, sampling_params, request_id, **kwargs):
        async def _gen():
            processor = OutputProcessor(
                tokenizer=_tokenizer() if sampling_params.stop else None,
                log_stats=False,
            )
            request = EngineCoreRequest(
                request_id=request_id + "-int",
                external_req_id=request_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=None,
                arrival_time=0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                sampling_params=sampling_params,
                pooling_params=None,
            )
            queue = RequestOutputCollector(sampling_params.output_kind, request_id)
            n = sampling_params.n
            if n == 1:
                requests = [request]
                processor.add_request(request, None, queue=queue)
            else:
                parent, requests = ParentRequest(request), []
                for idx in range(n):
                    child_id, child_params = parent.get_child_info(idx)
                    child = request if idx == n - 1 else copy(request)
                    child.request_id = child_id
                    child.sampling_params = child_params
                    processor.add_request(child, None, parent, idx, queue)
                    requests.append(child)
            prompt_k = sampling_params.prompt_logprobs
            for step, (ids, lps, ranks) in enumerate(self.chunks):
                outputs = []
                for idx, child in enumerate(requests):
                    child_ids = ids + 7 * idx  # distinct rows per child
                    outputs.append(
                        EngineCoreOutput(
                            request_id=child.request_id,
                            new_token_ids=child_ids[:, 0].tolist(),
                            new_logprobs=(
                                None
                                if sampling_params.logprobs is None
                                else LogprobsLists(child_ids, lps, ranks)
                            ),
                            new_prompt_logprobs_tensors=(
                                _prompt_logprobs_tensors(prompt_k)
                                if step == 0 and prompt_k is not None
                                else None
                            ),
                        )
                    )
                processor.process_outputs(outputs)
                if (out := queue.get_nowait()) is not None:
                    yield out
            processor.abort_requests([r.request_id for r in requests], internal=True)
            if (out := queue.get_nowait()) is not None:
                yield out

        return _gen()


def make_app(chunks, middleware=None):
    app = FastAPI()
    app.state.args = Namespace(
        log_error_stack=False, tokens_only=False, middleware=middleware or []
    )
    engine = _mock_engine()
    feeder = Feeder(chunks)
    engine.generate = MagicMock(side_effect=feeder.generate)
    app.state.serving_tokens = _build_serving_tokens(engine)
    api_router.attach_router(app)
    if attach_endpoint_plugins is not None:
        attach_endpoint_plugins(app, ("generate",))
        asyncio.run(init_endpoint_plugins_state(engine, app.state, app.state.args))
    from vllm.entrypoints.serve.exception_handling.register import (
        init_exception_handler,
    )

    init_exception_handler(app)
    return app


async def call(app, body):
    raw = json.dumps(body).encode()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/inference/v1/generate",
        "raw_path": b"/inference/v1/generate",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
        ],
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("x", 80),
        "client": ("y", 1),
        "root_path": "",
    }
    sent = [False]
    done = asyncio.Event()

    async def receive():
        if not sent[0]:
            sent[0] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await done.wait()  # disconnect only after the response completed
        return {"type": "http.disconnect"}

    msgs = []

    async def send(m):
        msgs.append(m)
        if m["type"] == "http.response.body" and not m.get("more_body", False):
            done.set()

    await app(scope, receive, send)
    return msgs


def summarize(msgs):
    start = msgs[0]
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    bodies = [m["body"] for m in msgs if m["type"] == "http.response.body"]
    body = b"".join(bytes(b) for b in bodies)
    if os.environ.get("GOLDEN_LEAN_SWITCHES") == "1":
        slot = b'"sampled_slot":false,'
        if body.count(slot) and headers.get("content-length"):
            headers["content-length"] = str(
                int(headers["content-length"]) - body.count(slot) * len(slot)
            )
        body = body.replace(slot, b"")
    return {
        "status": start["status"],
        "msgs": len(bodies),
        "content_length": headers.get("content-length"),
        "len": len(body),
        "sha": hashlib.sha256(body).hexdigest()[:16],
        "head": body[:160].decode("utf-8", "replace")
        if start["status"] != 200
        else None,
    }


def req(k=3, fmt=None, stream=False, positions=None, sp=None, **extra):
    body = {
        "token_ids": [1, 2, 3],
        "request_id": "golden",
        "sampling_params": {
            "max_tokens": positions or 100,
            "logprobs": k,
            **(sp or {}),
        },
        "stream": stream,
    }
    if k is None:
        del body["sampling_params"]["logprobs"]
    if fmt:
        body["logprobs_format"] = fmt
        if os.environ.get("GOLDEN_LEAN_SWITCHES") == "1":
            body["compact_include_sampled"] = False
            body["compact_include_ranks"] = False
    body.update(extra)
    return body


def chunks(n, width, sizes=None, seed=0):
    sizes = sizes or [n]
    out, start = [], 0
    for size in sizes:
        out.append(rows(start, size, width, seed))
        start += size
    return out


SCENARIOS = {
    # default format
    "def-k3-1chunk": (chunks(8, 4), req(3)),
    "def-k3-3chunks": (chunks(10, 4, [3, 3, 4]), req(3)),
    "def-k0": (chunks(6, 1), req(0)),
    "def-k128": (chunks(40, 129, [16, 24]), req(128)),
    "def-nologprobs": (chunks(5, 1), req(None)),
    "def-stream-k3": (chunks(6, 4, [2, 4]), req(3, stream=True)),
    "def-k3-mw": (chunks(8, 4), req(3)),
    # compact
    "cmp-k3": (chunks(8, 4), req(3, "compact")),
    "cmp-k3-3chunks": (chunks(10, 4, [3, 3, 4]), req(3, "compact")),
    "cmp-k0": (chunks(6, 1), req(0, "compact")),
    "cmp-k128": (chunks(40, 129, [16, 24]), req(128, "compact")),
    "cmp-k128-big": (chunks(3000, 129, [1024, 1024, 952]), req(128, "compact")),
    "cmp-rl-lean": (chunks(40, 129, [16, 24]), req(128, "compact")),
    "cmp-stream": (chunks(6, 4, [2, 4]), req(3, "compact", stream=True)),
    "cmp-stream-lean": (chunks(6, 4, [2, 4]), req(3, "compact", stream=True)),
    "cmp-mw": (chunks(8, 4), req(3, "compact")),
    "cmp-narrow-500": ([rows(0, 2, 4), rows(2, 2, 3)], req(3, "compact")),
    "cmp-k-1-400": (chunks(4, 4), req(-1, "compact")),
    # A former switch is an unknown field now: ignored, as on base.
    "removed-switch-ignored": (chunks(4, 4), req(3, compact_include_ranks=False)),
    # r17: n > 1, stop strings, prompt logprobs (default and compact)
    "def-n2-k3": (chunks(6, 4, [3, 3]), req(3, sp={"n": 2})),
    "def-stop-k3": (chunks(6, 4, [3, 3]), req(3, sp={"stop": ["@@@"]})),
    "def-prompt-lp-k3": (chunks(6, 4, [3, 3]), req(3, sp={"prompt_logprobs": 1})),
    "cmp-n2-k3": (chunks(6, 4, [3, 3]), req(3, "compact", sp={"n": 2})),
    "cmp-stop-k3": (chunks(6, 4, [3, 3]), req(3, "compact", sp={"stop": ["@@@"]})),
    "cmp-prompt-lp-k3": (
        chunks(6, 4, [3, 3]),
        req(3, "compact", sp={"prompt_logprobs": 1}),
    ),
}


def run_scenario(name):
    scenario_chunks, body = SCENARIOS[name]
    mw = ["x.Middleware"] if name.endswith("-mw") else None
    return summarize(asyncio.run(call(make_app(scenario_chunks, mw), body)))


def main():
    freeze_time()
    tree = os.environ.get("TREE", "?")
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, (scenario_chunks, body) in SCENARIOS.items():
        if flt and flt not in name:
            continue
        mw = ["x.Middleware"] if name.endswith("-mw") else None
        app = make_app(scenario_chunks, mw)
        try:
            result = summarize(asyncio.run(call(app, body)))
        except BaseException as e:  # noqa: BLE001
            result = {"exception": f"{type(e).__name__}: {e}"[:200]}
        print(json.dumps({"tree": tree, "scenario": name, **result}), flush=True)


if __name__ == "__main__":
    main()
