# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin-level tests of ``vllm_rl_compact`` (design C): byte/framing parity with
the in-tree compact format of 214f522fbf and with base for everything else
(golden), loading/gating through ``vllm.endpoint_plugins``, route replacement,
OpenAPI, container wiring."""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import golden_harness  # noqa: E402

from vllm.logprobs import set_sample_logprobs_container  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.engine import EngineCoreRequest  # noqa: E402
from vllm.v1.engine.output_processor import OutputProcessor  # noqa: E402
from vllm.v1.serial_utils import MsgpackEncoder  # noqa: E402

# (status, body messages, content-length, sha256[:12]) of each golden scenario
# with the plugin: the in-tree implementation (HEAD 214f522fbf) with
# compact_include_sampled=false + compact_include_ranks=false on every compact
# request and "sampled_slot":false, removed (golden_harness.py with
# GOLDEN_LEAN_SWITCHES=1). Two scenarios do not depend on the format:
# "cmp-k-1-400" (in-tree without switches) and "removed-switch-ignored" (an
# unknown field: base).
PLUGIN = {
    "def-k3-1chunk": [200, 1, "2819", "1c0e101976ee"],
    "def-k3-3chunks": [200, 1, "3411", "1b0dfc3920ce"],
    "def-k0": [200, 1, "1374", "30926d38d028"],
    "def-k128": [200, 1, "358873", "934b323ea236"],
    "def-nologprobs": [200, 1, "447", "c6c04690e849"],
    "def-stream-k3": [200, 4, None, "e3cb8134b31f"],
    "def-k3-mw": [200, 1, "2819", "1c0e101976ee"],
    "cmp-k3": [200, 1, "877", "525e1ad85335"],
    "cmp-k3-3chunks": [200, 1, "955", "5e372944be48"],
    "cmp-k0": [200, 1, "609", "72bd10fa44e5"],
    "cmp-k128": [200, 1, "55430", "4c38a708c16d"],
    "cmp-k128-big": [200, 5, "4114229", "3c3ab4d2c7f6"],
    "cmp-rl-lean": [200, 1, "55430", "4c38a708c16d"],
    "cmp-stream": [200, 4, None, "5ebf15f5066e"],
    "cmp-stream-lean": [200, 4, None, "5ebf15f5066e"],
    "cmp-mw": [200, 1, "877", "525e1ad85335"],
    "cmp-narrow-500": [500, 1, "195", "7c76a8c7eaf7"],
    "cmp-k-1-400": [400, 1, "499", "5ab12771b3c6"],
    "removed-switch-ignored": [200, 1, "1617", "a79837ef238f"],
    "def-n2-k3": [200, 1, "4124", "c3c70d43b317"],
    "def-stop-k3": [200, 1, "2215", "a39e59252a7d"],
    "def-prompt-lp-k3": [200, 1, "2432", "44ad5e67e908"],
    "cmp-n2-k3": [200, 1, "1296", "b63b86eb66b0"],
    "cmp-stop-k3": [200, 1, "801", "c03b51e25724"],
    "cmp-prompt-lp-k3": [200, 1, "1018", "3323b1019003"],
}
BASE_0FD2E8D = {
    "def-k3-1chunk": [200, 1, "2819", "1c0e101976ee"],
    "def-k3-3chunks": [200, 1, "3411", "1b0dfc3920ce"],
    "def-k0": [200, 1, "1374", "30926d38d028"],
    "def-k128": [200, 1, "358873", "934b323ea236"],
    "def-nologprobs": [200, 1, "447", "c6c04690e849"],
    "def-stream-k3": [200, 4, None, "e3cb8134b31f"],
    "def-k3-mw": [200, 1, "2819", "1c0e101976ee"],
    "cmp-k3": [200, 1, "2819", "1c0e101976ee"],
    "cmp-k3-3chunks": [200, 1, "3411", "1b0dfc3920ce"],
    "cmp-k0": [200, 1, "1374", "30926d38d028"],
    "cmp-k128": [200, 1, "358873", "934b323ea236"],
    "cmp-k128-big": [200, 1, "26877154", "be2e9f72c6ca"],
    "cmp-rl-lean": [200, 1, "358873", "934b323ea236"],
    "cmp-stream": [200, 4, None, "e3cb8134b31f"],
    "cmp-stream-lean": [200, 4, None, "e3cb8134b31f"],
    "cmp-mw": [200, 1, "2819", "1c0e101976ee"],
    "cmp-narrow-500": [200, 1, "1536", "b9571f1d91f8"],
    "cmp-k-1-400": [200, 1, "1072", "4d8089539f7b"],
    "removed-switch-ignored": [200, 1, "1617", "a79837ef238f"],
    "def-n2-k3": [200, 1, "4124", "c3c70d43b317"],
    "def-stop-k3": [200, 1, "2215", "a39e59252a7d"],
    "def-prompt-lp-k3": [200, 1, "2432", "44ad5e67e908"],
    "cmp-n2-k3": [200, 1, "4124", "c3c70d43b317"],
    "cmp-stop-k3": [200, 1, "2215", "a39e59252a7d"],
    "cmp-prompt-lp-k3": [200, 1, "2432", "44ad5e67e908"],
}


def _key(result):
    return [
        result["status"],
        result["msgs"],
        result["content_length"],
        result["sha"][:12],
    ]


@pytest.fixture
def frozen_time(monkeypatch):
    import time

    monkeypatch.setattr(time, "time", lambda: 1700000000.0)


@pytest.mark.parametrize("scenario", sorted(PLUGIN))
def test_plugin_bytes_match_in_tree_compact(monkeypatch, frozen_time, scenario):
    """With the plugin allowlisted, every compact scenario has exactly the
    status, framing, Content-Length and bytes of the in-tree RL-lean format;
    default-format scenarios equal base."""
    monkeypatch.setenv("VLLM_PLUGINS", "rl_compact")
    assert _key(golden_harness.run_scenario(scenario)) == PLUGIN[scenario]
    if scenario.startswith("def-"):
        assert PLUGIN[scenario] == BASE_0FD2E8D[scenario]


@pytest.mark.parametrize("scenario", sorted(BASE_0FD2E8D))
def test_without_allowlist_requests_behave_like_base(
    monkeypatch, frozen_time, scenario
):
    """Endpoint plugins load only when named in VLLM_PLUGINS: without it the
    compact switches are ignored exactly like on base (bytes identical)."""
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    assert _key(golden_harness.run_scenario(scenario)) == BASE_0FD2E8D[scenario]


def test_plugin_replaces_the_core_route(monkeypatch):
    """One POST /inference/v1/generate operation, served by the plugin, with
    the plugin's request schema in OpenAPI."""
    monkeypatch.setenv("VLLM_PLUGINS", "rl_compact")
    app = golden_harness.make_app([])
    routes = [
        r
        for r in app.router.routes
        if getattr(r, "path", None) == "/inference/v1/generate"
        and "POST" in getattr(r, "methods", ())
    ]
    assert len(routes) == 1
    assert routes[0].endpoint.__module__ == "vllm_rl_compact.router"
    assert app.state.rl_compact_serving is not None
    schema = app.openapi()
    operation = schema["paths"]["/inference/v1/generate"]["post"]
    ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_schema = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert "logprobs_format" in request_schema["properties"]
    assert "CompactResponseChoice" in schema["components"]["schemas"]


def test_second_claim_of_the_route_fails_loudly(monkeypatch):
    from vllm_rl_compact import router

    monkeypatch.setenv("VLLM_PLUGINS", "rl_compact")
    app = golden_harness.make_app([])
    with pytest.raises(RuntimeError, match="already replaced"):
        router.attach_router(app)


def _app_with_catch_all(before_core: bool):
    """Core generate plus another plugin's
    ``POST /inference/{rest:path}``, before or after the core route."""
    from fastapi import APIRouter
    from fastapi.testclient import TestClient

    from vllm.entrypoints.scale_out.token_in_token_out import api_router

    app = golden_harness.make_app([])
    other = APIRouter()

    @other.post("/inference/{rest:path}")
    async def fallback(rest: str):
        return {"served_by": "fallback"}

    app.include_router(other)
    if before_core:
        app.router.routes.insert(0, app.router.routes.pop())
    return app, TestClient(app), api_router


def test_replacement_keeps_route_precedence(monkeypatch):
    from vllm_rl_compact import router

    app, client, api_router = _app_with_catch_all(before_core=False)
    paths = [getattr(r, "path", None) for r in app.router.routes]
    core_index = paths.index("/inference/v1/generate")
    # make_app already attached the plugin (VLLM_PLUGINS unset: not loaded)
    router.attach_router(app)
    endpoint = getattr(app.router.routes[core_index], "endpoint", None)
    assert endpoint.__module__ == "vllm_rl_compact.router"
    core = app.state.serving_tokens
    calls = []

    async def serve_tokens(request, raw_request=None):
        calls.append(request)
        return core.create_error_response("served by core")

    # Non-compact requests reach the replaced core endpoint (its handler).
    monkeypatch.setattr(core, "serve_tokens", serve_tokens)
    result = client.post(
        "/inference/v1/generate", json={"token_ids": [1], "sampling_params": {}}
    )
    assert "served by core" in result.text and len(calls) == 1
    assert client.post("/inference/v1/other", json={}).json() == {
        "served_by": "fallback"
    }


def test_overlapping_earlier_route_is_a_conflict():
    from vllm_rl_compact import router

    app, _, _ = _app_with_catch_all(before_core=True)
    with pytest.raises(RuntimeError, match="refusing to replace"):
        router.attach_router(app)


@pytest.mark.parametrize("container", ["rl_compact.array", "rl_compact.wire"])
def test_container_choice_stays_off_engine_wire(container):
    """The frontend-only container name is reset on the request sent to
    EngineCore (the caller's params object is not modified)."""
    import vllm_rl_compact

    vllm_rl_compact.register_containers()
    params = SamplingParams(max_tokens=4, logprobs=2)
    set_sample_logprobs_container(params, container)
    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=params,
        pooling_params=None,
    )
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    processor.add_request(request, None)
    assert request.sampling_params._sample_logprobs_container is None
    assert params._sample_logprobs_container == container
    encoded = b"".join(bytes(b) for b in MsgpackEncoder().encode(request))
    assert container.encode() not in encoded
    handle = processor.request_states["r"].logprobs_processor.logprobs
    assert type(handle).__name__ == "SampleLogprobsHandle"
    logprobs = handle.unwrap()
    assert type(logprobs).__name__ == "ArrayLogprobs"
    assert (logprobs.wire_parts() is not None) is (container != "rl_compact.array")


def test_wire_container_rows_roundtrip_and_contain_failures(monkeypatch):
    import vllm_rl_compact.storage as storage

    container = storage.ArrayLogprobs(wire_base64=True)
    ids = np.arange(12, dtype=np.int64).reshape(3, 4)
    lps = -np.arange(12, dtype=np.float32).reshape(3, 4)
    ranks = np.array([1, 2, 3])
    container.append_rows(ids, lps, ranks)
    assert container.wire_parts() is not None and container.num_slots == 4
    t, _, _ = container.arrays()  # leaves wire mode
    np.testing.assert_array_equal(t, ids)
    assert container.wire_parts() is None

    broken = storage.ArrayLogprobs(wire_base64=True)

    def boom(self, data):
        raise MemoryError("injected")

    monkeypatch.setattr(storage._Base64Stream, "write", boom)
    broken.append_rows(ids, lps, ranks)  # never raises
    assert broken.broken and len(broken) == 3 and broken.wire_parts() is None


@pytest.mark.parametrize(
    ("stream", "expected"), [(False, "rl_compact.wire"), (True, "rl_compact.array")]
)
def test_compact_requests_select_their_container(stream, expected):
    """Non-streaming compact rows are encoded as they arrive; streams slice
    array rows per chunk."""
    from vllm_rl_compact.protocol import CompactGenerateRequest
    from vllm_rl_compact.serving import compact_container

    request = CompactGenerateRequest.model_validate(
        {
            "token_ids": [1],
            "sampling_params": {"logprobs": 2},
            "stream": stream,
            "logprobs_format": "compact",
        }
    )
    assert compact_container(request) == expected


def test_compact_request_without_logprobs_skips_sampled_text(monkeypatch):
    """The sampled-text detokenizer skip follows the selected
    container also when no logprobs (hence no container) are requested."""
    import vllm_rl_compact

    vllm_rl_compact.register_containers()
    params = SamplingParams(max_tokens=4)
    set_sample_logprobs_container(params, "rl_compact.wire")
    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=params,
        pooling_params=None,
    )
    processor = OutputProcessor(tokenizer=MagicMock(), log_stats=False)
    monkeypatch.setattr(
        "vllm.v1.engine.output_processor.IncrementalDetokenizer.from_new_request",
        lambda tokenizer, request: ("detok", tokenizer),
    )
    processor.add_request(request, None)
    state = processor.request_states["r"]
    assert state.logprobs_processor.logprobs is None
    assert state.detokenizer == ("detok", None)


@pytest.mark.parametrize("before_core", [True, False])
def test_host_routes_before_the_core_route_are_refused(before_core):
    """A complete synthetic scope (no KeyError from Host routes);
    a Host route ahead of the core route may shadow it for some hosts."""
    from starlette.routing import Host, Router
    from vllm_rl_compact import router

    app = golden_harness.make_app([])
    host = Host("api.example.com", app=Router())
    if before_core:
        app.router.routes.insert(0, host)
    else:
        app.router.routes.append(host)
    if before_core:
        with pytest.raises(RuntimeError, match="Host route"):
            router.attach_router(app)
    else:
        router.attach_router(app)


def test_dependency_overrides_apply_to_the_plugin_route(monkeypatch):
    """The plugin route is added through the app router, so
    app.dependency_overrides apply as for the core route."""
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    from vllm_rl_compact import router

    from vllm.entrypoints.serve.utils.api_utils import validate_json_request

    app = golden_harness.make_app([])
    router.attach_router(app)

    def deny():
        raise HTTPException(status_code=418, detail="overridden")

    app.dependency_overrides[validate_json_request] = deny
    with TestClient(app) as client:
        result = client.post(
            "/inference/v1/generate", json={"token_ids": [1], "sampling_params": {}}
        )
    assert result.status_code == 418
