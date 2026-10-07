# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin-level tests: golden bytes with and without the plugin loaded, and
the route takeover."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import golden_harness  # noqa: E402

# (status, body messages, content-length, sha256[:12]) of each golden scenario
# with the plugin: the in-tree implementation (HEAD 214f522fbf) with
# compact_include_sampled=false + compact_include_ranks=false on every compact
# request and "sampled_slot":false, removed (golden_harness.py with
# GOLDEN_LEAN_SWITCHES=1). Compact streaming is a 400. Two
# scenarios do not depend on the format:
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
    "cmp-stream": [400, 1, "465", "8ec7ee0148cf"],
    "cmp-stream-lean": [400, 1, "465", "8ec7ee0148cf"],
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
