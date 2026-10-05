# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`replace_route`: an endpoint plugin taking over a core route in place, with
conflict detection."""

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.routing import Host, Router

from vllm.plugins.endpoint_plugins.interface import replace_route

PATH = "/inference/v1/generate"


def _dependency() -> str:
    return "core"


async def core(dep: str = Depends(_dependency)):
    return {"served_by": "core", "dep": dep}


async def other(rest: str = ""):
    return {"served_by": "other"}


def _app(*, before=None, after=None) -> FastAPI:
    app = FastAPI()
    for router in before or ():
        app.include_router(router)
    app.add_api_route(PATH, core, methods=["POST"])
    for router in after or ():
        app.include_router(router)
    return app


def _catch_all() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/inference/{rest:path}", other, methods=["POST"])
    return router


def _plugin(delegate: list):
    async def plugin(request: Request, dep: str = Depends(_dependency)):
        if request.query_params.get("core"):
            return await delegate[0](dep=dep)
        return {"served_by": "plugin", "dep": dep}

    return plugin


def _replace(app: FastAPI):
    delegate: list = []
    delegate.append(replace_route(app, PATH, _plugin(delegate)))
    return delegate[0]


def test_replaces_in_place_and_returns_the_replaced_endpoint():
    app = _app(after=[_catch_all()])
    paths = [getattr(r, "path", None) for r in app.router.routes]
    app.openapi()  # cached schema is reset by the replacement
    assert _replace(app) is core
    assert [getattr(r, "path", None) for r in app.router.routes] == paths
    client = TestClient(app)
    assert client.post(PATH).json() == {"served_by": "plugin", "dep": "core"}
    assert client.post(PATH + "?core=1").json() == {"served_by": "core", "dep": "core"}
    # Precedence kept: other paths still reach the catch-all registered after.
    assert client.post("/inference/v1/other").json() == {"served_by": "other"}
    operations = app.openapi()["paths"][PATH]
    assert list(operations) == ["post"]
    assert operations["post"]["operationId"].startswith("plugin")


def test_dependency_overrides_apply_to_the_new_route():
    app = _app()
    _replace(app)
    app.dependency_overrides[_dependency] = lambda: "override"
    assert TestClient(app).post(PATH).json() == {
        "served_by": "plugin",
        "dep": "override",
    }


def _unchanged(app: FastAPI, call):
    routes = list(app.router.routes)
    with pytest.raises(RuntimeError) as e:
        call()
    assert app.router.routes == routes and all(
        a is b for a, b in zip(app.router.routes, routes)
    )
    return str(e.value)


def test_an_earlier_catch_all_is_a_conflict():
    app = _app(before=[_catch_all()])
    assert "refusing" in _unchanged(app, lambda: _replace(app))


def test_a_second_replacement_is_a_conflict():
    app = _app()
    _replace(app)
    assert "already replaced" in _unchanged(app, lambda: _replace(app))


def test_no_route_or_another_method_is_a_conflict():
    app = FastAPI()
    assert "refusing" in _unchanged(app, lambda: _replace(app))
    app = _app()
    _unchanged(
        app, lambda: replace_route(app, PATH, _plugin([]), methods=["POST", "GET"])
    )


def test_different_routes_per_method_are_a_conflict():
    app = _app()
    app.add_api_route(PATH, other, methods=["GET"])
    assert "different routes" in _unchanged(
        app, lambda: replace_route(app, PATH, _plugin([]), methods=["POST", "GET"])
    )


@pytest.mark.parametrize("before_core", [True, False])
def test_host_routes_before_the_route_are_refused(before_core):
    app = FastAPI()
    host = Host("admin.example", app=Router())
    if before_core:
        app.router.routes.append(host)
    app.add_api_route(PATH, core, methods=["POST"])
    if not before_core:
        app.router.routes.append(host)
    if before_core:
        assert "Host route" in _unchanged(app, lambda: _replace(app))
    else:
        assert _replace(app) is core


def test_a_route_that_cannot_be_checked_is_a_conflict():
    app = _app()

    class Raising:
        def matches(self, scope):
            raise ValueError("broken")

    app.router.routes.insert(0, Raising())
    assert "Cannot check" in _unchanged(app, lambda: _replace(app))
