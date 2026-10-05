# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`replace_route`: an endpoint plugin taking over a core route in place, with
conflict detection."""

import gc

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.routing import Host, Router

from vllm.plugins.endpoint_plugins import routing
from vllm.plugins.endpoint_plugins.routing import replace_route

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
    operation_id = app.openapi()["paths"][PATH]["post"]["operationId"]
    assert _replace(app) is core
    assert [getattr(r, "path", None) for r in app.router.routes] == paths
    client = TestClient(app)
    assert client.post(PATH).json() == {"served_by": "plugin", "dep": "core"}
    assert client.post(PATH + "?core=1").json() == {"served_by": "core", "dep": "core"}
    # Precedence kept: other paths still reach the catch-all registered after.
    assert client.post("/inference/v1/other").json() == {"served_by": "other"}
    operations = app.openapi()["paths"][PATH]
    assert list(operations) == ["post"]
    # The cached schema was reset; the inherited route name keeps the
    # operation id of the replaced route.
    assert operations["post"]["operationId"] == operation_id


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


def test_partial_method_sets_are_a_conflict():
    """Stack audit: replacing POST of a POST + PUT route would drop PUT."""
    app = FastAPI()
    app.add_api_route(PATH, core, methods=["POST", "PUT"])
    assert "would drop" in _unchanged(app, lambda: _replace(app))
    assert TestClient(app).put(PATH).json()["served_by"] == "core"
    assert replace_route(app, PATH, _plugin([]), methods=["post", "put"]) is core


@pytest.mark.parametrize("methods", ["POST", b"POST", [], (), [1]])
def test_invalid_methods_are_rejected(methods):
    app = _app()
    routes = list(app.router.routes)
    with pytest.raises(ValueError):
        replace_route(app, PATH, _plugin([]), methods=methods)
    assert all(a is b for a, b in zip(app.router.routes, routes, strict=True))


def test_router_prefix_is_applied_once():
    """Stack audit: a prefixed router must not end up at /api/api/..."""
    app = FastAPI()
    app.router.prefix = "/api"
    app.router.add_api_route("/x", core, methods=["POST"])
    assert replace_route(app, "/api/x", _plugin([])) is core
    paths = [getattr(r, "path", None) for r in app.router.routes]
    assert "/api/x" in paths and "/api/api/x" not in paths


def test_failure_after_the_route_was_added_restores_the_routes(monkeypatch):
    """Kimi stack audit: the post-add section is atomic."""
    app = _app()
    routes = list(app.router.routes)

    class Failing(list):
        def append(self, item):
            raise OSError("late failure")

    monkeypatch.setattr(routing, "_REPLACEMENTS", Failing())
    with pytest.raises(OSError):
        _replace(app)
    assert all(a is b for a, b in zip(app.router.routes, routes, strict=True))


def test_inherited_app_dependencies_and_tags_are_not_added_twice():
    """Stack v2 audit: the app router's own dependencies and tags are part of
    the replaced route's and are added again by add_api_route: inherited
    once, an uncached nested dependency runs once, as before."""
    calls = []

    def inner():
        calls.append("inner")

    def outer(_: None = Depends(inner, use_cache=False)):
        calls.append("outer")

    app = FastAPI(dependencies=[Depends(outer)])
    app.router.tags = ["app"]
    app.add_api_route(PATH, core, methods=["POST"])
    client = TestClient(app)
    client.post(PATH)
    before, calls[:] = list(calls), []
    _replace(app)
    client.post(PATH)
    assert calls == before == ["inner", "outer"]
    new = next(r for r in app.router.routes if getattr(r, "path", None) == PATH)
    assert new.tags == ["app"] and len(new.dependencies) == 1


def test_rollback_keeps_the_cached_openapi_schema(monkeypatch):
    """Stack v2 audit: a late failure leaves the cached schema as it was."""
    app = _app()
    schema = app.openapi()

    class Failing(list):
        def append(self, item):
            raise OSError("late failure")

    monkeypatch.setattr(routing, "_REPLACEMENTS", Failing())
    with pytest.raises(OSError):
        _replace(app)
    assert app.openapi_schema is schema


def test_replacement_registry_forgets_collected_routes(monkeypatch):
    """Kimi stack v2 audit: the registry does not grow with every app."""
    monkeypatch.setattr(routing, "_REPLACEMENTS", [])
    for _ in range(3):
        _replace(_app())
    gc.collect()
    keep = _app()
    _replace(keep)
    assert len(routing._REPLACEMENTS) == 1


def test_given_dependencies_are_added_to_the_inherited_ones():
    """Claude stack v2 audit: passing dependencies must not drop the core
    route's (e.g. its router's validation or auth)."""
    calls = []

    def router_dep():
        calls.append("router")

    def plugin_dep():
        calls.append("plugin")

    router = APIRouter(dependencies=[Depends(router_dep)])
    router.add_api_route(PATH, core, methods=["POST"], name="core_generate")
    app = FastAPI()
    app.include_router(router)
    replace_route(app, PATH, _plugin([]), dependencies=[Depends(plugin_dep)])
    TestClient(app).post(PATH)
    assert calls == ["router", "plugin"]
    # The route name is inherited: url_path_for still resolves it.
    assert app.url_path_for("core_generate") == PATH


def test_replaced_route_settings_are_inherited():
    """Stack audit: the replaced route's (and its router's) dependencies,
    response class, tags and schema visibility carry over unless given."""
    calls = []

    def router_dep():
        calls.append("router")

    router = APIRouter(dependencies=[Depends(router_dep)], tags=["core"])
    router.add_api_route(PATH, core, methods=["POST"], include_in_schema=False)
    app = FastAPI()
    app.include_router(router)
    _replace(app)
    new = next(r for r in app.router.routes if getattr(r, "path", None) == PATH)
    assert new.tags == ["core"] and new.include_in_schema is False
    TestClient(app).post(PATH)
    assert calls == ["router"]


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
