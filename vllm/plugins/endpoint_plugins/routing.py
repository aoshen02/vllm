# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route replacement for endpoint plugins (see `EndpointPlugin.attach_router`)."""

import weakref
from collections.abc import Callable, Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import Host, Match

# Routes installed by `replace_route`. APIRoute is not hashable (it defines
# __eq__), so weak references are compared by identity.
_REPLACEMENTS: list[weakref.ref[APIRoute]] = []

# Settings of the replaced route that the new route takes unless given
# (dependencies: always, given ones are added after them).
_INHERITED = (
    "dependencies",
    "response_class",
    "tags",
    "include_in_schema",
    "name",
    "responses",
)


def replace_route(
    app: FastAPI,
    path: str,
    endpoint: Callable[..., Any],
    *,
    methods: Sequence[str] = ("POST",),
    **route_kwargs: Any,
) -> Callable[..., Any]:
    """Replace the API route that serves `methods` `path` with `endpoint`
    (an endpoint plugin taking over a core route), for use in
    `EndpointPlugin.attach_router`. Returns the replaced route's endpoint
    function, e.g. to delegate the requests the plugin does not handle.

    The new route is added through `app.router`, so `app.dependency_overrides`
    and app-level dependencies apply, and takes the replaced route's place:
    precedence relative to every other route is unchanged and OpenAPI shows
    one operation. `route_kwargs` are those of `APIRouter.add_api_route`.
    The new route keeps the replaced route's effective dependencies (its
    router's included; those of `app.router` itself are added once, by
    `add_api_route`, as for any route); given `dependencies` run after them,
    so a core dependency (e.g. validation or auth) cannot be dropped by
    accident. `response_class`, `tags`, `include_in_schema`, `name` (so
    `url_path_for` keeps working) and `responses` default to the replaced
    route's; given values replace them. Nothing else is inherited (e.g.
    `operation_id`, `summary`, `status_code`, `response_model`).
    The returned endpoint is the plain function: calling it does not run the
    replaced route's dependencies again (the new route's ran for the request).
    `methods` is the route's exact method set (Starlette routes that serve GET
    also serve HEAD: pass both).

    Raises `ValueError` for invalid `methods` and `RuntimeError` if the
    replacement is not safe; errors from `add_api_route` itself (e.g. a
    `TypeError` for invalid `route_kwargs`) propagate. In every case `app` is
    left unchanged (routes and cached OpenAPI schema). Not safe:

    - for some method, the first route matching the request is not an API
      route registered at exactly `path` (e.g. a catch-all or a `Mount`);
    - the methods are served by different routes, or the route also serves
      methods that are not being replaced (they would be dropped);
    - the route was already replaced (e.g. by another plugin);
    - a `Host` route precedes it (it may serve the path for some hosts);
    - a route cannot be checked, or the new route does not end up at `path`
      (e.g. a router prefix that `path` does not start with).

    Meant for app construction (before serving): any failure after the new
    route was added restores the previous route list.
    """
    # Forget routes that no longer exist (e.g. apps built in tests).
    _REPLACEMENTS[:] = [ref for ref in _REPLACEMENTS if ref() is not None]
    if (
        isinstance(methods, (str, bytes))
        or not methods
        or not all(isinstance(method, str) for method in methods)
    ):
        raise ValueError(
            f"methods must be a non-empty sequence such as ['POST'], got {methods!r}"
        )
    wanted = {method.upper() for method in methods}
    replaced: APIRoute | None = None
    for method in sorted(wanted):
        first = _first_full_match(app, path, method)
        if not isinstance(first, APIRoute) or first.path != path:
            raise RuntimeError(
                f"{method} {path} is served by {first!r}, not by an API route "
                "at that path; refusing to replace it"
            )
        if replaced is not None and first is not replaced:
            raise RuntimeError(f"{path} is served by different routes per method")
        if any(ref() is first for ref in _REPLACEMENTS):
            raise RuntimeError(f"{method} {path} was already replaced")
        replaced = first
    assert replaced is not None
    if replaced.methods != wanted:
        raise RuntimeError(
            f"{path} serves {sorted(replaced.methods)}; replacing "
            f"{sorted(wanted)} only would drop the others; refusing"
        )
    prefix = app.router.prefix
    if not path.startswith(prefix):
        raise RuntimeError(f"{path} does not start with the router prefix {prefix!r}")
    for name in _INHERITED:
        inherited = _inherited(app, replaced, name)
        if name == "dependencies":
            route_kwargs[name] = inherited + list(route_kwargs.get(name) or [])
        elif name not in route_kwargs:
            route_kwargs[name] = inherited

    routes = app.router.routes
    before = list(routes)
    schema = app.openapi_schema
    try:
        app.router.add_api_route(
            path[len(prefix) :], endpoint, methods=sorted(wanted), **route_kwargs
        )
        new_route = routes[-1]
        if (
            len(routes) != len(before) + 1
            or not isinstance(new_route, APIRoute)
            or new_route.endpoint is not endpoint
            or new_route.path != path
        ):
            raise RuntimeError(f"The replacement of {path} was not added at {path}")
        routes.pop()
        routes[next(i for i, r in enumerate(routes) if r is replaced)] = new_route
        _REPLACEMENTS.append(weakref.ref(new_route))
        app.openapi_schema = None
    except BaseException:
        routes[:] = before
        app.openapi_schema = schema
        raise
    return replaced.endpoint


def _inherited(app: FastAPI, route: APIRoute, name: str) -> Any:
    """The replaced route's setting `name`, as `app.router.add_api_route` takes
    it. A route's dependencies and tags start with those `app.router` had when
    the route was added, and `add_api_route` prepends them again: that leading
    part is left out (dependencies matched by identity, tags by equality), so
    they are neither repeated (an uncached sub-dependency would run twice) nor
    reordered."""
    value = getattr(route, name)
    if name not in ("dependencies", "tags"):
        return value
    own = list(getattr(app.router, name))
    value = list(value)
    if name == "dependencies":
        prefix = all(a is b for a, b in zip(value, own))
    else:
        prefix = all(a == b for a, b in zip(value, own))
    if len(value) >= len(own) and prefix:
        return value[len(own) :]
    return value


def _first_full_match(app: FastAPI, path: str, method: str) -> Any:
    """The route Starlette dispatches `method` `path` to (None if none)."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"localhost")],
        "scheme": "http",
        "server": ("localhost", 80),
        "http_version": "1.1",
        "app": app,
    }
    for route in app.router.routes:
        if isinstance(route, Host):
            raise RuntimeError(
                f"Host route {route.host!r} precedes {method} {path}; "
                "refusing to replace it (it may serve the path for some hosts)"
            )
        try:
            if route.matches(scope)[0] == Match.FULL:
                return route
        except Exception as e:
            raise RuntimeError(f"Cannot check route {route!r}: {e!r}") from e
    return None
