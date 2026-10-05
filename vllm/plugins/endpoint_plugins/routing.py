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

# Settings of the replaced route that the new route takes unless given.
_INHERITED = ("dependencies", "response_class", "tags", "include_in_schema")


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
    one operation. `route_kwargs` are those of `APIRouter.add_api_route`;
    `dependencies`, `response_class`, `tags` and `include_in_schema` default
    to the replaced route's (which include its router's dependencies). The
    returned endpoint is the plain function: calling it does not run the
    replaced route's dependencies again (the new route's ran for the request).

    Raises (`ValueError` for invalid `methods`, else `RuntimeError`) and leaves
    `app` unchanged if the replacement is not safe:

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
    if isinstance(methods, str) or not methods:
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
        route_kwargs.setdefault(name, getattr(replaced, name))

    routes = app.router.routes
    before = list(routes)
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
        app.openapi_schema = None
        _REPLACEMENTS.append(weakref.ref(new_route))
    except BaseException:
        routes[:] = before
        raise
    return replaced.endpoint


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
