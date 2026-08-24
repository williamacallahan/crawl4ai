# deploy/docker/mcp_bridge.py

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any
from urllib.parse import unquote, urlsplit

import anyio
import httpx
import mcp.types as t
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from pydantic import AnyUrl, BaseModel
from starlette.routing import Mount, Route


# ── opt‑in decorators ───────────────────────────────────────────
def mcp_resource(name: str | None = None):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "resource", name
        return fn
    return deco

def mcp_template(name: str | None = None):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "template", name
        return fn
    return deco

def mcp_tool(name: str | None = None):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "tool", name
        return fn
    return deco

# ── HTTP‑proxy helper for FastAPI endpoints ─────────────────────
def _service_auth_headers() -> dict:
    """Authenticate the internal loopback call to our own gated endpoints.

    The MCP transport is now behind the AuthGateMiddleware, so by the time a
    tool is invoked the MCP client is already authenticated. The loopback HTTP
    call this proxy makes must therefore carry a credential too, or the gate
    would 401 it. We mint a short-lived, data-scope service token: MCP exposes
    only data-plane tools (no admin/monitor actions), so this grants no
    privilege escalation. (Requires a shared SECRET_KEY across workers, which
    the auth startup check already mandates for any real deployment.)
    """
    from auth import create_access_token
    token = create_access_token({"sub": "mcp-service"}, scope="data")
    return {"Authorization": f"Bearer {token}"}


def _make_http_proxy(
    base_url: str,
    route: Route,
    *,
    timeout: float | None = None,
    auth_headers_provider: Callable[[], dict] = _service_auth_headers,
) -> Callable[..., Awaitable[Any]]:
    methods = route.methods
    if methods is None:
        raise ValueError("MCP tools require an HTTP route with an explicit method")
    method = next((candidate for candidate in methods if candidate not in {"HEAD", "OPTIONS"}), None)
    if method is None:
        raise ValueError("MCP tools require a method other than HEAD or OPTIONS")

    async def proxy(**kwargs: Any) -> Any:
        # replace `/items/{id}` style params first
        path = route.path
        for k, v in list(kwargs.items()):
            placeholder = "{" + k + "}"
            if placeholder in path:
                path = path.replace(placeholder, str(v))
                kwargs.pop(k)
        url = base_url.rstrip("/") + path

        headers = auth_headers_provider()
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = (
                    await client.get(url, params=kwargs, headers=headers)
                    if method == "GET"
                    else await client.request(method, url, json=kwargs, headers=headers)
                )
                r.raise_for_status()
                return r.text if method == "GET" else r.json()
            except httpx.HTTPStatusError as e:
                # surface FastAPI error details instead of plain 500
                raise HTTPException(e.response.status_code, e.response.text)
            except httpx.TimeoutException:
                raise HTTPException(504, "upstream request timed out")
    return proxy

# ── main entry point ────────────────────────────────────────────
def attach_mcp(
    app: FastAPI,
    *,                          # keyword‑only
    base: str = "/mcp",
    name: str | None = None,
    base_url: str,              # eg. "http://127.0.0.1:8020"
    timeout: float | None = None,  # httpx timeout in seconds; None = no limit
    auth_headers_provider: Callable[[], dict] = _service_auth_headers,
) -> None:
    """Call once after all routes are declared to expose WS+SSE MCP endpoints."""
    server_name = name or app.title or "FastAPI-MCP"
    mcp = Server(server_name)

    # tools: Dict[str, Callable] = {}
    tools: dict[str, tuple[Callable[..., Awaitable[Any]], Callable[..., Any]]] = {}
    resources: dict[str, tuple[str, Callable[..., Any]]] = {}
    templates: dict[str, tuple[Route, Callable[..., Any]]] = {}

    # register decorated FastAPI routes
    for route in app.routes:
        if not isinstance(route, Route):
            continue

        fn = route.endpoint
        kind = getattr(fn, "__mcp_kind__", None)
        if not kind:
            continue

        configured_name = getattr(fn, "__mcp_name__", None)
        key = configured_name if isinstance(configured_name, str) and configured_name else _route_name(route.path)

        # if kind == "tool":
        #     tools[key] = _make_http_proxy(base_url, route)
        if kind == "tool":
            proxy = _make_http_proxy(
                base_url,
                route,
                timeout=timeout,
                auth_headers_provider=auth_headers_provider,
            )
            tools[key] = (proxy, fn)
            continue
        if kind == "resource":
            resources[_mcp_uri(route.path)] = (key, fn)
        if kind == "template":
            templates[key] = (route, fn)

    # helpers for JSON‑Schema
    def _schema(model: type[BaseModel] | None) -> dict[str, Any]:
        return {"type": "object"} if model is None else model.model_json_schema()

    def _body_model(fn: Callable) -> type[BaseModel] | None:
        for p in inspect.signature(fn).parameters.values():
            a = p.annotation
            if inspect.isclass(a) and issubclass(a, BaseModel):
                return a
        return None

    # MCP handlers
    @mcp.list_tools()
    async def _list_tools() -> list[t.Tool]:
        out = []
        for k, (proxy, orig_fn) in tools.items():
            desc   = getattr(orig_fn, "__mcp_description__", None) or inspect.getdoc(orig_fn) or ""
            schema = getattr(orig_fn, "__mcp_schema__", None) or _schema(_body_model(orig_fn))
            out.append(
                t.Tool(name=k, description=desc, inputSchema=schema)
            )
        return out
             

    @mcp.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[t.TextContent]:
        if name not in tools:
            raise HTTPException(404, "tool not found")
        
        proxy, _ = tools[name]
        try:
            res = await proxy(**(arguments or {}))
        except HTTPException as exc:
            # map server‑side errors into MCP "text/error" payloads
            err = {"error": exc.status_code, "detail": exc.detail}
            return [t.TextContent(type = "text", text=json.dumps(err, ensure_ascii=False))]
        return [t.TextContent(type = "text", text=json.dumps(res, default=str, ensure_ascii=False))]

    @mcp.list_resources()
    async def _list_resources() -> list[t.Resource]:
        return [
            t.Resource(
                uri=uri,
                name=name,
                description=inspect.getdoc(fn) or "",
                mimeType="application/json",
            )
            for uri, (name, fn) in resources.items()
        ]

    @mcp.read_resource()
    async def _read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        resource = resources.get(str(uri))
        if resource is not None:
            _, handler = resource
            result = await _invoke(handler)
        else:
            result = await _read_template(uri, templates)
        return [
            ReadResourceContents(
                content=json.dumps(result, default=str, ensure_ascii=False),
                mime_type="application/json",
            )
        ]

    @mcp.list_resource_templates()
    async def _list_templates() -> list[t.ResourceTemplate]:
        return [
            t.ResourceTemplate(
                name=k,
                uriTemplate=_mcp_uri(route.path),
                description=inspect.getdoc(f) or "",
                mimeType="application/json",
            )
            for k, (route, f) in templates.items()
        ]

    init_opts = InitializationOptions(
        server_name=server_name,
        server_version="0.1.0",
        capabilities=mcp.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )

    # ── WebSocket transport ────────────────────────────────────
    @app.websocket_route(f"{base}/ws")
    async def _ws(ws: WebSocket):
        await ws.accept()
        c2s_send, c2s_recv = anyio.create_memory_object_stream(100)
        s2c_send, s2c_recv = anyio.create_memory_object_stream(100)

        from mcp.types import JSONRPCMessage
        from pydantic import TypeAdapter
        adapter = TypeAdapter(JSONRPCMessage)

        init_done = anyio.Event()

        async def srv_to_ws():
            first = True 
            try:
                async for msg in s2c_recv:
                    await ws.send_json(msg.model_dump())
                    if first:
                        init_done.set()
                        first = False
            finally:
                # make sure cleanup survives TaskGroup cancellation
                with anyio.CancelScope(shield=True):
                    with suppress(RuntimeError):       # idempotent close
                        await ws.close()

        async def ws_to_srv():
            try:
                # 1st frame is always "initialize"
                first = adapter.validate_python(await ws.receive_json())
                await c2s_send.send(first)
                await init_done.wait()          # block until server ready
                while True:
                    data = await ws.receive_json()
                    await c2s_send.send(adapter.validate_python(data))
            except WebSocketDisconnect:
                await c2s_send.aclose()

        async with anyio.create_task_group() as tg:
            tg.start_soon(mcp.run, c2s_recv, s2c_send, init_opts)
            tg.start_soon(ws_to_srv)
            tg.start_soon(srv_to_ws)

    # ── SSE transport (raw ASGI — avoids Starlette middleware conflict) ──
    sse = SseServerTransport(f"{base}/messages/")

    # Starlette's Route wraps plain async functions in request_response(),
    # which calls handler(request) instead of handler(scope, receive, send).
    # Using a callable class bypasses this — Route passes classes through
    # as raw ASGI apps.  See #1594, #1850.
    class _MCPSseApp:
        async def __call__(self, scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await mcp.run(read_stream, write_stream, init_opts)

    app.routes.append(Route(f"{base}/sse", endpoint=_MCPSseApp()))
    app.routes.append(Mount(f"{base}/messages", app=sse.handle_post_message))

    # ── schema endpoint ───────────────────────────────────────
    @app.get(f"{base}/schema")
    async def _schema_endpoint():
        return JSONResponse({
            "tools": [x.model_dump(mode="json") for x in await _list_tools()],
            "resources": [x.model_dump(mode="json") for x in await _list_resources()],
            "resource_templates": [x.model_dump(mode="json") for x in await _list_templates()],
        })


# ── helpers ────────────────────────────────────────────────────
def _route_name(path: str) -> str:
    return re.sub(r"[/{}}]", "_", path).strip("_")


def _mcp_uri(path: str) -> str:
    """Map a FastAPI route path to the URI namespace used by this bridge."""
    uri_path = re.sub(r"{([^}:]+):[^}]+}", r"{\1}", path)
    return f"crawl4ai://resources{uri_path}"


async def _invoke(handler: Callable[..., Any], **arguments: Any) -> Any:
    result = handler(**arguments)
    if inspect.isawaitable(result):
        return await result
    return result


async def _read_template(
    uri: AnyUrl,
    templates: dict[str, tuple[Route, Callable[..., Any]]],
) -> Any:
    parsed = urlsplit(str(uri))
    if parsed.scheme != "crawl4ai" or parsed.netloc != "resources":
        raise HTTPException(404, "resource not found")

    for route, handler in templates.values():
        match = route.path_regex.fullmatch(unquote(parsed.path))
        if match is None:
            continue

        arguments: dict[str, Any] = {}
        for name, value in match.groupdict().items():
            if value is None:
                break
            arguments[name] = route.param_convertors[name].convert(value)
        else:
            return await _invoke(handler, **arguments)

    raise HTTPException(404, "resource not found")
