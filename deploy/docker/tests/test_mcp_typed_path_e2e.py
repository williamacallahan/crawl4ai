"""Regression coverage for typed-converter ``@mcp_tool`` routes.

Drives the real MCP dispatch path (``Server.request_handlers[
CallToolRequest]`` — i.e. ``mcp_bridge._call_tool``) to confirm that FastAPI
path parameters declared with Starlette converters (e.g.
``/items/{item_id:int}``) are substituted into the loopback URL by
``mcp_bridge._make_http_proxy``.

Regression context
------------------
Before this fix, the proxy built the upstream URL from ``route.path``,
which Starlette keeps verbatim with the typed-converter syntax
(``/items/{item_id:int}``). The bare ``{item_id}`` placeholder never
matched, the literal ``{item_id:int}`` was sent upstream, FastAPI's path
regex did not match, and the upstream returned 404. The proxy now reads
``route.path_format`` (the bare ``{name}`` form), so substitution
succeeds.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" \\
        .venv/bin/python -m pytest -xvs deploy/docker/tests/test_mcp_typed_path_e2e.py
"""

import asyncio
import json

import httpx
import mcp.types as mcp_types
import mcp_bridge
from fastapi import FastAPI


# ── helpers ─────────────────────────────────────────────────────
def _capture_server(monkeypatch):
    """Patch ``mcp_bridge.Server`` with a capturing subclass.

    Mirrors the pattern in ``test_mcp_bridge_compat_20260824.py``: the
    subclass records the instance created inside ``attach_mcp`` so the test
    can drive its ``request_handlers`` directly (the real MCP dispatch path).
    """
    original = mcp_bridge.Server

    class CapturingServer(original):
        instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).instance = self

    monkeypatch.setattr(mcp_bridge, "Server", CapturingServer)
    return CapturingServer


def _asgi_client_factory(app: FastAPI, base_url: str):
    """An ``httpx.AsyncClient`` subclass bound to ``app`` via ASGI.

    The proxy inside ``_make_http_proxy`` constructs ``httpx.AsyncClient``
    and dials the loopback URL. To keep tests in-process (no real port
    binding) this subclass injects an ``ASGITransport`` bound to ``app``,
    so the loopback HTTP call actually reaches FastAPI.
    """

    class _ASGIClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.ASGITransport(app=app))
            kwargs.setdefault("base_url", base_url)
            super().__init__(*args, **kwargs)

    return _ASGIClient


def _build_app_with_typed_tool() -> FastAPI:
    """An app with a typed-converter @mcp_tool that advertises its path
    param via the ``__mcp_schema__`` override hook."""
    app = FastAPI()

    @app.get("/items/{item_id:int}")
    @mcp_bridge.mcp_tool("get_item")
    async def get_item(item_id: int):
        return {"item_id": item_id}

    # The override hook that ``_list_tools`` checks first. Without this,
    # ``_body_model`` would not see a BaseModel and the param would not be
    # advertised to the MCP client.
    get_item.__mcp_schema__ = {
        "type": "object",
        "properties": {"item_id": {"type": "integer"}},
        "required": ["item_id"],
    }
    return app


def _build_app_with_untyped_tool() -> FastAPI:
    """An app with a plain (no-converter) path-param @mcp_tool, used to guard
    against regressing the non-typed case the original code handled fine."""
    app = FastAPI()

    @app.get("/notes/{note_id}")
    @mcp_bridge.mcp_tool("get_note")
    async def get_note(note_id: str):
        return {"note_id": note_id}

    get_note.__mcp_schema__ = {
        "type": "object",
        "properties": {"note_id": {"type": "string"}},
        "required": ["note_id"],
    }
    return app


def _build_app_with_paramless_post_tool() -> FastAPI:
    """An app with one of the shipped-style parameter-less POST tools, used
    to confirm the fix does not regress the only kind of tool the deployed
    server actually registers today."""
    app = FastAPI()

    @app.post("/crawl")
    @mcp_bridge.mcp_tool("crawl")
    async def crawl():
        return {"ok": True}

    return app


async def _call_tool(server, name, arguments):
    result = await server.request_handlers[mcp_types.CallToolRequest](
        mcp_types.CallToolRequest(
            params={"name": name, "arguments": arguments}
        )
    )
    # ServerResult wraps the typed result in ``.root`` (see the compat test).
    return result.root


def _decode_text_content(text: str) -> object:
    """Decode the TextContent text the MCP client receives.

    The proxy returns ``r.text`` for GET tools (a JSON string) and
    ``r.json()`` for non-GET tools (a dict); ``_call_tool`` then
    ``json.dumps``-encodes that return value into the TextContent text. So
    GET tools are double-encoded (the MCP client must parse twice) and POST
    tools are single-encoded. Parse once; if we got another string, parse
    again.
    """
    val = json.loads(text)
    if isinstance(val, str):
        val = json.loads(val)
    return val


# ── tests ───────────────────────────────────────────────────────
def test_call_typed_tool_returns_item_through_real_dispatch(monkeypatch):
    """End-to-end (with fix): a typed-converter tool advertised via
    ``__mcp_schema__`` substitutes the path param and returns the item to the
    MCP client through the real ``_call_tool`` dispatch handler."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_app_with_typed_tool()
    base_url = "http://127.0.0.1:9999"
    mcp_bridge.attach_mcp(
        app, base_url=base_url, auth_headers_provider=lambda: {}
    )
    server = CapturingServer.instance

    monkeypatch.setattr(
        mcp_bridge.httpx, "AsyncClient", _asgi_client_factory(app, base_url)
    )

    result = asyncio.run(_call_tool(server, "get_item", {"item_id": 42}))
    payload = _decode_text_content(result.content[0].text)
    assert payload == {"item_id": 42}


def test_call_untyped_tool_still_substitutes(monkeypatch):
    """Regression guard: plain (no-converter) path params still substitute."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_app_with_untyped_tool()
    base_url = "http://127.0.0.1:9999"
    mcp_bridge.attach_mcp(
        app, base_url=base_url, auth_headers_provider=lambda: {}
    )
    server = CapturingServer.instance

    monkeypatch.setattr(
        mcp_bridge.httpx, "AsyncClient", _asgi_client_factory(app, base_url)
    )

    result = asyncio.run(_call_tool(server, "get_note", {"note_id": "abc"}))
    payload = _decode_text_content(result.content[0].text)
    assert payload == {"note_id": "abc"}


def test_call_paramless_post_tool_still_works(monkeypatch):
    """Regression guard: the only kind of tool the shipped server registers
    today (parameter-less POST) still works end-to-end."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_app_with_paramless_post_tool()
    base_url = "http://127.0.0.1:9999"
    mcp_bridge.attach_mcp(
        app, base_url=base_url, auth_headers_provider=lambda: {}
    )
    server = CapturingServer.instance

    monkeypatch.setattr(
        mcp_bridge.httpx, "AsyncClient", _asgi_client_factory(app, base_url)
    )

    result = asyncio.run(_call_tool(server, "crawl", {}))
    payload = _decode_text_content(result.content[0].text)
    assert payload == {"ok": True}
