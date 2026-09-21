"""Regression coverage for the MCP ``inputSchema`` of query/path-param tools.

Before the fix, ``mcp_bridge._list_tools`` derived ``inputSchema`` from
``_schema(_body_model(orig_fn))``: ``_body_model`` only returned a pydantic
``BaseModel`` body, so any ``@mcp_tool`` route whose endpoint takes
``Query(...)``/``Path(...)`` params and no ``BaseModel`` body (the shipped
``GET /ask`` is the canonical example) was advertised with the empty
``{"type": "object"}`` schema — none of its query params were exposed to MCP
clients. Tools that relied on ``inputSchema`` to construct their arguments
would therefore call ``ask`` with ``{}`` and receive the full unfiltered
context corpus.

The fix routes the schema through ``_route_schema(route)``, which:

* for a route with a ``BaseModel`` body, returns that model's
  ``model_json_schema()`` *verbatim* (preserving ``$defs``/``$ref``/``title``
  — the behaviour the six shipped POST tools already rely on); and
* for a route with no body, synthesizes a flat
  ``{"type":"object","properties":{...},"required":[...]}`` from
  ``route.dependant.query_params``/``path_params`` via ``pydantic.create_model``
  so each param's type, default, description, and constraints (``pattern``,
  ``ge``, ``le``, ``min_length``, ...) are emitted by pydantic itself.

These tests drive the real MCP ``tools/list`` dispatch handler
(``server.request_handlers[ListToolsRequest]``) and the real ``tools/call``
dispatch handler, so they verify what MCP clients actually receive — not a
direct call to the schema helper.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" \\
        .venv/bin/python -m pytest -xvs deploy/docker/tests/test_mcp_tool_schema_query_params.py
"""

import asyncio
import json
from typing import Optional

import httpx
import mcp.types as mcp_types
import mcp_bridge
from fastapi import Depends, FastAPI, Path, Query, Request
from pydantic import BaseModel


# ── helpers (mirror test_mcp_typed_path_e2e.py) ─────────────────────
def _capture_server(monkeypatch):
    """Patch ``mcp_bridge.Server`` with a capturing subclass so the test can
    drive its ``request_handlers`` directly (the real MCP dispatch path)."""
    original = mcp_bridge.Server

    class CapturingServer(original):
        instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).instance = self

    monkeypatch.setattr(mcp_bridge, "Server", CapturingServer)
    return CapturingServer


def _list_tools(server):
    """Drive the real ``tools/list`` dispatch handler; return the tools."""
    result = asyncio.run(
        server.request_handlers[mcp_types.ListToolsRequest](
            mcp_types.ListToolsRequest()
        )
    )
    return result.root.tools


def _asgi_client_factory(app, base_url):
    """An ``httpx.AsyncClient`` subclass bound to ``app`` via ASGI so the
    loopback proxy stays in-process (no real port binding)."""

    class _ASGIClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.ASGITransport(app=app))
            kwargs.setdefault("base_url", base_url)
            super().__init__(*args, **kwargs)

    return _ASGIClient


def _call_tool(server, name, arguments):
    result = asyncio.run(
        server.request_handlers[mcp_types.CallToolRequest](
            mcp_types.CallToolRequest(
                params={"name": name, "arguments": arguments}
            )
        )
    )
    return result.root


def _decode_text_content(text: str):
    """Decode the double-encoded TextContent text the MCP client receives.

    GET tools return ``r.text`` (a JSON string); ``_call_tool`` then
    ``json.dumps``-encodes that string, so GET tools are double-encoded."""
    val = json.loads(text)
    if isinstance(val, str):
        val = json.loads(val)
    return val


# ── app builders ───────────────────────────────────────────────────
def _build_ask_app() -> FastAPI:
    """A GET @mcp_tool that mirrors the shipped ``/ask`` endpoint: only
    ``Query(...)`` params (plus ``Request``/``Depends``), no BaseModel body."""
    app = FastAPI()

    def _token_dep():
        return {}

    @app.get("/ask")
    @mcp_bridge.mcp_tool("ask")
    async def get_context(
        request: Request,
        _td: dict = Depends(_token_dep),
        context_type: str = Query("all", pattern="^(code|doc|all)$"),
        query: Optional[str] = Query(
            None, description="search query to filter chunks"
        ),
        score_ratio: float = Query(
            0.5, ge=0.0, le=1.0, description="min score as fraction of max_score"
        ),
        max_results: int = Query(
            20, ge=1, description="absolute cap on returned chunks"
        ),
    ):
        """Ask about Crawl4AI. query is RECOMMENDED to filter the response."""
        return {"context_type": context_type, "query": query}

    return app


def _build_path_param_app() -> FastAPI:
    """A GET @mcp_tool with a required ``Path(...)`` param + constraints."""
    app = FastAPI()

    @app.get("/items/{item_id:int}")
    @mcp_bridge.mcp_tool("get_item")
    async def get_item(
        item_id: int = Path(..., ge=1, le=1000, description="the item id"),
    ):
        return {"item_id": item_id}

    return app


def _build_mixed_params_app() -> FastAPI:
    """A GET @mcp_tool with both query and (required) path params."""
    app = FastAPI()

    @app.get("/notes/{note_id}")
    @mcp_bridge.mcp_tool("get_note")
    async def get_note(
        note_id: str = Path(..., description="the note id"),
        highlight: bool = Query(False, description="highlight matches"),
    ):
        return {"note_id": note_id, "highlight": highlight}

    return app


def _build_required_query_app() -> FastAPI:
    """A GET @mcp_tool with a required ``Query(...)`` (no default)."""
    app = FastAPI()

    @app.get("/search")
    @mcp_bridge.mcp_tool("search")
    async def search(
        q: str = Query(..., min_length=1, description="the search term"),
    ):
        return {"q": q}

    return app


def _build_body_post_app() -> FastAPI:
    """A POST @mcp_tool with a ``BaseModel`` body — the six shipped POST tools'
    shape. The schema must stay the model's ``model_json_schema()`` verbatim."""
    app = FastAPI()

    class CrawlRequest(BaseModel):
        url: str
        max_results: int = 20

    @app.post("/crawl")
    @mcp_bridge.mcp_tool("crawl")
    async def crawl(req: CrawlRequest):
        return {"url": req.url}

    return app


def _build_nested_body_post_app() -> FastAPI:
    """A POST @mcp_tool whose body model references another model — guards
    that ``$defs``/``$ref`` survive verbatim (a flat synthesis would not)."""
    app = FastAPI()

    class Tag(BaseModel):
        name: str

    class Item(BaseModel):
        title: str
        tag: Tag

    @app.post("/items")
    @mcp_bridge.mcp_tool("create_item")
    async def create_item(item: Item):
        return {"title": item.title}

    return app


def _build_paramless_post_app() -> FastAPI:
    """A POST @mcp_tool with no params at all — old ``_schema(None)`` fallback."""
    app = FastAPI()

    @app.post("/ping")
    @mcp_bridge.mcp_tool("ping")
    async def ping():
        return {"ok": True}

    return app


def _build_override_app() -> FastAPI:
    """A GET @mcp_tool that sets ``__mcp_schema__`` explicitly — the override
    must take precedence over the synthesized schema."""
    app = FastAPI()

    @app.get("/items/{item_id:int}")
    @mcp_bridge.mcp_tool("get_item")
    async def get_item(item_id: int):
        return {"item_id": item_id}

    get_item.__mcp_schema__ = {
        "type": "object",
        "properties": {"item_id": {"type": "integer"}},
        "required": ["item_id"],
    }
    return app


# ── the original bug: /ask advertises all four query params ────────
def test_ask_tool_advertises_query_params_in_schema(monkeypatch):
    """The shipped ``/ask`` shape: a query-only @mcp_tool must expose all four
    ``Query(...)`` params (with constraints/defaults/descriptions) in
    ``inputSchema`` instead of the empty ``{"type": "object"}`` that the bug
    produced."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_ask_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    assert [t.name for t in tools] == ["ask"]

    schema = tools[0].inputSchema
    # The bug: schema was {"type": "object"} with no "properties" key.
    assert schema["type"] == "object"
    assert "properties" in schema, (
        "inputSchema must list query params as properties; got the empty "
        f"fallback {schema!r}"
    )

    props = schema["properties"]
    assert set(props) == {"context_type", "query", "score_ratio", "max_results"}
    # Request / Depends params must NOT leak into the advertised schema.
    assert "request" not in props
    assert "_td" not in props

    # context_type: str, pattern constraint, default
    assert props["context_type"]["type"] == "string"
    assert props["context_type"]["pattern"] == "^(code|doc|all)$"
    assert props["context_type"]["default"] == "all"

    # query: Optional[str] (nullable), default None, description carried
    assert props["query"]["default"] is None
    assert props["query"]["description"] == "search query to filter chunks"
    assert props["query"]["anyOf"] == [{"type": "string"}, {"type": "null"}]

    # score_ratio: float -> number, ge/le -> minimum/maximum, default
    assert props["score_ratio"]["type"] == "number"
    assert props["score_ratio"]["minimum"] == 0.0
    assert props["score_ratio"]["maximum"] == 1.0
    assert props["score_ratio"]["default"] == 0.5
    assert props["score_ratio"]["description"] == "min score as fraction of max_score"

    # max_results: int -> integer, ge -> minimum, default
    assert props["max_results"]["type"] == "integer"
    assert props["max_results"]["minimum"] == 1
    assert props["max_results"]["default"] == 20
    assert props["max_results"]["description"] == "absolute cap on returned chunks"

    # Every /ask param has a default -> none are required.
    assert schema.get("required", []) == []


def test_ask_tool_description_still_carried_from_docstring(monkeypatch):
    """The free-text ``description`` (from ``inspect.getdoc``) was relayed
    correctly even before the fix; guard against regressing it while the
    schema path changes."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_ask_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    assert tools[0].description
    assert "RECOMMENDED" in tools[0].description


def test_ask_call_with_query_args_still_filters(monkeypatch):
    """A client that sends query args (despite the previously-empty schema)
    must still get them forwarded to the loopback GET — confirms the fix is a
    schema-advertisement change, NOT a call-path change."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_ask_app()
    base_url = "http://127.0.0.1:9999"
    mcp_bridge.attach_mcp(
        app, base_url=base_url, auth_headers_provider=lambda: {}
    )
    server = CapturingServer.instance

    monkeypatch.setattr(
        mcp_bridge.httpx, "AsyncClient", _asgi_client_factory(app, base_url)
    )

    result = _call_tool(
        server,
        "ask",
        {
            "context_type": "code",
            "query": "browser",
            "score_ratio": 0.5,
            "max_results": 5,
        },
    )
    payload = _decode_text_content(result.content[0].text)
    assert payload == {"context_type": "code", "query": "browser"}


# ── path-param-only tool: required + constraints ───────────────────
def test_path_param_tool_advertises_required_and_constraints(monkeypatch):
    CapturingServer = _capture_server(monkeypatch)
    app = _build_path_param_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    assert set(schema["properties"]) == {"item_id"}
    assert schema["properties"]["item_id"]["type"] == "integer"
    assert schema["properties"]["item_id"]["minimum"] == 1
    assert schema["properties"]["item_id"]["maximum"] == 1000
    assert schema["properties"]["item_id"]["description"] == "the item id"
    # Path params are always required.
    assert schema["required"] == ["item_id"]


# ── mixed query + path params ───────────────────────────────────────
def test_mixed_query_and_path_params(monkeypatch):
    CapturingServer = _capture_server(monkeypatch)
    app = _build_mixed_params_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    assert set(schema["properties"]) == {"note_id", "highlight"}
    assert schema["properties"]["note_id"]["type"] == "string"
    assert schema["properties"]["highlight"]["type"] == "boolean"
    assert schema["properties"]["highlight"]["default"] is False
    # note_id (path) is required; highlight (query, has default) is not.
    assert schema["required"] == ["note_id"]


# ── required Query (no default) appears in required ────────────────
def test_required_query_param_in_required(monkeypatch):
    CapturingServer = _capture_server(monkeypatch)
    app = _build_required_query_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    assert set(schema["properties"]) == {"q"}
    assert schema["properties"]["q"]["type"] == "string"
    assert schema["properties"]["q"]["minLength"] == 1
    assert "default" not in schema["properties"]["q"]
    assert schema["required"] == ["q"]


# ── body-case preservation: POST tools keep model_json_schema() verbatim
def test_body_model_schema_preserved_verbatim(monkeypatch):
    """Regression guard for the six shipped POST tools: a ``BaseModel`` body
    must advertise the model's ``model_json_schema()`` verbatim, including
    the ``title`` and top-level properties — NOT the synthesized flat shape
    (which would wrap the body in a ``req`` property or strip the title)."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_body_post_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    # The body model's own properties at the top level (no `req` wrapper).
    assert set(schema["properties"]) == {"url", "max_results"}
    assert schema["properties"]["url"]["type"] == "string"
    assert schema["properties"]["max_results"]["type"] == "integer"
    assert schema["properties"]["max_results"]["default"] == 20
    assert schema["required"] == ["url"]
    # The model's class name is preserved as the top-level title (existing
    # behaviour for body tools — not stripped like the synthesized "Input").
    assert schema["title"] == "CrawlRequest"


def test_nested_body_model_keeps_defs_and_refs(monkeypatch):
    """A body model that references another model must advertise ``$defs`` and
    ``$ref`` verbatim — a flat synthesis would have collapsed these."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_nested_body_post_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    assert "$defs" in schema, "nested body model must keep its $defs"
    # The referenced model goes in $defs; the root model lives at the top level.
    assert "Tag" in schema["$defs"]
    assert schema["title"] == "Item"
    assert schema["required"] == ["title", "tag"]
    # The nested field is a $ref, not an inlined object.
    assert schema["properties"]["tag"] == {"$ref": "#/$defs/Tag"}


# ── paramless POST keeps the {"type": "object"} fallback ────────────
def test_paramless_post_keeps_empty_object_schema(monkeypatch):
    CapturingServer = _capture_server(monkeypatch)
    app = _build_paramless_post_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    assert schema == {"type": "object"}


# ── __mcp_schema__ override still takes precedence ─────────────────
def test_mcp_schema_override_takes_precedence(monkeypatch):
    """The explicit ``__mcp_schema__`` hook must still win over the synthesized
    schema (``test_mcp_typed_path_e2e`` relies on this)."""
    CapturingServer = _capture_server(monkeypatch)
    app = _build_override_app()
    mcp_bridge.attach_mcp(
        app, base_url="http://127.0.0.1:9999", auth_headers_provider=lambda: {}
    )

    tools = _list_tools(CapturingServer.instance)
    schema = tools[0].inputSchema
    # The override is returned verbatim — note the absence of `description`,
    # `minimum`, `maximum` that the synthesized schema would have added.
    assert schema == {
        "type": "object",
        "properties": {"item_id": {"type": "integer"}},
        "required": ["item_id"],
    }
