"""Focused behavioral checks for the PR #6 runtime/UI repair slice."""

from __future__ import annotations

import asyncio
import base64
import re
import time
from pathlib import Path

import egress_broker
import egress_proxy
import httpx
import pytest
from auth_gate import AuthGateMiddleware
from egress_broker import PinnedTarget
from fastapi import BackgroundTasks, Request
from governor import BodySizeLimitMiddleware
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from webhook import WebhookDeliveryService

pytestmark = pytest.mark.posture

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = Path(__file__).resolve().parents[1]


async def _run_asgi(app, messages, *, headers=None):
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "path": "/upload",
            "raw_path": b"/upload",
            "query_string": b"",
            "headers": headers or [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
            "scheme": "http",
        },
        receive,
        send,
    )
    return sent


@pytest.mark.asyncio
async def test_body_limit_counts_chunked_receive_bytes():
    received = []

    async def endpoint(scope, receive, send):
        while True:
            message = await receive()
            received.append(message)
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = BodySizeLimitMiddleware(endpoint, max_bytes=5)
    sent = await _run_asgi(
        app,
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ],
    )

    assert sent[0]["status"] == 413
    assert received == []


@pytest.mark.asyncio
async def test_zero_body_limit_is_unbounded_and_preserves_receive():
    body = bytearray()

    async def endpoint(scope, receive, send):
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = BodySizeLimitMiddleware(endpoint, max_bytes=0)
    sent = await _run_asgi(
        app,
        [{"type": "http.request", "body": b"unbounded", "more_body": False}],
    )

    assert sent[0]["status"] == 204
    assert body == b"unbounded"


@pytest.mark.asyncio
async def test_body_limit_preserves_streaming_response_after_body_replay():
    async def endpoint(request):
        assert await request.body() == b"bounded"

        async def content():
            yield b"first\n"
            await asyncio.sleep(0)
            yield b"second\n"

        return StreamingResponse(content(), media_type="text/plain")

    app = BodySizeLimitMiddleware(
        Starlette(routes=[Route("/stream", endpoint, methods=["POST"])]),
        max_bytes=1024,
    )

    async def decorate(_request, call_next):
        response = await call_next(_request)
        response.headers["x-decorated"] = "true"
        return response

    app = BaseHTTPMiddleware(app, dispatch=decorate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/stream", content=b"bounded")

    assert response.status_code == 200
    assert response.text == "first\nsecond\n"
    assert response.headers["x-decorated"] == "true"


@pytest.mark.asyncio
async def test_cors_preflight_reaches_cors_middleware_without_opening_data_route():
    async def data(_request):
        return JSONResponse({"secret": True})

    inner = Starlette(routes=[Route("/data", data, methods=["POST"])])
    cors = CORSMiddleware(
        inner,
        allow_origins=["https://allowed.example"],
        allow_methods=["POST"],
        allow_headers=["authorization", "content-type"],
    )
    app = AuthGateMiddleware(
        cors,
        token_provider=lambda: "operator-token",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        preflight = await client.options(
            "/data",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == (
            "https://allowed.example"
        )
        assert (await client.options("/data")).status_code == 401
        assert (await client.post("/data")).status_code == 401


def test_websocket_subprotocol_carries_token_but_query_string_does_not():
    token = "operator.token-with_symbols"
    encoded = base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")
    protocol_scope = {
        "type": "websocket",
        "headers": [],
        "subprotocols": ["other", f"crawl4ai.bearer.{encoded}"],
        "query_string": b"",
    }
    query_scope = {
        "type": "websocket",
        "headers": [],
        "query_string": b"token=must-not-be-read",
    }

    assert AuthGateMiddleware._extract_token(protocol_scope) == token
    assert AuthGateMiddleware._extract_token(query_scope) is None


@pytest.mark.asyncio
async def test_websocket_gate_selects_authenticated_bearer_protocol():
    token = "operator.token-with_symbols"
    encoded = base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")
    protocol = f"crawl4ai.bearer.{encoded}"
    sent = []

    async def websocket_app(_scope, _receive, send):
        await send({"type": "websocket.accept"})
        await send({"type": "websocket.close", "code": 1000})

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    app = AuthGateMiddleware(websocket_app, token_provider=lambda: token)
    await app(
        {
            "type": "websocket",
            "path": "/monitor/ws",
            "headers": [],
            "subprotocols": [protocol],
            "query_string": b"",
        },
        receive,
        send,
    )

    assert sent[0] == {"type": "websocket.accept", "subprotocol": protocol}


@pytest.mark.asyncio
async def test_websocket_gate_does_not_select_protocol_authenticated_by_header():
    token = "operator-token"
    sent = []

    async def websocket_app(_scope, _receive, send):
        await send({"type": "websocket.accept"})
        await send({"type": "websocket.close", "code": 1000})

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    app = AuthGateMiddleware(websocket_app, token_provider=lambda: token)
    await app(
        {
            "type": "websocket",
            "path": "/monitor/ws",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "subprotocols": ["crawl4ai.bearer.not-the-authenticated-token"],
            "query_string": b"",
        },
        receive,
        send,
    )

    assert sent[0] == {"type": "websocket.accept"}


@pytest.mark.asyncio
async def test_dns_resolution_runs_off_the_event_loop(monkeypatch):
    def slow_resolve(url):
        time.sleep(0.05)
        return PinnedTarget("https", "example.com", 443, "93.184.216.34")

    monkeypatch.setattr(egress_broker, "resolve_and_pin", slow_resolve)
    ticked = asyncio.Event()

    async def tick():
        await asyncio.sleep(0.005)
        ticked.set()

    result, _ = await asyncio.gather(
        egress_broker.resolve_and_pin_async("https://example.com"), tick()
    )
    assert ticked.is_set()
    assert result.ip == "93.184.216.34"


@pytest.mark.asyncio
@pytest.mark.parametrize("job_kind", ["llm", "crawl"])
async def test_job_webhook_validation_runs_off_the_event_loop(monkeypatch, job_kind):
    import job
    import utils

    def slow_validate(_url):
        time.sleep(0.05)

    async def accepted(*_args, **_kwargs):
        return {"accepted": True}

    monkeypatch.setattr(utils, "validate_webhook_url", slow_validate)
    if job_kind == "llm":
        payload = job.LlmJobPayload(
            url="https://example.com",
            q="summary",
            webhook_config={"webhook_url": "https://hooks.example/callback"},
        )
        monkeypatch.setattr(job, "handle_llm_request", accepted)
        enqueue = job.llm_job_enqueue(
            payload,
            BackgroundTasks(),
            Request({"type": "http", "scheme": "http", "server": ("test", 80)}),
            None,
        )
    else:
        payload = job.CrawlJobPayload(
            urls=["https://example.com"],
            webhook_config={"webhook_url": "https://hooks.example/callback"},
        )
        monkeypatch.setattr(job, "handle_crawl_job", accepted)
        enqueue = job.crawl_job_enqueue(payload, BackgroundTasks(), None)

    task = asyncio.create_task(enqueue)
    await asyncio.sleep(0.01)
    assert not task.done()
    assert await task == {"accepted": True}


@pytest.mark.asyncio
async def test_pinning_proxy_lifecycle_registers_and_clears_global_proxy(monkeypatch):
    for name in (
        "CRAWL4AI_UPSTREAM_PROXY",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    proxy = await egress_proxy.start_pinning_proxy()
    try:
        assert egress_broker.get_egress_proxy() == proxy.url
    finally:
        await egress_proxy.stop_pinning_proxy(proxy)
    assert egress_broker.get_egress_proxy() is None


def test_upstream_proxy_parsing_falls_through_and_honors_no_proxy(monkeypatch):
    for name in (
        "CRAWL4AI_UPSTREAM_PROXY",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://broken.example:not-a-port")
    assert egress_proxy.upstream_proxy() == ("proxy.example", 3128, None)
    monkeypatch.setenv("HTTPS_PROXY", "https://unsupported.example")
    assert egress_proxy.upstream_proxy() == ("proxy.example", 3128, None)

    pin = PinnedTarget("https", "site.example", 443, "93.184.216.34")
    monkeypatch.setenv("NO_PROXY", "site.example:443")
    assert egress_proxy._use_upstream(pin) is None
    monkeypatch.setenv("NO_PROXY", "site.example:444")
    assert egress_proxy._use_upstream(pin) == ("proxy.example", 3128, None)


@pytest.mark.asyncio
async def test_upstream_proxy_connect_receives_only_the_pinned_ip(monkeypatch):
    seen = []

    async def corporate_proxy(reader, writer):
        seen.append(await reader.readline())
        while await reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await reader.read(65536)
        writer.write(b"TUNNEL-OK")
        await writer.drain()
        writer.close()

    corporate = await asyncio.start_server(corporate_proxy, "127.0.0.1", 0)
    corporate_port = corporate.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{corporate_port}")
    monkeypatch.delenv("CRAWL4AI_UPSTREAM_PROXY", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(
        egress_proxy,
        "resolve_and_pin",
        lambda _url: PinnedTarget(
            "https", "target.example", 443, "203.0.113.7"
        ),
    )

    proxy = await egress_proxy.start_pinning_proxy()
    try:
        reader, writer = await asyncio.open_connection(
            proxy.bound_host, proxy.bound_port
        )
        writer.write(b"CONNECT target.example:443 HTTP/1.1\r\n\r\n")
        await writer.drain()
        assert b"200" in await asyncio.wait_for(reader.readline(), timeout=2)
        await reader.readline()
        writer.write(b"hello")
        await writer.drain()
        assert await asyncio.wait_for(reader.read(100), timeout=2) == b"TUNNEL-OK"
        writer.close()
        await writer.wait_closed()
    finally:
        await egress_proxy.stop_pinning_proxy(proxy)
        corporate.close()
        await corporate.wait_closed()

    assert seen == [b"CONNECT 203.0.113.7:443 HTTP/1.1\r\n"]


@pytest.mark.asyncio
async def test_relative_webhook_redirect_is_joined_before_next_pinned_dial(monkeypatch):
    validated = []

    async def handle(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        if request.startswith(b"POST /start "):
            response = (
                b"HTTP/1.1 302 Found\r\nLocation: /next\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
        else:
            response = b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n"
        writer.write(response)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    def pin(url):
        validated.append(("pin", url))
        return PinnedTarget("http", "hooks.example", port, "127.0.0.1")

    monkeypatch.setattr(egress_broker, "resolve_and_pin", pin)
    service = WebhookDeliveryService(
        {"webhooks": {"retry": {"max_attempts": 1, "timeout_ms": 5000}}}
    )
    try:
        status = await service._deliver(
            f"http://hooks.example:{port}/start",
            {"ok": True},
            {"Content-Type": "application/json"},
        )
    finally:
        server.close()
        await server.wait_closed()

    absolute = f"http://hooks.example:{port}/next"
    assert status == 204
    assert validated.count(("pin", absolute)) == 1


@pytest.mark.asyncio
async def test_server_auth_posture_token_issuance_and_public_redirect(
    server_module, monkeypatch
):
    monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
    monkeypatch.setattr(server_module, "verify_email_domain", lambda _email: True)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server_module.app),
        base_url="http://test",
    ) as client:
        root = await client.get("/", follow_redirects=False)
        assert root.status_code in {302, 307}
        assert root.headers["location"] == "/playground"
        tailwind = await client.get("/static/assets/tailwind-3.4.17.min.css")
        assert tailwind.status_code == 200
        assert ".bg-dark" in tailwind.text
        bad_bearer = await client.get(
            "/schema", headers={"Authorization": "Bearer not-a-jwt"}
        )
        assert bad_bearer.status_code == 401
        assert bad_bearer.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in bad_bearer.headers[
            "content-security-policy"
        ]

        monkeypatch.setenv("CRAWL4AI_API_TOKEN", "effective-operator-token")
        monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "true")
        monkeypatch.setenv("SECRET_KEY", "test-only-secret-key-value-000000")
        issued = await client.post(
            "/token",
            json={
                "email": "operator@example.com",
                "api_token": "effective-operator-token",
            },
        )
        monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "false")
        disabled = await client.post(
            "/token",
            json={
                "email": "operator@example.com",
                "api_token": "effective-operator-token",
            },
        )
    assert issued.status_code == 200
    assert issued.json()["token_type"] == "bearer"
    assert disabled.status_code == 403


def test_exposed_jwt_only_posture_does_not_create_or_log_static_admin_token(
    server_module, monkeypatch, caplog
):
    monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
    monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "true")
    monkeypatch.setenv("GUNICORN_BIND", "0.0.0.0:11235")
    monkeypatch.setenv("SECRET_KEY", "test-only-secret-key-value-000000")

    server_module._resolve_auth()

    assert "CRAWL4AI_API_TOKEN" not in server_module.os.environ
    assert "CRAWL4AI_API_TOKEN=" not in caplog.text


def test_loopback_without_credentials_never_creates_a_token(
    server_module, monkeypatch, caplog
):
    monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
    monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "false")
    monkeypatch.setenv("GUNICORN_BIND", "127.0.0.1:11235")

    server_module._resolve_auth()

    assert "CRAWL4AI_API_TOKEN" not in server_module.os.environ
    assert "generated" not in caplog.text.lower()


def test_runtime_api_token_override_has_one_owner(server_module, monkeypatch):
    monkeypatch.setitem(server_module.config["security"], "api_token", "config-token")
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", "runtime-token")
    assert server_module._current_api_token() == "runtime-token"


def test_internal_mcp_auth_prefers_existing_static_operator_token(
    server_module, monkeypatch
):
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", "existing-operator-token")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    assert server_module._internal_service_auth_headers() == {
        "Authorization": "Bearer existing-operator-token"
    }


@pytest.mark.asyncio
async def test_health_uses_effective_redis_client_when_lifespan_is_active(
    server_module, monkeypatch
):
    class ReadyRedis:
        async def ping(self):
            return True

    monkeypatch.setattr(server_module, "redis", ReadyRedis())
    monkeypatch.setenv("C4AI_GIT_SHA", "0123456789abcdef")
    monkeypatch.setenv("HOSTNAME", "crawl4ai.1.test")
    server_module.app.state.readiness_checks_active = True
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server_module.app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
    finally:
        server_module.app.state.readiness_checks_active = False
    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["redis"] == "ready"
    assert payload["revision"] == "0123456789abcdef"
    assert payload["instance"] == "crawl4ai.1.test"


@pytest.mark.asyncio
async def test_health_reports_unavailable_effective_redis_without_details(
    server_module, monkeypatch
):
    class UnavailableRedis:
        async def ping(self):
            raise ConnectionError("internal redis topology detail")

    monkeypatch.setattr(server_module, "redis", UnavailableRedis())
    server_module.app.state.readiness_checks_active = True
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server_module.app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
    finally:
        server_module.app.state.readiness_checks_active = False
    assert response.status_code == 503
    assert response.json()["components"]["redis"] == "unavailable"
    assert "topology detail" not in response.text


def test_swarm_drain_keeps_established_vip_connections_ready():
    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text()
    drain = entrypoint.split("begin_drain()", maxsplit=1)[1]
    assert "CRAWL4AI_DRAIN_PATH" not in entrypoint
    assert drain.index('sleep "${CRAWL4AI_DRAIN_DELAY_SECONDS:-2}"') < drain.index(
        'kill -TERM "${SUPERVISORD_PID}"'
    )


def test_ui_tokens_are_ephemeral_and_cdn_assets_are_version_pinned():
    playground = (DOCKER_DIR / "static" / "playground" / "index.html").read_text()
    monitor = (DOCKER_DIR / "static" / "monitor" / "index.html").read_text()

    for html in (playground, monitor):
        assert "sessionStorage" not in html
        assert 'autocomplete="off"' in html
        assert "cdn.tailwindcss.com" not in html
        assert "tailwind-3.4.17.min.css" in html
        for tag in re.findall(r"<script\b[^>]*\bsrc=\"https://[^>]+>", html):
            assert "integrity=\"sha384-" in tag
            assert "crossorigin=\"anonymous\"" in tag

        tailwind = re.search(r'href="([^"]*tailwind-3\.4\.17\.min\.css)"', html)
        assert tailwind
        assert tailwind.group(1) == "/static/assets/tailwind-3.4.17.min.css"

    tailwind_asset = DOCKER_DIR / "static" / "assets" / "tailwind-3.4.17.min.css"
    assert tailwind_asset.is_file()
    compiled_css = tailwind_asset.read_text()
    assert all(
        selector in compiled_css
        for selector in (".bg-dark", ".bg-accent", ".text-primary", ".hidden")
    )

    assert "${CRAWL4AI_API_TOKEN}" in playground
    assert "os.environ['CRAWL4AI_API_TOKEN']" in playground
    assert "?token=" not in monitor
    assert "crawl4ai.bearer." in monitor


def test_playground_controls_are_named_and_header_can_wrap():
    from bs4 import BeautifulSoup

    playground = (DOCKER_DIR / "static" / "playground" / "index.html").read_text()
    document = BeautifulSoup(playground, "html.parser")

    assert document.select_one("#endpoint")["aria-label"] == "API endpoint"
    for control_id in ("urls", "st-total", "st-chunk", "st-conc"):
        assert document.select_one(f'label[for="{control_id}"]') is not None

    dialog = document.select_one("#stress-modal")
    assert dialog["role"] == "dialog"
    assert dialog["aria-modal"] == "true"
    assert dialog["aria-labelledby"] == "stress-title"
    assert document.select_one("#stress-title") is not None

    header = document.find("header")
    assert {"flex-wrap", "gap-4"} <= set(header["class"])
    assert {"flex-wrap", "gap-4"} <= set(header.find("h1")["class"])
    token_bar = document.select_one("#token-bar")
    assert {"flex-wrap", "gap-2"} <= set(token_bar["class"])
    assert {"flex-wrap", "gap-4"} <= set(token_bar.parent["class"])


class _StubMonitor:
    async def get_health_summary(self):
        return {}

    async def get_browser_list(self):
        return []

    def get_active_requests(self):
        return []

    def get_completed_requests(self, limit=10):
        return []

    def get_timeline_data(self, metric, window):
        return []

    def get_janitor_log(self, limit=10):
        return []

    def get_errors_log(self, limit=10):
        return []


class _StubWebSocket:
    def __init__(self, state):
        from starlette.websockets import WebSocketState

        self.application_state = state
        self.sends = 0
        self._connected = WebSocketState.CONNECTED

    async def accept(self):
        return None

    async def send_json(self, data):
        self.sends += 1
        # Exactly what Starlette raises once the socket has been closed.
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
        )


@pytest.mark.asyncio
async def test_monitor_websocket_stops_instead_of_looping_on_a_closed_socket(
    monkeypatch,
):
    """A send on a closed socket raises RuntimeError, not WebSocketDisconnect.

    The old handler retried every 2s, so one rollout logged the same traceback
    for the whole 390s stop-grace window (observed 2026-08-31 on haiku-18).
    """
    import monitor_routes
    from starlette.websockets import WebSocketState

    monkeypatch.setattr(monitor_routes, "get_monitor", lambda: _StubMonitor())

    closed = _StubWebSocket(WebSocketState.DISCONNECTED)
    await asyncio.wait_for(monitor_routes.websocket_endpoint(closed), timeout=5)
    assert closed.sends == 1

    # A still-connected socket keeps the retry: the loop must not exit.
    live = _StubWebSocket(WebSocketState.CONNECTED)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(monitor_routes.websocket_endpoint(live), timeout=0.5)
    assert live.sends == 1


def _extract_js_function(source: str, signature: str) -> str:
    """Slice one brace-balanced function out of a single-file HTML page."""
    start = source.index(signature)
    depth = 0
    for i in range(source.index("{", start), len(source)):
        depth += 1 if source[i] == "{" else -1 if source[i] == "}" else 0
        if depth == 0:
            return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def test_playground_api_error_message_survives_every_fastapi_error_shape():
    """Runs the real function; the other UI checks only read source text.

    `detail` is whatever the server sent: a string, an array of {loc,msg}, an
    array holding nulls or bare strings, or absent. Interpolating it printed
    "[object Object]" (2026-08-31 dogfood ISSUE-005), and a naive rewrite throws
    on a null entry or prints "undefined".
    """
    import json
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    assert node, "node is required (the workflow's JS actions already need it)"

    playground = (DOCKER_DIR / "static" / "playground" / "index.html").read_text()
    fn = _extract_js_function(playground, "function apiErrorMessage(body)")

    cases = {
        "null": (None, "Request failed"),
        "empty": ({}, "Request failed"),
        "null entry": ({"detail": [None]}, "Request failed"),
        "empty array": ({"detail": []}, "Request failed"),
        "entry with no msg": ({"detail": [{"loc": ["body", "urls"]}]}, "Request failed"),
        "string entries": ({"detail": ["Not authenticated"]}, "Not authenticated"),
        "real 422": (
            {"detail": [{"loc": ["body", "urls"], "msg": "List should have at least 1 item"}]},
            "body.urls: List should have at least 1 item",
        ),
        "two entries": (
            {"detail": [{"loc": ["body", "a"], "msg": "m1"}, {"loc": ["body", "b"], "msg": "m2"}]},
            "body.a: m1; body.b: m2",
        ),
        "detail string": ({"detail": "Token invalid"}, "Token invalid"),
        # 7017316 returns a correlation id instead of internal detail; losing it
        # from the UI leaves an operator nothing to grep for.
        "correlation id": (
            {"error": "Internal server error", "correlation_id": "abc123"},
            "Internal server error (correlation_id: abc123)",
        ),
    }

    script = [fn, "const out = {};"]
    for name, (body, _) in cases.items():
        script.append(
            f"try {{ out[{json.dumps(name)}] = apiErrorMessage({json.dumps(body)}); }}"
            f" catch (e) {{ out[{json.dumps(name)}] = 'THREW ' + e.message; }}"
        )
    script.append("console.log(JSON.stringify(out));")

    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as handle:
        handle.write("\n".join(script))
        path = handle.name
    result = subprocess.run([node, path], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr

    actual = json.loads(result.stdout)
    assert actual == {name: expected for name, (_, expected) in cases.items()}


def test_response_pane_never_parses_crawled_markup_into_the_document():
    """`innerHTML = textContent` re-parsed the crawl response as HTML.

    2026-08-31 dogfood: a crawled page's <img>/<link> were inserted into the
    Playground document and issued real requests from this origin, and
    highlight.js warned "unescaped HTML" on every run.
    """
    playground = (DOCKER_DIR / "static" / "playground" / "index.html").read_text()
    reset = playground.split("function forceHighlightElement", maxsplit=1)[1]
    reset = reset.split("hljs.highlightElement", maxsplit=1)[0]
    assert "element.textContent = text;" in reset
    assert "element.innerHTML" not in reset
    # One highlight per render. The deleted call sat after the if/else and so ran
    # a second time on the block a branch had just highlighted, which is what
    # produced the hljs warning. Match the shape (a highlight immediately after a
    # closing brace at the branch's indent), not a call count, so hoisting the
    # repeated querySelector stays allowed.
    body = playground.split("async function runCrawl", maxsplit=1)[1]
    body = body.split("async function runStressTest", maxsplit=1)[0]
    assert not re.search(
        r"\n {16}}\n+\s*forceHighlightElement\(", body
    ), "runCrawl highlights the response block twice on a success path"


def test_stress_test_fails_closed_and_averages_over_completed_chunks():
    """2026-08-31 dogfood ISSUE-001/ISSUE-002.

    Every chunk reported a tick on HTTP 401, and the average divided by the
    last-finished chunk's index rather than the number of timed chunks.
    """
    playground = (DOCKER_DIR / "static" / "playground" / "index.html").read_text()
    stress = playground.split("async function runStressTest", maxsplit=1)[1]

    success, _, failure = stress.partition("} catch (error) {")

    # Both request paths reject a non-2xx response instead of ticking it green,
    # and neither may parse the body in a way that can throw first (a proxy 502
    # answers with HTML, and that would hide the status).
    assert stress.count("if (!response.ok)") == 2
    assert stress.count("await response.json().catch(() => null)") == 2
    assert "await response.json();" not in stress

    # The average divides by the chunks actually timed, never by a chunk index.
    assert "totalTime / succeededChunks" in stress
    assert "totalTime / (index + 1)" not in stress

    # Success and failure bookkeeping each live on their own side of the catch —
    # a positive assertion, so deleting either one fails rather than passing.
    assert "succeededChunks++" in success and "succeededChunks++" not in failure
    assert "completed += batch.length;" in success
    assert "completed += batch.length;" not in failure
    assert "failedChunks++" in failure and "failedChunks++" not in success
    assert "Stress test finished with ${failedChunks}" in stress

    # Unknown memory reads the same in the per-chunk log and in the footer; the
    # footer used to print a fabricated 0MB beside a log line saying n/a.
    assert "Number.isFinite(memory) ? `${memory}MB` : 'n/a'" in stress
    assert "Number.isFinite(peakMemory) ? `${peakMemory}MB` : 'n/a'" in stress
    assert "let maxMem;" in stress  # never seeded with a real-looking 0


def test_monitor_controls_are_named_and_layout_survives_a_mobile_viewport():
    """2026-08-31 dogfood ISSUE-003/ISSUE-004: two critical axe `select-name`
    violations, and a header group 334px wider than a 390px viewport."""
    from bs4 import BeautifulSoup

    monitor_text = (DOCKER_DIR / "static" / "monitor" / "index.html").read_text()
    document = BeautifulSoup(monitor_text, "html.parser")

    assert document.select_one("#filter-requests")["aria-label"] == "Filter requests"
    assert document.select_one("#timeline-metric")["aria-label"] == "Timeline metric"

    header = document.find("header")
    assert {"flex-wrap", "gap-4"} <= set(header["class"])
    assert {"flex-wrap", "gap-4"} <= set(header.find("h1")["class"])
    controls = document.select_one("header > div.ml-auto")
    assert {"flex-wrap", "gap-4"} <= set(controls["class"])

    # The vendored Tailwind build ships no responsive variants, so the stacking
    # has to come from the page's own media query. Assert the rule and the class
    # it targets together — a rule keyed to some other selector would leave every
    # grid multi-column at 390px while both halves "exist".
    grids = document.select(".responsive-grid")
    assert len(grids) == 4
    assert all("grid" in grid["class"] for grid in grids)
    rule = re.search(
        r"@media \(max-width: 767px\) \{(.*?)\n {8}\}", monitor_text, re.DOTALL
    )
    assert rule, "no narrow-viewport media query"
    assert ".responsive-grid" in rule.group(1)
    assert "grid-template-columns: minmax(0, 1fr);" in rule.group(1)

    # Any responsive utility the page uses must exist in the vendored build,
    # which is generated per-page: a `md:` class that is not compiled is a silent
    # no-op that reads like a working fix.
    compiled_css = (
        DOCKER_DIR / "static" / "assets" / "tailwind-3.4.17.min.css"
    ).read_text()
    used_variants = {
        cls
        for tag in document.find_all(class_=True)
        for cls in tag["class"]
        if ":" in cls
    }
    missing = {
        cls for cls in used_variants if cls.replace(":", "\\:") not in compiled_css
    }
    assert not missing, f"classes absent from the vendored Tailwind build: {missing}"


def test_ui_error_and_websocket_fallback_paths_are_explicit():
    playground = (DOCKER_DIR / "static" / "playground" / "index.html").read_text()
    monitor = (DOCKER_DIR / "static" / "monitor" / "index.html").read_text()

    assert "const errorData = await response.json().catch(() => ({}));" in playground
    # FastAPI 422 sends `detail` as an array of objects; interpolating it printed
    # "[object Object]" and dropped the only actionable text (2026-08-31 dogfood).
    assert "throw new Error(apiErrorMessage(errorData));" in playground
    assert "function apiErrorMessage(body" in playground
    run_crawl = playground.split("async function runCrawl", maxsplit=1)[1]
    run_crawl = run_crawl.split("async function runStressTest", maxsplit=1)[0]
    assert ".detail || " not in run_crawl
    assert "Number.isFinite(memory) && Number.isFinite(peakMemory)" in playground

    assert "if (!token)" in monitor
    assert "websocket.onclose = null;" in monitor
    assert "updateConnectionStatus('disconnected', 'Token required');" in monitor


def test_container_contracts_use_app_readiness_and_compose_v5_shape():
    import yaml

    compose_text = (REPO_ROOT / "docker-compose.yml").read_text()
    compose = yaml.safe_load(compose_text)
    base = compose["x-base-config"]
    assert base["env_file"] == [{"path": ".llm.env", "required": False}]
    assert (
        "CRAWL4AI_API_TOKEN_FROM_HOST=${CRAWL4AI_API_TOKEN:-}"
        in base["environment"]
    )
    assert "REDIS_PASSWORD_FROM_HOST=${REDIS_PASSWORD:-}" in base["environment"]
    assert not any(
        value.startswith("CRAWL4AI_API_TOKEN=") for value in base["environment"]
    )
    assert "pids_limit" not in base
    assert base["deploy"]["resources"]["limits"]["pids"] == 512

    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "redis-cli ping" not in dockerfile
    assert "http://127.0.0.1:11235/health" in dockerfile

    ignore = (REPO_ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in ignore
    assert ".llm.env" in ignore

    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text()
    assert "EMBEDDED_REDIS" in entrypoint
    assert 'export REDIS_PASSWORD="${REDIS_PASSWORD:-}"' in entrypoint
    assert "embedded Redis requires an existing operator-managed" in entrypoint
    assert "CRAWL4AI_JWT_ENABLED" in entrypoint
    assert "token_hex" not in entrypoint
    assert "requires an existing operator-managed REDIS_PASSWORD" in entrypoint

    supervisord = (DOCKER_DIR / "supervisord.conf").read_text()
    assert "--bind 127.0.0.1 -::1" in supervisord

    server = (DOCKER_DIR / "server.py").read_text()
    assert "await init_permanent(get_default_browser_config())" in server
    assert 'await asyncio.to_thread(_store_artifact, "pdf", pdf_data)' in server
    assert "await asyncio.to_thread(resolve_artifact, artifact_id)" in server


def test_ci_build_reclaims_repo_disk_before_and_after_native_build():
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci-deploy.yml").read_text()
    )
    build_script = next(
        step["run"]
        for step in workflow["jobs"]["build"]["steps"]
        if step.get("name") == "Build and push ${{ matrix.arch }}"
    )

    assert "reclaim_build_disk()" in build_script
    assert "trap reclaim_build_disk EXIT" in build_script
    assert build_script.index("reclaim_build_disk\n") < build_script.index(
        "docker build --provenance"
    )
    assert "docker builder prune -af --filter until=10m" in build_script
    assert "docker system prune" not in build_script
    assert build_script.count('docker push "${IMAGE}:${GIT_SHA}-${ARCH}"') == 1
    assert "for attempt in" not in build_script


def test_coolify_keeps_external_durable_redis_without_client_only_password():
    import yaml

    compose = yaml.safe_load((REPO_ROOT / "docker-compose.coolify.yml").read_text())
    app = compose["services"]["crawl4ai"]
    redis = compose["services"]["redis"]
    assert "REDIS_HOST=redis" in app["environment"]
    assert (
        "CRAWL4AI_API_TOKEN_FROM_HOST=${CRAWL4AI_API_TOKEN:-}"
        in app["environment"]
    )
    assert "--requirepass" not in redis["command"]
    assert "crawl4ai-redis:/data" in redis["volumes"]
