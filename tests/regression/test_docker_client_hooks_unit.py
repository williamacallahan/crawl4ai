"""Regression coverage for the Docker client hooks payload contract.

The Docker server's ``HookConfig`` schema (since 0.9.0) accepts only a
declarative ``hooks`` list; the legacy ``code``/``timeout`` fields were
removed (replaced by fixed actions to prevent RCE) - see
``deploy/docker/MIGRATION.md``. These tests pin the client/server hook
contract so a future server schema change (or a client regression to the
legacy shape) is caught here instead of causing hooks to be silently
ignored again. No browser, network, or Redis required.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
import pytest
import pytest_asyncio

# deploy/docker holds ``schemas`` (imported as a bare module by the server
# tests); put it on sys.path so we can cross-validate the client payload
# against the real ``HookConfig`` without starting the server.
_DEPLOY_DOCKER = Path(__file__).resolve().parents[2] / "deploy" / "docker"
if str(_DEPLOY_DOCKER) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DOCKER))

from crawl4ai.docker_client import (  # noqa: E402
    Crawl4aiClientError,
    Crawl4aiDockerClient,
    ConnectionError,
    RequestError,
)
from crawl4ai.async_configs import CrawlerRunConfig  # noqa: E402
from schemas import HookConfig  # noqa: E402


@pytest.fixture
def client():
    """A client whose HTTP transport never touches the network.

    ``_prepare_request`` is synchronous; the constructor's ``httpx.AsyncClient``
    is harmless without I/O and is closed to avoid resource warnings.
    """
    c = Crawl4aiDockerClient(base_url="http://localhost:0", verbose=False)
    try:
        yield c
    finally:
        asyncio.run(c.close())


def _spec(action="block_resources", **params):
    return {"action": action, "params": params or {}}


# --- server contract pin (would have caught this bug when the server changed) ---
def test_server_hook_config_accepts_only_the_declarative_hooks_field():
    # If the server re-adds ``code``/``timeout`` or renames ``hooks``, the client
    # payload format must be revisited. Mirrors the server-side contract test
    # in deploy/docker/tests/test_crawl_hook_lifecycle.py.
    assert set(HookConfig.model_json_schema()["properties"]) == {"hooks"}


# --- _build_hooks_payload: declarative input ---
def test_declarative_wrapper_dict_is_forwarded_unchanged():
    spec = _spec("block_resources", resource_types=["image", "font"])
    assert Crawl4aiDockerClient._build_hooks_payload({"hooks": [spec]}) == {"hooks": [spec]}


def test_declarative_bare_list_is_wrapped_under_the_hooks_key():
    spec = _spec("scroll_to_bottom", max_steps=10, delay_ms=500)
    assert Crawl4aiDockerClient._build_hooks_payload([spec]) == {"hooks": [spec]}


def test_declarative_payload_carries_no_legacy_code_or_timeout_keys():
    # Guards against regressing the bug: the client must never emit the shape
    # the server removed in 0.9.0.
    payload = Crawl4aiDockerClient._build_hooks_payload(
        {"hooks": [_spec("wait_for_timeout", timeout_ms=500)]}
    )
    assert set(payload) == {"hooks"}
    assert "code" not in payload and "timeout" not in payload


# --- _build_hooks_payload: legacy code-based input must not fail silently ---
def test_legacy_callable_hooks_emit_deprecation_warning_and_return_none():
    async def my_hook(page, context, **kwargs):
        return page

    with pytest.warns(FutureWarning, match="Code-based hooks"):
        assert Crawl4aiDockerClient._build_hooks_payload(
            {"on_page_context_created": my_hook}
        ) is None


def test_legacy_string_hooks_emit_deprecation_warning_and_return_none():
    legacy = {"before_goto": "async def hook(page, context, url, **kwargs):\n    return page\n"}
    with pytest.warns(FutureWarning, match="Code-based hooks"):
        assert Crawl4aiDockerClient._build_hooks_payload(legacy) is None


# --- _prepare_request ---
def test_no_hooks_omits_the_hooks_key(client):
    assert "hooks" not in client._prepare_request(["https://example.com"])


def test_declarative_request_body_validates_against_server_schema(client):
    # Cross-contract: the body the client builds must round-trip through the
    # server's HookConfig with the specs populated (not silently dropped).
    spec = _spec("block_resources", resource_types=["image", "font"])
    data = client._prepare_request(["https://example.com"], hooks={"hooks": [spec]})

    config = HookConfig.model_validate(data["hooks"])
    assert len(config.hooks) == 1
    assert config.hooks[0].action == "block_resources"
    assert config.hooks[0].params == {"resource_types": ["image", "font"]}


# --- public crawl() wire contract ---
@pytest.mark.asyncio
async def test_crawl_sends_declarative_hooks_payload(client, monkeypatch):
    captured = {}

    async def fake_check_server(_self):
        return None

    async def fake_request(_self, method, endpoint, **kwargs):
        captured["json"] = kwargs.get("json")
        captured["timeout"] = kwargs.get("timeout")

        class FakeResponse:
            def json(self):
                return {"success": True, "results": []}

        return FakeResponse()

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    monkeypatch.setattr(Crawl4aiDockerClient, "_request", fake_request)

    spec = _spec("block_resources", resource_types=["image"])
    await client.crawl(
        ["https://example.com"],
        hooks={"hooks": [spec]},
        hooks_timeout=42,
    )

    assert captured["json"]["hooks"] == {"hooks": [spec]}
    assert "code" not in captured["json"]["hooks"]
    assert "timeout" not in captured["json"]["hooks"]
    # hooks_timeout is now the HTTP request timeout (the per-hook timeout
    # field was removed from the server schema).
    assert captured["timeout"] == 42


@pytest.mark.asyncio
async def test_crawl_with_legacy_hooks_warns_and_sends_no_hooks(client, monkeypatch):
    captured = {}

    async def fake_check_server(_self):
        return None

    async def fake_request(_self, method, endpoint, **kwargs):
        captured["json"] = kwargs.get("json")

        class FakeResponse:
            def json(self):
                return {"success": True, "results": []}

        return FakeResponse()

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    monkeypatch.setattr(Crawl4aiDockerClient, "_request", fake_request)

    async def my_hook(page, context, **kwargs):
        return page

    with pytest.warns(FutureWarning, match="Code-based hooks"):
        await client.crawl(
            ["https://example.com"],
            hooks={"on_page_context_created": my_hook},
        )

    # Legacy hooks are dropped (not sent as a payload the server would ignore).
    assert "hooks" not in captured["json"]


def test_api_token_sets_bearer_header():
    """A static CRAWL4AI_API_TOKEN must authenticate every request.

    The server binds beyond loopback only with a credential configured, and
    every endpoint except /health rejects requests without a Bearer token, so
    the client needs a static-token lane besides the JWT authenticate() flow.
    """
    c = Crawl4aiDockerClient(
        base_url="http://localhost:0", verbose=False, api_token="static-token"
    )
    try:
        assert c._http_client.headers["Authorization"] == "Bearer static-token"
    finally:
        asyncio.run(c.close())


# --- streaming crawl() error contract ---
# crawl()'s docstring promises that a server-side rejection such as the
# hooks-disabled 403 is "raised here as RequestError", unqualified by mode.
# The non-streaming path honours this via _request(); these tests pin the
# streaming path to the same contract so httpx.HTTPStatusError never leaks
# (regression coverage for the bug introduced in commit 392c9239, where the
# streaming refactor dropped the surrounding try/except). The responses are
# served by an in-process ASGI app via httpx.ASGITransport, mirroring how
# FastAPI's http_exception_handler emits a JSON 403/400 before any stream body
# starts — so no docker server, browser, or Redis is required.

def _asgi_json_error(status_code: int, detail: str):
    async def app(scope, receive, send):
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status_code,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
    return app


def _asgi_non_json_error(status_code: int, body: bytes, content_type="text/html"):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status_code,
                    "headers": [(b"content-type", content_type.encode())]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
    return app


def _asgi_ndjson(lines):
    async def app(scope, receive, send):
        body = ("".join(line + "\n" for line in lines)).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/x-ndjson")]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
    return app


@pytest_asyncio.fixture
async def streaming_client(monkeypatch):
    """A client whose transport is swapped per-test with an in-process ASGI app.

    ``_check_server`` is a no-op so the client never reaches ``/health``. The
    default ``_http_client`` from ``__init__`` is closed up front (it is never
    used) and whatever client is installed at teardown is closed in the same
    event loop, so no ``httpx.AsyncClient`` leaks regardless of which path the
    test took.
    """
    async def fake_check_server(_self):
        return None

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    c = Crawl4aiDockerClient(base_url="http://test", verbose=False)
    await c._http_client.aclose()
    try:
        yield c
    finally:
        if c._http_client is not None and not c._http_client.is_closed:
            await c._http_client.aclose()


def _install_asgi(client, app):
    client._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={"Content-Type": "application/json"},
    )


@pytest.mark.asyncio
async def test_streaming_crawl_raises_request_error_on_403_hooks_rejection(streaming_client):
    """The documented 403 hooks-disabled rejection surfaces as RequestError.

    This is the exact case the crawl() docstring promises ("raised here as
    RequestError"); before the fix the streaming branch leaked
    httpx.HTTPStatusError instead. The message must carry the JSON ``detail``
    verbatim, proving the body was read inside the stream context (the naive
    port of _request() -- catching outside the async with -- would yield
    ``httpx.StreamClosed`` and lose the body).
    """
    _install_asgi(streaming_client, _asgi_json_error(
        403, "Hooks are disabled. Set CRAWL4AI_HOOKS_ENABLED=true to enable."))
    cfg = CrawlerRunConfig(stream=True)

    with pytest.raises(RequestError) as exc_info:
        gen = await streaming_client.crawl(["https://example.com"], crawler_config=cfg)
        async for _ in gen:
            pass

    assert not isinstance(exc_info.value, httpx.HTTPStatusError)
    assert str(exc_info.value) == (
        "Server error 403: Hooks are disabled. Set CRAWL4AI_HOOKS_ENABLED=true to enable.")


@pytest.mark.asyncio
async def test_streaming_crawl_non_json_body_falls_back_to_str(streaming_client):
    """A non-JSON body (e.g. text/html 502 from a proxy) falls back to str(e).

    Mirrors _request()'s content-type branch so the streaming path never
    attempts ``response.json()`` on a non-JSON body and crash with a second
    unhandled exception.
    """
    _install_asgi(streaming_client, _asgi_non_json_error(502, b"<html>Bad Gateway</html>"))
    cfg = CrawlerRunConfig(stream=True)

    with pytest.raises(RequestError, match=r"Server error 502: ") as exc_info:
        gen = await streaming_client.crawl(["https://example.com"], crawler_config=cfg)
        async for _ in gen:
            pass

    assert "Bad Gateway" in str(exc_info.value)


@pytest.mark.parametrize("body", [b"[]", b"{"])
@pytest.mark.asyncio
async def test_streaming_crawl_invalid_json_shape_still_raises_request_error(
    streaming_client, body
):
    _install_asgi(
        streaming_client,
        _asgi_non_json_error(502, body, content_type="application/json"),
    )
    cfg = CrawlerRunConfig(stream=True)

    with pytest.raises(RequestError, match=r"Server error 502: "):
        gen = await streaming_client.crawl(
            ["https://example.com"], crawler_config=cfg
        )
        async for _ in gen:
            pass


@pytest.mark.parametrize("body", [b"[]", b"{"])
@pytest.mark.asyncio
async def test_non_streaming_request_invalid_json_shape_raises_request_error(
    streaming_client, body
):
    _install_asgi(
        streaming_client,
        _asgi_non_json_error(502, body, content_type="application/json"),
    )

    with pytest.raises(RequestError, match=r"Server error 502: "):
        await streaming_client._request("POST", "/crawl", json={})


@pytest.mark.asyncio
async def test_streaming_crawl_2xx_still_streams_results(streaming_client):
    """Happy path regression: a 200 NDJSON stream still yields CrawlResult objects.

    The error-handling fix must not change the 2xx behaviour. Also exercises the
    per-URL ``error`` line (logged + skipped) and the ``status == "completed"``
    line (skipped) so those branches are not regressed by the new try/except.
    """
    lines = [
        json.dumps({"url": "https://done.example.com", "html": "", "success": True,
                    "status": "completed"}),
        json.dumps({"error": "boom", "url": "https://failed.example.com"}),
        json.dumps({"url": "https://ok.example.org", "html": "<p>hi</p>", "success": True}),
    ]
    _install_asgi(streaming_client, _asgi_ndjson(lines))
    cfg = CrawlerRunConfig(stream=True)

    gen = await streaming_client.crawl(["https://example.com"], crawler_config=cfg)
    yielded = [r.url async for r in gen]

    assert yielded == ["https://ok.example.org"]


@pytest.mark.asyncio
async def test_streaming_crawl_error_does_not_leave_body_unread(streaming_client):
    """The response body is fully read before the stream context closes.

    Guards against the naive refactor the bug report warned about: catching
    HTTPStatusError *outside* the ``async with`` would read the body after
    stream close. Here we assert the consumed body is reflected in the error
    message, which is only possible if ``aread()`` ran inside the context.
    """
    long_detail = "x" * 500
    _install_asgi(streaming_client, _asgi_json_error(403, long_detail))
    cfg = CrawlerRunConfig(stream=True)

    with pytest.raises(RequestError) as exc_info:
        gen = await streaming_client.crawl(["https://example.com"], crawler_config=cfg)
        async for _ in gen:
            pass

    assert long_detail in str(exc_info.value)


# --- streaming crawl() transport-error contract ---
# crawl() exposes two response modes. The non-streaming path's _request()
# normalizes every httpx failure family: TimeoutException/RequestError ->
# ConnectionError, HTTPStatusError -> RequestError. The streaming branch
# previously only caught HTTPStatusError, so transport failures such as
# httpx.ReadTimeout (mid-stream) and httpx.ConnectError (on entry) leaked to
# the caller as raw httpx exceptions outside the client's own hierarchy.
# These tests pin the streaming path to the same transport-error contract as
# _request() (regression coverage for the leak introduced when commit
# baee824 added the HTTPStatusError handler without its transport
# counterparts). httpx.ASGITransport cannot synthesise transport-level
# exceptions (it serves in-process responses), so these cases use
# httpx.MockTransport, the natural way to raise ReadTimeout/ConnectError.
# No docker server, browser, or Redis required.

@pytest_asyncio.fixture
async def mock_client():
    """A client whose transport is swapped per-test with ``httpx.MockTransport``.

    Unlike ``streaming_client``, ``_check_server`` is NOT bypassed: each
    test's handler serves ``/health`` so the real prelude runs (mirroring
    production, where the server is reachable when the stream opens and the
    transport failure happens during/after stream entry). The default
    ``_http_client`` from ``__init__`` is closed up front (it is never used)
    and whatever client is installed at teardown is closed in the same event
    loop, so no ``httpx.AsyncClient`` leaks regardless of which path the
    test took.
    """
    c = Crawl4aiDockerClient(base_url="http://test", verbose=False)
    await c._http_client.aclose()
    try:
        yield c
    finally:
        if c._http_client is not None and not c._http_client.is_closed:
            await c._http_client.aclose()


def _install_mock(client, handler):
    client._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Content-Type": "application/json"},
    )


def _health_ok(req):
    return httpx.Response(200, content=b'{"status":"ok"}')


def _ndjson_first_line():
    return json.dumps(
        {"url": "https://example.com", "html": "", "success": True}
    ) + "\n"


def test_httpx_exception_hierarchy_supports_fix_ordering():
    """Pin the class-hierarchy facts the fix relies on.

    The fix must catch ``httpx.TimeoutException`` before ``httpx.RequestError``
    (TimeoutException is a subclass of RequestError, so the broader clause
    would otherwise shadow it and lose the "timed out" message), and the
    client's own ``RequestError``/``ConnectionError`` must remain outside
    httpx's hierarchy so the new outer transport handlers do not re-catch the
    ``RequestError`` re-raised by the inner ``HTTPStatusError`` handler. A
    future httpx upgrade or refactor that violates these assumptions would
    re-open the leak; this test makes that visible here.
    """
    assert issubclass(httpx.TimeoutException, httpx.RequestError)
    assert issubclass(httpx.ReadTimeout, httpx.TimeoutException)
    assert issubclass(httpx.ConnectTimeout, httpx.TimeoutException)
    assert issubclass(httpx.ConnectError, httpx.RequestError)
    assert not issubclass(RequestError, httpx.RequestError)
    assert not issubclass(RequestError, httpx.HTTPStatusError)
    assert not issubclass(ConnectionError, httpx.RequestError)
    assert issubclass(ConnectionError, Crawl4aiClientError)


@pytest.mark.asyncio
async def test_streaming_crawl_raises_connection_error_on_read_timeout(mock_client):
    """A mid-stream read timeout is normalized to ``ConnectionError``.

    This is the production-reachable case from the bug report: the stream
    opens, one CrawlResult is delivered, then a read timeout while waiting for
    the next line must surface as ``ConnectionError`` (as the non-streaming
    ``_request()`` would), not leak as a raw ``httpx.ReadTimeout``. Asserting
    that exactly one result was delivered proves the failure is genuinely
    mid-stream, not on entry.
    """
    first_line = _ndjson_first_line()

    async def body():
        yield first_line.encode()
        raise httpx.ReadTimeout("read timeout after first line")

    def handler(req):
        if "/health" in str(req.url):
            return _health_ok(req)
        return httpx.Response(
            200, content=body(), headers={"content-type": "application/x-ndjson"}
        )

    _install_mock(mock_client, handler)

    delivered = []
    with pytest.raises(ConnectionError) as exc_info:
        gen = await mock_client.crawl(
            ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
        )
        async for result in gen:
            delivered.append(result)

    assert len(delivered) == 1
    assert delivered[0].url == "https://example.com"
    assert isinstance(exc_info.value, Crawl4aiClientError)
    assert not isinstance(exc_info.value, httpx.HTTPStatusError)
    assert "Request timed out" in str(exc_info.value)
    assert "read timeout after first line" in str(exc_info.value)


@pytest.mark.asyncio
async def test_streaming_crawl_raises_connection_error_on_connect_error(mock_client):
    """A connection failure on stream entry is normalized to ``ConnectionError``.

    The handler serves ``/health`` (prelude passes) and raises
    ``httpx.ConnectError`` for ``/crawl/stream``. The streaming branch must
    surface this as ``ConnectionError`` with the "Failed to connect" message,
    matching ``_request()``'s ``RequestError -> ConnectionError`` mapping.
    """
    def handler(req):
        if "/health" in str(req.url):
            return _health_ok(req)
        raise httpx.ConnectError("connection refused")

    _install_mock(mock_client, handler)

    with pytest.raises(ConnectionError) as exc_info:
        gen = await mock_client.crawl(
            ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
        )
        async for _ in gen:
            pass

    assert isinstance(exc_info.value, Crawl4aiClientError)
    assert not isinstance(exc_info.value, httpx.HTTPStatusError)
    assert "Failed to connect" in str(exc_info.value)
    assert "connection refused" in str(exc_info.value)


@pytest.mark.asyncio
async def test_streaming_crawl_raises_connection_error_on_connect_timeout(mock_client):
    """A connect timeout surfaces as ``ConnectionError`` with the timeout message.

    ``httpx.ConnectTimeout`` is a ``TimeoutException`` (itself a subclass of
    ``RequestError``), so the fix must catch the ``TimeoutException`` clause
    first and produce the "Request timed out" message rather than the broader
    "Failed to connect" message. This guards against reordering the two
    ``except`` clauses.
    """
    def handler(req):
        if "/health" in str(req.url):
            return _health_ok(req)
        raise httpx.ConnectTimeout("connect timeout")

    _install_mock(mock_client, handler)

    with pytest.raises(ConnectionError) as exc_info:
        gen = await mock_client.crawl(
            ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
        )
        async for _ in gen:
            pass

    assert "Request timed out" in str(exc_info.value)
    assert "connect timeout" in str(exc_info.value)
    assert "Failed to connect" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_streaming_and_non_streaming_read_timeout_raise_same_exception_type(
    mock_client,
):
    """Same transport failure, same exception type regardless of ``stream`` flag.

    Pins the "no exception type drift" guarantee from the bug report: a
    ``httpx.ReadTimeout`` that the non-streaming ``_request()`` surfaces as
    ``ConnectionError("Request timed out: ...")`` must surface identically on
    the streaming path, so the ``stream`` flag never changes the exception
    family a caller has to handle.
    """
    def handler(req):
        if "/health" in str(req.url):
            return _health_ok(req)
        raise httpx.ReadTimeout("simulated read timeout")

    _install_mock(mock_client, handler)

    streaming_exc: Optional[ConnectionError] = None
    try:
        gen = await mock_client.crawl(
            ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
        )
        async for _ in gen:
            pass
    except ConnectionError as e:
        streaming_exc = e

    request_exc: Optional[ConnectionError] = None
    try:
        await mock_client._request("POST", "/crawl", json={})
    except ConnectionError as e:
        request_exc = e

    assert isinstance(streaming_exc, ConnectionError)
    assert isinstance(request_exc, ConnectionError)
    assert type(streaming_exc) is type(request_exc)
    assert str(streaming_exc) == str(request_exc)
    assert str(streaming_exc) == "Request timed out: simulated read timeout"
