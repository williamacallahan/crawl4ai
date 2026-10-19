"""Regression coverage for the Docker client ``base_url`` path-prefix contract.

``Crawl4aiDockerClient`` previously built non-streaming request URLs with
``urllib.parse.urljoin(self.base_url, endpoint)`` while the streaming
endpoint was built with ``f"{self.base_url}/crawl/stream"``. Because every
non-streaming endpoint passed to ``urljoin`` begins with ``/``, the helper
resolves it against the host root and discards any path prefix in
``base_url``: ``urljoin("http://host/c4ai", "/health")`` returns
``http://host/health``. A path-prefixed ``base_url`` (the form a reverse-
proxy subpath deployment requires) thus silently routed ``/health``,
``/token``, ``/crawl`` and ``/schema`` to the wrong service while the
streaming path kept working. The 0.9.0 ``_check_server()`` further masked
the wrong-URL 404 by never calling ``raise_for_status()``.

These tests pin the prefix-preserving behavior at every call site the bug
touched (``authenticate``, ``_check_server``, ``_request`` -> non-streaming
``crawl`` and ``get_schema``) and the fail-fast health check on a 404. No
browser, network, or Redis required — ``httpx.MockTransport`` records the
request path each handler receives, mirroring
``tests/regression/test_docker_client_hooks_unit.py`` and
``tests/regression/test_docker_client_authenticate_unit.py``.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

# Keep deploy/docker on sys.path so a future cross-contract test can
# `import server` without starting the server (mirrors the other
# docker_client regression test files).
_DEPLOY_DOCKER = Path(__file__).resolve().parents[2] / "deploy" / "docker"
if str(_DEPLOY_DOCKER) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DOCKER))

from crawl4ai.async_configs import CrawlerRunConfig  # noqa: E402
from crawl4ai.docker_client import (  # noqa: E402
    Crawl4aiClientError,
    Crawl4aiDockerClient,
    ConnectionError,
    RequestError,
)
from crawl4ai.models import CrawlResult  # noqa: E402


# Representative subpath reverse-proxy prefix. The exact value doesn't
# matter — any non-empty path component would be dropped by the pre-fix
# ``urljoin``; ``/c4ai`` matches the bug report's example.
PREFIX = "/c4ai"
ROOT_BASE_URL = "http://testserver"
PREFIXED_BASE_URL = f"{ROOT_BASE_URL}{PREFIX}"
OPERATOR_TOKEN = "operator-secret-0123456789"


# --- shared helpers --------------------------------------------------------
def _install_mock(client, handler):
    """Swap a client's transport for ``httpx.MockTransport(handler)``.

    The handler receives ``request.url`` as the absolute URL the client
    built, so ``request.url.path`` is exactly the path dispatched on the
    wire — the value these tests assert on. No ``base_url`` is set on the
    ``AsyncClient`` because the client always passes absolute URLs built
    from ``self.base_url``.
    """
    client._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Content-Type": "application/json"},
    )


def _health_ok():
    return httpx.Response(200, content=b'{"status":"ok"}')


def _token_ok():
    return httpx.Response(
        200, json={"email": "u@x.com", "access_token": "jwt", "token_type": "bearer"}
    )


def _ndjson_body(line_dicts):
    return "".join(json.dumps(d) + "\n" for d in line_dicts).encode()


@pytest_asyncio.fixture
async def prefixed_client():
    """Client with a path-prefixed ``base_url`` (the bug's trigger)."""
    c = Crawl4aiDockerClient(base_url=PREFIXED_BASE_URL, verbose=False)
    await c._http_client.aclose()
    try:
        yield c
    finally:
        if c._http_client is not None and not c._http_client.is_closed:
            await c._http_client.aclose()


@pytest_asyncio.fixture
async def root_client():
    """Client with a bare host ``base_url`` (the existing-test posture).

    Every existing test and documentation snippet uses a root ``base_url``
    (``http://localhost:8000`` etc.) for which ``urljoin(base, "/endpoint")``
    happened to produce the right URL. A fix that's correct for a prefixed
    ``base_url`` must not break this no-prefix posture (e.g. by introducing
    a leading-slash duplication); this fixture guards against an over-broad
    fix.
    """
    c = Crawl4aiDockerClient(base_url=ROOT_BASE_URL, verbose=False)
    await c._http_client.aclose()
    try:
        yield c
    finally:
        if c._http_client is not None and not c._http_client.is_closed:
            await c._http_client.aclose()


# --- _check_server --------------------------------------------------------
@pytest.mark.asyncio
async def test_check_server_preserves_base_url_path_prefix(prefixed_client):
    """``_check_server()`` GETs ``/c4ai/health`` for a prefixed ``base_url``.

    Pre-fix ``urljoin("http://testserver/c4ai", "/health")`` returned
    ``http://testserver/health`` (prefix dropped), so a subpath
    deployment's prelude hit the wrong service.
    """
    sent = []

    def handler(request):
        sent.append(request.url.path)
        if request.url.path == f"{PREFIX}/health":
            return _health_ok()
        return httpx.Response(404)

    _install_mock(prefixed_client, handler)
    await prefixed_client._check_server()

    assert sent == [f"{PREFIX}/health"]


@pytest.mark.asyncio
async def test_check_server_raises_connection_error_on_404_with_prefix(prefixed_client):
    """A 404 from ``/c4ai/health`` raises ``ConnectionError`` (was masked).

    Pre-fix ``_check_server()`` never called ``raise_for_status()``, so a
    wrong-URL 404 (the silently-dropped-prefix case against a real proxy)
    was logged as a successful connection; the user only saw a later
    ``RequestError: Server error 404`` from ``/crawl``. The fix surfaces
    the 404 as ``ConnectionError`` so a subpath misconfiguration fails fast
    at the prelude.
    """
    def handler(request):
        return httpx.Response(404, json={"detail": "not found"})

    _install_mock(prefixed_client, handler)
    with pytest.raises(ConnectionError) as exc_info:
        await prefixed_client._check_server()
    assert isinstance(exc_info.value, Crawl4aiClientError)
    assert "404" in str(exc_info.value)
    assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_server_propagates_request_error_as_connection_error(prefixed_client):
    """A transport failure still surfaces as ``ConnectionError`` (no regression).

    The fix adds an ``except httpx.HTTPStatusError`` clause to
    ``_check_server()``; the existing ``except httpx.RequestError`` clause
    must remain so transport failures still normalize to ``ConnectionError``
    with the "Cannot connect" message, not leak as raw httpx exceptions.
    """
    def handler(request):
        raise httpx.ConnectError("connection refused")

    _install_mock(prefixed_client, handler)
    with pytest.raises(ConnectionError) as exc_info:
        await prefixed_client._check_server()
    assert "Cannot connect to server" in str(exc_info.value)
    assert "connection refused" in str(exc_info.value)


# --- authenticate ---------------------------------------------------------
@pytest.mark.asyncio
async def test_authenticate_preserves_base_url_path_prefix(prefixed_client):
    """``authenticate()`` POSTs ``/c4ai/token`` for a prefixed ``base_url``."""
    sent = []

    def handler(request):
        sent.append((request.method, request.url.path))
        if request.url.path == f"{PREFIX}/token":
            return _token_ok()
        return httpx.Response(404)

    _install_mock(prefixed_client, handler)
    await prefixed_client.authenticate("u@x.com", api_token=OPERATOR_TOKEN)

    assert sent == [("POST", f"{PREFIX}/token")]
    assert prefixed_client._token == "jwt"
    assert prefixed_client._http_client.headers["Authorization"] == "Bearer jwt"


# --- get_schema -----------------------------------------------------------
@pytest.mark.asyncio
async def test_get_schema_preserves_base_url_path_prefix(prefixed_client):
    """``get_schema()`` GETs ``/c4ai/schema`` for a prefixed ``base_url``.

    ``get_schema()`` routes through ``_request("GET", "/schema")``; the fix
    replaces ``urljoin(self.base_url, endpoint)`` there with
    ``f"{self.base_url}{endpoint}"`` so the prefix is preserved.
    """
    sent = []
    payload = {"browser": {}, "crawler": {}}

    def handler(request):
        sent.append((request.method, request.url.path))
        if request.url.path == f"{PREFIX}/schema":
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    _install_mock(prefixed_client, handler)
    schema = await prefixed_client.get_schema()

    assert sent == [("GET", f"{PREFIX}/schema")]
    assert schema == payload


# --- _request (dynamic endpoint) -----------------------------------------
@pytest.mark.asyncio
async def test_request_preserves_base_url_path_prefix_for_dynamic_endpoint(
    prefixed_client,
):
    """``_request()`` preserves the prefix for any endpoint it's called with.

    Drives ``_request`` directly rather than via ``crawl``/``get_schema`` so
    the prefix-preservation pin covers any future caller too, not just the
    two endpoints currently routed through it.
    """
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    _install_mock(prefixed_client, handler)
    await prefixed_client._request("GET", "/schema")
    await prefixed_client._request("POST", "/crawl", json={})

    assert seen == [f"{PREFIX}/schema", f"{PREFIX}/crawl"]


# --- non-streaming crawl() ------------------------------------------------
@pytest.mark.asyncio
async def test_non_streaming_crawl_preserves_base_url_path_prefix(prefixed_client):
    """Non-streaming ``crawl()`` hits ``/c4ai/health`` then ``/c4ai/crawl``."""
    sent = []
    result_dict = {"url": "https://example.com", "html": "", "success": True}

    def handler(request):
        sent.append((request.method, request.url.path))
        if request.url.path == f"{PREFIX}/health":
            return _health_ok()
        if request.url.path == f"{PREFIX}/crawl":
            return httpx.Response(200, json={"success": True, "results": [result_dict]})
        return httpx.Response(404)

    _install_mock(prefixed_client, handler)
    result = await prefixed_client.crawl(
        ["https://example.com"], crawler_config=CrawlerRunConfig()
    )

    assert sent == [("GET", f"{PREFIX}/health"), ("POST", f"{PREFIX}/crawl")]
    # Single-URL crawl returns a bare CrawlResult, not a list.
    assert isinstance(result, CrawlResult)
    assert result.url == "https://example.com"


# --- streaming crawl() ----------------------------------------------------
@pytest.mark.asyncio
async def test_streaming_crawl_preserves_base_url_path_prefix(prefixed_client):
    """Streaming ``crawl()`` hits ``/c4ai/health`` then ``/c4ai/crawl/stream``.

    Streaming has always preserved the prefix (it builds the URL as
    ``f"{self.base_url}/crawl/stream"``); this guards against a refactor
    that "harmonizes" streaming onto ``urljoin`` or another prefix-dropping
    helper, and confirms the prelude (which goes through ``_check_server``)
    now also preserves the prefix.
    """
    sent = []
    result_dict = {"url": "https://example.com", "html": "", "success": True}
    body = _ndjson_body([result_dict])

    def handler(request):
        sent.append((request.method, request.url.path))
        if request.url.path == f"{PREFIX}/health":
            return _health_ok()
        if request.url.path == f"{PREFIX}/crawl/stream":
            return httpx.Response(
                200, content=body, headers={"content-type": "application/x-ndjson"}
            )
        return httpx.Response(404)

    _install_mock(prefixed_client, handler)
    gen = await prefixed_client.crawl(
        ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
    )
    yielded = [r async for r in gen]

    assert sent == [("GET", f"{PREFIX}/health"), ("POST", f"{PREFIX}/crawl/stream")]
    assert len(yielded) == 1
    assert isinstance(yielded[0], CrawlResult)
    assert yielded[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_streaming_crawl_404_with_prefix_raises_request_error(prefixed_client):
    """A 404 on the prefixed streaming path still raises ``RequestError``.

    Pins that the streaming branch's existing ``RequestError("Server error
    404: ...")`` classification (commit baee824) is unchanged by the
    URL-building fix on the non-streaming side.
    """
    def handler(request):
        if request.url.path == f"{PREFIX}/health":
            return _health_ok()
        return httpx.Response(404, json={"detail": "not found"})

    _install_mock(prefixed_client, handler)
    with pytest.raises(RequestError) as exc_info:
        gen = await prefixed_client.crawl(
            ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
        )
        async for _ in gen:
            pass
    assert "404" in str(exc_info.value)


# --- root base_url: no regression for the happy path ----------------------
@pytest.mark.asyncio
async def test_root_base_url_paths_unchanged_for_every_call_site(root_client):
    """Every call site still routes to the bare path with a root ``base_url``.

    Consolidates the no-regression posture for the documented
    host:port-only ``base_url`` (every existing test and example uses
    one). Drives ``authenticate``, a non-streaming ``crawl``, a streaming
    ``crawl`` and ``get_schema`` against a strict handler that 404s any
    prefixed path, so an over-broad fix that prepends a stray ``/`` or
    munges the path visibly fails here.
    """
    sent = []

    def handler(request):
        sent.append((request.method, request.url.path))
        path = request.url.path
        if path == "/health":
            return _health_ok()
        if path == "/token":
            return _token_ok()
        if path == "/schema":
            return httpx.Response(200, json={"browser": {}, "crawler": {}})
        if path == "/crawl":
            return httpx.Response(200, json={"success": True, "results": []})
        if path == "/crawl/stream":
            return httpx.Response(
                200,
                content=_ndjson_body([{"url": "https://example.com", "html": "", "success": True}]),
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(404, json={"detail": f"unexpected {path}"})

    _install_mock(root_client, handler)
    # authenticate sets Authorization from the response; starting from a
    # clean header proves authenticate populates it rather than relying on
    # the constructor's api_token.
    root_client._http_client.headers.pop("Authorization", None)

    await root_client.authenticate("u@x.com", api_token=OPERATOR_TOKEN)
    await root_client.crawl(["https://example.com"], crawler_config=CrawlerRunConfig())
    gen = await root_client.crawl(
        ["https://example.com"], crawler_config=CrawlerRunConfig(stream=True)
    )
    [r async for r in gen]
    await root_client.get_schema()

    assert sent == [
        ("POST", "/token"),
        ("GET", "/health"),        # prelude of the non-streaming crawl
        ("POST", "/crawl"),
        ("GET", "/health"),        # prelude of the streaming crawl
        ("POST", "/crawl/stream"),
        ("GET", "/schema"),
    ]


# --- structural pin: urljoin no longer drives URL construction -------------
def test_docker_client_module_does_not_import_urljoin():
    """``urllib.parse.urljoin`` is no longer imported by the client module.

    A future commit that re-introduces ``urljoin`` for URL construction
    would re-open the bug; this pin makes the re-introduction visible at
    review time and fails the suite if ``urljoin`` returns through any
    import path (including an ``import x as y`` alias or a fully-qualified
    ``urllib.parse.urljoin`` call).
    """
    import crawl4ai.docker_client as mod

    assert not hasattr(mod, "urljoin")
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "urljoin" not in source
