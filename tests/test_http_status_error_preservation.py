"""Regression coverage for HTTPStatusError preservation in
AsyncHTTPCrawlerStrategy.

The HTTP crawler strategy defines a typed exception family
(`HTTPCrawlerError`, `ConnectionTimeoutError`, `HTTPStatusError`) so callers
can distinguish failure modes. A non-2xx response raises `HTTPStatusError`
(carrying `.status_code`) inside `_handle_http`'s try block. Before the fix,
the except chain had no clause for `HTTPStatusError`, so it fell into the
catch-all `except Exception` and was re-wrapped as a generic
`HTTPCrawlerError`, destroying both the exception type and the structured
`.status_code` field. These tests pin the documented contract: non-2xx
responses surface as `HTTPStatusError` with a readable `.status_code`, the
`on_error` hook still fires for that path, and the adjacent timeout branch
is undisturbed.
"""

import asyncio
import socket

import pytest
import pytest_asyncio
from aiohttp import web

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_crawler_strategy import (
    AsyncHTTPCrawlerStrategy,
    ConnectionTimeoutError,
    HTTPCrawlerError,
    HTTPStatusError,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_server() -> tuple[web.AppRunner, int]:
    """Start a local aiohttp server returning various status codes."""
    app = web.Application()

    async def status_handler(request):
        code = int(request.match_info["code"])
        return web.Response(text=f"status {code}", status=code)

    async def slow_handler(request):
        await asyncio.sleep(30)
        return web.Response(text="slow", status=200)

    app.router.add_get("/status/{code}", status_handler)
    app.router.add_get("/slow", slow_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    port = _find_free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner, port


@pytest_asyncio.fixture
async def http_server():
    """Yield a base URL for a local aiohttp test server returning status codes."""
    runner, port = await _start_server()
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        await runner.cleanup()


@pytest_asyncio.fixture
async def strategy():
    """Yield a fresh AsyncHTTPCrawlerStrategy and ensure its session is closed."""
    s = AsyncHTTPCrawlerStrategy()
    try:
        yield s
    finally:
        await s.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [400, 401, 403, 404, 410, 418, 500, 502, 503])
async def test_non_2xx_raises_http_status_error_with_status_code(
    http_server, strategy, code
):
    """A non-2xx response must surface as exactly `HTTPStatusError` (not a
    generic `HTTPCrawlerError` re-wrap) so the documented
    `except HTTPStatusError` branch fires, and the wire status code must be
    readable on `.status_code` across the 4xx and 5xx bands."""
    with pytest.raises(HTTPStatusError) as exc_info:
        await strategy.crawl(f"{http_server}/status/{code}")

    assert type(exc_info.value) is HTTPStatusError
    assert exc_info.value.status_code == code


@pytest.mark.asyncio
async def test_http_status_error_still_subclass_of_crawler_error(
    http_server, strategy
):
    """HTTPStatusError must remain a subclass of HTTPCrawlerError so callers
    catching the base exception (the documented fallback branch) still match."""
    with pytest.raises(HTTPCrawlerError) as exc_info:
        await strategy.crawl(f"{http_server}/status/403")

    assert isinstance(exc_info.value, HTTPStatusError)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_on_error_hook_fires_with_http_status_error(http_server, strategy):
    """The on_error hook must be invoked with the original HTTPStatusError
    instance for the status-error path, preserving the side-effect the
    catch-all branch used to provide (now without the type-destroying re-wrap)."""
    seen = []

    async def on_error(err):
        seen.append(err)

    strategy.set_hook("on_error", on_error)

    with pytest.raises(HTTPStatusError):
        await strategy.crawl(f"{http_server}/status/503")

    assert len(seen) == 1
    assert isinstance(seen[0], HTTPStatusError)
    assert seen[0].status_code == 503


@pytest.mark.asyncio
async def test_timeout_still_raises_connection_timeout_error(http_server, strategy):
    """A network timeout must still surface as ConnectionTimeoutError. The
    `except HTTPStatusError` clause sits after the timeout clauses and must
    not disturb the existing timeout handling."""
    with pytest.raises(ConnectionTimeoutError):
        await strategy.crawl(
            f"{http_server}/slow",
            config=CrawlerRunConfig(page_timeout=500),
        )
