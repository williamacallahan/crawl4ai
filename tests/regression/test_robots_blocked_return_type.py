"""
Regression guard for the contract that AsyncWebCrawler.arun()'s robots.txt
blocked branch returns a CrawlResultContainer (like every other return path),
not a bare CrawlResult.

The bug history: the blocked branch (added before the CrawlResultContainer
refactor) was overlooked when every other return path in arun was wrapped.
Attribute access (result.success) coincidentally worked on both types, so
the divergence only manifested for callers that treat the return value as a
sequence (results[0], len(results), for r in results, async for r in results)
— patterns used by the bundled server (results[0].success) and by the
tutorial example (async for). These tests pin the contract so a future
refactor cannot silently reintroduce the bare-CrawlResult branch.

Each test drives arun() end-to-end against a real local aiohttp server that
serves a blocking /robots.txt (Disallow: /). No mocking.
"""

import socket

import pytest
from aiohttp import web

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import CrawlResult, CrawlResultContainer


# Running as root in CI requires --no-sandbox; harmless elsewhere.
_BROWSER_CONFIG = BrowserConfig(
    headless=True, verbose=False, extra_args=["--no-sandbox"]
)

_BLOCKING_ROBOTS = "User-agent: *\nDisallow: /\n"
_ALLOWING_ROBOTS = "User-agent: *\nAllow: /\n"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def _start_server(port: int, robots_body: str):
    """Start a server with /robots.txt and a substantial HTML index page."""
    app = web.Application()

    async def robots(_req):
        return web.Response(text=robots_body, content_type="text/plain")

    async def index(_req):
        paras = "".join(
            f"<p>Paragraph number {i} with some real sentences.</p>"
            for i in range(20)
        )
        return web.Response(
            text=(
                "<html><head><title>Test Page</title></head><body>"
                f"<h1>Hello World</h1>{paras}</body></html>"
            ),
            content_type="text/html",
        )

    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", index)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


def _blocked_config():
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, check_robots_txt=True, verbose=False
    )


# ---------------------------------------------------------------------------
# Contract: arun() returns CrawlResultContainer on the robots-blocked branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_robots_blocked_returns_crawl_result_container():
    """The blocked branch must wrap its CrawlResult in a CrawlResultContainer,
    matching the fresh-fetch, cache-hit, and exception return paths in arun.
    Before the fix this branch returned a bare CrawlResult, breaking the
    declared -> CrawlResultContainer contract."""
    port = _free_port()
    runner = await _start_server(port, _BLOCKING_ROBOTS)
    try:
        url = f"http://127.0.0.1:{port}/"
        async with AsyncWebCrawler(config=_BROWSER_CONFIG) as crawler:
            crawler.robots_parser.clear_cache()
            result = await crawler.arun(url=url, config=_blocked_config())
            assert isinstance(result, CrawlResultContainer), (
                f"Expected CrawlResultContainer, got {type(result).__name__}"
            )
            assert not isinstance(result, CrawlResult), (
                "robots-blocked branch returned bare CrawlResult (regression)"
            )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_robots_blocked_supports_sequence_protocol():
    """The blocked-branch container must support indexing, len, sync iteration,
    and async iteration. These are the patterns the bundled server (results[0])
    and the tutorial example (async for) rely on; on a bare CrawlResult each
    raises TypeError (or, for `for r in result`, silently yields Pydantic
    field tuples instead of CrawlResult objects)."""
    port = _free_port()
    runner = await _start_server(port, _BLOCKING_ROBOTS)
    try:
        url = f"http://127.0.0.1:{port}/"
        async with AsyncWebCrawler(config=_BROWSER_CONFIG) as crawler:
            crawler.robots_parser.clear_cache()
            results = await crawler.arun(url=url, config=_blocked_config())
            assert len(results) == 1
            assert isinstance(results[0], CrawlResult)
            assert isinstance(results[-1], CrawlResult)
            iter_items = list(results)
            assert len(iter_items) == 1 and isinstance(iter_items[0], CrawlResult)
            async_iter_items = [r async for r in results]
            assert len(async_iter_items) == 1
            assert isinstance(async_iter_items[0], CrawlResult)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_robots_blocked_result_payload():
    """The wrapped CrawlResult must carry the documented blocked payload:
    success=False, status_code=403, the robots error message, the
    X-Robots-Status response header, empty html, and the requested url."""
    port = _free_port()
    runner = await _start_server(port, _BLOCKING_ROBOTS)
    try:
        url = f"http://127.0.0.1:{port}/"
        async with AsyncWebCrawler(config=_BROWSER_CONFIG) as crawler:
            crawler.robots_parser.clear_cache()
            r = await crawler.arun(url=url, config=_blocked_config())
            # Attribute access works on both container (proxied) and bare model,
            # but pin it here so a future change to the proxy can't slip through.
            assert r.success is False
            assert r.status_code == 403
            assert r.error_message == "Access denied by robots.txt"
            assert r.response_headers["X-Robots-Status"] == "Blocked by robots.txt"
            assert r.html == ""
            assert r.url == url
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_robots_allowed_path_still_returns_crawl_result_container():
    """Regression: when robots.txt allows the crawl, arun() must take the
    fresh-fetch path and still return a CrawlResultContainer."""
    port = _free_port()
    runner = await _start_server(port, _ALLOWING_ROBOTS)
    try:
        url = f"http://127.0.0.1:{port}/"
        async with AsyncWebCrawler(config=_BROWSER_CONFIG) as crawler:
            crawler.robots_parser.clear_cache()
            result = await crawler.arun(url=url, config=_blocked_config())
            assert isinstance(result, CrawlResultContainer)
            assert result.success is True
    finally:
        await runner.cleanup()
