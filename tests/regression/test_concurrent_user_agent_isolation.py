"""Regression tests for concurrent per-crawl User-Agent / sec-ch-ua isolation.

Bug (introduced in commit fdd989785): ``AsyncPlaywrightCrawlerStrategy._crawl_web``
wrote the per-crawl UA / sec-ch-ua onto the shared mutable ``self.browser_config``
object, then - after an ``await self.browser_manager.get_page(...)`` that yields
to the event loop - read those same shared fields back to set the page-level
HTTP headers via ``page.set_extra_http_headers(...)``. A single strategy
instance and a single ``BrowserConfig`` are shared across every concurrent
crawl issued by ``AsyncWebCrawler.arun`` / ``arun_many``; nothing serializes
``_crawl_web``. Two concurrent crawls that each request a distinct UA interleave
at the ``await``, so the first crawl's post-``await`` ``set_extra_http_headers``
ships the second crawl's UA on the wire (TOCTOU on the page-level HTTP
User-Agent / sec-ch-ua headers).

The fix has two coordinated layers. (1) The per-crawl UA and client hint are
captured into local variables BEFORE the await and used for the post-await
``set_extra_http_headers`` (the shared ``browser_config`` is still mutated
because it is load-bearing for ``navigator.userAgent`` / context-level
headers at context creation, but that mutation no longer feeds the racy
page-level read). (2) The per-crawl UA is threaded through
``BrowserManager.get_page`` -> ``_acquire_context`` ->
``_make_config_signature`` (which now includes the UA, so distinct UAs get
distinct cached contexts) + ``_new_context`` / ``create_browser_context``
(which bakes the crawl's own UA via ``new_context(user_agent=...)`` - the
value that actually goes on the wire, since Playwright's context-level
``user_agent`` overrides ``page.set_extra_http_headers`` for the on-wire
``User-Agent``) + ``setup_context`` (context-level ``sec-ch-ua`` header).

Most tests here are offline (browser-free) using a mocked browser surface
modeled on ``tests/regression/test_ssl_certificate_http_scheme_guard.py`` and
``tests/regression/test_browser_lifecycle_residuals.py``. The end-to-end test at
the bottom is ``@pytest.mark.browser`` and uses a real Chromium against a
small UA-echoing local server.
"""

import asyncio
import socket
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from crawl4ai.async_configs import BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.async_logger import AsyncLogger
from crawl4ai.user_agent_generator import UAGen

UA_CHROME_121 = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
UA_CHROME_122 = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

HINT_121 = UAGen.generate_client_hints(UA_CHROME_121)
HINT_122 = UAGen.generate_client_hints(UA_CHROME_122)


# ---------------------------------------------------------------------------
# Offline helpers: a mocked browser surface that captures set_extra_http_headers
# ---------------------------------------------------------------------------

def _build_strategy_with_capturing_pages():
    """Build a real ``AsyncPlaywrightCrawlerStrategy`` with a fully mocked
    browser surface. Each ``page.set_extra_http_headers`` call captures its
    argument so the on-the-wire UA / sec-ch-ua are observable per crawl.
    ``get_page`` yields (``await asyncio.sleep(0)``) so concurrent
    ``_crawl_web`` invocations interleave around it, exactly as they do in
    production against the shared ``BrowserConfig``.

    Construction is lightweight (``__init__`` only allocates a
    ``BrowserManager``; it does not start Chromium), so this stays offline.
    """
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True),
        logger=AsyncLogger(verbose=False),
    )
    captured = {}

    def make_page(tag):
        p = MagicMock()
        p.captured_headers = None
        p.close = AsyncMock()

        async def set_extra_http_headers(headers):
            p.captured_headers = dict(headers)

        p.set_extra_http_headers = set_extra_http_headers
        # page.goto is the first mandatory await AFTER set_extra_http_headers;
        # raising here isolates the UA-leak surface from real navigation while
        # guaranteeing set_extra_http_headers has already run.
        p.goto = AsyncMock(side_effect=RuntimeError(f"STOP_AFTER_UA_{tag}"))
        captured[tag] = p
        return p

    queue: asyncio.Queue = asyncio.Queue()

    async def get_page(crawlerRunConfig=None, user_agent=None, browser_hint=None, **_):
        # Two awaits model the real get_page yield to the event loop so
        # concurrent _crawl_web invocations interleave around this point.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        tag = await queue.get()
        return make_page(tag), MagicMock()

    strategy.browser_manager = MagicMock()
    strategy.browser_manager.get_page = get_page
    strategy.browser_manager.release_page_with_context = AsyncMock()
    return strategy, captured, queue


async def _run_crawl(strategy, queue, tag, url, **cfg_kwargs):
    await queue.put(tag)
    cfg = CrawlerRunConfig(**cfg_kwargs)
    with pytest.raises(RuntimeError, match=f"STOP_AFTER_UA_{tag}"):
        await strategy._crawl_web(url, cfg)


# ---------------------------------------------------------------------------
# Offline tests: the concurrent TOCTOU race on User-Agent / sec-ch-ua
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_distinct_user_agents_no_leak():
    """Core regression: two concurrent ``_crawl_web`` with distinct per-crawl
    ``user_agent`` values interleave at ``get_page``; each page's
    ``set_extra_http_headers`` must carry the crawl's OWN UA and sec-ch-ua,
    not the other crawl's. Under the bug, the first-resumed crawl ships the
    last writer's UA on the wire."""
    strategy, captured, queue = _build_strategy_with_capturing_pages()
    await asyncio.gather(
        _run_crawl(strategy, queue, "A", "https://example.test/a", user_agent=UA_CHROME_121),
        _run_crawl(strategy, queue, "B", "https://example.test/b", user_agent=UA_CHROME_122),
    )

    a = captured["A"].captured_headers
    b = captured["B"].captured_headers
    assert a is not None, "set_extra_http_headers was not called for crawl A"
    assert b is not None, "set_extra_http_headers was not called for crawl B"
    assert a["User-Agent"] == UA_CHROME_121, f"Page A leaked crawl B's UA: {a['User-Agent']}"
    assert b["User-Agent"] == UA_CHROME_122, f"Page B leaked crawl A's UA: {b['User-Agent']}"
    assert a["sec-ch-ua"] == HINT_121, f"Page A leaked crawl B's sec-ch-ua: {a['sec-ch-ua']}"
    assert b["sec-ch-ua"] == HINT_122, f"Page B leaked crawl A's sec-ch-ua: {b['sec-ch-ua']}"


@pytest.mark.asyncio
async def test_locals_used_not_post_await_shared_state():
    """Defense-in-depth: even if a concurrent crawl (or a hook) overwrites
    the shared ``browser_config`` fields while this crawl is suspended at
    ``get_page``, this crawl's page-level HTTP headers must come from the
    locals captured BEFORE the await, not from the post-await shared state.
    This test deterministically fails on the buggy code (which reads shared
    state post-await) and passes on the fix."""
    strategy, captured, queue = _build_strategy_with_capturing_pages()
    original_get_page = strategy.browser_manager.get_page

    async def sabotaging_get_page(crawlerRunConfig=None, user_agent=None, browser_hint=None, **_):
        result = await original_get_page(
            crawlerRunConfig=crawlerRunConfig,
            user_agent=user_agent,
            browser_hint=browser_hint,
        )
        # Stand-in for a concurrent crawl that mutates the shared config
        # during this crawl's await window.
        strategy.browser_config.user_agent = "SABOTAGED-UA"
        strategy.browser_config.browser_hint = "SABOTAGED-HINT"
        strategy.browser_config.headers["sec-ch-ua"] = "SABOTAGED-HINT"
        return result

    strategy.browser_manager.get_page = sabotaging_get_page
    await _run_crawl(strategy, queue, "A", "https://example.test/a", user_agent=UA_CHROME_121)
    h = captured["A"].captured_headers
    assert h["User-Agent"] == UA_CHROME_121, "must use local UA captured before await"
    assert h["sec-ch-ua"] == HINT_121, "must use local sec-ch-ua captured before await"


@pytest.mark.asyncio
async def test_shared_browser_config_still_mutated_for_context_creation():
    """The fix still mutates the shared ``browser_config`` (load-bearing for
    ``navigator.userAgent`` / context-level headers at context creation, see
    ``BrowserManager.create_browser_context`` / ``setup_context``). The
    leftover (last writer's UA) is the documented behavior; the page-level
    HTTP headers remain correct per-crawl regardless."""
    strategy, _, queue = _build_strategy_with_capturing_pages()
    await asyncio.gather(
        _run_crawl(strategy, queue, "A", "https://example.test/a", user_agent=UA_CHROME_121),
        _run_crawl(strategy, queue, "B", "https://example.test/b", user_agent=UA_CHROME_122),
    )
    # Shared config holds the last writer's UA - documented leftover; this
    # is NOT the page-level HTTP race fixed here, see the bug report's
    # "Scope: the leftover corruption" note.
    assert strategy.browser_config.user_agent in (UA_CHROME_121, UA_CHROME_122)
    last = strategy.browser_config.user_agent
    assert strategy.browser_config.browser_hint == UAGen.generate_client_hints(last)
    assert strategy.browser_config.headers["sec-ch-ua"] == UAGen.generate_client_hints(last)


# ---------------------------------------------------------------------------
# Offline tests: single-crawl happy paths (no regression on the non-racy path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_magic_mode_generates_and_applies_ua_to_page():
    """``config.magic=True`` generates a UA and applies it to the page. The
    sec-ch-ua is regenerated from the generated UA so they stay in sync
    (the original fdd989785 anti-bot fix's intent). Covers the shared
    ``elif config.magic or config.user_agent_mode == "random"`` generate
    branch."""
    strategy, captured, queue = _build_strategy_with_capturing_pages()
    await _run_crawl(strategy, queue, "A", "https://example.test/a", magic=True)
    h = captured["A"].captured_headers
    assert h is not None
    assert h.get("User-Agent"), "magic mode must generate a non-empty User-Agent"
    assert h["sec-ch-ua"] == UAGen.generate_client_hints(h["User-Agent"])
    assert strategy.browser_config.user_agent == h["User-Agent"]


@pytest.mark.asyncio
async def test_no_user_agent_change_skips_set_extra_http_headers():
    """When no per-crawl UA is requested (no ``user_agent``, no ``magic``,
    no ``random``), ``set_extra_http_headers`` must NOT be called and the
    shared ``browser_config`` UA fields must not be mutated by the UA path."""
    strategy, captured, queue = _build_strategy_with_capturing_pages()
    original_ua = strategy.browser_config.user_agent
    original_hint = strategy.browser_config.browser_hint
    original_hdr = strategy.browser_config.headers.get("sec-ch-ua")
    await _run_crawl(strategy, queue, "A", "https://example.test/a")
    assert captured["A"].captured_headers is None, (
        "set_extra_http_headers must not be called when UA is unchanged"
    )
    assert strategy.browser_config.user_agent == original_ua
    assert strategy.browser_config.browser_hint == original_hint
    assert strategy.browser_config.headers.get("sec-ch-ua") == original_hdr


@pytest.mark.asyncio
async def test_persistent_context_skips_per_crawl_ua_block():
    """For persistent contexts (``use_persistent_context=True``), the per-crawl
    UA block is skipped entirely: ``set_extra_http_headers`` is not called and
    the shared config is not mutated - the UA is locked at browser launch
    time (documented behavior preserved by the fix)."""
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(use_persistent_context=True, headless=True),
        logger=AsyncLogger(verbose=False),
    )
    page = MagicMock()
    page.captured_headers = None
    page.close = AsyncMock()

    async def set_extra_http_headers(headers):
        page.captured_headers = dict(headers)

    page.set_extra_http_headers = set_extra_http_headers
    page.goto = AsyncMock(side_effect=RuntimeError("STOP_AFTER_UA"))
    strategy.browser_manager = MagicMock()
    strategy.browser_manager.get_page = AsyncMock(return_value=(page, MagicMock()))
    strategy.browser_manager.release_page_with_context = AsyncMock()

    original_ua = strategy.browser_config.user_agent
    with pytest.raises(RuntimeError, match="STOP_AFTER_UA"):
        await strategy._crawl_web(
            "https://example.test/a", CrawlerRunConfig(user_agent=UA_CHROME_121)
        )
    assert page.captured_headers is None, (
        "set_extra_http_headers must be skipped for persistent context"
    )
    assert strategy.browser_config.user_agent == original_ua, (
        "shared config must not be mutated for persistent context"
    )


@pytest.mark.asyncio
async def test_user_custom_headers_preserved_with_per_crawl_ua():
    """Custom headers set on ``BrowserConfig.headers`` (other than UA /
    sec-ch-ua) are still sent, alongside the per-crawl UA and sec-ch-ua.
    Sec-ch-ua set by a concurrent crawl in the shared ``headers`` dict does
    NOT win over this crawl's per-crawl hint."""
    strategy, captured, queue = _build_strategy_with_capturing_pages()
    strategy.browser_config.headers["X-Custom-Header"] = "custom-value"
    await _run_crawl(strategy, queue, "A", "https://example.test/a", user_agent=UA_CHROME_121)
    h = captured["A"].captured_headers
    assert h is not None
    assert h["X-Custom-Header"] == "custom-value"
    assert h["User-Agent"] == UA_CHROME_121
    assert h["sec-ch-ua"] == HINT_121


# ---------------------------------------------------------------------------
# E2E test: real browser, public arun_many entry point, per-URL UA configs
# ---------------------------------------------------------------------------

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_echo_server(app, host, port, ready):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, host, port)
    loop.run_until_complete(site.start())
    ready.set()
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(runner.cleanup())
        loop.close()


def _start_echo_server():
    """Start an aiohttp server that echoes the request ``User-Agent`` header
    in the page body so the on-the-wire UA is observable per crawl."""
    port = _find_free_port()

    async def echo_ua(request):
        ua = request.headers.get("User-Agent", "")
        # Rich enough to clear the anti-bot "no_content_elements on small
        # page" structural check: several semantic content elements, several
        # hundred visible chars, metadata. The on-the-wire UA is echoed in
        # multiple places so the assertion can find it.
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <title>UA echo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="User-Agent echo page for crawl4ai regression tests">
</head>
<body>
    <header>
        <h1>User-Agent Echo Service</h1>
        <p>This page echoes the User-Agent header of the requesting HTTP
        client. It is served by the crawl4ai regression test suite to verify
        per-crawl User-Agent isolation under concurrency.</p>
    </header>
    <main>
        <article>
            <h2>Observed User-Agent</h2>
            <p id="ua">{ua}</p>
            <p>The string above is the verbatim value of the
            <code>User-Agent</code> HTTP request header received by the
            server. If two concurrent crawls each request a distinct
            per-crawl User-Agent, each result's body must contain only its
            own requested User-Agent and not the other crawl's.</p>
        </article>
        <article>
            <h2>Notes</h2>
            <p>This page intentionally contains a few hundred visible
            characters across multiple semantic elements so that the
            crawler's anti-bot structural heuristics do not flag it as a
            placeholder / error page.</p>
        </article>
    </main>
    <footer>
        <p>End of page. The User-Agent was: <span id="ua-footer">{ua}</span>.</p>
    </footer>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", echo_ua)
    app.router.add_get("/{path:.*}", echo_ua)

    ready = threading.Event()
    thread = threading.Thread(
        target=_run_echo_server, args=(app, "localhost", port, ready), daemon=True
    )
    thread.start()
    assert ready.wait(timeout=10), "UA-echo server failed to start"
    time.sleep(0.2)
    return f"http://localhost:{port}", thread


@pytest.mark.browser
@pytest.mark.asyncio
async def test_arun_many_per_url_user_agent_isolation_e2e():
    """End-to-end through the public ``crawler.arun_many`` entry point with a
    per-URL ``CrawlerRunConfig`` list (distinct ``user_agent`` values, routed
    via ``url_matcher``), against a UA-echoing local server. The body of each
    result must echo its OWN requested User-Agent and NOT contain the other
    crawl's UA (no cross-leak on the wire). This reproduces the production
    symptom from the bug report and confirms the fix end-to-end."""
    from crawl4ai import AsyncWebCrawler

    base_url, _thread = _start_echo_server()
    urls = [f"{base_url}/a", f"{base_url}/b"]
    cfg_a = CrawlerRunConfig(
        user_agent=UA_CHROME_121,
        url_matcher=urls[0],
        cache_mode=CacheMode.BYPASS,
        verbose=False,
    )
    cfg_b = CrawlerRunConfig(
        user_agent=UA_CHROME_122,
        url_matcher=urls[1],
        cache_mode=CacheMode.BYPASS,
        verbose=False,
    )

    async with AsyncWebCrawler(
        config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])
    ) as crawler:
        results = await crawler.arun_many(urls, config=[cfg_a, cfg_b])

    assert len(results) == 2, f"expected 2 results, got {len(results)}"
    by_url = {r.url: r for r in results}
    for url, expected_ua in ((urls[0], UA_CHROME_121), (urls[1], UA_CHROME_122)):
        r = by_url.get(url)
        assert r is not None, f"no result for {url}"
        assert r.success, f"crawl failed for {url}: {r.error_message}"
        body = r.html
        assert expected_ua in body, (
            f"on-wire UA mismatch for {url}: expected {expected_ua!r}, body={body[:200]!r}"
        )
        other_ua = UA_CHROME_122 if expected_ua == UA_CHROME_121 else UA_CHROME_121
        assert other_ua not in body, (
            f"cross-leak for {url}: other UA {other_ua!r} present in body={body[:200]!r}"
        )
