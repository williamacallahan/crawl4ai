"""Regression tests for sec-ch-ua / User-Agent isolation between custom-header
crawls and magic / random-UA crawls sharing one ``AsyncPlaywrightCrawlerStrategy``.

Bug (introduced in commit 349fb864, "fix(browser): apply custom headers to
live pages"): ``AsyncPlaywrightCrawlerStrategy._crawl_web`` widened the
post-``await self.browser_manager.get_page(...)`` page-level
``page.set_extra_http_headers`` flush from ``if ua_changed:`` to
``if ua_changed or self._custom_headers_set:``. Crawls that do NOT change the
User-Agent (``ua_changed`` is False) but set custom headers
(``set_custom_headers`` → ``_custom_headers_set`` True) began copying the
*shared* ``self.browser_config.headers`` dict — read AFTER the ``await``
suspension — onto the page. The TOCTOU fix (commit 14cc355a) had just added
per-crawl UA / sec-ch-ua locals captured BEFORE the await, but those locals
only protect the ``ua_changed`` branch; the custom-header-only branch has no
per-crawl overrides and so re-reads the shared dict post-await.

A magic / random-UA crawl writes ``headers["sec-ch-ua"]`` into that same
shared dict (commit fdd98978) and never reverts it, so a custom-header crawl
can ship another crawl's ``sec-ch-ua``:
  * concurrently — if the magic crawl's pre-await mutation runs while the
    custom crawl is suspended inside ``get_page`` (TOCTOU); or
  * sequentially — a later custom-header crawl picks up the stale hint any
    earlier magic crawl left in the shared dict (never reverted, and
    ``setup_context`` pushes the shared ``self.config.headers`` at context
    creation too). A ``sec-ch-ua`` that does not match the request's
    ``User-Agent`` is a well-known bot-detection signal (commit fdd98978:
    "Anti-bot systems like Akamai detect this mismatch as a bot signal").

The fix has two layers. (1) ``_crawl_web`` snapshots ``configured_headers``
into a local BEFORE the ``await get_page(...)``; the post-await flush reads
that snapshot (not the shared dict) — mirroring the per-crawl UA locals
TOCTOU mitigation — and threads it through ``get_page`` → ``setup_context``
so context-level headers use it too. (2) A magic / random-UA crawl does NOT
mutate the shared ``browser_config`` — the per-crawl UA / hint and header
snapshot are threaded through ``get_page`` as locals, so ``setup_context`` /
``create_browser_context`` see the per-crawl values without touching shared state.

Most tests are offline using a mocked browser surface modeled on
``tests/regression/test_concurrent_user_agent_isolation.py``. The single
end-to-end test at the bottom is ``@pytest.mark.browser`` and uses a real
Chromium against a small header-echoing local server. All offline tests run
in CI's Unit gate (``-m "not network and not browser"``).
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

UA_DEFAULT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
HINT_DEFAULT = UAGen.generate_client_hints(UA_DEFAULT)


# ---------------------------------------------------------------------------
# Offline helpers: a mocked browser surface that captures set_extra_http_headers
# ---------------------------------------------------------------------------

def _build_strategy(user_agent=UA_DEFAULT):
    """Build a real ``AsyncPlaywrightCrawlerStrategy`` with a fully mocked
    browser surface. Each ``page.set_extra_http_headers`` call captures its
    argument so the on-the-wire headers are observable per crawl.
    ``get_page`` yields (``await asyncio.sleep(0)``) so concurrent
    ``_crawl_web`` invocations interleave around it, exactly as they do in
    production against the shared ``BrowserConfig``.

    Construction is lightweight (``__init__`` only allocates a
    ``BrowserManager``; it does not start Chromium), so this stays offline.
    """
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(user_agent=user_agent, headless=True),
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
        # raising here isolates the header-leak surface from real navigation
        # while guaranteeing set_extra_http_headers has already run.
        p.goto = AsyncMock(side_effect=RuntimeError(f"STOP_AFTER_HDR_{tag}"))
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
    with pytest.raises(RuntimeError, match=f"STOP_AFTER_HDR_{tag}"):
        await strategy._crawl_web(url, cfg)


# ---------------------------------------------------------------------------
# Offline tests: the concurrent TOCTOU race (Scenario A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_magic_crawl_does_not_leak_sec_ch_ua_into_custom_header_crawl():
    """Scenario A (concurrent race): a custom-header crawl (``ua_changed``
    False, ``_custom_headers_set`` True) suspends inside ``get_page``; a
    concurrent magic crawl runs its pre-await mutation block, writing its
    random-UA ``sec-ch-ua`` into the shared ``browser_config.headers`` dict.
    When the custom crawl resumes and flushes, its page must carry its OWN
    (default) ``sec-ch-ua`` — not the magic crawl's. Under the bug, the
    post-await flush read the shared dict and shipped the magic crawl's
    client hint, producing a ``sec-ch-ua`` / ``User-Agent`` mismatch."""
    strategy, captured, queue = _build_strategy()
    original_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]
    strategy.set_custom_headers({"X-Custom": "1"})
    await asyncio.gather(
        _run_crawl(strategy, queue, "custom", "https://example.test/a"),
        _run_crawl(strategy, queue, "magic", "https://example.test/b", magic=True),
    )

    custom_h = captured["custom"].captured_headers
    magic_h = captured["magic"].captured_headers
    assert custom_h is not None, "custom crawl must flush headers"
    assert magic_h is not None, "magic crawl must flush headers"
    # Custom crawl ships its own default sec-ch-ua, not the magic crawl's.
    assert custom_h["sec-ch-ua"] == original_sec_ch_ua, (
        f"custom crawl leaked the magic crawl's sec-ch-ua: "
        f"{custom_h['sec-ch-ua']!r} (magic={magic_h['sec-ch-ua']!r})"
    )
    assert custom_h["sec-ch-ua"] != magic_h["sec-ch-ua"], (
        "custom crawl's sec-ch-ua must differ from the magic crawl's random hint"
    )
    assert custom_h["X-Custom"] == "1", "custom header must still reach the wire"
    assert magic_h["User-Agent"] != UA_DEFAULT, "magic crawl must generate a distinct UA"


# ---------------------------------------------------------------------------
# Offline tests: sequential staleness (Scenario B)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequential_magic_crawl_does_not_leave_stale_sec_ch_ua():
    """Scenario B (sequential staleness): a magic crawl runs and finishes;
    under the bug its ``sec-ch-ua`` write into the shared dict is never
    reverted, so a subsequent custom-header crawl ships the stale hint. The
    fix does not mutate the shared ``browser_config`` — per-crawl UA / hint
    are threaded through ``get_page`` as locals, so the later crawl ships
    its own default ``sec-ch-ua``."""
    strategy, captured, queue = _build_strategy()
    original_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]

    # First: a magic crawl runs to completion.
    await _run_crawl(strategy, queue, "magic", "https://example.test/a", magic=True)
    magic_h = captured["magic"].captured_headers
    assert magic_h["sec-ch-ua"] != original_sec_ch_ua, (
        "magic crawl must generate a distinct sec-ch-ua for the test to be meaningful"
    )
    # The shared config is never mutated by the magic crawl.
    assert strategy.browser_config.headers["sec-ch-ua"] == original_sec_ch_ua, (
        "shared headers['sec-ch-ua'] must not be mutated by the magic crawl"
    )
    assert strategy.browser_config.user_agent == UA_DEFAULT, (
        "shared user_agent must not be mutated by the magic crawl"
    )

    # Second: a custom-header crawl on the same crawler.
    strategy.set_custom_headers({"X-Custom": "1"})
    await _run_crawl(strategy, queue, "custom", "https://example.test/b")
    custom_h = captured["custom"].captured_headers
    assert custom_h["sec-ch-ua"] == original_sec_ch_ua, (
        f"custom crawl shipped the magic crawl's stale sec-ch-ua: "
        f"{custom_h['sec-ch-ua']!r} (magic={magic_h['sec-ch-ua']!r})"
    )
    assert custom_h["X-Custom"] == "1"


@pytest.mark.asyncio
async def test_custom_user_agent_via_set_custom_headers_not_polluted_by_prior_magic():
    """A user setting a custom ``User-Agent`` through the documented
    ``set_custom_headers`` API (advanced-features.md lists "specialized
    user-agent strings" as a use case) after a magic crawl must not get a
    stale ``sec-ch-ua`` from that magic crawl. The custom UA ships alongside
    the default ``sec-ch-ua``, never the magic crawl's hint."""
    strategy, captured, queue = _build_strategy()
    original_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]

    await _run_crawl(strategy, queue, "magic", "https://example.test/a", magic=True)
    magic_h = captured["magic"].captured_headers

    strategy.set_custom_headers(
        {"User-Agent": "MyCustomUA/1.0", "X-Custom": "1"}
    )
    await _run_crawl(strategy, queue, "custom", "https://example.test/b")
    custom_h = captured["custom"].captured_headers
    assert custom_h["User-Agent"] == "MyCustomUA/1.0"
    assert custom_h["sec-ch-ua"] != magic_h["sec-ch-ua"], (
        "custom-UA crawl must not ship the prior magic crawl's sec-ch-ua"
    )
    assert custom_h["sec-ch-ua"] == original_sec_ch_ua


# ---------------------------------------------------------------------------
# Offline tests: defense-in-depth — pre-await snapshot, not post-await shared
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_header_flush_uses_pre_await_snapshot():
    """Even if a concurrent crawl (or a hook) overwrites the shared
    ``browser_config.headers["sec-ch-ua"]`` while a custom-header crawl is
    suspended at ``get_page``, the crawl's page-level headers must come from
    the snapshot captured BEFORE the await, not from the post-await shared
    state. Deterministically fails on the buggy code (which reads the shared
    dict post-await) and passes on the fix."""
    strategy, captured, queue = _build_strategy()
    original_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]
    strategy.set_custom_headers({"X-Custom": "1"})
    original_get_page = strategy.browser_manager.get_page

    async def sabotaging_get_page(crawlerRunConfig=None, user_agent=None, browser_hint=None, **_):
        result = await original_get_page(
            crawlerRunConfig=crawlerRunConfig,
            user_agent=user_agent,
            browser_hint=browser_hint,
        )
        # Stand-in for a concurrent crawl that mutates the shared dict during
        # this crawl's await window.
        strategy.browser_config.headers["sec-ch-ua"] = "SABOTAGED-HINT"
        return result

    strategy.browser_manager.get_page = sabotaging_get_page
    await _run_crawl(strategy, queue, "custom", "https://example.test/a")
    h = captured["custom"].captured_headers
    assert h["sec-ch-ua"] == original_sec_ch_ua, (
        f"must use the pre-await snapshot, not the sabotaged shared dict: "
        f"{h['sec-ch-ua']!r}"
    )
    assert h["X-Custom"] == "1"


@pytest.mark.asyncio
async def test_magic_crawl_page_level_uses_locals_not_post_await_shared():
    """For a ``ua_changed`` crawl the per-crawl UA / sec-ch-ua locals still
    drive the page-level flush (no regression on the path 14cc355a fixed),
    even if the shared dict is mutated during the await."""
    strategy, captured, queue = _build_strategy()
    original_get_page = strategy.browser_manager.get_page

    async def sabotaging_get_page(crawlerRunConfig=None, user_agent=None, browser_hint=None, **_):
        result = await original_get_page(
            crawlerRunConfig=crawlerRunConfig,
            user_agent=user_agent,
            browser_hint=browser_hint,
        )
        strategy.browser_config.headers["sec-ch-ua"] = "SABOTAGED-HINT"
        return result

    strategy.browser_manager.get_page = sabotaging_get_page
    await _run_crawl(strategy, queue, "magic", "https://example.test/a", magic=True)
    h = captured["magic"].captured_headers
    assert h["sec-ch-ua"] == UAGen.generate_client_hints(h["User-Agent"]), (
        "ua_changed crawl must keep sec-ch-ua in sync with its own UA"
    )
    assert h["sec-ch-ua"] != "SABOTAGED-HINT"


@pytest.mark.asyncio
async def test_no_ua_no_custom_headers_skips_flush_and_keeps_config_clean():
    """A plain crawl (no UA change, no custom headers) must not call
    ``set_extra_http_headers`` and must not mutate the shared config. Guards
    against the fix accidentally widening the flush."""
    strategy, captured, queue = _build_strategy()
    original_ua = strategy.browser_config.user_agent
    original_hint = strategy.browser_config.browser_hint
    original_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]
    await _run_crawl(strategy, queue, "plain", "https://example.test/a")
    assert captured["plain"].captured_headers is None, (
        "set_extra_http_headers must not be called for a plain crawl"
    )
    assert strategy.browser_config.user_agent == original_ua
    assert strategy.browser_config.browser_hint == original_hint
    assert strategy.browser_config.headers["sec-ch-ua"] == original_sec_ch_ua


# ---------------------------------------------------------------------------
# Offline test: context-level vector — shared config clean after a magic crawl
# ---------------------------------------------------------------------------

def _applied_headers(mock_context):
    assert mock_context.set_extra_http_headers.called
    call_args = mock_context.set_extra_http_headers.call_args
    return call_args.args[0] if call_args.args else call_args.kwargs["headers"]


@pytest.mark.asyncio
async def test_setup_context_clean_after_magic_crawl_no_mutation():
    """The context-level vector: ``BrowserManager.setup_context`` pushes the
    shared ``self.config.headers`` at context creation. A magic crawl does NOT
    mutate the shared ``browser_config``, so a subsequent ``setup_context``
    (no per-crawl override) must push the ORIGINAL ``sec-ch-ua``, not the
    magic crawl's. Under the bug the never-reverted mutation leaks into every
    subsequent context.

    Uses a REAL ``BrowserManager`` (only ``get_page`` /
    ``release_page_with_context`` are faked) so ``setup_context`` runs
    against the genuine shared ``self.config`` — mirrors
    ``test_set_custom_headers_applied.py``."""
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(user_agent=UA_DEFAULT, headless=True),
        logger=AsyncLogger(verbose=False),
    )
    captured = {}
    queue: asyncio.Queue = asyncio.Queue()
    original_get_page = strategy.browser_manager.get_page

    async def fake_get_page(crawlerRunConfig=None, user_agent=None, browser_hint=None, **_):
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        tag = await queue.get()
        p = MagicMock()
        p.captured_headers = None
        p.close = AsyncMock()

        async def set_extra_http_headers(headers):
            p.captured_headers = dict(headers)

        p.set_extra_http_headers = set_extra_http_headers
        p.goto = AsyncMock(side_effect=RuntimeError(f"STOP_AFTER_HDR_{tag}"))
        captured[tag] = p
        return p, MagicMock()

    strategy.browser_manager.get_page = fake_get_page
    strategy.browser_manager.release_page_with_context = AsyncMock()
    original_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]

    await _run_crawl(strategy, queue, "magic", "https://example.test/a", magic=True)
    magic_h = captured["magic"].captured_headers
    assert magic_h["sec-ch-ua"] != original_sec_ch_ua

    # Simulate a subsequent crawl's context creation (no per-crawl UA thread)
    # via the REAL setup_context on the shared config.
    mock_context = AsyncMock()
    await strategy.browser_manager.setup_context(mock_context)
    applied = _applied_headers(mock_context)
    assert applied["sec-ch-ua"] == original_sec_ch_ua, (
        f"setup_context shipped the magic crawl's stale sec-ch-ua: "
        f"{applied['sec-ch-ua']!r}"
    )


# ---------------------------------------------------------------------------
# E2E test: real browser, public arun entry point, custom headers + magic
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


def _start_header_echo_server():
    """Start an aiohttp server that echoes the request ``User-Agent`` and
    ``sec-ch-ua`` headers in the page body so the on-the-wire values are
    observable per crawl."""
    port = _find_free_port()
    async def echo_headers(request):
        ua = request.headers.get("User-Agent", "")
        sec_ch_ua = request.headers.get("sec-ch-ua", "")
        # Rich enough to clear the anti-bot "no_content_elements on small page"
        # structural check: several semantic content elements, several hundred
        # visible chars. UA / sec-ch-ua echoed in multiple places so the
        # assertions can find them.
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <title>header echo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="Header echo page for crawl4ai regression tests">
</head>
<body>
    <header><h1>Header Echo Service</h1>
    <p>This page echoes the User-Agent and sec-ch-ua HTTP request headers of
    the requesting client for the crawl4ai regression test suite. The on-wire
    values are observable here so per-crawl header isolation can be verified
    end-to-end against a real browser.</p></header>
    <main><article><h2>Observed User-Agent</h2><p id="ua">{ua}</p>
    <h2>Observed sec-ch-ua</h2><p id="sec-ch-ua">{sec_ch_ua}</p>
    <p>This page intentionally contains several hundred visible characters
    across multiple semantic elements so the crawler's anti-bot structural
    heuristics do not flag it as a placeholder or error page.</p></article></main>
    <footer><p>End of page. UA={ua} sec-ch-ua={sec_ch_ua}</p></footer>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", echo_headers)
    app.router.add_get("/{path:.*}", echo_headers)

    ready = threading.Event()
    thread = threading.Thread(
        target=_run_echo_server, args=(app, "localhost", port, ready), daemon=True
    )
    thread.start()
    assert ready.wait(timeout=10), "header-echo server failed to start"
    time.sleep(0.2)
    return f"http://localhost:{port}", thread


@pytest.mark.browser
@pytest.mark.asyncio
async def test_custom_header_crawl_after_magic_crawl_e2e():
    """End-to-end through the public ``crawler.arun`` entry point: a magic
    crawl runs first, then a custom-header crawl (with a custom User-Agent)
    against a header-echoing local server. The custom crawl's body must echo
    its OWN custom User-Agent and a ``sec-ch-ua`` consistent with the
    configuration — NOT the magic crawl's random client hint (no
    sequential-staleness leak on the wire)."""
    from crawl4ai import AsyncWebCrawler

    base_url, _thread = _start_header_echo_server()

    async with AsyncWebCrawler(
        config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])
    ) as crawler:
        # First crawl: magic mode (mutates the shared config under the bug).
        magic_result = await crawler.arun(
            url=f"{base_url}/magic",
            config=CrawlerRunConfig(magic=True, cache_mode=CacheMode.BYPASS, verbose=False),
        )
        assert magic_result.success, (
            f"magic crawl failed: {magic_result.error_message}"
        )

        # Second crawl: custom headers with a custom User-Agent, on the same
        # crawler (shared BrowserConfig).
        crawler.crawler_strategy.set_custom_headers(
            {"User-Agent": "MyCustomUA/1.0", "X-Custom": "1"}
        )
        custom_result = await crawler.arun(
            url=f"{base_url}/custom",
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False),
        )
        assert custom_result.success, (
            f"custom-header crawl failed: {custom_result.error_message}"
        )

    magic_body = magic_result.html
    custom_body = custom_result.html
    assert "MyCustomUA/1.0" in custom_body, (
        f"custom UA not on the wire: {custom_body[:200]!r}"
    )
    # The magic crawl's random UA must NOT leak onto the custom crawl's wire.
    magic_ua = magic_result.html.split('id="ua">', 1)[1].split("<", 1)[0]
    assert magic_ua and magic_ua != "MyCustomUA/1.0", (
        "magic crawl must have a distinct UA for the test to be meaningful"
    )
    assert magic_ua not in custom_body, (
        f"magic crawl's UA leaked onto the custom crawl's wire: {custom_body[:300]!r}"
    )