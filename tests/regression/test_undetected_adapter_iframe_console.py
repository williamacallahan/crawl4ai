"""End-to-end regression for the iframe console-capture bug in
`UndetectedAdapter.retrieve_console_messages`.

Pre-fix, `retrieve_console_messages` drained `window.__capturedConsole` /
`window.__capturedErrors` from the main frame only (`page.evaluate(...)`),
so every console message and uncaught error emitted from an http(s)-scheme
iframe was silently dropped from `AsyncCrawlResponse.console_messages` on
undetected-mode crawls. `add_init_script` runs in every http(s)-scheme
document, so each iframe had its own populated buffer that was never drained.

These tests drive the real `UndetectedAdapter` through the real
`AsyncPlaywrightCrawlerStrategy` and a real patchright-launched Chromium
against a local http server (`tests/regression/conftest.py::local_server`)
serving a page whose same-origin http iframe logs a message and raises an
uncaught error. The fix iterates `page.frames`, so both the main and
subframe buffers must reach the response.

Marked `@pytest.mark.browser` — requires the installed Chromium (present in
`~/.cache/ms-playwright`) and is excluded from the fast unit gate
(`-m "not network and not browser"`); it runs in the CI build job and
locally where Chromium is available.
"""

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.browser_adapter import PlaywrightAdapter, UndetectedAdapter


def _console_texts(result):
    return [m.get("text") for m in (result.console_messages or [])]


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_console_from_http_iframe(local_server):
    """UndetectedAdapter crawl of a page whose http iframe calls console.log
    must surface both the main-frame and subframe messages on
    `AsyncCrawlResponse.console_messages`. Pre-fix only the main-frame
    message (`FROM_MAIN`) reached the response; `FROM_HTTP_IFRAME` was
    silently discarded because `page.evaluate` drained only the top frame.
    """
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True, verbose=False),
        browser_adapter=UndetectedAdapter(),
    )
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="networkidle",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(
            url=local_server + "/iframe-http-page", config=config
        )

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert "FROM_MAIN" in texts, f"main-frame console.log missing: {texts}"
    assert "FROM_HTTP_IFRAME" in texts, (
        f"http-iframe console.log missing (the bug): {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_error_from_http_iframe(local_server):
    """The iframe's uncaught error must also reach the response. Pre-fix the
    subframe's `window.__capturedErrors` buffer was never drained, so the
    iframe-originated error was lost. Patchright's `error` event listener
    records `event.message` (e.g. `"Uncaught Error: FROM_HTTP_IFRAME_ERROR"`),
    so the assertion matches on the substring rather than the exact text."""
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True, verbose=False),
        browser_adapter=UndetectedAdapter(),
    )
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="networkidle",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(
            url=local_server + "/iframe-http-page", config=config
        )

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert any("FROM_HTTP_IFRAME_ERROR" in (t or "") for t in texts), (
        f"http-iframe uncaught error missing (the bug): {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_playwright_adapter_still_captures_http_iframe_console(local_server):
    """Baseline parity guard: the default `PlaywrightAdapter` (event-based
    `page.on("console")`) was never affected by the bug — it captures iframe
    messages via the Playwright console event. Pin that behavior so a future
    change to the adapter interface does not silently regress the default
    path while fixing the undetected path."""
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True, verbose=False),
        browser_adapter=PlaywrightAdapter(),
    )
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="networkidle",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(
            url=local_server + "/iframe-http-page", config=config
        )

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert "FROM_MAIN" in texts, f"main-frame console.log missing: {texts}"
    assert "FROM_HTTP_IFRAME" in texts, (
        f"http-iframe console.log missing on default adapter: {texts}"
    )
