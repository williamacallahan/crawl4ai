"""End-to-end regression for silent console-capture loss on `raw:` / `file://`
crawls when the `UndetectedAdapter` is in use.

Bug lineage
-----------
Commit 2550f3d2 ("Add browser pipeline support for raw:/file:// URLs",
part of v0.8.0) routed `raw:` / `file://` crawls with
`capture_console_messages=True` into `_crawl_web()`'s full browser
pipeline. The local-content branch loads the HTML with
`page.set_content(...)` instead of `page.goto(...)`. The
`UndetectedAdapter` armed its console/error capture solely via
`page.add_init_script(...)`, which in patchright 1.56 is implemented by
rewriting the response of *network* `document` requests whose URL starts
with `http`. `page.set_content` produces no network document request and
`file://` is not an `http` URL, so the init script never ran for those
documents, the in-page `window.__capturedConsole` buffer was never
created, and retrieval's `window.__capturedConsole || []` fallback plus
`except Exception: pass` then silently yielded `[]`. The crawl reported
`success=True` with empty `console_messages` — a documented feature
combination (`raw:`/`file://` browser pipeline + undetected adapter +
console capture) silently dropping the captured data with no warning,
error, or log.

Fix
---
`UndetectedAdapter.bootstrap_capture_html` prepends the same capture
bootstrap that the init script would have installed into the HTML as a
`<script>` tag at the top of `<head>`, so it runs synchronously
*before* any user inline script during `page.set_content` parsing.
The strategy calls it from the `is_local_content` branch of `_crawl_web`
when `capture_console_messages` is on. Adapter-internal init scripts
through `add_init_script` remain for `http(s)`-scheme documents; the
prepend path is the fallback for the schemes the init scripts miss.

These tests drive the real `UndetectedAdapter` through the real
`AsyncPlaywrightCrawlerStrategy` and a real patchright-launched Chromium
against both `raw:` (in-process HTML, loaded via `page.set_content`) and
`file://` (a temp-file crawl) inputs. They assert that:

  * `console.log` from an inline `<script>` reaches `console_messages`.
  * An uncaught `throw new Error(...)` from an inline `<script>` reaches
    `console_messages`.
  * The `PlaywrightAdapter` baseline still captures the same content
    (parity guard: a fix to the undetected path must not break the
    default adapter).
  * The same crawl with `capture_console_messages=False` returns
    `console_messages=None` (the feature flag still gates the capture).

Marked `@pytest.mark.browser` — requires the installed Chromium (present
in `~/.cache/ms-playwright`); excluded from the fast unit gate
(`-m "not network and not browser"`); runs in CI's browser-tests stage
and locally where Chromium is available.
"""

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.browser_adapter import PlaywrightAdapter, UndetectedAdapter


# Inline-script HTML that exercises both code paths the bug dropped:
#   * a synchronous `console.log` that fires during `set_content` parsing
#     (so the test proves capture does not depend on an init script that
#     would have run before the inline script),
#   * an uncaught `Error` thrown on the next microtask, proving pageerror
#     capture is also wired for non-http documents.
RAW_CONSOLE_HTML = (
    "<html><head><title>raw console capture</title></head>"
    "<body>"
    "<p>Body text so the antibot minimal-text check passes.</p>"
    "<script>console.log('RAW_LOG');</script>"
    "<script>setTimeout(function () { throw new Error('RAW_ERROR'); }, 0);</script>"
    "</body></html>"
)


def _console_texts(result):
    return [m.get("text") for m in (result.console_messages or [])]


def _console_timestamps(result):
    return [
        m.get("timestamp")
        for m in (result.console_messages or [])
        if isinstance(m.get("timestamp"), (int, float))
    ]


# `time.time()` seconds fall in [1e9, 4e9) for years 2001–2100; raw JS `Date.now()`
# milliseconds fall in [1e12, 4e12). A timestamp below the ms threshold therefore
# pins the seconds unit and rejects raw-ms values (1000x too large) regardless
# of the current wall clock. Shared with `test_undetected_adapter_iframe_console`.
_SECONDS_LOWER = 1e9
_SECONDS_UPPER = 1e11


def _build_strategy(adapter):
    return AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True, verbose=False),
        browser_adapter=adapter,
    )


# ---------------------------------------------------------------------------
# UndetectedAdapter — the adapter the bug was silently broken on.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_console_log_from_raw_url():
    """`UndetectedAdapter` + `raw:` (loaded via `page.set_content`) +
    `capture_console_messages=True` must surface the inline `console.log`
    on `console_messages`. Pre-fix this returned `[]` with `success=True`
    — silent data loss. The crawl must also report success so the bug
    cannot hide behind a failure flag."""
    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="domcontentloaded",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw:{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert "RAW_LOG" in texts, (
        f"raw: inline console.log missing on UndetectedAdapter (the bug): {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_uncaught_error_from_raw_url():
    """`UndetectedAdapter` + `raw:` + `capture_console_messages=True` must
    also surface uncaught errors thrown from the inline script — pre-fix
    the pageerror init-script listener was never installed either, so the
    `RAW_ERROR` was silently dropped. Patchright's `pageerror` event
    message is `Uncaught Error: RAW_ERROR`, so match on the substring."""
    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="networkidle",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw:{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert any("RAW_ERROR" in (t or "") for t in texts), (
        f"raw: uncaught error missing on UndetectedAdapter (the bug): {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_console_timestamps_from_raw_url_are_in_seconds():
    """Captured timestamps must be in Python seconds (matching
    `PlaywrightAdapter`'s `time.time()`), not raw JS milliseconds — the
    `UndetectedAdapter` used to read `Date.now()` ms from the init-script
    buffer and divide by 1000; under the event-based fix it uses
    `time.time()` directly. Pin the seconds unit so a regression that
    drops the unit normalization cannot re-introduce 1000x-off values in
    downstream consumers that compare timestamps."""
    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="domcontentloaded",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw:{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    timestamps = _console_timestamps(result)
    assert timestamps, f"no numeric timestamps: {result.console_messages!r}"
    assert all(_SECONDS_LOWER < ts < _SECONDS_UPPER for ts in timestamps), (
        f"timestamp not in seconds (expected ~1e9, got raw ms ~1e12?): {timestamps!r}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_console_log_from_raw_double_slash_url():
    """The `raw://` prefix (alternative to `raw:`) is parsed slightly
    differently (`url[6:]` vs `url[4:]`); both routes go through
    `_crawl_web()`'s `is_local_content` branch and `page.set_content`,
    so capture must work on both. Pin the `raw://` form so a regex/path
    tweak cannot silently re-break it."""
    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="domcontentloaded",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw://{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert "RAW_LOG" in texts, (
        f"raw:// inline console.log missing on UndetectedAdapter: {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_console_log_from_file_url(tmp_path):
    """`UndetectedAdapter` + `file://` (the other half of the bug) must
    surface `console.log` from the loaded file. Patchright's `add_init_script`
    does not run for `file://` either, so pre-fix this was also `[]`."""
    html_file = tmp_path / "console.html"
    html_file.write_text(RAW_CONSOLE_HTML, encoding="utf-8")

    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="domcontentloaded",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"file://{html_file}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert "RAW_LOG" in texts, (
        f"file:// inline console.log missing on UndetectedAdapter (the bug): {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_captures_errors_from_file_url(tmp_path):
    """`file://` + `UndetectedAdapter` must surface the thrown `Error` from
    the inline script too — the pageerror capture path is also affected by
    the same scheme limitation and must be fixed by the same event-based
    redesign."""
    html_file = tmp_path / "error.html"
    html_file.write_text(RAW_CONSOLE_HTML, encoding="utf-8")

    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="networkidle",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"file://{html_file}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert any("RAW_ERROR" in (t or "") for t in texts), (
        f"file:// uncaught error missing on UndetectedAdapter: {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_undetected_adapter_capture_disabled_still_returns_none_on_raw():
    """The fix must not regress the feature flag: when
    `capture_console_messages` is False, no listeners are registered, no
    retrieval runs, and `console_messages` is `None` (not `[]`) — the
    strategy gates the field on the config flag and the adapter is a
    no-op. Guards against an over-eager fix that always registers
    listeners regardless of the caller's flag."""
    strategy = _build_strategy(UndetectedAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=False,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="domcontentloaded",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw:{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    assert result.console_messages is None, (
        f"console_messages must be None when capture is disabled, got "
        f"{result.console_messages!r}"
    )


# ---------------------------------------------------------------------------
# PlaywrightAdapter — parity guard so the fix to the undetected path
# cannot silently regress the default adapter that was already capturing
# correctly on `raw:`/`file://`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.browser
async def test_playwright_adapter_still_captures_console_log_from_raw_url():
    """Baseline parity: the default `PlaywrightAdapter` (event-based
    `page.on("console")`) was never affected by the bug — it always
    captured `raw:` console output. Pin that behavior so the fix to
    `UndetectedAdapter` cannot regress the default path while reworking
    the adapter contract."""
    strategy = _build_strategy(PlaywrightAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="domcontentloaded",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw:{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert "RAW_LOG" in texts, (
        f"raw: inline console.log missing on PlaywrightAdapter (parity): {texts}"
    )


@pytest.mark.asyncio
@pytest.mark.browser
async def test_playwright_adapter_still_captures_errors_from_raw_url():
    """Same parity guard for the pageerror path of the default adapter."""
    strategy = _build_strategy(PlaywrightAdapter())
    config = CrawlerRunConfig(
        capture_console_messages=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        wait_until="networkidle",
    )
    async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
        result = await crawler.arun(url=f"raw:{RAW_CONSOLE_HTML}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    texts = _console_texts(result)
    assert any("RAW_ERROR" in (t or "") for t in texts), (
        f"raw: uncaught error missing on PlaywrightAdapter (parity): {texts}"
    )
