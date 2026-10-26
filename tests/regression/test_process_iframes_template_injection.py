"""Regression tests for the ``process_iframes`` JavaScript-injection fix.

The async crawler strategy's ``process_iframes`` used to extract each
iframe's ``document.body.innerHTML`` and re-inject it into the parent page
by interpolating it into a JavaScript template literal
(``parser.parseFromString(`${iframe_content}`, 'text/html')``) with only
backticks escaped. That left two injection channels into the main page's
origin:

  - ``${...}`` sequences in the iframe body HTML were evaluated as JavaScript
    expressions in the PARENT page's origin (a same-origin-policy bypass),
    letting attacker-controlled iframe content read parent cookies, DOM,
    and storage in the crawler's browser session.
  - backslashes in the iframe body were interpreted as JS escape sequences,
    corrupting the content; a trailing backslash produced an unterminated
    template literal which raised a ``SyntaxError`` that the per-iframe
    ``except Exception`` block swallowed, silently dropping that iframe.

The fix passes the iframe HTML as a structured ``evaluate`` argument so
Playwright's serializer handles it as data, never as JS source. These
tests guard against re-introducing the template-literal interpolation
pattern and verify the actual security behaviour end-to-end against a real
Chromium with a cross-origin iframe (different ports on 127.0.0.1, which
browsers treat as different origins). Binding to the literal IP 127.0.0.1
means the fixture does not depend on ``/etc/hosts`` containing ``localhost``.

The source-inspection guards run in the unit gate
(``-m "not network and not browser"``). The end-to-end tests are marked
``@pytest.mark.browser`` — they require an installed Chromium
(``~/.cache/ms-playwright``) and run in the CI browser-tests stage.
"""

import asyncio
import inspect
import socket
import threading
import time

import pytest
import pytest_asyncio
from aiohttp import web

from crawl4ai.async_configs import BrowserConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


# ===========================================================================
# Source-inspection guards (deterministic, no browser, run in unit gate)
# ===========================================================================


def _source() -> str:
    return inspect.getsource(AsyncPlaywrightCrawlerStrategy.process_iframes)


class TestProcessIframesSourceGuards:
    """Guard against re-introducing the unsafe JS-interpolation pattern."""

    def test_iframe_html_not_interpolated_into_template_literal(self):
        """The iframe HTML local must not be embedded into a JS template
        literal. Specifically the ``DOMParser.parseFromString`` call must NOT
        receive a backtick-wrapped string."""
        src = _source()
        assert "iframe_content" in src, (
            "this guard tracks the iframe_content local; rename and update"
        )
        assert "parseFromString(`" not in src, (
            "parseFromString must use args.html, not a backtick template literal"
        )
        for ln in src.splitlines():
            if "parseFromString" in ln:
                assert "`" not in ln, (
                    f"parseFromString line still contains a template literal: "
                    f"{ln.strip()!r}"
                )

    def test_backtick_escape_line_removed(self):
        """The ``iframe_content.replace('`', '\\`')`` line is unnecessary
        once the HTML is passed as a structured arg and must not come back."""
        src = _source()
        assert ".replace(\"`\", \"\\\\`\")" not in src, (
            "backtick-escape line is unnecessary once iframe HTML is passed as "
            "an evaluate arg; its presence signals a regression to "
            "template-literal interpolation"
        )
        assert ".replace('`', '\\\\`')" not in src, (
            "single-quoted backtick-escape line must also be absent"
        )

    def test_iframe_html_passed_as_evaluate_argument(self):
        """The evaluate call must take a structured ``args`` payload and
        reference it as ``args.html``/``args.className``/``args.id`` inside
        the JS body — i.e. iframe HTML flows through Playwright's
        serializer, not through JS source."""
        src = _source()
        assert "(args) =>" in src, (
            "evaluate must declare an arg-taking arrow function"
        )
        assert "args.html" in src
        assert "args.className" in src
        assert "args.id" in src
        assert (
            '"html": iframe_content' in src or "'html': iframe_content" in src
        ), "iframe_content must be forwarded as the 'html' field of the eval arg"

    def test_domparser_pattern_preserved(self):
        """The DOMParser-based migration (PR #1796) must remain intact: the
        fix changes only the embedding channel, not the parsing strategy."""
        src = _source()
        assert "new DOMParser()" in src
        assert "parseFromString" in src
        assert "doc.body.firstChild" in src
        assert "appendChild" in src
        for ln in src.splitlines():
            stripped = ln.strip()
            if "div.innerHTML" in stripped and "=" in stripped:
                pytest.fail(
                    f"unsafe innerHTML assignment reintroduced: {stripped!r}"
                )


# ===========================================================================
# End-to-end browser tests (require installed Chromium)
# ===========================================================================


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(
    app: web.Application, host: str, port: int, ready_event: threading.Event
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, host, port).start())
    ready_event.set()
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(runner.cleanup())
        loop.close()


class _MutableProvider:
    """A small callable whose return value tests can rewrite per test."""

    def __init__(self) -> None:
        self.value = ""

    def __call__(self) -> str:
        return self.value


def _make_app(handler) -> web.Application:
    app = web.Application()
    app.router.add_get("/", handler)
    return app


def _parent_page_html(parent_base: str, iframe_url: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><title>parent</title></head><body>
  <h1>parent content visible enough</h1>
  <p>main paragraph so the page is not empty: {parent_base}</p>
  <iframe id="f0" src="{iframe_url}" title="f0"></iframe>
</body></html>"""


class _CrossOriginHarness:
    """Starts two aiohttp servers on 127.0.0.1 (different ports => different
    browser origins) plus a Playwright Chromium, and exposes a fresh page
    plus mutable providers for the parent HTML and the iframe body HTML."""

    def __init__(self) -> None:
        self.parent_provider = _MutableProvider()
        self.iframe_provider = _MutableProvider()
        parent_port = _find_free_port()
        iframe_port = _find_free_port()

        async def parent_handler(_request):
            return web.Response(
                text=self.parent_provider(), content_type="text/html"
            )

        async def iframe_handler(_request):
            return web.Response(
                text=self.iframe_provider(), content_type="text/html"
            )

        self.parent_base = f"http://127.0.0.1:{parent_port}/"
        self.iframe_base = f"http://127.0.0.1:{iframe_port}/"
        parent_ready = threading.Event()
        iframe_ready = threading.Event()
        self._parent_thread = threading.Thread(
            target=_run_server,
            args=(_make_app(parent_handler), "127.0.0.1", parent_port, parent_ready),
            daemon=True,
        )
        self._iframe_thread = threading.Thread(
            target=_run_server,
            args=(_make_app(iframe_handler), "127.0.0.1", iframe_port, iframe_ready),
            daemon=True,
        )
        self._parent_ready = parent_ready
        self._iframe_ready = iframe_ready
        self._playwright = None
        self._browser = None

    def start_servers(self) -> "_CrossOriginHarness":
        self._parent_thread.start()
        self._iframe_thread.start()
        assert self._parent_ready.wait(10), "parent aiohttp failed to start"
        assert self._iframe_ready.wait(10), "iframe aiohttp failed to start"
        time.sleep(0.2)
        return self

    async def start_browser(self) -> "_CrossOriginHarness":
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        return self

    async def new_page(self):
        context = await self._browser.new_context()
        return context, await context.new_page()

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


@pytest_asyncio.fixture
async def _harness():
    """Yield a started cross-origin harness; tear down at fixture exit."""
    h = _CrossOriginHarness().start_servers()
    await h.start_browser()
    try:
        yield h
    finally:
        await h.close()


def _strategy() -> AsyncPlaywrightCrawlerStrategy:
    """Build the real strategy without invoking ``start()``. process_iframes
    only touches ``self.adapter`` (``PlaywrightAdapter`` forwards to
    ``page.evaluate``), ``self.logger`` (default AsyncLogger), and
    ``self._raise_if_page_crashed`` (uses ``self._page_crash_errors``,
    initialised in __init__), so it is safe to skip the browser manager
    start and use our own Playwright page."""
    return AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True, verbose=False)
    )


async def _inline_one(harness, page, iframe_body_html: str) -> None:
    """Drive the real process_iframes against a fresh parent payload."""
    harness.parent_provider.value = _parent_page_html(
        harness.parent_base, harness.iframe_base
    )
    harness.iframe_provider.value = iframe_body_html
    await page.goto(harness.parent_base, wait_until="load")
    # Allow the cross-origin iframe to finish loading before reading body.
    await page.wait_for_timeout(500)
    await _strategy().process_iframes(page, max_iframes=10)
    await page.wait_for_load_state("load", timeout=5000)


async def _replaced_div_html(page) -> str:
    return await page.evaluate(
        "() => document.querySelector('.extracted-iframe-content-0')?.innerHTML"
    )


@pytest.mark.browser
@pytest.mark.asyncio
async def test_dollar_curly_in_iframe_body_does_not_execute(_harness):
    """The core security regression: an attacker-controlled iframe body
    containing ``${...}`` must be treated as data, not evaluated as JS in
    the parent origin. Pre-fix, ``window.parentPwned`` would be set by the
    interpolation and the inlined HTML would show ``price is 2 and true``."""
    context, page = await _harness.new_page()
    try:
        payload = (
            "<!DOCTYPE html><html><head><title>iframe-doc</title></head>"
            "<body><p id='x'>price is ${1+1} and "
            "${window.parentPwned = String(window === window.top)}</p></body></html>"
        )
        await _inline_one(_harness, page, payload)
        parent_pwned = await page.evaluate("() => window.parentPwned")
        assert parent_pwned is None, (
            f"BUG: ${{...}} executed in parent origin "
            f"(parentPwned={parent_pwned!r})"
        )
        div_html = await _replaced_div_html(page)
        assert div_html is not None, "iframe replacement div not created"
        assert "${1+1}" in div_html, f"literal ${{1+1}} mangled: {div_html!r}"
        assert (
            "${window.parentPwned = String(window === window.top)}" in div_html
        ), f"interpolation payload not preserved verbatim: {div_html!r}"
        assert "2 and true" not in div_html, (
            f"BUG: expressions evaluated to literal text: {div_html!r}"
        )
    finally:
        await context.close()


@pytest.mark.browser
@pytest.mark.asyncio
async def test_backslashes_preserved_verbatim(_harness):
    """Literal backslashes in iframe body must be preserved
    character-for-character. Pre-fix, ``\\t`` and ``\\n`` became real TAB
    and NEWLINE inside the inlined HTML."""
    context, page = await _harness.new_page()
    try:
        await _inline_one(
            _harness,
            page,
            "<!DOCTYPE html><html><head><title>b</title></head>"
            "<body><p id='p1'>path C:\\temp\\nname and a\\\\b</p></body></html>",
        )
        div_html = await _replaced_div_html(page)
        assert div_html is not None
        assert "\t" not in div_html, f"literal \\t turned into a TAB: {div_html!r}"
        assert "\n" not in div_html, (
            f"literal \\n turned into a NEWLINE: {div_html!r}"
        )
        assert "C:\\temp" in div_html, f"backslash path corrupted: {div_html!r}"
    finally:
        await context.close()


@pytest.mark.browser
@pytest.mark.asyncio
async def test_trailing_backslash_does_not_drop_iframe(_harness):
    """A trailing backslash as the LAST character of the iframe body's
    ``innerHTML`` previously abutted the injected closing backtick of the
    buggy template literal, producing an unterminated template literal
    that raised a ``SyntaxError`` the per-iframe ``except Exception`` block
    swallowed — the iframe was silently omitted from the crawl.

    To actually end ``innerHTML`` with a backslash (and thus trigger the
    SyntaxError on the buggy code), the backslash must be the last text
    node of the body with no closing element tag after it; a payload like
    ``<p>foo\\</p>`` serialises to ``<p>foo\\</p>`` which ends with
    ``</p>``, NOT with a backslash, and so does NOT exercise the bug. The
    payload below (a ``<p>`` followed by a bare trailing text node ending
    in a backslash) is engineered to make ``document.body.innerHTML`` end
    with a literal ``\\``.

    The fix must replace the iframe and preserve the trailing backslash
    verbatim (the iframe element MUST be removed from the parent DOM:
    no remaining ``<iframe>`` for that slot)."""
    context, page = await _harness.new_page()
    try:
        await _inline_one(
            _harness,
            page,
            "<!DOCTYPE html><html><head><title>tb</title></head>"
            "<body><p id='p2'>before</p>trailing backslash\\</body></html>",
        )
        div_html = await _replaced_div_html(page)
        iframes_left = await page.evaluate(
            "() => document.querySelectorAll('iframe').length"
        )
        assert div_html is not None, (
            "BUG: trailing backslash aborted evaluate; iframe silently dropped"
        )
        assert "trailing backslash\\" in div_html, (
            f"trailing-backslash content not preserved: {div_html!r}"
        )
        assert "before" in div_html, f"prior content lost: {div_html!r}"
        assert iframes_left == 0, (
            f"BUG: iframe not replaced (silent drop); {iframes_left} left"
        )
    finally:
        await context.close()


@pytest.mark.browser
@pytest.mark.asyncio
async def test_backticks_in_iframe_body_preserved(_harness):
    """Backticks and nested ``${`...`}`` (a nested template literal in JS
    terms) inside iframe body HTML must round-trip as data."""
    context, page = await _harness.new_page()
    try:
        await _inline_one(
            _harness,
            page,
            "<!DOCTYPE html><html><head><title>bt</title></head>"
            "<body><p id='p3'>a `b` c and ${`nested`}</p></body></html>",
        )
        div_html = await _replaced_div_html(page)
        assert div_html is not None
        assert "`b`" in div_html, f"backticks mangled: {div_html!r}"
        assert "${`nested`}" in div_html, (
            f"nested template literal not preserved as data: {div_html!r}"
        )
    finally:
        await context.close()


@pytest.mark.browser
@pytest.mark.asyncio
async def test_benign_iframe_content_still_inlines(_harness):
    """Happy-path regression: ordinary iframe bodies without any of
    ``${``, ``\\``, or backticks must still inline exactly as before."""
    context, page = await _harness.new_page()
    try:
        await _inline_one(
            _harness,
            page,
            "<!DOCTYPE html><html><head><title>b</title></head><body>"
            "<p>Iframe 1 content: embedded text</p></body></html>",
        )
        combined = await page.evaluate("() => document.body.innerHTML")
        assert "Iframe 1 content" in combined, (
            f"benign content regression: {combined!r}"
        )
        assert "extracted-iframe-content-0" in combined, (
            "replacement div class missing for benign content"
        )
    finally:
        await context.close()
