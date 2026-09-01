"""
Regression tests guarding CDP session cleanup in ``async_crawler_strategy.py``.

Both CDP call sites (viewport adjustment via ``Emulation.setDeviceMetricsOverride``
and MHTML capture via ``Page.captureSnapshot``) must detach their CDP session
in a ``finally`` block so the session is released even when ``send()`` raises.
The leak pattern (``detach()`` inside the ``try``) has regressed more than once
in this codebase, so these tests assert the exact detach contract for both the
success and exception paths on both code paths.
"""

import pytest
import pytest_asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


# ---------------------------------------------------------------------------
# Test doubles for the CDP session / page surface (no browser required)
# ---------------------------------------------------------------------------


class FakeCDPSession:
    """Minimal stand-in for a Playwright ``CDPSession``.

    Records ``send``/``detach`` calls so tests can assert that ``detach`` runs
    even when ``send`` raises. ``send_raises``/``detach_raises`` drive the
    exception paths that previously leaked the session.
    """

    def __init__(self, *, send_result=None, send_raises=None, detach_raises=None):
        self.send_result = send_result
        self.send_raises = send_raises
        self.detach_raises = detach_raises
        self.send_calls = []
        self.detach_calls = 0

    async def send(self, method, params=None):
        self.send_calls.append((method, params))
        if self.send_raises is not None:
            raise self.send_raises
        return self.send_result

    async def detach(self):
        self.detach_calls += 1
        if self.detach_raises is not None:
            raise self.detach_raises


class FakeContext:
    def __init__(self, cdp_session):
        self._cdp_session = cdp_session
        self.new_cdp_session_calls = 0

    async def new_cdp_session(self, page):
        self.new_cdp_session_calls += 1
        return self._cdp_session


class FakePage:
    def __init__(self, context):
        self.context = context

    async def wait_for_load_state(self, state, timeout=None):
        return None

    async def wait_for_timeout(self, timeout):
        return None

    async def evaluate(self, expression, arg=None):
        return None


def _make_strategy():
    # Construction is lightweight: it does not launch a browser, so we can
    # exercise capture_mhtml() in isolation with a fake page.
    return AsyncPlaywrightCrawlerStrategy(browser_config=BrowserConfig(headless=True))


# ---------------------------------------------------------------------------
# capture_mhtml() cleanup contract (deterministic, no browser)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_mhtml_success_detaches_cdp_session():
    """On success the CDP session is created, used, and detached exactly once."""
    cdp = FakeCDPSession(send_result={"data": "MIME-Version: 1.0"})
    page = FakePage(FakeContext(cdp))
    strategy = _make_strategy()

    result = await strategy.capture_mhtml(page)

    assert result == "MIME-Version: 1.0"
    assert page.context.new_cdp_session_calls == 1
    assert cdp.send_calls == [("Page.captureSnapshot", {"format": "mhtml"})]
    assert cdp.detach_calls == 1


@pytest.mark.asyncio
async def test_capture_mhtml_send_failure_still_detaches_cdp_session():
    """If cdp_session.send() raises, detach() must still run.

    Before the fix, detach() lived inside the try block and was skipped when
    send() raised, leaking the session. With try/finally detach always runs.
    """
    cdp = FakeCDPSession(send_raises=RuntimeError("Target closed"))
    page = FakePage(FakeContext(cdp))
    strategy = _make_strategy()

    result = await strategy.capture_mhtml(page)

    assert result is None  # capture_mhtml swallows the error and returns None
    assert page.context.new_cdp_session_calls == 1
    assert cdp.send_calls == [("Page.captureSnapshot", {"format": "mhtml"})]
    assert cdp.detach_calls == 1


# ---------------------------------------------------------------------------
# Real-browser fixture and end-to-end leak guards
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def crawler():
    """A started AsyncWebCrawler backed by a real headless browser."""
    c = AsyncWebCrawler(config=BrowserConfig(headless=True))
    await c.start()
    try:
        yield c
    finally:
        await c.close()


def _patch_new_cdp_session_on_context(crawler, fake_cdp):
    """Install an on_page_context_created hook that swaps the real
    ``context.new_cdp_session`` with one returning ``fake_cdp``.

    This lets us force and observe the CDP exception path while keeping the
    rest of the browser/crawl real. Only the CDP session is a test double.
    """

    async def hook(page, context, config):
        async def fake_new_cdp_session(_page):
            return fake_cdp

        context.new_cdp_session = fake_new_cdp_session

    crawler.crawler_strategy.set_hook("on_page_context_created", hook)


@pytest.mark.asyncio
async def test_viewport_adjustment_send_failure_detaches_cdp_session(
    crawler, local_server
):
    """When Emulation.setDeviceMetricsOverride raises, the CDP session is
    still detached, the viewport error is logged, and the crawl still succeeds.
    """
    fake_cdp = FakeCDPSession(send_raises=RuntimeError("Emulation failed"))
    _patch_new_cdp_session_on_context(crawler, fake_cdp)

    run_config = CrawlerRunConfig(adjust_viewport_to_content=True)
    result = await crawler.arun(f"{local_server}/", config=run_config)

    assert result.success is True, f"crawl should survive viewport failure: {result.error_message}"
    assert "Welcome to the Crawl4AI Test Site" in result.html
    assert any(
        method == "Emulation.setDeviceMetricsOverride"
        for method, _ in fake_cdp.send_calls
    )
    # The session was detached despite the failure (no leak).
    assert fake_cdp.detach_calls == 1


@pytest.mark.asyncio
async def test_session_based_viewport_reuse_no_leak_on_repeated_failures(
    crawler, local_server
):
    """Across repeated session-based crawls where the CDP command fails every
    time, detach() must run on every crawl so sessions never accumulate/leak
    on the reused page (session pages are not closed between crawls).
    """
    fake_cdp = FakeCDPSession(send_raises=RuntimeError("Emulation failed"))
    _patch_new_cdp_session_on_context(crawler, fake_cdp)

    run_config = CrawlerRunConfig(
        session_id="reuse-sess-2", adjust_viewport_to_content=True
    )

    for _ in range(3):
        r = await crawler.arun(f"{local_server}/", config=run_config)
        assert r.success is True, f"crawl failed: {r.error_message}"

    # One Emulation attempt per crawl, and one detach per crawl -> no leak.
    emulation_attempts = [
        method for method, _ in fake_cdp.send_calls
        if method == "Emulation.setDeviceMetricsOverride"
    ]
    assert len(emulation_attempts) == 3
    assert fake_cdp.detach_calls == 3
