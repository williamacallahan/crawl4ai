"""Tests for BrowserManager._get_page_by_target_id.

The method implements the BrowserConfig.target_id feature: attaching Crawl4AI
to a specific pre-created browser tab identified by its CDP target id.

Regression coverage for the bug where _get_page_by_target_id compared against
``page._impl_obj._target_id`` (an attribute Playwright never exposes, making
the matching loop dead code) and then fell back to ``context.pages[0]``,
silently attaching the crawler to the wrong tab whenever target_id did not
happen to refer to the first page.

These tests cover two layers:

1. Mock-based tests (no browser, no network) that pin the matching contract:
   the right page is returned, a missing/stale target_id yields None (never
   pages[0]), and CDP-session failures degrade to None instead of crashing.
2. A real-browser test that opens real Chromium pages, reads their real CDP
   targetIds, and asserts the method selects the correct Page object.
"""

import asyncio
import collections
import time

import pytest
from playwright.async_api import async_playwright

from crawl4ai.browser_manager import (
    BrowserManager,
    TARGET_ID_DISCOVERY_SECONDS,
)


# ---------------------------------------------------------------------------
# Fakes for the mock-based tests.
# ---------------------------------------------------------------------------

class FakePage:
    """A stand-in for a Playwright Page. Carries the target id that a CDP
    Target.getTargetInfo call would report for it. Deliberately has NO
    ``_impl_obj`` attribute so any regression that reintroduces a probe of
    ``page._impl_obj._target_id`` cannot accidentally match."""

    def __init__(self, name, target_id):
        self.name = name
        self.target_id = target_id

    def __repr__(self):
        return f"FakePage(name={self.name!r}, target_id={self.target_id!r})"


class FakeCDPSession:
    """A stand-in for the object returned by BrowserContext.new_cdp_session.
    Only the methods the code-under-test touches are implemented."""

    def __init__(self, target_info_factory):
        self._factory = target_info_factory
        self.send_calls = 0
        self.detached = False

    async def send(self, method, *args, **kwargs):
        self.send_calls += 1
        assert method == "Target.getTargetInfo", method
        return self._factory()

    async def detach(self):
        self.detached = True


class FakeContext:
    """A stand-in for playwright.BrowserContext. ``pages`` is a plain mutable
    list so tests can simulate late target surfacing by appending to it."""

    def __init__(self, pages=None, target_info_factory=None):
        self.pages = list(pages) if pages is not None else []

        # target_info_factory(page) -> dict returned by Target.getTargetInfo
        if target_info_factory is not None:
            self._factory = target_info_factory
        else:
            self._factory = lambda page: {
                "targetInfo": {"targetId": page.target_id, "type": "page"}
            }
        self.new_cdp_session_calls = []

    async def new_cdp_session(self, page):
        self.new_cdp_session_calls.append(page)
        return FakeCDPSession(lambda: self._factory(page))


def _make_manager(wait_seconds=0):
    """Build a BrowserManager wired only with what _get_page_by_target_id
    needs (logger + the optional wait override). __new__ bypasses __init__ on
    purpose: the method under test touches neither the browser nor the
    default context, and we want an isolated unit test."""
    manager = BrowserManager.__new__(BrowserManager)
    manager.logger = None
    manager._target_id_wait_seconds = wait_seconds
    return manager


# ---------------------------------------------------------------------------
# Mock-based matching contract tests.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_matching_page_not_first_page():
    """In a multi-page context the page whose targetId matches is returned,
    not context.pages[0]. This is the core regression: the old code returned
    pages[0] regardless of target_id."""
    page_a = FakePage("A", "tid-A")
    page_b = FakePage("B", "tid-B")
    page_c = FakePage("C", "tid-C")
    ctx = FakeContext([page_a, page_b, page_c])
    manager = _make_manager(wait_seconds=0)

    result = await manager._get_page_by_target_id(ctx, "tid-B")

    assert result is page_b, f"expected page B, got {result!r}"
    assert result is not page_a, "must not return the first page when it does not match"


@pytest.mark.asyncio
async def test_returns_first_page_when_first_page_matches():
    """When the first page is the match, it is returned (sanity)."""
    page_a = FakePage("A", "tid-A")
    page_b = FakePage("B", "tid-B")
    ctx = FakeContext([page_a, page_b])
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-A") is page_a


@pytest.mark.asyncio
async def test_returns_last_page_when_last_page_matches():
    """Matching the last page in the list proves the loop scans the whole
    list rather than only the first page."""
    page_a = FakePage("A", "tid-A")
    page_b = FakePage("B", "tid-B")
    page_c = FakePage("C", "tid-C")
    ctx = FakeContext([page_a, page_b, page_c])
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-C") is page_c


@pytest.mark.asyncio
async def test_returns_none_when_no_page_matches():
    """A target_id that no page has must return None, NOT context.pages[0].
    The old code silently returned pages[0] here (Branch C)."""
    page_a = FakePage("A", "tid-A")
    page_b = FakePage("B", "tid-B")
    ctx = FakeContext([page_a, page_b])
    manager = _make_manager(wait_seconds=0)

    result = await manager._get_page_by_target_id(ctx, "does-not-exist")

    assert result is None, f"expected None for unmatched target_id, got {result!r}"


@pytest.mark.asyncio
async def test_returns_none_for_empty_context():
    """An empty context.pages yields None, not an exception."""
    ctx = FakeContext([])
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-X") is None


@pytest.mark.asyncio
async def test_non_page_target_type_not_matched():
    """A target whose CDP type is not 'page' (e.g. shared_worker) is ignored,
    so the method does not return a non-page target as if it were a Page."""
    page = FakePage("worker", "tid-W")
    ctx = FakeContext(
        [page],
        target_info_factory=lambda p: {
            "targetInfo": {"targetId": p.target_id, "type": "shared_worker"}
        },
    )
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-W") is None


@pytest.mark.asyncio
async def test_wait_seconds_zero_returns_immediately_on_miss():
    """With _target_id_wait_seconds=0 a miss resolves immediately (no
    polling), so callers that opt out of the discovery window pay no latency
    on unmatched ids."""
    page_a = FakePage("A", "tid-A")
    ctx = FakeContext([page_a])
    manager = _make_manager(wait_seconds=0)

    start = time.monotonic()
    result = await manager._get_page_by_target_id(ctx, "nope")
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed < 0.05, f"wait_seconds=0 should not poll, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_discovery_window_finds_late_surfacing_page():
    """A target created via raw CDP may not be in context.pages immediately.
    While the discovery window is open, a page that appears after the first
    scan must be matched once it surfaces."""
    early = FakePage("early", "tid-EARLY")
    late = FakePage("late", "tid-LATE")
    ctx = FakeContext([early])
    manager = _make_manager(wait_seconds=2.0)

    async def append_late_after_delay():
        await asyncio.sleep(0.15)
        ctx.pages.append(late)

    asyncio.create_task(append_late_after_delay())

    result = await manager._get_page_by_target_id(ctx, "tid-LATE")

    assert result is late, f"expected the late page to be discovered, got {result!r}"


@pytest.mark.asyncio
async def test_discovery_window_expires_to_none_when_page_never_surfaces():
    """If the page never surfaces within the window, the method returns None
    rather than pages[0]."""
    page_a = FakePage("A", "tid-A")
    ctx = FakeContext([page_a])
    manager = _make_manager(wait_seconds=0.3)

    start = time.monotonic()
    result = await manager._get_page_by_target_id(ctx, "tid-MISSING")
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed >= 0.25, f"expected to wait out the window, returned in {elapsed:.3f}s"
    assert elapsed < 1.0, f"should not exceed the window by much, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_target_id_queried_once_per_page_within_call():
    """The poll loop caches each page's targetId, so a single
    _get_page_by_target_id call does not open a new CDP session for the same
    page more than once. Guards against redundant CDP churn during the
    discovery window."""
    page_a = FakePage("A", "tid-A")
    page_b = FakePage("B", "tid-B")
    ctx = FakeContext([page_a, page_b])
    manager = _make_manager(wait_seconds=0.3)

    await manager._get_page_by_target_id(ctx, "tid-MISSING")

    counts = collections.Counter(id(p) for p in ctx.new_cdp_session_calls)
    assert counts[id(page_a)] == 1, f"page A queried {counts[id(page_a)]} times"
    assert counts[id(page_b)] == 1, f"page B queried {counts[id(page_b)]} times"


@pytest.mark.asyncio
async def test_new_cdp_session_failure_returns_none():
    """If context.new_cdp_session raises for every page, the method returns
    None instead of propagating or falling back to pages[0]."""
    page_a = FakePage("A", "tid-A")
    ctx = FakeContext([page_a])

    async def boom(page):
        raise RuntimeError("cdp unavailable")

    ctx.new_cdp_session = boom
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-A") is None


@pytest.mark.asyncio
async def test_send_failure_returns_none():
    """If Target.getTargetInfo fails for every page, the method returns None
    rather than falling back to pages[0]."""

    class FailingSession:
        async def send(self, method, *a, **kw):
            raise RuntimeError("send failed")

        async def detach(self):
            pass

    page_a = FakePage("A", "tid-A")
    ctx = FakeContext([page_a])

    async def failing_session(page):
        return FailingSession()

    ctx.new_cdp_session = failing_session
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-A") is None


@pytest.mark.asyncio
async def test_malformed_get_target_info_returns_none():
    """If Target.getTargetInfo returns a result without a targetInfo dict, the
    page is treated as non-matching and the call falls through to None."""
    page_a = FakePage("A", "tid-A")
    ctx = FakeContext(
        [page_a],
        target_info_factory=lambda p: {"unexpected": "shape"},
    )
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-A") is None


@pytest.mark.asyncio
async def test_default_discovery_seconds_is_bounded_and_positive():
    """The module-level default window must be a small positive number so a
    missing target id fails fast instead of hanging the crawler."""
    assert isinstance(TARGET_ID_DISCOVERY_SECONDS, (int, float))
    assert 0 < TARGET_ID_DISCOVERY_SECONDS <= 10


@pytest.mark.asyncio
async def test_does_not_probe_private_playwright_target_id():
    """A FakePage has no ``_impl_obj`` attribute. If the method regressed to
    probing ``page._impl_obj._target_id`` again, it would AttributeError.
    Passing a match and an unrelated first page proves the code path does not
    rely on the (non-existent) Playwright attribute."""
    first = FakePage("first", "tid-FIRST")  # has no _impl_obj at all
    wanted = FakePage("wanted", "tid-WANTED")
    ctx = FakeContext([first, wanted])
    manager = _make_manager(wait_seconds=0)

    assert await manager._get_page_by_target_id(ctx, "tid-WANTED") is wanted


# ---------------------------------------------------------------------------
# Real-browser test (requires installed Chromium; no network needed).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_browser_matches_correct_page_in_multi_page_context():
    """End-to-end against real Chromium: open two pages, read each one's real
    CDP targetId, and assert _get_page_by_target_id selects the correct Page
    object for each. Also asserts an unmatched id returns None instead of the
    first page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        try:
            context = await browser.new_context()
            page1 = await context.new_page()
            page2 = await context.new_page()
            page3 = await context.new_page()

            async def target_id_of(page):
                session = await context.new_cdp_session(page)
                try:
                    info = await session.send("Target.getTargetInfo")
                finally:
                    await session.detach()
                return info["targetInfo"]["targetId"]

            tid1 = await target_id_of(page1)
            tid2 = await target_id_of(page2)
            tid3 = await target_id_of(page3)
            assert len({tid1, tid2, tid3}) == 3, "pages must have distinct target ids"

            manager = _make_manager(wait_seconds=0)

            assert await manager._get_page_by_target_id(context, tid2) is page2
            assert await manager._get_page_by_target_id(context, tid1) is page1
            assert await manager._get_page_by_target_id(context, tid3) is page3
            assert await manager._get_page_by_target_id(context, "NOT-A-TARGET") is None
        finally:
            await browser.close()
