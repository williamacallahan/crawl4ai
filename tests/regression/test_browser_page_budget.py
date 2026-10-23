"""Deadline behavior without launching a browser or mocking renderer internals."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.browser_budget import page_deadline, within_page_budget


@pytest.mark.asyncio
async def test_budget_cancels_a_blocked_operation():
    stopped = asyncio.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    token = page_deadline.set(asyncio.get_running_loop().time() + 0.05)
    task = asyncio.create_task(within_page_budget(blocked()))
    try:
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "The page budget must bound work that never completes"
        with pytest.raises(asyncio.TimeoutError):
            await task
        assert stopped.is_set(), "Timed-out work must be cancelled before returning"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        page_deadline.reset(token)


@pytest.mark.asyncio
async def test_sequential_operations_share_an_absolute_deadline():
    loop = asyncio.get_running_loop()
    token = page_deadline.set(loop.time() + 0.05)
    second_started = asyncio.Event()

    async def second_operation():
        second_started.set()
        return "finished"

    try:
        assert await within_page_budget(asyncio.sleep(0, result="first")) == "first"
        # Time between renderer calls also spends the crawl's budget.
        await asyncio.sleep(0.06)
        with pytest.raises(asyncio.TimeoutError):
            await within_page_budget(second_operation())
        assert not second_started.is_set()
    finally:
        page_deadline.reset(token)


@pytest.mark.asyncio
async def test_concurrent_crawls_keep_independent_deadlines():
    short_started = asyncio.Event()
    long_started = asyncio.Event()
    release_long = asyncio.Event()

    async def crawl(seconds, started, release):
        token = page_deadline.set(asyncio.get_running_loop().time() + seconds)

        async def operation():
            started.set()
            await release.wait()
            return "completed"

        try:
            return await within_page_budget(operation())
        finally:
            page_deadline.reset(token)

    parent_deadline = page_deadline.get()
    short = asyncio.create_task(crawl(0.05, short_started, asyncio.Event()))
    long = asyncio.create_task(crawl(2, long_started, release_long))
    try:
        await asyncio.wait_for(asyncio.gather(short_started.wait(), long_started.wait()), 1)
        done, _ = await asyncio.wait({short}, timeout=1)
        assert short in done, "The shorter crawl must exhaust its own budget"
        with pytest.raises(asyncio.TimeoutError):
            await short
        assert not long.done(), "One crawl timing out must not cancel another"
        release_long.set()
        assert await asyncio.wait_for(long, 1) == "completed"
        assert page_deadline.get() == parent_deadline
    finally:
        for task in (short, long):
            task.cancel()
        await asyncio.gather(short, long, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_deadline", [False, True])
async def test_caller_cancellation_reaches_the_operation(has_deadline):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    deadline = asyncio.get_running_loop().time() + 10 if has_deadline else None
    token = page_deadline.set(deadline)
    task = asyncio.create_task(within_page_budget(operation()))
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        page_deadline.reset(token)


@pytest.mark.asyncio
async def test_operation_without_a_deadline_can_complete():
    token = page_deadline.set(None)
    try:
        assert await within_page_budget(asyncio.sleep(0, result="content")) == "content"
    finally:
        page_deadline.reset(token)


def _scan_strategy(*, navigation_delay=0):
    """Keep the real scan/adapter pipeline; fake only Playwright page calls."""
    page = MagicMock()
    page.url = "https://example.test/"
    page.viewport_size = {"width": 800, "height": 600}
    page.scroll_count = 0
    page.scan_cancelled = asyncio.Event()
    page.capture_remaining = None
    page.wait_for_selector = AsyncMock(return_value=True)
    page.context.browser.contexts = []
    page.context.new_cdp_session = AsyncMock(
        return_value=MagicMock(send=AsyncMock(), detach=AsyncMock())
    )

    async def navigate(*_args, **_kwargs):
        if navigation_delay:
            await asyncio.sleep(navigation_delay)
        return None

    async def evaluate(expression):
        if "window.scrollTo(" in expression:
            page.scroll_count += 1
            return {"success": True}
        if "const {scrollWidth, scrollHeight}" in expression:
            return {"width": 800, "height": 2400}
        return True  # The body visibility check succeeds.

    async def render_delay(milliseconds):
        try:
            await asyncio.sleep(milliseconds / 1000)
        except asyncio.CancelledError:
            page.scan_cancelled.set()
            raise

    async def content():
        deadline = page_deadline.get()
        if deadline is not None:
            page.capture_remaining = deadline - asyncio.get_running_loop().time()
        body = "partial scroll" if page.scroll_count else "initial content"
        return f"<html><body>{body}</body></html>"

    page.goto = AsyncMock(side_effect=navigate)
    page.evaluate = AsyncMock(side_effect=evaluate)
    page.wait_for_timeout = AsyncMock(side_effect=render_delay)
    page.content = AsyncMock(side_effect=content)
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True, text_mode=True),
        logger=MagicMock(),
    )
    strategy.browser_manager = MagicMock()
    strategy.browser_manager.get_page = AsyncMock(return_value=(page, page.context))
    strategy.browser_manager.release_page_with_context = AsyncMock()
    return strategy, page


def _scan_warnings(strategy):
    return [
        call.kwargs
        for call in strategy.logger.warning.call_args_list
        if call.kwargs.get("tag") == "PAGE_SCAN"
    ]


@pytest.mark.asyncio
async def test_slow_full_page_scan_returns_partial_html_before_page_deadline():
    strategy, page = _scan_strategy()
    previous_deadline = page_deadline.get()

    result = await strategy._crawl_web(
        page.url,
        CrawlerRunConfig(
            page_timeout=2250,
            scan_full_page=True,
            scroll_delay=10,
            delay_before_return_html=0,
        ),
    )

    assert result.status_code == 200
    assert result.html == "<html><body>partial scroll</body></html>"
    assert page.scan_cancelled.is_set()
    assert page.scroll_count == 1
    assert page.capture_remaining >= 1.9
    assert len(_scan_warnings(strategy)) == 1
    strategy.browser_manager.release_page_with_context.assert_awaited_once_with(page)
    assert page_deadline.get() == previous_deadline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page_timeout", "remaining", "reserve"),
    [(10_000, 2.15, 2.0), (30_000, 3.15, 3.0)],
)
async def test_full_page_scan_uses_remaining_absolute_budget(
    page_timeout, remaining, reserve
):
    strategy, page = _scan_strategy()
    config = CrawlerRunConfig(
        page_timeout=page_timeout,
        scan_full_page=True,
        scroll_delay=10,
        delay_before_return_html=0,
    )
    # Model admission/navigation already spending most of a longer crawl's
    # budget, without waiting tens of seconds before reaching the scan.
    token = page_deadline.set(asyncio.get_running_loop().time() + remaining)
    try:
        result = await within_page_budget(strategy._crawl_web_impl(page.url, config))
    finally:
        page_deadline.reset(token)

    assert result.status_code == 200
    assert result.html == "<html><body>partial scroll</body></html>"
    assert page.scan_cancelled.is_set()
    assert page.capture_remaining >= reserve - 0.1
    warnings = _scan_warnings(strategy)
    assert len(warnings) == 1
    assert 0 < warnings[0]["params"]["timeout"] <= 0.15


@pytest.mark.asyncio
async def test_navigation_spending_most_deadline_skips_scan_and_returns_html():
    strategy, page = _scan_strategy(navigation_delay=0.4)

    result = await strategy._crawl_web(
        page.url,
        CrawlerRunConfig(
            page_timeout=600,
            scan_full_page=True,
            scroll_delay=10,
            delay_before_return_html=0,
        ),
    )

    assert result.status_code == 200
    assert result.html == "<html><body>initial content</body></html>"
    assert page.capture_remaining > 0
    assert page.scroll_count == 0
    page.wait_for_timeout.assert_not_awaited()
    assert _scan_warnings(strategy)[0]["params"]["timeout"] == 0
    strategy.browser_manager.release_page_with_context.assert_awaited_once_with(page)


@pytest.mark.asyncio
async def test_full_page_scan_without_deadline_keeps_configured_timeout():
    strategy, page = _scan_strategy()
    token = page_deadline.set(None)
    try:
        result = await asyncio.wait_for(
            strategy._crawl_web_impl(
                page.url,
                CrawlerRunConfig(
                    page_timeout=50,
                    scan_full_page=True,
                    scroll_delay=10,
                    delay_before_return_html=0,
                ),
            ),
            timeout=1,
        )
    finally:
        page_deadline.reset(token)

    assert result.status_code == 200
    assert result.html == "<html><body>partial scroll</body></html>"
    assert page.scan_cancelled.is_set()
    assert page.capture_remaining is None
    assert _scan_warnings(strategy)[0]["params"]["timeout"] == 0.05


def test_page_limits_survive_serialization_and_clone():
    original = CrawlerRunConfig(max_iframes=3, page_termination_timeout=0.25)
    serialized = json.loads(json.dumps(original.dump()))
    restored = CrawlerRunConfig.load(serialized)
    for config in (original.clone(), restored):
        assert config.max_iframes == 3
        assert config.page_termination_timeout == 0.25
    updated = original.clone(max_iframes=1, page_termination_timeout=0.5)
    assert updated.max_iframes == 1
    assert updated.page_termination_timeout == 0.5
    assert original.max_iframes == 3
    assert original.page_termination_timeout == 0.25


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_iframe_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_iframes"):
        CrawlerRunConfig(max_iframes=value)


@pytest.mark.parametrize(
    "value", [0, -1, True, False, "1", None, float("nan"), float("inf"), -float("inf")]
)
def test_termination_timeout_requires_a_positive_number(value):
    with pytest.raises(ValueError, match="page_termination_timeout"):
        CrawlerRunConfig(page_termination_timeout=value)
