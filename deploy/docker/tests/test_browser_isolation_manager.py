"""BrowserManager allocation and disposal contracts, without a browser process."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import (
    BrowserManager,
    ManagedBrowser,
    PLAYWRIGHT_DISABLED_FEATURES,
)
import crawl4ai.browser_manager as browser_manager_module


class Page:
    def __init__(self, context):
        self.context = context
        self.closed = False
        self.close = AsyncMock(side_effect=self._close)
        self.evaluate = AsyncMock()

    async def _close(self):
        self.closed = True

    def is_closed(self):
        return self.closed


class Context:
    def __init__(self):
        self.pages = []
        self.closed = False
        self.handlers = {}
        self.page_error = None
        self.page_gate = None
        self.page_created = asyncio.Event()
        self.route = AsyncMock()
        self.close = AsyncMock(side_effect=self._close)

    def on(self, event, handler):
        self.handlers[event] = handler

    def remove_listener(self, event, handler):
        if self.handlers.get(event) is handler:
            self.handlers.pop(event)

    def emit_page(self, page):
        self.pages.append(page)
        self.handlers["page"](page)

    async def new_page(self):
        if self.page_error:
            raise self.page_error
        page = Page(self)
        self.emit_page(page)
        self.page_created.set()
        if self.page_gate:
            await self.page_gate.wait()
        return page

    async def _close(self):
        self.closed = True
        for page in self.pages:
            if not page.closed:
                await page.close()


class Browser:
    def __init__(self):
        self.contexts = []
        self.next_context = None
        self.closed = False

    async def close(self):
        self.closed = True
        for context in self.contexts:
            if not context.closed:
                await context.close()

    async def new_context(self, **kwargs):
        context = self.next_context or Context()
        self.next_context = None
        self.contexts.append(context)
        return context


def manager(**config):
    instance = BrowserManager(BrowserConfig(headless=True, sleep_on_close=False, **config))
    instance.browser = Browser()
    instance.setup_context = AsyncMock()

    async def start():
        instance.browser = Browser()
        await instance._started_browser()

    instance.start = AsyncMock(side_effect=start)
    return instance


@pytest.mark.asyncio
async def test_same_config_crawls_own_contexts_and_release_them():
    instance = manager(use_managed_browser=True)
    first, second = await asyncio.gather(
        instance.get_page(CrawlerRunConfig()),
        instance.get_page(CrawlerRunConfig()),
    )
    assert first[1] is not second[1]
    assert not first[0].closed and not second[0].closed
    await instance.release_page_with_context(first[0])
    assert first[0].closed and first[1].closed
    assert not second[0].closed and not second[1].closed
    await instance.release_page_with_context(second[0])
    assert not instance.contexts_by_config
    assert not instance._active_acquisitions
    await instance._drain_background_tasks()


@pytest.mark.asyncio
async def test_session_is_reused_and_disposed_only_when_killed():
    instance = manager()
    config = CrawlerRunConfig(session_id="retained")
    first, second = await asyncio.gather(
        instance.get_page(config), instance.get_page(config)
    )
    assert first == second
    assert len(instance.browser.contexts) == 1
    await instance.release_page_with_context(first[0])
    assert not first[0].closed and not first[1].closed
    await instance.kill_session("retained")
    assert first[0].closed and first[1].closed
    assert not instance._active_acquisitions
    await instance._drain_background_tasks()


@pytest.mark.asyncio
async def test_cancelled_session_acquisition_is_not_a_browser_fault():
    instance = manager()
    config = CrawlerRunConfig(session_id="retained")
    page, context = await instance.get_page(config)
    evaluate_started = asyncio.Event()

    async def busy_evaluate(*args, **kwargs):
        evaluate_started.set()
        await asyncio.Event().wait()

    page.evaluate.side_effect = busy_evaluate
    task = asyncio.create_task(instance.get_page(config))
    await asyncio.wait_for(evaluate_started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # A client disconnect says nothing about the browser: no recycle, and
    # other crawls keep being admitted.
    assert instance.pool_recycle_reason is None
    page.evaluate.side_effect = None
    other_page, _ = await instance.get_page(CrawlerRunConfig())
    await instance.release_page_with_context(other_page)
    await instance.kill_session("retained")
    assert page.closed and context.closed
    assert not instance._active_acquisitions
    await instance._drain_background_tasks()


@pytest.mark.asyncio
async def test_page_failure_disposes_context_and_counts_attempt():
    instance = manager()
    context = Context()
    context.page_error = RuntimeError("page failed")
    instance.browser.next_context = context
    with pytest.raises(RuntimeError, match="page failed"):
        await instance.get_page(CrawlerRunConfig())
    assert context.closed
    assert not instance.contexts_by_config
    assert not instance._active_acquisitions
    assert instance._pages_served == 1


@pytest.mark.asyncio
async def test_setup_cancellation_disposes_partial_context():
    instance = manager()
    setup_started = asyncio.Event()

    async def setup(*args, **kwargs):
        setup_started.set()
        await asyncio.Event().wait()

    instance.setup_context = setup
    task = asyncio.create_task(instance.get_page(CrawlerRunConfig()))
    await setup_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert instance.browser.contexts[0].closed
    assert not instance.contexts_by_config
    assert not instance._active_acquisitions


@pytest.mark.asyncio
async def test_route_failure_closes_context_created_before_setup():
    instance = manager(avoid_css=True)
    context = Context()
    context.route.side_effect = RuntimeError("route failed")
    instance.browser.next_context = context
    with pytest.raises(RuntimeError, match="route failed"):
        await instance.get_page(CrawlerRunConfig())
    assert context.closed
    assert not instance._active_acquisitions


@pytest.mark.asyncio
async def test_popup_is_closed_while_exact_pending_crawl_page_is_allowed():
    instance = manager()
    context = Context()
    context.page_gate = asyncio.Event()
    instance.browser.next_context = context
    task = asyncio.create_task(instance.get_page(CrawlerRunConfig()))
    await context.page_created.wait()
    intended = context.pages[0]
    popup = Page(context)
    context.emit_page(popup)
    # Both events arrived during the same explicit allocation. Identity of its
    # returned result, rather than a context-wide creation flag, decides.
    context.page_gate.set()
    page, _ = await task
    await instance._drain_background_tasks()
    assert page is intended and not intended.closed
    assert popup.closed
    await instance.release_page_with_context(page)


@pytest.mark.asyncio
async def test_release_cancellation_finishes_page_and_context_cleanup():
    instance = manager()
    page, context = await instance.get_page(CrawlerRunConfig())
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    async def close_page():
        close_started.set()
        await finish_close.wait()
        page.closed = True

    page.close.side_effect = close_page
    task = asyncio.create_task(instance.release_page_with_context(page))
    await close_started.wait()
    task.cancel()
    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert page.closed and context.closed
    assert not instance._active_acquisitions


@pytest.mark.asyncio
async def test_page_close_timeout_drains_and_closes_context(monkeypatch):
    instance = manager()
    page, context = await instance.get_page(CrawlerRunConfig())
    other_page, other_context = await instance.get_page(CrawlerRunConfig())
    original_browser = instance.browser
    monkeypatch.setattr(browser_manager_module, "PAGE_CLOSE_SECONDS", 0.01)

    async def close_page():
        await asyncio.Event().wait()

    page.close.side_effect = close_page
    # Actual context shutdown owns its pages even when individual page.close
    # did not settle. Keep that boundary separate in this double.
    context.close.side_effect = None
    await instance.release_page_with_context(page)
    # The browser drains and restarts instead of closing under other crawls.
    await instance._pool_recycle_task
    assert instance.pool_recycle_reason == "page_close_timeout"
    assert instance._recycle_due
    assert not instance.pool_recycle_requested
    assert not instance._recycle_done.is_set()
    assert not other_page.closed and not other_context.closed
    assert not original_browser.closed
    instance.start.assert_not_awaited()
    context.close.assert_awaited_once()
    waiter = asyncio.create_task(instance.get_page(CrawlerRunConfig()))
    await asyncio.sleep(0)
    assert not waiter.done()
    await instance.release_page_with_context(other_page)
    next_page, _ = await asyncio.wait_for(waiter, 1)
    await instance._recycle_task
    assert original_browser.closed and instance.browser is not original_browser
    instance.start.assert_awaited_once()
    await instance.release_page_with_context(next_page)
    assert not instance.contexts_by_config
    assert not instance._active_acquisitions


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "retained"])
async def test_unhealthy_browser_restarts_at_drain_deadline_even_with_a_session(
    monkeypatch, session_id
):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MIN_SECONDS", 0.05)
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MAX_SECONDS", 0.05)
    instance = manager()
    config = CrawlerRunConfig(session_id=session_id)
    page, context = await instance.get_page(config)
    original_browser = instance.browser

    instance.request_pool_recycle("renderer unhealthy")
    assert not instance._recycle_done.is_set()
    await instance._pool_recycle_task
    assert not page.closed and not context.closed
    assert not original_browser.closed
    waiter = asyncio.create_task(instance.get_page(config))
    await asyncio.sleep(0)
    assert not waiter.done()

    next_page, next_context = await asyncio.wait_for(waiter, 1)
    await instance._recycle_task
    assert page.closed and context.closed and original_browser.closed
    assert next_page is not page and next_context is not context
    instance.start.assert_awaited_once()
    assert not instance._pool_recycle_pending
    # A late release from the expired generation cannot release the new page.
    await instance.release_page_with_context(page)
    assert not next_page.closed and len(instance._active_acquisitions) == 1
    if session_id:
        await instance.kill_session(session_id)
    else:
        await instance.release_page_with_context(next_page)
    assert not instance._active_acquisitions


@pytest.mark.asyncio
async def test_new_session_already_allocating_cannot_reopen_unhealthy_admission(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MIN_SECONDS", 0.05)
    monkeypatch.setattr(browser_manager_module, "RECYCLE_DRAIN_MAX_SECONDS", 0.05)
    instance = manager(max_pages_before_recycle=1)
    context = Context()
    context.page_gate = asyncio.Event()
    instance.browser.next_context = context
    allocation = asyncio.create_task(
        instance.get_page(CrawlerRunConfig(session_id="retained"))
    )
    await asyncio.wait_for(context.page_created.wait(), 1)
    instance.request_pool_recycle("renderer unhealthy")
    await instance._pool_recycle_task
    drain = instance._drain_deadline
    assert drain is not None

    context.page_gate.set()
    page, _ = await asyncio.wait_for(allocation, 1)
    assert not instance._recycle_done.is_set()
    assert instance._drain_deadline is drain and not drain.cancelled()
    assert not page.closed
    await asyncio.wait_for(drain, 1)
    await asyncio.wait_for(instance._recycle_task, 1)
    assert page.closed and context.closed
    assert not instance.sessions and not instance._active_acquisitions
    instance.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_closing_before_scheduled_health_drain_does_not_restart():
    instance = manager()
    page, _ = await instance.get_page(CrawlerRunConfig())
    instance.request_pool_recycle("renderer unhealthy")
    await instance.close()
    await instance._pool_recycle_task
    assert page.closed
    assert instance._closing and instance._recycle_done.is_set()
    instance.start.assert_not_awaited()


def test_launch_merges_default_and_custom_features_and_retains_ipc_protection():
    instance = manager(light_mode=True, extra_args=["--disable-features=CustomFeature"])
    kwargs = instance._build_browser_args()
    switches = [arg for arg in kwargs["args"] if arg.startswith("--disable-features=")]
    assert len(switches) == 1
    features = set(switches[0].split("=", 1)[1].split(","))
    assert set(PLAYWRIGHT_DISABLED_FEATURES) <= features
    assert {"OptimizationHints", "CustomFeature", "TranslateUI"} <= features
    assert "--disable-ipc-flooding-protection" not in kwargs["args"]
    assert "--disable-ipc-flooding-protection" in kwargs["ignore_default_args"]
    managed = ManagedBrowser.build_browser_flags(instance.config)
    managed_switches = [arg for arg in managed if arg.startswith("--disable-features=")]
    assert len(managed_switches) == 1
    assert set(PLAYWRIGHT_DISABLED_FEATURES) <= set(managed_switches[0].split("=", 1)[1].split(","))
    assert "--disable-ipc-flooding-protection" not in managed
