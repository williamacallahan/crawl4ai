"""Browser-free residual lifecycle regressions for issue 10."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai import browser_manager as browser_manager_module
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.browser_manager import BrowserManager


class _Page:
    def __init__(self, context=None, stop_error=None):
        self.context = context
        self.closed = False
        self.stop_error = stop_error

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True

    async def evaluate(self, script):
        if script == "window.stop()" and self.stop_error:
            raise self.stop_error


class _Context:
    def __init__(self, *, new_page_started=None, new_page_release=None):
        self.pages = []
        self.closed = False
        self.new_page_started = new_page_started
        self.new_page_release = new_page_release

    async def new_page(self):
        if self.new_page_started:
            self.new_page_started.set()
        if self.new_page_release:
            await self.new_page_release.wait()
        page = _Page(self)
        self.pages.append(page)
        return page

    async def close(self):
        self.closed = True
        for page in self.pages:
            await page.close()


class _FailingContext(_Context):
    async def new_page(self):
        if self.new_page_started:
            self.new_page_started.set()
        if self.new_page_release:
            await self.new_page_release.wait()
        raise RuntimeError("page allocation failed")


class _BlockingStealth:
    def __init__(self, started, release):
        self.started = started
        self.release = release

    async def apply_stealth(self, page):
        self.started.set()
        await self.release.wait()


class _CancellingConsoleAdapter:
    async def setup_console_capture(self, *_args):
        return None

    async def setup_error_capture(self, *_args):
        return None

    async def cleanup_console_capture(self, *_args):
        raise asyncio.CancelledError


def _manager(config=None):
    manager = BrowserManager(config or BrowserConfig(headless=True), logger=None)
    manager.managed_browser = None
    manager._browser_endpoint_key = f"instance:{id(manager)}"
    return manager


@pytest.fixture(autouse=True)
def reset_global_pages():
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    BrowserManager._global_pages_lock_loop = None
    yield
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    BrowserManager._global_pages_lock_loop = None


@pytest.mark.asyncio
async def test_page_registry_prunes_on_release_and_close():
    manager = _manager(BrowserConfig(use_managed_browser=True, headless=True))
    page = _Page()

    async with BrowserManager._get_global_lock():
        manager._mark_page_in_use_locked(page)
    assert BrowserManager._global_pages_in_use[manager._browser_endpoint_key] == {page}

    manager.release_page(page)
    assert manager._browser_endpoint_key not in BrowserManager._global_pages_in_use

    async with BrowserManager._get_global_lock():
        manager._mark_page_in_use_locked(page)
    await manager._release_page_from_use(page)
    assert manager._browser_endpoint_key not in BrowserManager._global_pages_in_use

    async with BrowserManager._get_global_lock():
        manager._mark_page_in_use_locked(page)
    await manager.close()
    assert manager._browser_endpoint_key not in BrowserManager._global_pages_in_use


@pytest.mark.asyncio
async def test_managed_page_is_reused_after_release():
    manager = _manager(BrowserConfig(use_managed_browser=True, headless=True))
    manager.default_context = _Context()

    first, _ = await manager.get_page(CrawlerRunConfig())
    await manager.release_page_with_context(first)
    second, _ = await manager.get_page(CrawlerRunConfig())

    assert second is first
    await manager.release_page_with_context(second)


@pytest.mark.asyncio
async def test_closing_one_shared_endpoint_manager_preserves_the_other_page():
    config = BrowserConfig(use_managed_browser=True, headless=True)
    first, second = _manager(config), _manager(config)
    first._browser_endpoint_key = second._browser_endpoint_key = "cdp:http://shared:9222"
    first_page, second_page = _Page(), _Page()

    async with BrowserManager._get_global_lock():
        first._mark_page_in_use_locked(first_page)
        second._mark_page_in_use_locked(second_page)

    await first.close()
    assert BrowserManager._global_pages_in_use[first._browser_endpoint_key] == {second_page}

    await second._release_page_from_use(second_page)
    assert first._browser_endpoint_key not in BrowserManager._global_pages_in_use


@pytest.mark.asyncio
async def test_cancelled_new_page_restores_its_refcount_and_closes_the_page():
    started, release = asyncio.Event(), asyncio.Event()
    context = _Context(new_page_started=started, new_page_release=release)
    manager = _manager(BrowserConfig(headless=True))

    async def create_context(_):
        return context

    async def setup_context(*_):
        return None

    manager.create_browser_context = create_context
    manager.setup_context = setup_context
    task = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.pages[0].closed
    assert not manager._page_to_sig
    assert not manager._page_to_admission
    assert sum(manager._context_refcounts.values()) == 0
    assert not manager._active_acquisitions


@pytest.mark.asyncio
async def test_cancelled_new_page_keeps_cancellation_when_allocation_fails():
    started, release = asyncio.Event(), asyncio.Event()
    context = _FailingContext(new_page_started=started, new_page_release=release)
    manager = _manager(BrowserConfig(headless=True))

    async def create_context(_):
        return context

    async def setup_context(*_):
        return None

    manager.create_browser_context = create_context
    manager.setup_context = setup_context
    task = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not manager._page_to_admission
    assert not manager._active_acquisitions


@pytest.mark.asyncio
async def test_cancelled_stealth_restores_its_refcount_and_closes_the_page():
    started, release = asyncio.Event(), asyncio.Event()
    context = _Context()
    manager = _manager(BrowserConfig(headless=True))
    manager._stealth_adapter = _BlockingStealth(started, release)

    async def create_context(_):
        return context

    async def setup_context(*_):
        return None

    manager.create_browser_context = create_context
    manager.setup_context = setup_context
    task = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.pages[0].closed
    assert not manager._page_to_sig
    assert not manager._page_to_admission
    assert sum(manager._context_refcounts.values()) == 0
    assert not manager._active_acquisitions
    release.set()


@pytest.mark.asyncio
async def test_close_rejects_an_acquisition_that_was_already_in_flight():
    started, release = asyncio.Event(), asyncio.Event()
    context = _Context(new_page_started=started, new_page_release=release)
    manager = _manager(BrowserConfig(headless=True))

    async def create_context(_):
        return context

    async def setup_context(*_):
        return None

    manager.create_browser_context = create_context
    manager.setup_context = setup_context
    acquisition = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
    await started.wait()
    await manager.close()
    release.set()

    with pytest.raises(RuntimeError, match="closed"):
        await acquisition

    assert context.pages[0].closed
    assert not manager._active_acquisitions


@pytest.mark.asyncio
async def test_window_stop_cancellation_kills_only_a_new_session():
    page = _Page(stop_error=asyncio.CancelledError())
    context = _Context()
    context.pages.append(page)
    page.context = context
    manager = _manager(BrowserConfig(use_managed_browser=True))
    manager.default_context = context

    with pytest.raises(asyncio.CancelledError):
        await manager.get_page(CrawlerRunConfig(session_id="session"))

    assert "session" not in manager.sessions
    assert page.closed
    assert not manager._active_acquisitions

    existing_page = _Page(stop_error=asyncio.CancelledError())
    existing = _manager(BrowserConfig(use_managed_browser=True))
    existing.sessions["session"] = (_Context(), existing_page, 0)
    with pytest.raises(asyncio.CancelledError):
        await existing.get_page(CrawlerRunConfig(session_id="session"))

    assert "session" in existing.sessions
    assert not existing_page.closed


@pytest.mark.asyncio
async def test_repeated_cancellation_during_console_cleanup_still_releases_page():
    strategy = AsyncPlaywrightCrawlerStrategy(
        browser_config=BrowserConfig(headless=True),
        browser_adapter=_CancellingConsoleAdapter(),
    )
    page = MagicMock()
    page.goto = AsyncMock(side_effect=asyncio.CancelledError)
    page.context.browser.contexts = []
    strategy.browser_manager = MagicMock()
    strategy.browser_manager.get_page = AsyncMock(
        return_value=(page, MagicMock())
    )
    strategy.browser_manager.release_page_with_context = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await strategy._crawl_web(
            "https://example.test",
            CrawlerRunConfig(capture_console_messages=True),
        )

    strategy.browser_manager.release_page_with_context.assert_awaited_once_with(page)


@pytest.mark.asyncio
async def test_recycle_closes_admission_then_restarts_before_waking_waiters():
    config = BrowserConfig(
        use_managed_browser=True,
        headless=True,
        max_pages_before_recycle=1,
    )
    manager = _manager(config)
    manager.default_context = _Context()
    original_browser = object()
    manager.browser = original_browser
    close_started, permit_restart, started = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def close(*, _for_recycle=False):
        assert _for_recycle
        close_started.set()
        await permit_restart.wait()

    async def start():
        manager.default_context = _Context()
        manager.browser = object()
        started.set()

    manager.close = close
    manager.start = start

    first, _ = await manager.get_page(CrawlerRunConfig())
    assert not manager._recycle_done.is_set()

    waiter = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
    await asyncio.sleep(0)
    assert not waiter.done(), "later acquisition passed a closed recycle barrier"

    await manager.release_page_with_context(first)
    await close_started.wait()
    assert not waiter.done(), "waiter resumed before full close/start"

    permit_restart.set()
    await started.wait()
    second, _ = await asyncio.wait_for(waiter, timeout=1)
    assert manager.browser is not original_browser
    assert second is not first

    await manager.release_page_with_context(second)


@pytest.mark.asyncio
async def test_failed_recycle_wakes_waiters_with_a_closed_manager_error():
    manager = _manager(BrowserConfig(headless=True))

    async def restart():
        raise RuntimeError("restart failed")

    manager._restart_browser = restart
    await manager._recycle_browser()

    assert manager._recycle_done.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await manager.get_page(CrawlerRunConfig())


@pytest.mark.asyncio
async def test_killing_the_last_session_starts_a_due_recycle():
    manager = _manager(BrowserConfig(headless=True, max_pages_before_recycle=1))
    manager.sessions["session"] = (_Context(), _Page(), 0)
    manager._recycle_due = True
    started = asyncio.Event()

    async def recycle():
        started.set()

    manager._recycle_browser = recycle
    await manager.kill_session("session")
    await started.wait()
    await manager._recycle_task

    assert not manager._recycle_done.is_set()


def test_explicit_cdp_never_claims_process_recycling():
    manager = _manager(
        BrowserConfig(cdp_url="http://external.example:9222", max_pages_before_recycle=1)
    )
    manager._pages_served = 1
    assert not manager._should_recycle()


# ---------------------------------------------------------------------------
# Bounds. Every wait below replaced an unbounded one; without them a single
# wedged Playwright call takes the whole replica down instead of one request,
# which is the 2026-09-03 outage shape rather than a fix for it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_recycle_that_never_finishes_fails_the_request(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_WAIT_SECONDS", 0.05)
    manager = _manager()
    manager._recycle_done.clear()

    # Not a hang: PERMANENT has no janitor backstop, so a request parked here
    # would hold its pool permit until the container is restarted.
    with pytest.raises(RuntimeError, match="did not finish"):
        await manager._admit_page_acquisition()


@pytest.mark.asyncio
async def test_a_hung_allocation_stops_absorbing_its_caller_cancellation(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "CANCELLATION_GRACE_SECONDS", 0.05)
    manager = _manager()
    forever = asyncio.Event()

    async def never_returns():
        await forever.wait()

    hung = asyncio.create_task(never_returns())
    waiter = asyncio.create_task(manager._await_task_despite_cancellation(hung))
    await asyncio.sleep(0)
    waiter.cancel()

    # The pool's 60s close cap only SIGKILLs an abandoned Chromium if its
    # cancellation is actually delivered through here.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1)

    forever.set()
    hung.cancel()


@pytest.mark.asyncio
async def test_an_abandoned_page_is_closed_when_it_finally_arrives(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "CANCELLATION_GRACE_SECONDS", 0.05)
    manager = _manager()
    release = asyncio.Event()
    context = _Context(new_page_release=release)

    waiter = asyncio.create_task(manager._new_page(context))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1)

    release.set()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if context.pages and context.pages[0].closed:
            break
    assert context.pages, "the abandoned allocation never landed"
    assert context.pages[0].closed, "the abandoned page was left open"


@pytest.mark.asyncio
async def test_a_hung_page_close_still_returns_its_admission_token(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "PAGE_CLOSE_SECONDS", 0.05)
    manager = _manager()
    token = await manager._admit_page_acquisition()
    assert manager._active_acquisitions

    hung = asyncio.Event()

    async def never_closes(_page):
        await hung.wait()

    manager._close_page_quietly = never_closes
    await manager._rollback_page_acquisition(
        token, page=_Page(), close_page=True
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if not manager._active_acquisitions:
            break
    assert not manager._active_acquisitions, "a wedged close stranded the token"
    hung.set()


@pytest.mark.asyncio
async def test_a_failing_browser_close_still_ends_the_chromium_process():
    manager = _manager()
    cleaned = asyncio.Event()
    stopped = asyncio.Event()

    class _Browser:
        async def close(self):
            raise RuntimeError("connection already closed")

    class _Managed:
        async def cleanup(self):
            cleaned.set()

    class _Playwright:
        async def stop(self):
            stopped.set()

    manager.browser = _Browser()
    manager.managed_browser = _Managed()
    manager.playwright = _Playwright()
    manager.config.sleep_on_close = False

    await manager.close()

    # These two are what actually end the process group; a raising close() used
    # to skip both and leave the Chromium charged to the container for life.
    assert cleaned.is_set()
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_a_wedged_driver_does_not_deadlock_the_restart(monkeypatch):
    monkeypatch.setattr(browser_manager_module, "RECYCLE_CLOSE_SECONDS", 0.05)
    monkeypatch.setattr(browser_manager_module, "RECYCLE_START_SECONDS", 1.0)
    monkeypatch.setattr(browser_manager_module, "DRIVER_STOP_SECONDS", 0.05)
    manager = _manager()
    wedged = asyncio.Event()
    started = asyncio.Event()

    class _Playwright:
        async def stop(self):
            await wedged.wait()

    async def close(*, _for_recycle=False):
        await wedged.wait()

    async def start():
        # start() re-enters close() whenever playwright is still set, and that
        # close awaits the recycle task this restart is running inside. The
        # timeout branch has to drop the driver reference or the restart
        # deadlocks against itself and the barrier never reopens.
        assert manager.playwright is None
        started.set()

    manager.playwright = _Playwright()
    manager.close = close
    manager.start = start
    manager._recycle_task = asyncio.current_task()

    assert await asyncio.wait_for(manager._restart_browser(), timeout=3)
    assert started.is_set()
    wedged.set()


# ---------------------------------------------------------------------------
# Cross-event-loop reuse of the class-level _global_pages_lock.
#
# Each asyncio.run makes a fresh, independent event loop, but BrowserManager
# class state (including _global_pages_lock) persists across runs. Before the
# loop-tracking fix, the lock was bound to the first loop on its first
# contended acquire (a waiter parked on it while get_page held it across
# `await _new_page`), and a later contended acquire on a new loop raised:
#   RuntimeError: <asyncio.locks.Lock ...> is bound to a different event loop
#
# These tests are synchronous (no @pytest.mark.asyncio) on purpose: under
# pytest-asyncio strict mode each @pytest.mark.asyncio test gets a single
# event loop of its own, so to reproduce reuse across two loops they must call
# asyncio.run themselves. The reset_global_pages autouse fixture clears the
# class state before/after each test, so each test starts from a clean lock.
# ---------------------------------------------------------------------------


def test_get_global_lock_recreates_on_event_loop_change():
    """_get_global_lock tracks the running loop and recreates the lock when it
    changes, mirroring _CDPConnectionCache._get_lock. No contention needed:
    pre-fix the helper returned the same lock object forever, so a second
    asyncio.run (new loop) reused the loop-1-bound lock instead of making a
    fresh one.
    """

    async def capture_lock():
        loop = asyncio.get_running_loop()
        lock = BrowserManager._get_global_lock()
        assert BrowserManager._global_pages_lock is lock
        assert BrowserManager._global_pages_lock_loop is loop
        return lock

    lock1 = asyncio.run(capture_lock())
    # The class-level lock survives loop 1's close (the bug is precisely that
    # nothing resets it between runs).
    assert BrowserManager._global_pages_lock is lock1
    lock2 = asyncio.run(capture_lock())
    # A new running loop must yield a new lock bound to that loop. Pre-fix
    # lock2 was lock1 (still bound to the now-closed loop 1).
    assert lock2 is not lock1
    assert BrowserManager._global_pages_lock is lock2
    assert BrowserManager._global_pages_lock_loop is not None


def test_get_global_lock_survives_contended_reuse_across_event_loops():
    """The actual bug shape: a *contended* acquire of a loop-1-bound lock on a
    fresh loop 2 raised RuntimeError. Bind the class lock to loop 1 via a
    parked waiter (the path that calls asyncio.Lock._get_loop and binds the
    lock to its event loop), close loop 1, then contended-acquire on loop 2.
    """

    async def bind_lock_to_this_loop():
        lock = BrowserManager._get_global_lock()
        async with lock:
            parked = asyncio.Event()

            async def waiter():
                async with lock:
                    parked.set()

            task = asyncio.create_task(waiter())
            # Let the waiter run and park on the contended lock. This parking
            # is what calls asyncio.Lock._get_loop and binds the lock to this
            # event loop; if the waiter finished instead it never contended.
            await asyncio.sleep(0.05)
            assert not task.done(), "waiter did not park on the held lock"
            assert not parked.is_set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def acquire_contended_on_new_loop():
        # Pre-fix: the waiter's contended acquire raised
        #   RuntimeError: <Lock ...> is bound to a different event loop
        # because _get_global_lock returned the loop-1-bound lock.
        lock = BrowserManager._get_global_lock()
        async with lock:
            acquired = asyncio.Event()

            async def waiter():
                async with lock:
                    acquired.set()

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)
            if task.done():
                # Surface the actual failure (RuntimeError) instead of a
                # generic assertion so the failure mode is the bug's message.
                task.result()
            assert not task.done(), "contended waiter crashed instead of parking"
            assert not acquired.is_set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(bind_lock_to_this_loop())
    assert BrowserManager._global_pages_lock is not None  # survived loop 1
    asyncio.run(acquire_contended_on_new_loop())  # pre-fix: raises here


def test_get_page_third_branch_survives_concurrent_reuse_across_event_loops():
    """End-to-end regression for get_page's third branch (managed browser,
    default config) — the exact line in the bug's traceback. Two concurrent
    get_page calls park a waiter on _get_global_lock (binding it to the loop),
    exactly as arun_many's default 5-wide dispatcher does; a second asyncio.run
    with a fresh BrowserManager but the same class-level lock must not crash.
    """

    async def run_two_concurrent_get_page_calls():
        manager = _manager(BrowserConfig(use_managed_browser=True, headless=True))
        release = asyncio.Event()
        started = asyncio.Event()
        manager.default_context = _Context(
            new_page_started=started,
            new_page_release=release,
        )

        first = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
        await started.wait()  # first is inside _new_page, holding _get_global_lock
        second = asyncio.create_task(manager.get_page(CrawlerRunConfig()))
        # Let second reach _get_global_lock and park on it (or raise pre-fix).
        await asyncio.sleep(0.05)
        release.set()  # unblock first; it releases the lock; second acquires
        page_a, _ = await first
        page_b, _ = await second  # pre-fix: RuntimeError; post-fix: succeeds
        assert page_a is not page_b
        await manager.release_page_with_context(page_a)
        await manager.release_page_with_context(page_b)

    asyncio.run(run_two_concurrent_get_page_calls())  # loop 1: bind lock
    assert BrowserManager._global_pages_lock is not None
    asyncio.run(run_two_concurrent_get_page_calls())  # loop 2: must not crash
