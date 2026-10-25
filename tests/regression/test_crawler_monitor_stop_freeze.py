"""Regression tests for ``CrawlerMonitor.stop()`` lock-release order and the
``TerminalUI`` loop's stop-signal responsiveness.

Background
----------
``CrawlerMonitor.stop()`` is invoked synchronously on the asyncio event loop
from the ``finally`` blocks of ``MemoryAdaptiveDispatcher.run_urls`` /
``run_urls_stream`` / ``SemaphoreDispatcher.run_urls``. Pre-fix it acquired
the monitor ``RLock`` and, while still holding it, called ``terminal_ui.stop()``,
which joins the UI thread with a 5 s timeout. The UI thread's render pass
acquires that same lock on every call to ``get_summary`` /
``get_memory_status`` / ``get_queue_stats`` / ``get_all_task_stats``, so if
``stop()`` grabbed the lock while the UI thread was mid-render, the join
could only complete by timing out — a guaranteed 5 s freeze of the entire
event loop. Even in the non-contended baseline case, the UI loop slept via
``time.sleep(refresh_rate)`` between iterations, so it could not observe
``stop_event`` until the sleep elapsed: ~1 s on every shutdown at the default
``refresh_rate=1.0``.

The fix has two coordinated parts:

1. ``CrawlerMonitor.stop()`` releases ``self._lock`` *before* calling
   ``self.terminal_ui.stop()`` so the UI thread can always acquire the lock
   during its render pass and the join can complete.
2. ``TerminalUI._ui_loop_unix`` / ``_ui_loop_windows`` wait on
   ``self.stop_event.wait(refresh_rate)`` instead of
   ``time.sleep(refresh_rate)`` so the UI thread wakes immediately when
   ``stop_event`` is set.

The tests use a ``_LockAcquiringTerminalUI`` stand-in for the UI thread
because the real ``_ui_loop_*`` requires a tty (``termios.tcgetattr``) and a
real ``rich.Live`` render target — neither available in CI. The stand-in
faithfully reproduces the bug-relevant contract: its render pass calls the
monitor getters (each acquires ``monitor._lock``) and its inter-iter sleep is
selectable between ``time.sleep`` (pre-fix pattern) and ``stop_event.wait``
(post-fix pattern). The production ``TerminalUI.stop()`` join/wait logic is
exercised directly through the real class with a stub UI loop target, so the
actual ``stop()`` and its ``ui_thread.join(timeout=5.0)`` are tested.
"""

import asyncio
import inspect
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, SemaphoreDispatcher
from crawl4ai.components.crawler_monitor import CrawlerMonitor, TerminalUI
from crawl4ai.models import CrawlResult


# ---------------------------------------------------------------------------
# Stand-in UI used to drive CrawlerMonitor.stop() without a tty
# ---------------------------------------------------------------------------


class _LockAcquiringTerminalUI:
    """Stand-in for ``TerminalUI`` from the bug-relevant perspective.

    The render pass calls the same monitor getters as the production
    ``_update_display`` (each holds ``monitor._lock``); the inter-iteration
    sleep is selectable between ``time.sleep`` (pre-fix pattern) and
    ``stop_event.wait`` (post-fix pattern). ``stop()`` mirrors the production
    ``TerminalUI.stop()``: set ``stop_event`` and join the UI thread with a
    5 s timeout, skipping the self-join.
    """

    def __init__(self, refresh_rate: float = 1.0, use_stop_event_wait: bool = True):
        self.refresh_rate = refresh_rate
        self.stop_event = threading.Event()
        self.ui_thread = None
        self.monitor = None
        self.use_stop_event_wait = use_stop_event_wait

    def start(self, monitor: CrawlerMonitor) -> None:
        self.monitor = monitor
        self.stop_event.clear()
        self.ui_thread = threading.Thread(target=self._loop, daemon=True)
        self.ui_thread.start()

    def stop(self) -> None:
        if self.ui_thread and self.ui_thread.is_alive():
            self.stop_event.set()
            if threading.current_thread() != self.ui_thread:
                self.ui_thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            _ = self.monitor.get_summary()
            _ = self.monitor.get_memory_status()
            _ = self.monitor.get_queue_stats()
            _ = self.monitor.get_all_task_stats()
            if self.use_stop_event_wait:
                if self.stop_event.wait(self.refresh_rate):
                    break
            else:
                time.sleep(self.refresh_rate)
            if not self.monitor.is_running:
                break


def _make_monitor_with_fake_ui(
    refresh_rate: float = 1.0,
    use_stop_event_wait: bool = True,
    urls_total: int = 5,
):
    """Build a monitor wired to a ``_LockAcquiringTerminalUI`` (no real tty)."""
    monitor = CrawlerMonitor(urls_total=urls_total, enable_ui=False)
    fake_ui = _LockAcquiringTerminalUI(
        refresh_rate=refresh_rate, use_stop_event_wait=use_stop_event_wait
    )
    monitor.terminal_ui = fake_ui
    monitor.enable_ui = True
    fake_ui.start(monitor)
    monitor.is_running = True
    monitor.start_time = time.time()
    return monitor, fake_ui


# ---------------------------------------------------------------------------
# Fix #1: CrawlerMonitor.stop() releases the lock before terminal_ui.stop()
# ---------------------------------------------------------------------------


class TestCrawlerMonitorStopReleasesLockBeforeJoin:
    """Pin the lock-release ordering of ``CrawlerMonitor.stop()`` — the join
    must happen with the lock released so the UI thread can always make
    progress through a render pass and observe ``stop_event``."""

    def test_terminal_ui_stop_is_called_outside_the_lock(self):
        """Sentinel ``terminal_ui.stop()`` must observe the monitor lock as
        *free* when invoked. Uses ``RLock._is_owned()`` (CPython) to detect
        whether the calling thread still owns the lock. Pre-fix, the
        ``with self._lock:`` block wrapped the ``terminal_ui.stop()`` call,
        so this assertion would fail."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        observed = {"owned": None}
        sentinel = MagicMock()

        def fake_stop():
            try:
                observed["owned"] = monitor._lock._is_owned()
            except AttributeError:
                owned = monitor._lock.acquire(blocking=False)
                if owned:
                    monitor._lock.release()
                observed["owned"] = owned
        sentinel.stop = fake_stop
        monitor.terminal_ui = sentinel
        monitor.enable_ui = True

        monitor.stop()

        assert observed["owned"] is False, (
            "CrawlerMonitor.stop() held self._lock while calling "
            "terminal_ui.stop(); the join must happen with the lock released."
        )

    def test_stop_returns_promptly_under_lock_contention(self):
        """End-to-end through ``CrawlerMonitor.stop()`` with a UI thread that
        acquires the monitor lock mid-render (the exact contention pattern
        from the bug report). Pre-fix this deadlocked for the full 5 s join
        timeout and left the UI thread alive; post-fix the join returns
        promptly. We assert <1.0 s to leave margin from CI jitter."""
        monitor, fake_ui = _make_monitor_with_fake_ui(
            refresh_rate=1.0, use_stop_event_wait=True
        )
        for i in range(monitor.urls_total):
            monitor.add_task(f"t{i}", f"http://example.com/{i}")
        time.sleep(0.05)

        t0 = time.time()
        monitor.stop()
        dt = time.time() - t0

        assert dt < 1.0, (
            f"stop() took {dt:.3f}s under lock contention; expected well "
            f"under the 5 s join timeout (the pre-fix deadlock)."
        )
        assert not fake_ui.ui_thread.is_alive(), (
            "UI thread was still alive after stop() returned; pre-fix this "
            "happened because the join timed out while holding the lock."
        )
        assert monitor.is_running is False
        assert monitor.end_time is not None

    def test_concurrent_stops_do_not_deadlock(self):
        """Multiple threads calling ``monitor.stop()`` concurrently must
        serialize briefly at the lock and rejoin the UI thread once. No
        thread should hang, no deadlock, no exception."""
        monitor, fake_ui = _make_monitor_with_fake_ui(
            refresh_rate=0.5, use_stop_event_wait=True
        )
        time.sleep(0.05)
        errors = []

        def call_stop():
            try:
                monitor.stop()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=call_stop) for _ in range(4)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        dt = time.time() - t0

        assert not errors, errors
        assert all(not t.is_alive() for t in threads), "concurrent stop thread hung"
        assert dt < 1.5, f"concurrent stops took {dt:.3f}s; suspected deadlock"
        assert not fake_ui.ui_thread.is_alive()


# ---------------------------------------------------------------------------
# Fix #2: TerminalUI loop uses stop_event.wait(refresh_rate) (prompt wakeup)
# ---------------------------------------------------------------------------


class TestTerminalUILoopStopEventWait:
    """Pin that ``_ui_loop_unix`` / ``_ui_loop_windows`` use
    ``stop_event.wait(refresh_rate)`` rather than ``time.sleep(refresh_rate)``,
    so the UI thread wakes the instant ``stop_event`` is set and the join in
    ``stop()`` returns promptly rather than after up to one refresh cycle."""

    def test_ui_loop_unix_uses_stop_event_wait(self):
        src = inspect.getsource(TerminalUI._ui_loop_unix)
        assert "self.stop_event.wait(self.refresh_rate)" in src
        assert "time.sleep(self.refresh_rate)" not in src

    def test_ui_loop_windows_uses_stop_event_wait(self):
        src = inspect.getsource(TerminalUI._ui_loop_windows)
        assert "self.stop_event.wait(self.refresh_rate)" in src
        assert "time.sleep(self.refresh_rate)" not in src

    def test_real_terminal_ui_stop_returns_promptly_with_stop_event_wait_loop(self):
        """The real ``TerminalUI.stop()`` (production code) joins the UI
        thread with a 5 s timeout. With a UI loop body using
        ``stop_event.wait`` (the fixed pattern), the thread wakes
        immediately when ``stop_event`` is set and the join returns well
        under the timeout.

        We swap the UI loop target to a stub because the production
        ``_ui_loop_unix`` requires a tty; this still exercises the real
        ``stop()`` and its ``ui_thread.join(timeout=5.0)``."""
        ui = TerminalUI(refresh_rate=1.0)
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        monitor.is_running = True
        monitor.start_time = time.time()

        def stub_loop():
            while not ui.stop_event.is_set():
                if ui.stop_event.wait(ui.refresh_rate):
                    break
                if not monitor.is_running:
                    break

        ui.stop_event.clear()
        ui.monitor = monitor
        ui.ui_thread = threading.Thread(target=stub_loop, daemon=True)
        ui.ui_thread.start()
        time.sleep(0.05)

        t0 = time.time()
        ui.stop()
        dt = time.time() - t0

        assert dt < 0.5, (
            f"TerminalUI.stop() took {dt:.3f}s; with stop_event.wait the "
            f"join should return immediately, not wait refresh_rate=1.0"
        )
        assert not ui.ui_thread.is_alive()

    def test_stop_returns_promptly_in_baseline_uncontended_case(self):
        """The bug report's baseline defect: with no lock contention at all,
        pre-fix ``monitor.stop()`` still blocked for up to one full
        ``refresh_rate`` because the UI thread was in ``time.sleep`` and
        could not observe ``stop_event`` until it woke. Post-fix the UI
        thread uses ``stop_event.wait`` so the join returns in ms. We use
        ``refresh_rate=2.0`` so the pre-fix branch would block ~2 s — a
        clearly distinguishable signal."""
        monitor, fake_ui = _make_monitor_with_fake_ui(
            refresh_rate=2.0, use_stop_event_wait=True
        )
        time.sleep(0.05)

        t0 = time.time()
        monitor.stop()
        dt = time.time() - t0

        assert dt < 0.5, (
            f"stop() took {dt:.3f}s in the uncontended baseline; "
            f"with stop_event.wait(refresh_rate=2.0) this should be ~0, "
            f"not ~refresh_rate. A >0.5s result suggests time.sleep is back."
        )
        assert not fake_ui.ui_thread.is_alive()


# ---------------------------------------------------------------------------
# Control: confirm the test setup would *detect* the pre-fix behavior
# ---------------------------------------------------------------------------


class TestPreFixBehaviorWouldBeDetected:
    """Sanity-check that the test infrastructure above would *fail* on the
    pre-fix code. We construct a stand-in mimicking the buggy ``time.sleep``
    UI loop under the *fixed* ``CrawlerMonitor.stop()``: no deadlock (lock is
    released) but ``stop`` still blocks ~``refresh_rate`` waiting for the
    sleeping thread. If this drops to ~0, the test setup is no longer
    faithful to the pre-fix UI-loop behavior and the prompt-return
    assertions above lose their meaning."""

    def test_buggy_sleep_loop_reproduces_baseline_stall(self):
        monitor, fake_ui = _make_monitor_with_fake_ui(
            refresh_rate=0.4, use_stop_event_wait=False
        )
        time.sleep(0.05)

        t0 = time.time()
        monitor.stop()
        dt = time.time() - t0

        assert dt >= 0.3, (
            f"with time.sleep loop, stop() should block ~refresh_rate=0.4s, "
            f"got {dt:.3f}s; if this drops to ~0 the test setup is no longer "
            f"faithful to the pre-fix UI-loop behavior."
        )
        assert not fake_ui.ui_thread.is_alive()


# ---------------------------------------------------------------------------
# End-to-end: dispatcher finally no longer freezes the event loop
# ---------------------------------------------------------------------------


class _StubCrawler:
    """Minimal crawler returning a successful ``CrawlResult`` (mirrors the
    stubs in the existing dispatcher regression suite)."""

    def __init__(self):
        self.arun_calls: list[str] = []

    async def arun(self, url, config=None, session_id=None):
        self.arun_calls.append(url)
        return CrawlResult(url=url, html=f"<html>{url}</html>", success=True)


@pytest.mark.asyncio
class TestDispatcherCleanupDoesNotFreezeEventLoop:
    """End-to-end: the dispatcher ``finally`` block calls ``monitor.stop()``
    synchronously on the event loop. Pre-fix this blocked the loop for up to
    ~1 s unconditionally (baseline) and 5 s when the render/lock race hit;
    post-fix ``stop()`` returns in milliseconds. We wrap ``monitor.stop``
    with a timer so the assertion is isolated to the cleanup call (not the
    total ``run_urls`` wall-clock, which includes the actual crawl work)."""

    @staticmethod
    def _build(dispatcher_cls, **kwargs):
        urls = ["http://example.com/0", "http://example.com/1"]
        monitor = CrawlerMonitor(urls_total=len(urls), enable_ui=False)
        fake_ui = _LockAcquiringTerminalUI(
            refresh_rate=2.0, use_stop_event_wait=True
        )
        monitor.terminal_ui = fake_ui
        monitor.enable_ui = True
        dispatcher = dispatcher_cls(monitor=monitor, **kwargs)
        dispatcher._memory_monitor_task = lambda: asyncio.sleep(3600)
        return monitor, fake_ui, dispatcher, urls

    @staticmethod
    def _run(dispatcher_cls, dispatcher, crawler, urls):
        """Call ``run_urls`` with the per-class argument order. The two
        dispatchers historically swapped ``(urls, crawler)`` —
        ``MemoryAdaptiveDispatcher.run_urls(urls, crawler, config)`` vs
        ``SemaphoreDispatcher.run_urls(crawler, urls, config)`` — so a
        single positional call cannot cover both."""
        if dispatcher_cls is SemaphoreDispatcher:
            return dispatcher.run_urls(crawler, urls, None)
        return dispatcher.run_urls(urls, crawler, None)

    @staticmethod
    def _wrap_stop_with_timer(monitor):
        """Replace ``monitor.stop`` with a wrapper that records the elapsed
        wall-clock of the underlying call. Returns a dict carrying the
        measurement (filled when ``monitor.stop()`` actually runs)."""
        record = {"duration": None, "called": False}
        original = monitor.stop

        def timed():
            record["called"] = True
            t0 = time.time()
            original()
            record["duration"] = time.time() - t0

        monitor.stop = timed
        return record

    @pytest.mark.parametrize(
        "dispatcher_cls,kwargs",
        [
            (MemoryAdaptiveDispatcher, {"check_interval": 0.01, "max_session_permit": 2}),
            (SemaphoreDispatcher, {"semaphore_count": 2, "max_session_permit": 2}),
        ],
    )
    async def test_monitor_stop_in_finally_is_fast(
        self, dispatcher_cls, kwargs
    ):
        """The ``monitor.stop()`` issued from the dispatcher ``finally``
        block must return in milliseconds — not block the event loop for
        the pre-fix ~1 s baseline / 5 s contention timeout. We time the
        call directly so the assertion is isolated from the crawl work."""
        monitor, fake_ui, dispatcher, urls = self._build(dispatcher_cls, **kwargs)
        recorder = self._wrap_stop_with_timer(monitor)
        crawler = _StubCrawler()

        with patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            return_value=30.0,
        ):
            results = await self._run(dispatcher_cls, dispatcher, crawler, urls)

        assert recorder["called"], "monitor.stop() never ran from the finally"
        assert recorder["duration"] < 0.5, (
            f"monitor.stop() blocked the event loop for "
            f"{recorder['duration']:.3f}s inside the dispatcher finally; "
            f"pre-fix this baseline was ~1 s, 5 s contended."
        )
        assert not fake_ui.ui_thread.is_alive(), (
            "UI thread survived past the finally block; pre-fix this "
            "happened because the join timed out while holding the lock."
        )
        assert {r.url for r in results} == {"http://example.com/0", "http://example.com/1"}

    async def test_memory_dispatcher_does_not_reap_ui_thread_after_stop(self):
        """After ``run_urls`` returns, the UI thread must already be reaped
        by the in-``finally`` ``monitor.stop()`` — pre-fix the join timed out
        leaving the UI thread alive past the dispatcher's return."""
        monitor, fake_ui, dispatcher, urls = self._build(
            MemoryAdaptiveDispatcher,
            check_interval=0.01, max_session_permit=2,
        )
        crawler = _StubCrawler()
        with patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            return_value=30.0,
        ):
            await self._run(MemoryAdaptiveDispatcher, dispatcher, crawler, urls)

        assert not fake_ui.ui_thread.is_alive()
        # Idempotent re-stop is a fast no-op (UI thread already reaped).
        t0 = time.time()
        monitor.stop()
        dt = time.time() - t0
        assert dt < 0.5, f"idempotent re-stop took {dt:.3f}s"


# ---------------------------------------------------------------------------
# Negative case: enable_ui=False path remains instant
# ---------------------------------------------------------------------------


class TestEnableUiFalseStillInstant:
    """The ``enable_ui=False`` path used by every existing regression test
    must remain instant — the fix must not regress the no-UI branch."""

    def test_stop_without_ui_is_instant(self):
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        monitor.start()
        t0 = time.time()
        monitor.stop()
        dt = time.time() - t0
        assert dt < 0.01, f"enable_ui=False stop took {dt:.4f}s; expected ~0"
        assert monitor.is_running is False
        assert monitor.end_time is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
