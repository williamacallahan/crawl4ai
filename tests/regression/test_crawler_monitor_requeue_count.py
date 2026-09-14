"""Regression tests for ``CrawlerMonitor`` requeue accounting.

Background
----------
``MemoryAdaptiveDispatcher`` re-enqueues a task under memory pressure by
calling ``monitor.update_task(task_id, status=CrawlStatus.QUEUED, ...)``.
At that moment the task's *previous* status is ``IN_PROGRESS`` (set earlier
in ``crawl_url``), so the requeue transition observed by the monitor is
``IN_PROGRESS -> QUEUED``.

``CrawlerMonitor.update_task`` maintains ``requeued_count`` (the numerator of
the public ``requeue_rate`` metric returned by ``get_summary()`` and rendered
in the dashboard's "Requeued" row / "Requeue Rate" readout). The original
implementation only incremented this counter when the *old* status was
``COMPLETED`` or ``FAILED`` — a transition the dispatcher never produces — so
``requeued_count`` was always 0 and ``requeue_rate`` was always ``0.0``,
regardless of how many real requeues occurred.

These tests pin the corrected behavior: the counter increments exactly on the
``IN_PROGRESS -> QUEUED`` transition, is idempotent per task (a task requeued
twice counts once), is not triggered by ordinary ``QUEUED -> IN_PROGRESS ->
COMPLETED``/``FAILED`` progressions, and is reported accurately by
``get_summary()`` both in isolation and end-to-end through the real dispatcher
memory-pressure path.
"""

import asyncio
import uuid
from unittest.mock import patch

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, RateLimiter
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.models import CrawlResult, CrawlerTaskResult, CrawlStatus


# ---------------------------------------------------------------------------
# Direct monitor-unit tests (sync, no dispatcher / no asyncio)
# ---------------------------------------------------------------------------


def _make_monitor(n: int = 1) -> CrawlerMonitor:
    """A UI-less monitor. ``enable_ui=False`` avoids spawning the terminal
    thread so the tests stay deterministic and don't clobber the runner's
    tty."""
    return CrawlerMonitor(urls_total=n, enable_ui=False)


def _add(monitor: CrawlerMonitor, task_id: str, url: str) -> None:
    monitor.add_task(task_id, url)


class TestRequeueAccounting:
    """Direct ``update_task`` transition tests for ``requeued_count``."""

    def test_in_progress_to_queued_increments_counter(self):
        """The actual requeue transition (``IN_PROGRESS -> QUEUED``) must
        increment ``requeued_count`` and set ``counted_requeue``. This is the
        core regression: the buggy condition keyed on ``COMPLETED``/``FAILED``
        and so never fired."""
        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            tid = str(uuid.uuid4())
            _add(monitor, tid, "http://example.com/0")

            # QUEUED -> IN_PROGRESS (normal start)
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            assert monitor.requeued_count == 0

            # IN_PROGRESS -> QUEUED (requeue under memory pressure)
            monitor.update_task(
                tid,
                status=CrawlStatus.QUEUED,
                error_message="Requeued due to critical memory pressure",
            )

            assert monitor.requeued_count == 1
            assert monitor.stats[tid]["counted_requeue"] is True

            summary = monitor.get_summary()
            assert summary["requeued_count"] == 1
            assert summary["requeue_rate"] == 100.0  # 1 requeue / 1 task * 100
        finally:
            monitor.stop()

    def test_normal_completion_does_not_increment_requeue(self):
        """A clean ``QUEUED -> IN_PROGRESS -> COMPLETED`` life cycle must NOT
        be counted as a requeue. Guards against an over-broad condition that
        fires on any status change.        """
        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            tid = str(uuid.uuid4())
            _add(monitor, tid, "http://example.com/0")
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.COMPLETED, end_time=0.0)

            assert monitor.requeued_count == 0
            assert monitor.stats[tid]["counted_requeue"] is False

            summary = monitor.get_summary()
            assert summary["requeued_count"] == 0
            assert summary["requeue_rate"] == 0.0
            assert summary["urls_completed"] == 1
        finally:
            monitor.stop()

    def test_completed_to_queued_does_not_increment_requeue(self):
        """A ``COMPLETED -> QUEUED`` transition (the one the buggy condition
        *was* written to detect) must NOT be counted as a requeue under the new
        condition. This documents the deliberate behavior change: the counter
        now tracks only the real ``IN_PROGRESS -> QUEUED`` requeue path, not
        hypothetical completed/failed re-queueing that the dispatcher never
        emits.        """
        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            tid = str(uuid.uuid4())
            _add(monitor, tid, "http://example.com/0")
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.COMPLETED, end_time=0.0)
            # Hypothetical re-queue after completion (never produced by the
            # dispatcher, but used here to pin the new condition's intent).
            monitor.update_task(tid, status=CrawlStatus.QUEUED)

            assert monitor.requeued_count == 0
            assert monitor.stats[tid]["counted_requeue"] is False
        finally:
            monitor.stop()

    def test_repeated_requeue_counts_each_task_once(self):
        """A task requeued twice (``IN_PROGRESS -> QUEUED -> IN_PROGRESS ->
        QUEUED``) must increment ``requeued_count`` exactly once. The
        ``counted_requeue`` guard keeps ``requeue_rate`` meaningful as
        "percentage of tasks that experienced at least one requeue" rather
        than a count of requeue events.        """
        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            tid = str(uuid.uuid4())
            _add(monitor, tid, "http://example.com/0")

            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.QUEUED)  # requeue #1
            assert monitor.requeued_count == 1

            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.QUEUED)  # requeue #2
            assert monitor.requeued_count == 1  # still 1 (counted_requeue guard)

            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.COMPLETED, end_time=0.0)

            assert monitor.requeued_count == 1
            summary = monitor.get_summary()
            assert summary["requeued_count"] == 1
            assert summary["requeue_rate"] == 100.0
        finally:
            monitor.stop()

    def test_status_counts_still_balance_through_requeue(self):
        """Requeue transitions must keep ``status_counts`` internally
        consistent (decrement old, increment new) — the requeue-tracking
        branch must not skip or corrupt that bookkeeping.        """
        monitor = _make_monitor(n=2)
        monitor.start()
        try:
            tid = str(uuid.uuid4())
            _add(monitor, tid, "http://example.com/0")

            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.QUEUED)  # requeue
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(tid, status=CrawlStatus.COMPLETED, end_time=0.0)

            counts = monitor.get_summary()["status_counts"]
            assert counts == {
                CrawlStatus.QUEUED.name: 0,
                CrawlStatus.IN_PROGRESS.name: 0,
                CrawlStatus.COMPLETED.name: 1,
                CrawlStatus.FAILED.name: 0,
            }
        finally:
            monitor.stop()


# ---------------------------------------------------------------------------
# End-to-end dispatcher tests (async)
# ---------------------------------------------------------------------------


class StubCrawler:
    """Minimal crawler that records every ``arun`` call and returns a real
    ``CrawlResult`` with ``metadata=None`` (the production default)."""

    def __init__(self):
        self.arun_calls: list[str] = []

    async def arun(self, url, config=None, session_id=None):
        self.arun_calls.append(url)
        return CrawlResult(url=url, html=f"<html>{url}</html>", success=True)


class PhaseLimiter(RateLimiter):
    """Two-phase rate limiter for deterministic requeue testing — identical
    in spirit to the one in ``test_dispatcher_stream_requeue.py``.

    Phase 1: blocks every ``crawl_url`` at ``wait_if_needed`` until the
    dispatcher's cached memory reading reaches the critical threshold, so
    the blocked task observes critical memory and takes the requeue branch.

    Phase 2 (after ``release()``): passthrough, so re-enqueued retries sail
    through to ``arun`` and complete once memory recovers.
    """

    def __init__(self, dispatcher: MemoryAdaptiveDispatcher):
        super().__init__()
        self._disp = dispatcher
        self._released = False

    async def wait_if_needed(self, url: str) -> None:
        if self._released:
            return
        while self._disp.current_memory_percent < self._disp.critical_threshold_percent:
            await asyncio.sleep(0.001)

    def release(self) -> None:
        self._released = True


def _memory_holder(initial: float = 50.0):
    """Return a ``(mock_fn, setter)`` pair backed by a mutable cell so the
    test can flip memory between low / critical / recovered at will."""
    state = {"percent": initial}

    def mock_memory():
        return state["percent"]

    def setter(value: float):
        state["percent"] = value

    return mock_memory, setter


@pytest.mark.asyncio
class TestMonitorRequeueCountEndToEnd:
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_requeue_count_reflects_real_requeues(self, streaming):
        """End-to-end: drive ``MemoryAdaptiveDispatcher`` (streaming and
        non-streaming) through a memory spike that requeues every URL, let
        memory recover so retries complete, then assert the attached
        ``CrawlerMonitor`` reports the correct requeue metrics.

        Before the fix, ``get_summary()`` returned
        ``requeued_count=0, requeue_rate=0.0`` with ``counted_requeue=False``
        on every task even though all N tasks were requeued. After the fix it
        must report ``requeued_count=N`` and ``requeue_rate=100.0``.
        """
        N = 5
        urls = [f"http://example.com/{i}" for i in range(N)]
        monitor = CrawlerMonitor(urls_total=N, enable_ui=False)

        dispatcher = MemoryAdaptiveDispatcher(
            check_interval=0.005,
            max_session_permit=N,
            memory_wait_timeout=None,
            monitor=monitor,
        )
        limiter = PhaseLimiter(dispatcher)
        dispatcher.rate_limiter = limiter
        crawler = StubCrawler()

        mock_memory, set_memory = _memory_holder(initial=50.0)
        with patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            side_effect=mock_memory,
        ):
            yielded: list[CrawlerTaskResult] = []

            async def consume():
                if streaming:
                    async for r in dispatcher.run_urls_stream(
                        urls, crawler, CrawlerRunConfig()
                    ):
                        yielded.append(r)
                else:
                    yielded.extend(
                        await dispatcher.run_urls(urls, crawler, CrawlerRunConfig())
                    )

            consumer = asyncio.create_task(consume())

            # Phase 1: let the slot-filler pull all N (memory low), then spike
            # to critical so each blocked crawl_url requeues.
            await asyncio.sleep(0.05)
            set_memory(96.0)
            await asyncio.sleep(0.2)

            # Phase 2: drop memory below recovery and release the limiter so the
            # re-enqueued retries pass through and run arun.
            set_memory(50.0)
            limiter.release()

            await asyncio.wait_for(consumer, timeout=5.0)

        # Dispatcher-level invariants (the existing regression suite already
        # covers these; assert them here as a sanity gate so the monitor
        # assertions below are not measuring a broken run).
        sentinels = [
            r for r in yielded if (r.result.metadata or {}).get("status") == "requeued"
        ]
        assert sentinels == []
        assert len(yielded) == N
        assert all(r.success for r in yielded)
        assert set(r.url for r in yielded) == set(urls)
        assert len(crawler.arun_calls) == N

        # Monitor requeue-accounting invariants — the actual regression.
        requeued_task_ids = [
            tid for tid, s in monitor.stats.items() if s.get("counted_requeue")
        ]
        assert len(requeued_task_ids) == N, (
            f"all {N} tasks were requeued via IN_PROGRESS -> QUEUED, but only "
            f"{len(requeued_task_ids)} have counted_requeue=True: "
            f"{requeued_task_ids}"
        )

        summary = monitor.get_summary()
        assert summary["requeued_count"] == N, (
            f"expected requeued_count={N} (one per requeued task), got "
            f"{summary['requeued_count']}"
        )
        assert summary["requeue_rate"] == 100.0, (
            f"expected requeue_rate=100.0 ({N}/{N}*100), got "
            f"{summary['requeue_rate']}"
        )
        # Every task ended COMPLETED after its retry.
        assert summary["status_counts"]["COMPLETED"] == N
        assert summary["urls_completed"] == N

    async def test_no_requeue_when_memory_stays_low(self):
        """End-to-end negative case: when memory never crosses critical, no
        task is requeued and the monitor reports ``requeued_count=0``,
        ``requeue_rate=0.0``. Guards against the fix over-counting on
        ordinary completions."""
        N = 3
        urls = [f"http://example.com/{i}" for i in range(N)]
        monitor = CrawlerMonitor(urls_total=N, enable_ui=False)

        dispatcher = MemoryAdaptiveDispatcher(
            check_interval=0.01,
            max_session_permit=N,
            memory_wait_timeout=None,
            monitor=monitor,
        )
        crawler = StubCrawler()

        with patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            return_value=30.0,
        ):
            yielded = [
                r
                async for r in dispatcher.run_urls_stream(
                    urls, crawler, CrawlerRunConfig()
                )
            ]

        assert len(yielded) == N
        assert all(r.success for r in yielded)

        summary = monitor.get_summary()
        assert summary["requeued_count"] == 0
        assert summary["requeue_rate"] == 0.0
        assert all(
            not s.get("counted_requeue") for s in monitor.stats.values()
        )
        assert summary["urls_completed"] == N


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
