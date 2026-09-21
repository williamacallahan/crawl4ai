"""Regression tests for ``MemoryAdaptiveDispatcher.crawl_url`` rate-limit
exhaustion terminal-branch behavior.

Background
----------
``RateLimiter.update_delay`` returns ``False`` to signal that a domain's
rate-limit retry budget is exhausted (``fail_count > max_retries``). Both
dispatchers branch on this signal to set a dedicated
``"Rate limit retry count exceeded for domain ..."`` error message and a
``CrawlStatus.FAILED`` monitor status.

``SemaphoreDispatcher.crawl_url`` treats the branch as terminal: it returns a
``CrawlerTaskResult`` immediately, preserving both the ``FAILED`` status and
the exhaustion message.

``MemoryAdaptiveDispatcher.crawl_url`` lost its early ``return`` in commit
``1630fbd`` (the monitor-system rewrite). Without the return, execution falls
through into the generic ``if not result.success: ... elif self.monitor:
... COMPLETED`` block, which corrupts the exhausted task's record in two
production-reachable ways under the shipped default config
(``rate_limit_codes=[429, 503]``, ``max_retries=3``):

* **Variant A -- status/counter misreport** (503 whose body "looks like data",
  e.g. an API-gateway JSON overload response). ``antibot_detector.is_blocked``
  returns ``False`` for such a 503, so ``CrawlResult.success`` stays ``True``.
  The exhaustion branch sets ``status=FAILED``; the fall-through then takes the
  ``elif self.monitor:`` branch and overwrites the monitor status to
  ``COMPLETED``. The transition into ``COMPLETED`` spuriously increments
  ``monitor.urls_completed`` and ``monitor.status_counts["COMPLETED"]``,
  leaving an internally contradictory record (``status="COMPLETED"`` paired
  with ``error_message="Rate limit retry count exceeded for domain ..."``).
* **Variant B -- exhaustion message clobber** (a 429, which is unconditionally
  ``success=False``). The exhaustion branch's ``error_message`` is overwritten
  by ``result.error_message`` ("Blocked by anti-bot protection: HTTP 429 Too
  Many Requests") in the ``if not result.success:`` branch, erasing the
  exhaustion diagnostic from both the returned ``CrawlerTaskResult`` and the
  monitor record. The status stays ``FAILED`` (re-asserted), but the
  exhaustion-specific signal is lost.

The fix restores the early ``return CrawlerTaskResult(...)`` in
``MemoryAdaptiveDispatcher.crawl_url``, mirroring ``SemaphoreDispatcher`` and
preserving the input ``retry_count`` (which that method's signature accepts
and its own ``finally``-block monitor update already passes through). The
``finally`` block still runs on the early return and re-asserts the auxiliary
fields without passing ``status=``, so the ``FAILED`` status survives.

These tests pin the corrected behavior with deterministic, offline stubs (no
browser, no network): the exhausted branch is terminal, the monitor record and
returned ``CrawlerTaskResult`` carry the exhaustion message and ``FAILED``
status, the aggregate counters are not inflated, the ``finally`` block still
runs, the no-monitor path is unaffected, in-budget rate-limited responses are
not over-eagerly terminal, and the two dispatchers report the same exhaustion
message.
"""

import asyncio
import sys
from unittest.mock import patch

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_dispatcher import (
    MemoryAdaptiveDispatcher,
    RateLimiter,
    SemaphoreDispatcher,
)
from crawl4ai.components.crawler_monitor import CrawlerMonitor
from crawl4ai.models import CrawlResult, CrawlStatus


# ---------------------------------------------------------------------------
# Offline test scaffolding
# ---------------------------------------------------------------------------


class StubCrawler:
    """Stub crawler returning a fixed ``CrawlResult`` for every ``arun`` call.

    The response shape (status code, success, error_message, html) is what the
    dispatcher ultimately inspects; the rate-limit exhaustion boundary is driven
    by ``RateLimiter.fail_count`` (incremented once per rate-limited response),
    not by the response changing across calls.
    """

    def __init__(
        self,
        status_code: int,
        success: bool,
        error_message: str = "",
        html: str = "",
    ):
        self._status_code = status_code
        self._success = success
        self._error_message = error_message
        self._html = html
        self.call_count = 0

    async def arun(self, url, config=None, session_id=None, **kwargs):
        self.call_count += 1
        return CrawlResult(
            url=url,
            html=self._html,
            success=self._success,
            status_code=self._status_code,
            error_message=self._error_message,
        )


def _fast_rate_limiter() -> RateLimiter:
    """A ``RateLimiter`` whose inter-request pacing is zero but whose
    exhaustion semantics match the shipped defaults (``max_retries=3``,
    ``rate_limit_codes=[429, 503]``).

    With ``base_delay=(0.0, 0.0)`` and ``max_delay=0.0`` the per-domain
    ``current_delay`` stays 0 across the backoff branch, so ``wait_if_needed``
    never sleeps -- keeping the tests deterministic and sub-millisecond -- while
    ``update_delay`` still increments ``fail_count`` on every rate-limited
    response and returns ``False`` once ``fail_count > max_retries``.
    """
    return RateLimiter(
        base_delay=(0.0, 0.0),
        max_delay=0.0,
        max_retries=3,
        rate_limit_codes=[429, 503],
    )


EXHAUSTION_MESSAGE = "Rate limit retry count exceeded for domain example.com"
BLOCKED_429_MESSAGE = "Blocked by anti-bot protection: HTTP 429 Too Many Requests"


# ---------------------------------------------------------------------------
# Variant A: 503-with-data, success=True, monitor enabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimitExhaustionVariantA:
    """The 503-with-data path: ``result.success is True`` so the fall-through
    overwrites the monitor status ``FAILED -> COMPLETED`` and inflates the
    aggregate completion counters. The fix makes the exhaustion branch
    terminal."""

    async def test_monitor_status_stays_failed_not_overwritten_to_completed(self):
        """The per-task monitor status must end at ``FAILED`` (not
        ``COMPLETED``) after the exhaustion transition. Without the fix the
        ``elif self.monitor: update_task(status=COMPLETED)`` fall-through
        overwrites ``FAILED -> COMPLETED`` because ``result.success`` is True
        for a data-looking 503."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=503,
            success=True,
            html='<html><body><pre>{"err":"overloaded"}</pre></body></html>',
        )
        url = "https://example.com/overloaded.json"
        task_id = "T1"
        monitor.add_task(task_id, url)

        for _ in range(4):  # call 4 -> fail_count=4 > max_retries=3 -> exhausted
            await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)

        assert monitor.stats[task_id]["status"] == CrawlStatus.FAILED.name, (
            "Exhausted task's monitor status was overwritten from FAILED to "
            f"COMPLETED: got {monitor.stats[task_id]['status']!r}"
        )

    async def test_aggregate_completion_counters_not_inflated_by_exhausted_call(self):
        """The exhaustion transition must not itself increment
        ``urls_completed`` or ``status_counts["COMPLETED"]``. Calls 1-3 are
        in-budget (``success=True``) so they legitimately complete and bump
        ``urls_completed`` to 3; the 4th (exhausted) call must complete as
        ``FAILED`` without adding a spurious 4th completion."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=503,
            success=True,
            html='<html><body><pre>{"err":"overloaded"}</pre></body></html>',
        )
        url = "https://example.com/overloaded.json"
        task_id = "T1"
        monitor.add_task(task_id, url)

        for _ in range(4):
            await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)

        assert monitor.urls_completed == 3, (
            "Exhausted call's FAILED -> COMPLETED fall-through spuriously "
            f"incremented urls_completed: got {monitor.urls_completed}, expected 3"
        )
        assert monitor.status_counts[CrawlStatus.COMPLETED.name] == 0, (
            "Exhausted call left a stale COMPLETED counter: got "
            f"{monitor.status_counts[CrawlStatus.COMPLETED.name]}, expected 0"
        )
        assert monitor.status_counts[CrawlStatus.FAILED.name] == 1, (
            "Exhausted task's FAILED counter was not recorded: got "
            f"{monitor.status_counts[CrawlStatus.FAILED.name]}, expected 1"
        )


# ---------------------------------------------------------------------------
# Variant B: 429, success=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimitExhaustionVariantB:
    """The 429 path: ``result.success is False`` so the fall-through clobbers
    the exhaustion ``error_message`` with the generic blocked-protection
    string. The fix makes the exhaustion branch terminal, preserving the
    exhaustion diagnostic."""

    async def test_returned_error_message_is_exhaustion_not_blocked(self):
        """The returned ``CrawlerTaskResult.error_message`` (propagated to
        ``DispatchResult.error_message`` via ``async_webcrawler.py``) must
        carry the exhaustion sentence, not the generic anti-bot blocked
        string."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=429,
            success=False,
            error_message=BLOCKED_429_MESSAGE,
        )
        url = "https://example.com/page"
        task_id = "T1"
        monitor.add_task(task_id, url)

        results = [
            await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)
            for _ in range(4)
        ]

        assert results[-1].error_message == EXHAUSTION_MESSAGE, (
            "Exhaustion message was clobbered by result.error_message: got "
            f"{results[-1].error_message!r}"
        )

    async def test_monitor_error_message_is_exhaustion_not_blocked(self):
        """The monitor's per-task ``error_message`` field (written by the
        ``finally`` block) must carry the exhaustion sentence, not the generic
        anti-bot blocked string -- otherwise an operator cannot distinguish
        "still backing off within budget" from "retry budget exhausted"."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=429,
            success=False,
            error_message=BLOCKED_429_MESSAGE,
        )
        url = "https://example.com/page"
        task_id = "T1"
        monitor.add_task(task_id, url)

        for _ in range(4):
            await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)

        assert monitor.stats[task_id]["error_message"] == EXHAUSTION_MESSAGE
        assert monitor.stats[task_id]["status"] == CrawlStatus.FAILED.name


# ---------------------------------------------------------------------------
# Terminal-branch mechanics: finally block, no-monitor path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExhaustionBranchMechanics:
    """Mechanical guarantees of the early-return branch: the ``finally`` block
    still runs and the no-monitor path still returns the exhaustion message."""

    async def test_finally_block_still_runs_after_early_return(self):
        """The early ``return`` is inside the ``try`` block, so the ``finally``
        must still execute: it decrements ``concurrent_sessions`` back to the
        baseline and re-asserts the monitor's auxiliary fields (``end_time``,
        ``memory_usage``, ``retry_count``) without touching ``status``."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=429,
            success=False,
            error_message=BLOCKED_429_MESSAGE,
        )
        url = "https://example.com/page"
        task_id = "T1"
        monitor.add_task(task_id, url)

        result = None
        for _ in range(4):
            result = await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)

        # concurrent_sessions is incremented in the try and decremented in the
        # finally; if the finally did not run it would be stuck at 4.
        assert dispatcher.concurrent_sessions == 0, (
            "finally block did not run on early return: concurrent_sessions="
            f"{dispatcher.concurrent_sessions}"
        )
        # The finally block calls update_task(end_time=..., memory_usage=...,
        # retry_count=...) on the monitor; end_time must be populated.
        assert monitor.stats[task_id]["end_time"] is not None
        assert monitor.stats[task_id]["end_time"] >= monitor.stats[task_id]["start_time"]
        # status is NOT passed by the finally, so FAILED survives.
        assert monitor.stats[task_id]["status"] == CrawlStatus.FAILED.name

    async def test_no_monitor_path_returns_exhaustion_message(self):
        """When no ``CrawlerMonitor`` is attached (the literal default
        ``crawler.arun_many(urls)`` wiring), the early return must still
        produce a ``CrawlerTaskResult`` carrying the exhaustion sentence --
        this is the path Variant B reaches under the default call."""
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=None,
        )
        dispatcher.crawler = StubCrawler(
            status_code=429,
            success=False,
            error_message=BLOCKED_429_MESSAGE,
        )
        url = "https://example.com/page"
        task_id = "T1"

        result = None
        for _ in range(4):
            result = await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)

        assert result.error_message == EXHAUSTION_MESSAGE
        assert dispatcher.concurrent_sessions == 0


# ---------------------------------------------------------------------------
# Non-regression: in-budget rate-limited responses are not over-eagerly terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInBudgetRateLimitedResponses:
    """The fix must not make *in-budget* rate-limited responses (calls 1..
    ``max_retries``) terminal. Those must continue through the generic status
    block so that a 429 within the retry budget reports the generic blocked
    message and a 503-with-data reports success, exactly as before the fix."""

    async def test_in_budget_429_reports_blocked_message_not_exhaustion(self):
        """Calls 1-3 of a 429 streak (``fail_count`` 1, 2, 3, all <=
        ``max_retries=3``) must return ``update_delay=True`` and report the
        generic blocked-protection message -- not the exhaustion sentence.
        Only the 4th call (``fail_count=4 > 3``) is terminal."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=429,
            success=False,
            error_message=BLOCKED_429_MESSAGE,
        )
        url = "https://example.com/page"
        task_id = "T1"
        monitor.add_task(task_id, url)

        # Three in-budget calls: must NOT carry the exhaustion message.
        for i in range(3):
            result = await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)
            assert result.error_message == BLOCKED_429_MESSAGE, (
                f"In-budget 429 call #{i + 1} was treated as terminal/"
                f"exhausted: got {result.error_message!r}"
            )
            assert monitor.stats[task_id]["status"] == CrawlStatus.FAILED.name
        assert monitor.urls_completed == 0, (
            "In-budget 429 calls must not bump urls_completed"
        )

    async def test_in_budget_503_with_data_completes_normally(self):
        """Calls 1-3 of a 503-with-data streak (``success=True``) must complete
        normally (``status=COMPLETED``, ``urls_completed`` incrementing) -- the
        early return must fire only on the exhaustion boundary, not on every
        rate-limited response."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(),
            monitor=monitor,
        )
        dispatcher.crawler = StubCrawler(
            status_code=503,
            success=True,
            html='<html><body><pre>{"err":"overloaded"}</pre></body></html>',
        )
        url = "https://example.com/overloaded.json"
        task_id = "T1"
        monitor.add_task(task_id, url)

        for i in range(3):
            result = await dispatcher.crawl_url(url, CrawlerRunConfig(), task_id)
            assert monitor.stats[task_id]["status"] == CrawlStatus.COMPLETED.name, (
                f"In-budget 503-with-data call #{i + 1} did not complete: "
                f"status={monitor.stats[task_id]['status']!r}"
            )
            assert result.error_message == "", (
                f"In-budget 503-with-data call #{i + 1} was treated as "
                f"exhausted: {result.error_message!r}"
            )
        assert monitor.urls_completed == 3


# ---------------------------------------------------------------------------
# Sibling parity: MemoryAdaptiveDispatcher == SemaphoreDispatcher on exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSemaphoreDispatcherExhaustionParity:
    """``SemaphoreDispatcher.crawl_url`` already returned early from the
    exhaustion branch. This test guards that both dispatchers report the same
    exhaustion message on the boundary -- so a future refactor cannot
    re-introduce the asymmetry in either direction."""

    async def test_both_dispatchers_report_same_exhaustion_message(self):
        """On the exhaustion boundary both dispatchers must return a
        ``CrawlerTaskResult`` whose ``error_message`` is the exhaustion
        sentence -- no asymmetry."""
        url = "https://example.com/page"

        # MemoryAdaptiveDispatcher
        mad = MemoryAdaptiveDispatcher(
            rate_limiter=_fast_rate_limiter(), monitor=None
        )
        mad.crawler = StubCrawler(
            status_code=429, success=False, error_message=BLOCKED_429_MESSAGE
        )
        mad_result = None
        for _ in range(4):
            mad_result = await mad.crawl_url(url, CrawlerRunConfig(), "mad-T1")

        # SemaphoreDispatcher
        sd = SemaphoreDispatcher(rate_limiter=_fast_rate_limiter(), monitor=None)
        sd.crawler = StubCrawler(
            status_code=429, success=False, error_message=BLOCKED_429_MESSAGE
        )
        semaphore = asyncio.Semaphore(1)
        sd_result = None
        for _ in range(4):
            sd_result = await sd.crawl_url(
                url, CrawlerRunConfig(), "sd-T1", semaphore
            )

        assert mad_result.error_message == sd_result.error_message == EXHAUSTION_MESSAGE
        # Both must terminate without bumping any completion counter (no
        # monitor attached, so this is purely about the returned result).
        assert mad_result.result.status_code == sd_result.result.status_code == 429


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
