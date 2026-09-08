"""Regression tests for `MemoryAdaptiveDispatcher.run_urls_stream` requeue
handling.

Background
----------
When system memory crosses ``critical_threshold_percent`` (default 95%),
``crawl_url`` re-enqueues the URL for retry and returns a sentinel
``CrawlerTaskResult`` whose inner ``CrawlResult`` carries
``metadata={"status": "requeued"}`` and
``error_message="Requeued due to critical memory pressure"`` (capital "R").

``run_urls_stream`` must skip these sentinels: they must NOT be counted toward
``completed_count`` (which gates the ``while completed_count < total_urls``
loop) and they must NOT be yielded to the consumer. The re-enqueued URL stays
in ``task_queue`` to be re-pulled and crawled once memory recovers.

The original implementation filtered with the case-sensitive substring check
``"requeued" not in result.error_message``. Because Python's
``str.__contains__`` is case-sensitive, the lowercase literal never matched the
capital-"R" ``error_message``, so every sentinel inflated
``completed_count`` AND was yielded. The inflated counter terminated the loop
early, and the ``finally`` cleanup drained ``task_queue`` — discarding the
intended retries.

These tests pin the corrected behavior: the filter keys on the structured
``metadata["status"] == "requeued"`` marker, sentinels are skipped, retries
run after memory recovery, and real results (including results whose
``metadata`` is ``None``) flow through unaffected.
"""

import asyncio
from unittest.mock import patch

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, RateLimiter
from crawl4ai.models import CrawlResult, CrawlerTaskResult


class StubCrawler:
    """Minimal crawler that records every ``arun`` call and returns a real
    ``CrawlResult`` with ``metadata=None`` (the production default for a
    successfully crawled URL)."""

    def __init__(self, arun_delay: float = 0.0):
        self.arun_calls: list[str] = []
        self._arun_delay = arun_delay

    async def arun(self, url, config=None, session_id=None):
        self.arun_calls.append(url)
        if self._arun_delay:
            await asyncio.sleep(self._arun_delay)
        return CrawlResult(url=url, html=f"<html>{url}</html>", success=True)


class PhaseLimiter(RateLimiter):
    """A two-phase rate limiter for deterministic requeue testing.

    Phase 1 (default): blocks every ``crawl_url`` at the rate-limiter step
    (``await rate_limiter.wait_if_needed(url)``) until
    ``dispatcher.current_memory_percent`` reaches the critical threshold. The
    memory monitor refreshes that cached value on ``check_interval``, so the
    blocked task observes the critical reading and takes the line-289 requeue
    branch.

    Phase 2 (after ``release()``): a passthrough — every subsequent
    ``wait_if_needed`` returns immediately. Combined with low memory, this
    lets re-enqueued retries sail through to ``arun`` and produce real
    results.
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
    """Return a ``(mock_fn, setter)`` pair. The mock reads a mutable cell so
    the test can flip memory between low / critical / recovered at will and
    the monitor task picks up the new value on its next refresh."""
    state = {"percent": initial}

    def mock_memory():
        return state["percent"]

    def setter(value: float):
        state["percent"] = value

    return mock_memory, setter


@pytest.mark.asyncio
class TestRunUrlsStreamRequeue:
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_requeued_sentinel_is_not_returned_and_retry_completes(self, streaming):
        """A requeued sentinel must never reach the consumer, must not inflate
        ``completed_count``, and the re-enqueued URL must later be re-pulled
        and crawled for real once memory recovers — the core retry intent.

        This single test catches the original bug in all three of its
        symptoms: sentinels yielded to the consumer, ``completed_count``
        inflated (loop exits early), and re-enqueued retries discarded by
        the ``finally`` queue drain.
        """
        N = 5
        urls = [f"http://example.com/{i}" for i in range(N)]
        dispatcher = MemoryAdaptiveDispatcher(
            check_interval=0.005,
            max_session_permit=N,
            memory_wait_timeout=None,
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
                    yielded.extend(await dispatcher.run_urls(urls, crawler, CrawlerRunConfig()))

            consumer = asyncio.create_task(consume())

            # Phase 1: let the slot-filler pull all N (memory low), then spike
            # to critical so each blocked crawl_url requeues.
            await asyncio.sleep(0.05)
            set_memory(96.0)
            # Give the monitor time to refresh cached memory to 96 and let all
            # N tasks requeue (sentinels are skipped by the fix, so nothing is
            # yielded yet).
            await asyncio.sleep(0.2)

            # Phase 2: drop memory below recovery and release the limiter so the
            # re-enqueued retries pass through and run arun.
            set_memory(50.0)
            limiter.release()

            await asyncio.wait_for(consumer, timeout=5.0)

        sentinels = [
            r for r in yielded if (r.result.metadata or {}).get("status") == "requeued"
        ]
        assert sentinels == [], (
            f"no sentinel should be yielded, got {len(sentinels)}: "
            f"{[r.url for r in sentinels]}"
        )
        assert len(yielded) == N, f"expected {N} real results, got {len(yielded)}"
        assert all(r.success for r in yielded)
        assert set(r.url for r in yielded) == set(urls)
        assert len(crawler.arun_calls) == N, (
            f"each URL should be crawled exactly once after retry; "
            f"arun called {len(crawler.arun_calls)} times: {crawler.arun_calls}"
        )
        assert dispatcher.task_queue.empty()

    async def test_real_result_with_none_metadata_is_yielded(self):
        """A successfully crawled URL returns a ``CrawlResult`` with
        ``metadata=None`` (the production default). The fix's ``or {}`` guard
        must keep that path working — without it the filter raises
        ``AttributeError: 'NoneType' object has no attribute 'get'`` on the
        first real crawl. Guards against removing the ``or {}`` guard.
        """
        urls = ["http://example.com/0", "http://example.com/1"]
        dispatcher = MemoryAdaptiveDispatcher(
            check_interval=0.01,
            max_session_permit=2,
            memory_wait_timeout=None,
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

        assert len(yielded) == 2
        assert all(r.success for r in yielded)
        assert all(r.result.metadata is None for r in yielded), (
            "StubCrawler returns CrawlResult(metadata=None); the or-{} guard "
            "in the filter must not crash on it."
        )

    async def test_failed_real_result_is_still_yielded(self):
        """A genuine (non-requeued) failure must still be counted and
        yielded — the requeue filter must only swallow requeue sentinels.
        Guards against an over-broad filter (e.g. skipping every
        ``success=False`` result) that would silently drop real crawl
        failures.
        """
        class FailingCrawler:
            async def arun(self, url, config=None, session_id=None):
                return CrawlResult(
                    url=url, html="", success=False, error_message="boom"
                )

        urls = ["http://example.com/0"]
        dispatcher = MemoryAdaptiveDispatcher(
            check_interval=0.01,
            max_session_permit=2,
            memory_wait_timeout=None,
        )
        with patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            return_value=30.0,
        ):
            yielded = [
                r
                async for r in dispatcher.run_urls_stream(
                    urls, FailingCrawler(), CrawlerRunConfig()
                )
            ]

        assert len(yielded) == 1
        assert not yielded[0].success
        assert yielded[0].error_message == "boom"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
