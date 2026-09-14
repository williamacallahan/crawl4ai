"""Regression tests for ``RateLimiter.wait_if_needed`` per-domain pacing.

Background
----------
``RateLimiter.wait_if_needed`` is called before every outbound request by both
``MemoryAdaptiveDispatcher.crawl_url`` and ``SemaphoreDispatcher.crawl_url``.
Both dispatchers create up to ``max_session_permit`` (default 20) ``crawl_url``
tasks concurrently from a same-host URL batch (`MemoryAdaptiveDispatcher.run_urls`
greedy slot-fill; `SemaphoreDispatcher.run_urls` ``create_task`` loop), so N
same-domain callers enter ``wait_if_needed`` in one wave.

The original implementation had no per-domain ``asyncio.Lock`` and stamped
``state.last_request_time`` *after* the ``await asyncio.sleep(wait_time)``. With
N concurrent same-domain callers, they all observed the same stale
``last_request_time``, computed the same ``wait_time ~= current_delay``, slept
the identical duration, and woke together -- collapsing into a burst of N-1
requests within a single ``current_delay`` window instead of spacing them
``current_delay`` apart across ``(N-1) * current_delay``.

The fix serializes same-domain callers behind a per-domain ``asyncio.Lock`` and
advances ``last_request_time`` *inside* the lock before yielding, so each
caller reserves its slot before the next same-domain caller can read the
timestamp.

These tests pin the corrected behavior using deterministic, offline stubs (no
browser, no network): same-domain callers space ``current_delay`` apart,
different domains are not serialized against each other, the first call to a
fresh domain does not sleep, the per-domain lock map is keyed by netloc, and
``update_delay``'s backoff branch still spaces requests within a wave instead
of bursting.
"""

import asyncio
import random
import sys
import time
from unittest.mock import patch

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_dispatcher import (
    MemoryAdaptiveDispatcher,
    RateLimiter,
    SemaphoreDispatcher,
)
from crawl4ai.models import CrawlResult


class FireCrawler:
    """Stub crawler that records the relative time of every ``arun`` call and
    returns a real ``CrawlResult`` carrying a status code so ``update_delay``
    is exercised.
    """

    def __init__(self, status_code: int = 200, success: bool = True):
        self.fire_times: list[float] = []
        self._t0: float | None = None
        self._status_code = status_code
        self._success = success

    async def arun(self, url, config=None, session_id=None):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        self.fire_times.append(now - self._t0)
        return CrawlResult(
            url=url,
            html=f"<html>{url}</html>",
            success=self._success,
            status_code=self._status_code,
        )


def _inter(arrivals: list[float]) -> list[float]:
    return [arrivals[i + 1] - arrivals[i] for i in range(len(arrivals) - 1)]


@pytest.mark.asyncio
class TestRateLimiterPacing:
    async def test_same_domain_concurrent_callers_are_spaced(self):
        """N concurrent same-domain ``wait_if_needed`` calls must spread across
        ``(N-1) * current_delay``, not collapse into a burst within one
        ``current_delay`` window. ``base_delay=(D, D)`` makes ``current_delay``
        deterministic because ``random.uniform(D, D) == D``.
        """
        D = 0.1
        N = 6
        rl = RateLimiter(base_delay=(D, D), max_delay=60.0)
        arrivals: list[float] = []

        async def call(i):
            await rl.wait_if_needed(f"http://example.com/{i}")
            arrivals.append(time.time())

        await asyncio.gather(*[call(i) for i in range(N)])

        arr = sorted(arrivals)
        gaps = _inter(arr)
        assert min(gaps) >= D * 0.5, (
            f"same-domain callers arrived in a burst: min inter-arrival "
            f"{min(gaps):.4f}s < {D * 0.5}s; gaps={gaps}"
        )
        total_spread = arr[-1] - arr[0]
        assert total_spread >= (N - 1) * D * 0.7, (
            f"expected spread >= {(N - 1) * D * 0.7:.3f}s, got "
            f"{total_spread:.3f}s; arrivals={arr}"
        )

    async def test_different_domains_are_not_serialized(self):
        """A per-domain lock must not serialize different domains against each
        other. After pre-warming two domains, two concurrent callers (one per
        domain) must complete in ~``current_delay`` (parallel), not
        ~``2 * current_delay`` (serial) -- guards against a global lock.
        """
        D = 0.1
        rl = RateLimiter(base_delay=(D, D), max_delay=60.0)
        # Pre-warm both domains so the next call to each would sleep ~D.
        await rl.wait_if_needed("http://a.com/")
        await rl.wait_if_needed("http://b.com/")

        t0 = time.time()
        await asyncio.gather(
            rl.wait_if_needed("http://a.com/x"),
            rl.wait_if_needed("http://b.com/x"),
        )
        elapsed = time.time() - t0
        assert elapsed < D * 1.7, (
            f"different domains serialized together: elapsed {elapsed:.3f}s "
            f">= {D * 1.7}s (expected parallel ~{D}s)"
        )

    async def test_first_call_to_fresh_domain_does_not_sleep(self):
        """The first call to a fresh domain (``last_request_time == 0``) must
        return immediately -- guards against a regression that sleeps on the
        priming call.
        """
        rl = RateLimiter(base_delay=(0.5, 0.5), max_delay=60.0)
        t0 = time.time()
        await rl.wait_if_needed("http://fresh.example.com/")
        elapsed = time.time() - t0
        assert elapsed < 0.05, (
            f"first call to fresh domain slept {elapsed:.3f}s; expected no sleep"
        )

    async def test_domain_locks_are_isolated_per_domain(self):
        """Each domain gets its own ``asyncio.Lock``; the lock map is keyed by
        netloc and two different domains must share no lock object.
        """
        rl = RateLimiter(base_delay=(0.01, 0.01), max_delay=60.0)
        await asyncio.gather(
            rl.wait_if_needed("http://a.com/1"),
            rl.wait_if_needed("http://b.com/1"),
        )
        assert set(rl._domain_locks.keys()) == {"a.com", "b.com"}
        assert rl._domain_locks["a.com"] is not rl._domain_locks["b.com"]

    async def test_backoff_branch_still_spaces_within_wave(self):
        """Under 429 responses, ``update_delay`` doubles ``current_delay``
        after each request. Even as the delay grows, requests within a wave
        must still be spaced (no burst) -- the adversarial scenario from the
        bug report where ``current_delay`` grows but each wave still collapsed
        into a burst. Here the dispatcher interleaves ``arun``/``update_delay``
        between ``wait_if_needed`` calls, so ``current_delay`` grows across the
        wave. ``random.uniform`` is patched so base-delay selection is
        deterministic and the backoff jitter is 1.0 (clean doubling).
        """
        D = 0.1
        N = 4
        urls = [f"http://example.com/p{i}" for i in range(N)]

        def fake_uniform(a, b):
            # backoff jitter in update_delay is uniform(0.75, 1.25) -> 1.0
            if (a, b) == (0.75, 1.25):
                return 1.0
            # base_delay selection -> deterministic D
            return a

        rl = RateLimiter(
            base_delay=(D, D),
            max_delay=60.0,
            max_retries=20,
            rate_limit_codes=[429, 503],
        )
        dispatcher = MemoryAdaptiveDispatcher(
            max_session_permit=N,
            check_interval=0.05,
            memory_wait_timeout=None,
            rate_limiter=rl,
        )
        crawler = FireCrawler(status_code=429, success=False)

        with patch(
            "crawl4ai.async_dispatcher.random.uniform", side_effect=fake_uniform
        ), patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            return_value=30.0,
        ):
            results = await dispatcher.run_urls(urls, crawler, CrawlerRunConfig())

        assert len(results) == N
        arr = sorted(crawler.fire_times)
        gaps = _inter(arr)
        assert len(arr) == N, f"expected {N} fires, got {len(arr)}: {arr}"
        assert min(gaps) >= D * 0.5, (
            f"backoff wave burst: min inter-arrival {min(gaps):.4f}s; gaps={gaps}"
        )
        # Sanity: exponential backoff grew the delay across the wave, so the
        # last inter-arrival must exceed the first.
        assert gaps[-1] > gaps[0], f"backoff did not grow delays: gaps={gaps}"


@pytest.mark.asyncio
class TestDispatcherPacing:
    async def test_memory_adaptive_spaces_same_domain_urls(self):
        """``MemoryAdaptiveDispatcher.run_urls`` with N same-host URLs and
        ``max_session_permit >= N`` must space the outbound ``arun`` calls
        ``current_delay`` apart, not burst them within one window.
        """
        D = 0.1
        N = 6
        urls = [f"http://example.com/p{i}" for i in range(N)]
        rl = RateLimiter(base_delay=(D, D), max_delay=60.0)
        dispatcher = MemoryAdaptiveDispatcher(
            max_session_permit=N,
            check_interval=0.05,
            memory_wait_timeout=None,
            rate_limiter=rl,
        )
        crawler = FireCrawler(status_code=200, success=True)

        with patch(
            "crawl4ai.async_dispatcher.get_true_memory_usage_percent",
            return_value=30.0,
        ):
            results = await dispatcher.run_urls(urls, crawler, CrawlerRunConfig())

        assert len(results) == N
        arr = sorted(crawler.fire_times)
        gaps = _inter(arr)
        assert len(arr) == N, f"expected {N} fires, got {len(arr)}"
        assert min(gaps) >= D * 0.5, (
            f"memory-adaptive dispatcher burst: min inter-arrival "
            f"{min(gaps):.4f}s; gaps={gaps}"
        )
        assert arr[-1] - arr[0] >= (N - 1) * D * 0.7, (
            f"expected spread >= {(N - 1) * D * 0.7:.3f}s, got "
            f"{arr[-1] - arr[0]:.3f}s"
        )

    async def test_semaphore_dispatch_spaces_same_domain_urls(self):
        """``SemaphoreDispatcher.run_urls`` with N same-host URLs and a
        semaphore large enough to admit all N concurrently must still space
        the ``arun`` calls via the per-domain rate limiter.

        Note: ``SemaphoreDispatcher.run_urls`` takes ``(crawler, urls, config)``
        in that order (unlike ``MemoryAdaptiveDispatcher`` which takes
        ``(urls, crawler, config)``).
        """
        D = 0.1
        N = 6
        urls = [f"http://example.com/p{i}" for i in range(N)]
        rl = RateLimiter(base_delay=(D, D), max_delay=60.0)
        dispatcher = SemaphoreDispatcher(
            semaphore_count=N,
            max_session_permit=N,
            rate_limiter=rl,
        )
        crawler = FireCrawler(status_code=200, success=True)

        results = await dispatcher.run_urls(crawler, urls, CrawlerRunConfig())

        assert len(results) == N
        arr = sorted(crawler.fire_times)
        gaps = _inter(arr)
        assert len(arr) == N, f"expected {N} fires, got {len(arr)}"
        assert min(gaps) >= D * 0.5, (
            f"semaphore dispatcher burst: min inter-arrival {min(gaps):.4f}s; "
            f"gaps={gaps}"
        )
        assert arr[-1] - arr[0] >= (N - 1) * D * 0.7, (
            f"expected spread >= {(N - 1) * D * 0.7:.3f}s, got "
            f"{arr[-1] - arr[0]:.3f}s"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
