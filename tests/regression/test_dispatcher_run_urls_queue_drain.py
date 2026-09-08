"""Regression test pinning the task_queue drain in run_urls on abnormal termination.

``MemoryAdaptiveDispatcher`` stores pending URLs in ``self.task_queue``, an
``asyncio.PriorityQueue`` created once in ``__init__`` and never reset between
batches. ``run_urls`` enqueues every URL before consuming it; if a batch ends
abnormally (cancelled or raises) with items still queued, the ``finally`` block
must drain the queue. Without the drain, those URLs survive in the persistent
queue and are crawled by the next ``run_urls`` call on the same dispatcher
instance, breaking the return-value contract — the next batch's results
contain entries for URLs the caller never submitted.

``run_urls_stream`` has always drained its queue in its ``finally``; this test
pins the same cleanup for ``run_urls`` so a future refactor cannot drop it.
``crawl_url`` and ``_memory_monitor_task`` are stubbed so the test runs
offline, deterministically, and without a browser — the same pattern as
``tests/regression/test_dispatcher_task_cleanup.py``.
"""

import asyncio

import pytest

from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.models import CrawlResult, CrawlerTaskResult


def _install_crawl(dispatcher, processed, hang_url=None):
    """Stub ``crawl_url``: record processed URLs and hang forever on ``hang_url``.

    With ``max_session_permit=1`` only the hang URL starts; the remaining URLs
    stay queued until the batch is cancelled/raised — the condition under which
    the ``finally`` task_queue drain must run.
    """

    async def crawl_url(url, config, task_id, *args, **kwargs):
        processed.append(url)
        if url == hang_url:
            await asyncio.Event().wait()  # never set
        return CrawlerTaskResult(
            task_id=task_id, url=url, result=CrawlResult(url=url, html="", success=True),
            memory_usage=0, peak_memory=0, start_time=0, end_time=0,
        )

    dispatcher.crawl_url = crawl_url
    return dispatcher


async def _wait_for(predicate, iterations=20000):
    for _ in range(iterations):
        if predicate():
            return True
        await asyncio.sleep(0)
    return False


@pytest.mark.asyncio
async def test_reused_dispatcher_processes_only_new_urls_after_cancel(monkeypatch):
    """A cancelled ``run_urls`` must not leak queued URLs into the next batch.

    This is the contract from the bug report: pre-fix, ``run_urls(["D"])`` on a
    reused dispatcher returned results for the previous batch's ``B`` and ``C``
    as well, because they were never drained from the persistent queue. After
    the fix the queue is empty at the end of every batch, so the next batch
    crawls only the URLs the caller submits.
    """
    d = MemoryAdaptiveDispatcher(max_session_permit=1, check_interval=0.01)
    monkeypatch.setattr(d, "_memory_monitor_task", lambda: asyncio.sleep(3600))
    processed = []
    _install_crawl(d, processed, hang_url="A")

    # Batch 1: cancelled mid-flight. Pre-fix this left B and C queued.
    batch1 = asyncio.create_task(d.run_urls(["A", "B", "C"], None, None))
    await _wait_for(lambda: "A" in processed)
    batch1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch1
    assert d.task_queue.empty()

    # Batch 2 on the same dispatcher: must process only D.
    processed.clear()
    results = await d.run_urls(["D"], None, None)
    assert processed == ["D"]
    assert [r.url for r in results] == ["D"]

    # Batch 3 on the same dispatcher: stays clean.
    processed.clear()
    results = await d.run_urls(["E"], None, None)
    assert processed == ["E"]
    assert [r.url for r in results] == ["E"]


@pytest.mark.asyncio
async def test_run_urls_drains_task_queue_on_exception(monkeypatch):
    """When ``run_urls`` raises with URLs still queued, the ``finally`` block
    must still drain the persistent queue — the drain is not cancellation-only.

    ``run_urls_stream`` drains on both the cancellation and exception paths;
    ``run_urls`` must too.
    """
    d = MemoryAdaptiveDispatcher(max_session_permit=1, check_interval=0.01)
    processed = []

    async def boom_monitor():
        while "A" not in processed:
            await asyncio.sleep(0)
        raise MemoryError("simulated memory pressure")

    monkeypatch.setattr(d, "_memory_monitor_task", boom_monitor)
    _install_crawl(d, processed, hang_url="A")

    with pytest.raises(MemoryError):
        await d.run_urls(["A", "B", "C"], None, None)

    assert d.task_queue.empty(), "task_queue must be drained when run_urls raises"
    assert d.concurrent_sessions == 0

    # The next batch on the same dispatcher must process only its own URLs.
    processed.clear()
    results = await d.run_urls(["D"], None, None)
    assert processed == ["D"]
    assert [r.url for r in results] == ["D"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
