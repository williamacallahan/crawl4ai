"""A cancelled batch must not leave its per-URL crawls running detached.

`MemoryAdaptiveDispatcher.run_urls` creates one task per URL and waits on them with
`asyncio.wait`, which - unlike `asyncio.gather` - does not propagate the caller's
cancellation to them. When the caller is cancelled - a wall-clock deadline on /crawl, or a
client disconnect - those tasks used to survive the `finally`, each still holding a
Playwright page and browser context that nothing would ever release. On a
memory-pressured server that is the exact condition producing the most
cancellations, so the leak compounds under the load that causes it.
"""

import asyncio

import pytest

from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

URLS = ["https://a.test", "https://b.test"]


def _hanging(dispatcher):
    """Make every per-URL crawl hang so cancellation is what ends the batch."""
    dispatcher.started = []

    async def crawl_url(url, config, task_id, *args, **kwargs):
        dispatcher.started.append(asyncio.current_task())
        await asyncio.Event().wait()  # never set

    dispatcher.crawl_url = crawl_url
    return dispatcher


async def _cancel_mid_batch(dispatcher, coro):
    task = asyncio.create_task(coro)
    for _ in range(2000):
        if len(dispatcher.started) == len(URLS):
            break
        await asyncio.sleep(0)
    assert len(dispatcher.started) == len(URLS), "per-URL tasks never started"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for started in dispatcher.started:
        assert started.done(), "a per-URL crawl outlived its cancelled batch"


@pytest.mark.asyncio
async def test_memory_adaptive_cancels_its_url_tasks(monkeypatch):
    d = _hanging(MemoryAdaptiveDispatcher(check_interval=0.01))
    # Keep the batch out of memory-pressure mode without polling real memory.
    monkeypatch.setattr(d, "_memory_monitor_task", lambda: asyncio.sleep(3600))
    await _cancel_mid_batch(d, d.run_urls(URLS, None, None))
