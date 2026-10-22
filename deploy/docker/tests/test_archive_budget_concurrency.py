"""Concurrent starts share the real Redis pause published by a throttle."""
import asyncio
import uuid

import pytest

import archive_budget
from archive_budget import ArchiveBudgetSpent
from test_archive_budget import ARCHIVE_URL, budget_config, redis_client, redis_url  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [200, 429, 503, "refused"])
async def test_queued_fetches_check_pause_at_start(redis_client, monkeypatch, response):
    node = "queued-start-" + uuid.uuid4().hex
    monkeypatch.setenv("CRAWL4AI_NODE", node)
    cfg = budget_config(spacing_seconds=4, max_wait_seconds=12, throttle_pause_seconds=300)
    started, respond = asyncio.Event(), asyncio.Event()
    sleepers = asyncio.Queue()
    calls, tasks = [], []

    async def controlled_sleep(delay):
        release = asyncio.Event()
        await sleepers.put(release)
        await release.wait()

    async def opener(url, max_body):
        calls.append(url)
        if len(calls) == 1:
            started.set()
            await respond.wait()
            if response == "refused":
                raise archive_budget._ArchiveUpstreamRefused("Connection refused")
            return response, {"Content-Type": "text/plain"}, b"first"
        return 200, {"Content-Type": "text/plain"}, b"queued"

    monkeypatch.setattr(archive_budget.asyncio, "sleep", controlled_sleep)
    key = archive_budget.BUDGET_KEY_PREFIX + node
    try:
        async with asyncio.timeout(5):
            tasks.append(asyncio.create_task(archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=opener)))
            await started.wait()
            releases = []
            for _ in range(2):
                tasks.append(asyncio.create_task(archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=opener)))
                releases.append(await sleepers.get())
            respond.set()
            first = (await asyncio.gather(tasks[0], return_exceptions=True))[0]
            for release in releases:
                release.set()
            queued = await asyncio.gather(*tasks[1:], return_exceptions=True)
            if response == 200:
                assert first["body"] == "first"
                assert [result["body"] for result in queued] == ["queued", "queued"]
                assert len(calls) == 3
            else:
                for result in [first, *queued]:
                    assert isinstance(result, ArchiveBudgetSpent)
                    assert result.status_code == 429
                    assert result.node == node
                    assert 290 <= result.retry_after <= 300
                    assert result.headers["Retry-After"] == str(result.retry_after)
                assert calls == [ARCHIVE_URL]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await redis_client.delete(key, key + ":pause")
