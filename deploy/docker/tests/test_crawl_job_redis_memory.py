"""Real-Redis checks that crawl-job writes respect Redis maxmemory."""

from contextlib import contextmanager

import pytest
import redis
from redis import asyncio as aioredis

from test_crawl_job_queue_redis import (  # noqa: F401
    public_seed_validation, queue_config, redis_url,
)

MIB = 1024 * 1024


@contextmanager
def redis_maxmemory(url, maxmemory, policy=None):
    """Set server-wide maxmemory for one test; CI's Redis serves one serial run."""
    client = redis.from_url(url)
    original = client.config_get("maxmemory*")
    try:
        client.config_set("maxmemory", maxmemory)
        if policy:
            client.config_set("maxmemory-policy", policy)
        yield
    finally:
        client.config_set("maxmemory", original["maxmemory"])
        client.config_set("maxmemory-policy", original["maxmemory-policy"])
        client.close()


def used_memory(url):
    client = redis.from_url(url)
    try:
        return int(client.info("memory")["used_memory"])
    finally:
        client.close()


def memory_config(headroom):
    config = queue_config()
    config["crawl_jobs"]["redis_headroom_bytes"] = headroom
    config["limits"]["queue"]["per_principal"] = 4
    return config


async def enqueue(queue):
    return await queue.enqueue(urls=["https://example.com"], browser_config={},
        crawler_config={}, result_fields=None, webhook_config=None, owner="alice")


@pytest.mark.asyncio
async def test_enqueue_reserves_a_stored_result_until_completion_releases_it(redis_url):
    from crawl_job_queue import CrawlJobQueue, CrawlJobResultBudgetFull

    # 64 MiB above current use, 58 MiB of headroom: one 4 MiB result fits, two do not.
    with redis_maxmemory(redis_url, used_memory(redis_url) + 64 * MIB):
        client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            queue = CrawlJobQueue(client, memory_config(58 * MIB))
            await queue.ensure_group()
            first = await enqueue(queue)
            keys_before = await client.dbsize()
            with pytest.raises(CrawlJobResultBudgetFull):
                await enqueue(queue)
            assert await client.dbsize() == keys_before
            assert await client.xlen(queue.settings.stream) == 1

            entry = (await queue.read_new("worker-a"))[0]
            payload = await queue.load_payload(entry.task_id)
            attempt = await queue.start_attempt(entry, payload, "worker-a")
            await queue.complete(entry, payload, attempt, result={"ok": True})
            assert entry.task_id == first

            admitted = await enqueue(queue)
            assert await client.hget(queue.task_key(admitted), "status") == "processing"
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_enqueue_reserves_a_second_copy_for_a_webhook_carrying_the_result(redis_url):
    from crawl_job_queue import CrawlJobQueue, CrawlJobResultBudgetFull

    # 6 MiB of slack: one 4 MiB result fits, its outbox copy as well does not.
    with redis_maxmemory(redis_url, used_memory(redis_url) + 64 * MIB):
        client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            queue = CrawlJobQueue(client, memory_config(58 * MIB))
            await queue.ensure_group()
            with pytest.raises(CrawlJobResultBudgetFull):
                await queue.enqueue(urls=["https://example.com"], browser_config={},
                    crawler_config={}, result_fields=None, owner="alice",
                    webhook_config={"webhook_url": "https://example.com/hook",
                                    "webhook_data_in_payload": True})
            assert await client.xlen(queue.settings.stream) == 0
            task_id = await enqueue(queue)
            assert await client.hget(queue.task_key(task_id), "status") == "processing"
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_enqueue_is_uncapped_when_redis_has_no_maxmemory(redis_url):
    from crawl_job_queue import CrawlJobQueue

    with redis_maxmemory(redis_url, 0):
        client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            queue = CrawlJobQueue(client, memory_config(2**50))
            await queue.ensure_group()
            task_id = await enqueue(queue)
            assert await client.hget(queue.task_key(task_id), "status") == "processing"
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_script_write_over_noeviction_maxmemory_raises_out_of_memory(redis_url):
    from crawl_job_queue import CrawlJobQueue

    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        queue = CrawlJobQueue(client, memory_config(MIB))
        await queue.ensure_group()
        task_id = await enqueue(queue)
        entry = (await queue.read_new("worker-a"))[0]
        payload = await queue.load_payload(entry.task_id)
        attempt = await queue.start_attempt(entry, payload, "worker-a")
        with redis_maxmemory(redis_url, 1, "noeviction"):
            with pytest.raises(redis.exceptions.OutOfMemoryError):
                await queue.complete(entry, payload, attempt, result={"ok": True})
        assert await client.hget(queue.task_key(task_id), "status") == "processing"
        assert await client.xlen(queue.settings.stream) == 1
    finally:
        await client.aclose()
