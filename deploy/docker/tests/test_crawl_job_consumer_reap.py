"""Exited workers' stream consumers are retired; live or owning ones are kept."""

import asyncio
from uuid import uuid4

import pytest
from redis import asyncio as aioredis
from redis.exceptions import ResponseError

from crawl_job_consumers import reap_idle_consumers
from crawl_job_queue import CrawlJobQueue
from test_crawl_job_queue_redis import redis_url  # noqa: F401


def reap_config(batch=100):
    return {"crawl_jobs": {"stream": f"crawl-jobs-reap-{uuid4().hex}", "group": "crawl-workers",
                           "lease_seconds": 2, "heartbeat_seconds": 1, "read_block_ms": 1,
                           "max_attempt_seconds": 1, "consumer_reap_batch": batch}}


async def consumers(queue, stream, group):
    return {row["name"].decode() for row in await queue.redis.xinfo_consumers(stream, group)}


async def with_queue(redis_url, config, exercise):
    client = aioredis.from_url(redis_url)
    queue = CrawlJobQueue(client, config)
    try:
        await queue.ensure_group()
        await exercise(queue)
    finally:
        await client.delete(queue.settings.stream, queue.outbox_stream)
        await client.aclose()


def test_reaps_idle_empty_consumers_in_both_groups_and_keeps_owners(redis_url):
    async def exercise(queue):
        s, redis = queue.settings, queue.redis
        await redis.xadd(s.stream, {"task_id": "crawl_pending"})
        await redis.xreadgroup(s.group, "exited-owner", {s.stream: ">"}, count=1)
        await redis.xreadgroup(s.group, "exited-empty", {s.stream: ">"}, count=1)
        outbox_id = await redis.xadd(queue.outbox_stream, {"event": "{}"})
        await redis.xclaim(queue.outbox_stream, queue.outbox_group, "exited-owner:outbox", 0,
                           [outbox_id], force=True, justid=True)
        await redis.xautoclaim(queue.outbox_stream, queue.outbox_group, "exited-empty:outbox", 60000)
        await asyncio.sleep(1.2)  # past max_attempt_seconds
        await redis.xreadgroup(s.group, "live", {s.stream: ">"}, count=1)
        await redis.xautoclaim(queue.outbox_stream, queue.outbox_group, "live:outbox", 60000)

        assert await reap_idle_consumers(queue) == 2
        assert await consumers(queue, s.stream, s.group) == {"exited-owner", "live"}
        assert await consumers(queue, queue.outbox_stream, queue.outbox_group) == {
            "exited-owner:outbox", "live:outbox"}
        pending = await redis.xpending(queue.outbox_stream, queue.outbox_group)
        assert pending["pending"] == 1

    asyncio.run(with_queue(redis_url, reap_config(), exercise))


def test_reap_is_bounded_per_pass_and_tolerates_an_expired_outbox(redis_url):
    async def exercise(queue):
        s, redis = queue.settings, queue.redis
        for name in ("exited-a", "exited-b", "exited-c"):
            await redis.xreadgroup(s.group, name, {s.stream: ">"}, count=1)
        # The outbox stream expires with its group after OUTBOX_TTL_SECONDS idle.
        await redis.delete(queue.outbox_stream)
        await asyncio.sleep(1.2)

        assert await reap_idle_consumers(queue) == 2
        assert len(await consumers(queue, s.stream, s.group)) == 1
        await redis.xadd(queue.outbox_stream, {"event": "{}"})
        assert await reap_idle_consumers(queue) == 1  # group still missing on the live stream

        await redis.delete(queue.outbox_stream)
        await redis.set(queue.outbox_stream, "not a stream")
        with pytest.raises(ResponseError, match="WRONGTYPE"):
            await reap_idle_consumers(queue)

    asyncio.run(with_queue(redis_url, reap_config(batch=2), exercise))
