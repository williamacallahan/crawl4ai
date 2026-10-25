"""The cluster LLM permit survives redis-py's at-least-once command delivery.

redis-py's Retry resends a command whose reply was lost after Redis applied
it. These tests drive ``llm_permit`` against real Redis through a client that
applies commands without returning their replies, the way a dropped
connection does.
"""

import asyncio

import api
import pytest
from redis import asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from test_crawl_job_queue_redis import redis_url  # noqa: F401

CONFIG = {"llm": {"permit_ttl_seconds": 30}}


def test_replayed_acquisition_owns_the_permit_it_already_took(redis_url, monkeypatch):
    async def exercise():
        client = aioredis.from_url(redis_url)
        execute = client.execute_command

        async def replay(*args, **options):
            await execute(*args, **options)
            return await execute(*args, **options)

        monkeypatch.setattr(client, "execute_command", replay)
        try:
            await client.delete(api.LLM_PERMIT_KEY)
            async with api.llm_permit(client, CONFIG):
                assert await client.exists(api.LLM_PERMIT_KEY) == 1
            assert await client.exists(api.LLM_PERMIT_KEY) == 0
        finally:
            await client.delete(api.LLM_PERMIT_KEY)
            await client.aclose()

    asyncio.run(exercise())


def test_acquisition_that_loses_its_reply_releases_the_permit(redis_url, monkeypatch):
    async def exercise():
        client = aioredis.from_url(redis_url)
        execute = client.execute_command
        dropped = []

        async def drop_first_reply(*args, **options):
            result = await execute(*args, **options)
            if not dropped:
                dropped.append(args[0])
                raise RedisConnectionError("reply lost after Redis applied the command")
            return result

        monkeypatch.setattr(client, "execute_command", drop_first_reply)
        try:
            await execute("DEL", api.LLM_PERMIT_KEY)
            with pytest.raises(RedisConnectionError):
                async with api.llm_permit(client, CONFIG):
                    pytest.fail("a failed acquisition must not run protected work")
            assert await client.exists(api.LLM_PERMIT_KEY) == 0
        finally:
            await client.delete(api.LLM_PERMIT_KEY)
            await client.aclose()

    asyncio.run(exercise())


def test_another_holder_keeps_its_permit_and_callers_get_backpressure(redis_url):
    async def exercise():
        client = aioredis.from_url(redis_url)
        try:
            await client.set(api.LLM_PERMIT_KEY, "another-replica", ex=30)
            with pytest.raises(api.LlmCapacityBusy) as busy:
                async with api.llm_permit(client, CONFIG):
                    pytest.fail("a held permit must not admit a second LLM call")
            assert busy.value.status_code == 503
            assert busy.value.headers == {"Retry-After": "30"}
            assert await client.get(api.LLM_PERMIT_KEY) == b"another-replica"
        finally:
            await client.delete(api.LLM_PERMIT_KEY)
            await client.aclose()

    asyncio.run(exercise())
