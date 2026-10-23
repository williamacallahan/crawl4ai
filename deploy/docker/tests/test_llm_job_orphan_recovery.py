"""LLM orphan recovery and terminal fencing against real Redis."""

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import api
import pytest
from redis import asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from test_crawl_job_queue_redis import redis_url  # noqa: F401
from utils import TaskStatus, decode_redis_hash


CONFIG = {"llm": {"task_heartbeat_seconds": 30}, "redis": {"task_ttl_seconds": 60}}


@pytest.mark.parametrize(
    "case", ["heartbeat", "pre-heartbeat-release", "crawl-id"],
)
def test_stale_task_recovers_while_live_sibling_survives(redis_url, case):
    """Only a heartbeating LLM hash can be judged lost: a hash written by a
    release without heartbeats may still be running on an old replica during
    a rolling deploy, and a crawl job keeps its own lease recovery."""
    async def exercise():
        client = aioredis.from_url(redis_url)
        prefix = "crawl_" if case == "crawl-id" else "llm_"
        ids = [f"{prefix}{uuid4().hex}" for _ in range(2)]
        stale, live = ids
        old = time.time() - 91
        try:
            for task_id in ids:
                mapping = {
                    "status": TaskStatus.PROCESSING.value,
                    "created_at": datetime.fromtimestamp(old).isoformat(),
                    "url": "https://example.com",
                }
                if task_id == live or case != "pre-heartbeat-release":
                    mapping["heartbeat_at"] = time.time() if task_id == live else old
                await client.hset(f"task:{task_id}", mapping=mapping)

            for task_id in (stale, live):
                response = await api.handle_task_status(
                    client, task_id, "http://t/", collection="llm/job", config=CONFIG,
                )
                body = json.loads(response.body)
                stored = decode_redis_hash(await client.hgetall(f"task:{task_id}"))
                ttl = await client.ttl(f"task:{task_id}")
                assert response.status_code == 200
                if task_id == live or case != "heartbeat":
                    assert body["status"] == stored["status"] == TaskStatus.PROCESSING.value
                    assert ttl == -1
                else:
                    assert body["status"] == stored["status"] == TaskStatus.FAILED.value
                    assert "worker lost before completion" in body["error"] == stored["error"]
                    assert 59 <= ttl <= 60
                    # Reading a recovered terminal task must not renew retention.
                    await client.expire(f"task:{task_id}", 20)
                    again = await api.handle_task_status(
                        client, task_id, "http://t/", collection="llm/job", config=CONFIG,
                    )
                    assert json.loads(again.body) == body
                    assert 19 <= await client.ttl(f"task:{task_id}") <= 20
        finally:
            await client.delete(*(f"task:{task_id}" for task_id in ids))
            await client.aclose()

    asyncio.run(exercise())


@pytest.fixture
def extraction_worker(monkeypatch):
    from utils import load_config

    config = load_config()
    config["llm"].update(CONFIG["llm"])
    config["redis"].update(CONFIG["redis"])
    entered, finish = asyncio.Event(), asyncio.Event()

    class Crawler:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def arun(self, **_kwargs):
            entered.set()
            await finish.wait()
            return SimpleNamespace(success=True, extracted_content='{"answer": "late"}')

    from contextlib import asynccontextmanager
    import egress_broker
    import llm_broker

    @asynccontextmanager
    async def permit(*_args):
        yield

    webhook = SimpleNamespace(notify_job_completion=AsyncMock())
    monkeypatch.setattr(api, "AsyncWebCrawler", Crawler)
    monkeypatch.setattr(api, "llm_permit", permit)
    monkeypatch.setattr(api, "validate_llm_provider", lambda *_args: (True, None))
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "WebhookDeliveryService", lambda _config: webhook)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    monkeypatch.setattr(llm_broker, "resolve_llm", lambda *_args, **_kwargs: {
        "provider": "openai/test-model", "api_token": "test-only",
        "temperature": None, "base_url": None, "extra_args": {},
    })

    return config, entered, finish


def test_late_completion_cannot_overwrite_recovery(redis_url, extraction_worker):
    config, entered, finish = extraction_worker

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        worker = None
        try:
            await client.hset(key, mapping={
                "status": TaskStatus.PROCESSING.value,
                "created_at": datetime.now().isoformat(), "url": "https://example.com",
            })
            worker = asyncio.create_task(api.process_llm_extraction(
                client, config, task_id, "https://example.com", "extract",
            ))
            await asyncio.wait_for(entered.wait(), timeout=5)
            await client.hset(key, mapping={"heartbeat_at": time.time() - 91})
            await api.handle_task_status(
                client, task_id, "http://t/", collection="llm/job", config=config,
            )
            await client.expire(key, 20)
            finish.set()
            await asyncio.wait_for(worker, timeout=5)
            response = await api.handle_task_status(
                client, task_id, "http://t/", collection="llm/job", config=config,
            )
            body = json.loads(response.body)
            stored = decode_redis_hash(await client.hgetall(key))
            assert body["status"] == stored["status"] == TaskStatus.FAILED.value
            assert "worker lost before completion" in body["error"] == stored["error"]
            assert "result" not in stored
            assert 19 <= await client.ttl(key) <= 20
        finally:
            if worker is not None and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


def test_running_worker_refreshes_heartbeat_and_stops_at_completion(redis_url, extraction_worker):
    config, entered, finish = extraction_worker
    config["llm"]["task_heartbeat_seconds"] = 1

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        worker = None
        try:
            await client.hset(key, mapping={
                "status": TaskStatus.PROCESSING.value,
                "created_at": datetime.fromtimestamp(time.time() - 10).isoformat(),
                "url": "https://example.com",
            })
            worker = asyncio.create_task(api.process_llm_extraction(
                client, config, task_id, "https://example.com", "extract",
            ))
            await asyncio.wait_for(entered.wait(), timeout=5)
            first = await client.hget(key, "heartbeat_at")
            assert first is not None
            first = float(first)
            await asyncio.sleep(1.1)
            assert float(await client.hget(key, "heartbeat_at")) > first
            response = await api.handle_task_status(
                client, task_id, "http://t/", collection="llm/job", config=config,
            )
            assert json.loads(response.body)["status"] == TaskStatus.PROCESSING.value
            assert await client.ttl(key) == -1
            finish.set()
            await asyncio.wait_for(worker, timeout=5)
            completed = await client.hgetall(key)
            assert decode_redis_hash(completed)["status"] == TaskStatus.COMPLETED.value
            assert 59 <= await client.ttl(key) <= 60
            await asyncio.sleep(1.1)
            assert await client.hgetall(key) == completed
        finally:
            if worker is not None and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


def test_heartbeat_survives_a_redis_outage(redis_url):
    """A Redis restart takes seconds; the heartbeat must resume after it, or a
    later poll fails a job that is still running."""
    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        evaluate = client.eval
        outages = iter([True])

        async def eval_after_outage(*args):
            if next(outages, False):
                raise RedisConnectionError("Error 111 connecting to redis")
            return await evaluate(*args)

        client.eval = eval_after_outage
        config = {"llm": {"task_heartbeat_seconds": 1}}
        beating = asyncio.create_task(api._heartbeat_llm_task(client, task_id, config))
        try:
            await client.hset(key, mapping={"status": TaskStatus.PROCESSING.value})
            await asyncio.sleep(2.2)
            assert not beating.done()
            assert await client.hget(key, "heartbeat_at") is not None
        finally:
            beating.cancel()
            await asyncio.gather(beating, return_exceptions=True)
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())
