"""Unpolled LLM recovery and webhook fencing against real Redis and HTTP."""

import asyncio
import json
import os
import time as stdlib_time
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import api
import egress_broker
import pytest
from egress_broker import PinnedTarget
import redis as sync_redis
from redis import asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from test_crawl_job_queue_redis import redis_url  # noqa: F401
from utils import TaskStatus, decode_redis_hash


@pytest.fixture(autouse=True)
def isolated_redis(monkeypatch):
    """The reaper SCANs every task:llm_* key, so give it a database no other test uses.

    With a configured disposable Redis (CI's per-run container, a local test
    container) that is logical database 15, flushed before and after each test;
    without one, the imported redis_url fixture starts a private redis-server.
    """
    configured = os.getenv("REDIS_URL")
    if not configured:
        yield
        return
    isolated = urlunsplit(urlsplit(configured)._replace(path="/15"))
    monkeypatch.setenv("REDIS_URL", isolated)
    client = sync_redis.from_url(isolated)
    try:
        client.flushdb()
        yield
        client.flushdb()
    finally:
        client.close()


def reaper_config():
    return {
        "llm": {
            "task_heartbeat_seconds": 30,
            "orphan_scan_seconds": 60,
            "orphan_scan_count": 100,
        },
        "redis": {"task_ttl_seconds": 60},
        "webhooks": {"retry": {"max_attempts": 1, "timeout_ms": 2000}},
    }


async def seed_task(client, task_id, heartbeat_at=None, webhook_config=None):
    seconds, _microseconds = await client.time()
    mapping = {
        "status": TaskStatus.PROCESSING.value,
        "created_at": datetime.fromtimestamp(seconds - 120).isoformat(),
        "url": "https://example.com/article",
    }
    if heartbeat_at is not None:
        mapping["heartbeat_at"] = heartbeat_at
    if webhook_config is not None:
        mapping["webhook_config"] = json.dumps(webhook_config)
    await client.hset(f"task:{task_id}", mapping=mapping)


@asynccontextmanager
async def webhook_receiver(monkeypatch):
    receipts = []

    async def receive(reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            headers = dict(
                (name.lower(), value.strip())
                for name, value in (
                    line.split(":", 1) for line in lines[1:] if line
                )
            )
            body = await reader.readexactly(int(headers["content-length"]))
            receipts.append((lines[0], headers, json.loads(body)))
            writer.write(
                b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(receive, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"http://hooks.example:{port}/orphan"

    def pin(target):
        assert target == url
        return PinnedTarget("http", "hooks.example", port, "127.0.0.1")

    # Keep the actual WebhookDeliveryService, resolver, and HTTP request. Only
    # the validated address is redirected to the local receipt server.
    monkeypatch.setattr(egress_broker, "resolve_and_pin", pin)
    try:
        yield url, receipts
    finally:
        server.close()
        await server.wait_closed()


def test_one_scan_recovers_unpolled_orphan_and_preserves_live_and_legacy(redis_url):
    async def exercise():
        client = aioredis.from_url(redis_url)
        task_ids = [f"llm_{uuid4().hex}" for _ in range(3)]
        stale, live, legacy = task_ids
        crawl = f"crawl_{uuid4().hex}"
        keys = [f"task:{task_id}" for task_id in [*task_ids, crawl]]
        try:
            seconds, _microseconds = await client.time()
            await seed_task(client, stale, seconds - 91)
            await seed_task(client, live, seconds)
            await seed_task(client, legacy)
            await seed_task(client, crawl, seconds - 91)
            unchanged = {
                task_id: await client.hgetall(f"task:{task_id}")
                for task_id in (live, legacy, crawl)
            }

            # No status endpoint is called: one SCAN page must terminalize
            # the dead job even when its caller never polls again.
            cursor = await api.scan_llm_orphans(client, reaper_config())

            assert cursor == 0
            stored = decode_redis_hash(await client.hgetall(f"task:{stale}"))
            assert stored["status"] == TaskStatus.FAILED.value
            assert stored["error"] == "worker lost before completion"
            assert 59 <= await client.ttl(f"task:{stale}") <= 60
            for task_id, original in unchanged.items():
                assert await client.hgetall(f"task:{task_id}") == original
                assert await client.ttl(f"task:{task_id}") == -1
        finally:
            await client.delete(*keys)
            await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("competitor", ["reaper", "status"])
def test_recovery_race_delivers_one_real_webhook(redis_url, monkeypatch, competitor):
    """Two replicas, or a replica and status read, must share the Lua winner."""
    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        config = reaper_config()
        try:
            async with webhook_receiver(monkeypatch) as (url, receipts):
                seconds, _microseconds = await client.time()
                await seed_task(client, task_id, seconds - 91, {
                    "webhook_url": url,
                    "webhook_headers": {"X-Recovery-Test": task_id},
                })
                evaluate = client.eval
                ready = asyncio.Event()
                arrivals = 0

                async def race_recovery(script, *args, **kwargs):
                    nonlocal arrivals
                    if script == api._RECOVER_LLM_TASK_SCRIPT and arrivals < 2:
                        arrivals += 1
                        if arrivals == 2:
                            ready.set()
                        await ready.wait()
                    return await evaluate(script, *args, **kwargs)

                # Both contenders reach the real Redis Lua boundary with the
                # same stale job. Neither Redis nor webhook outcomes are faked.
                monkeypatch.setattr(client, "eval", race_recovery)
                other = (
                    api.scan_llm_orphans(client, config)
                    if competitor == "reaper"
                    else api.handle_task_status(
                        client, task_id, "http://crawl.example/",
                        collection="llm/job", config=config,
                    )
                )
                results = await asyncio.wait_for(asyncio.gather(
                    api.scan_llm_orphans(client, config), other,
                ), timeout=5)
                assert arrivals == 2
                assert results[0] == 0
                if competitor == "status":
                    assert json.loads(results[1].body)["status"] == "failed"

                stored = decode_redis_hash(await client.hgetall(key))
                assert stored["status"] == "failed"
                assert 59 <= await client.ttl(key) <= 60
                # Delivery runs in the background so a poll never waits on the
                # receiver; wait for it, then make sure no second one follows.
                for _ in range(50):
                    if receipts:
                        break
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.3)
                assert len(receipts) == 1
                request_line, headers, payload = receipts[0]
                assert request_line == "POST /orphan HTTP/1.1"
                assert headers["x-recovery-test"] == task_id
                assert payload["task_id"] == task_id
                assert payload["task_type"] == "llm_extraction"
                assert payload["status"] == "failed"
                assert payload["urls"] == [stored["url"]]
                assert payload["error"] == stored["error"] == "worker lost before completion"

                deadline = await client.pexpiretime(key)
                await api.scan_llm_orphans(client, config)
                await api.handle_task_status(
                    client, task_id, "http://crawl.example/",
                    collection="llm/job", config=config,
                )
                assert len(receipts) == 1
                assert await client.pexpiretime(key) == deadline
        finally:
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


def test_permit_renewal_connection_error_preserves_work_until_real_renewal(
    redis_url, monkeypatch,
):
    async def exercise():
        client = aioredis.from_url(redis_url)
        evaluate = client.eval
        renewal_succeeded = asyncio.Event()
        failure_ttls = []
        renewal_results = []

        async def fail_first_renewal(script, *args, **kwargs):
            if script == api._RENEW_LLM_PERMIT:
                if not failure_ttls:
                    failure_ttls.append(await client.pttl(api.LLM_PERMIT_KEY))
                    raise RedisConnectionError("temporary Redis connection interruption")
                result = await evaluate(script, *args, **kwargs)
                renewal_results.append(result)
                renewal_succeeded.set()
                return result
            return await evaluate(script, *args, **kwargs)

        monkeypatch.setattr(client, "eval", fail_first_renewal)
        try:
            async with api.llm_permit(client, {"llm": {"permit_ttl_seconds": 3}}):
                token = await client.get(api.LLM_PERMIT_KEY)
                initial_deadline = await client.pexpiretime(api.LLM_PERMIT_KEY)
                await asyncio.wait_for(renewal_succeeded.wait(), timeout=4)
                assert 0 < failure_ttls[0] <= 3000
                assert renewal_results == [1]
                assert await client.get(api.LLM_PERMIT_KEY) == token
                assert await client.pexpiretime(api.LLM_PERMIT_KEY) > initial_deadline
            assert await client.exists(api.LLM_PERMIT_KEY) == 0
        finally:
            await client.delete(api.LLM_PERMIT_KEY)
            await client.aclose()

    asyncio.run(exercise())


def test_accept_heartbeat_uses_redis_time_despite_host_clock_skew(redis_url, monkeypatch):
    queued = {}

    async def capture_job(_background_tasks, _factory, principal=None, on_cancel=None):
        queued["on_cancel"] = on_cancel

    monkeypatch.setattr(api, "time", SimpleNamespace(
        time=lambda: 0, monotonic=stdlib_time.monotonic,
    ))
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_enqueue_job", capture_job)

    async def exercise():
        client = aioredis.from_url(redis_url)
        key = None
        config = reaper_config()
        try:
            before = (await client.time())[0]
            response = await api.create_new_task(
                client, api.BackgroundTasks(), "https://example.com/article", "extract",
                schema=None, cache="0", base_url="http://crawl.example/", config=config,
            )
            after = (await client.time())[0]
            assert response.status_code == 202
            task_id = json.loads(response.body)["task_id"]
            key = f"task:{task_id}"
            snapshot = await client.hgetall(key)
            stored = decode_redis_hash(snapshot)
            assert before <= int(stored["heartbeat_at"]) <= after
            assert callable(queued["on_cancel"])

            status = await api.handle_task_status(
                client, task_id, "http://crawl.example/",
                collection="llm/job", config=config,
            )
            assert json.loads(status.body)["status"] == TaskStatus.PROCESSING.value
            assert await client.hgetall(key) == snapshot
            assert await client.ttl(key) == -1
        finally:
            try:
                if queued.get("on_cancel") is not None:
                    await queued["on_cancel"]()
            finally:
                if key is not None:
                    await client.delete(key)
                await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["connection-error", "hung-renewal"])
def test_permit_renewal_failure_cancels_only_at_confirmed_expiry(
    redis_url, monkeypatch, failure,
):
    async def exercise():
        client = aioredis.from_url(redis_url)
        evaluate = client.eval
        attempted = asyncio.Event()
        never = asyncio.Event()
        finished_normally = asyncio.Event()

        async def unavailable_renewal(script, *args, **kwargs):
            if script == api._RENEW_LLM_PERMIT:
                attempted.set()
                if failure == "connection-error":
                    raise RedisConnectionError("Redis remains unavailable for renewal")
                await never.wait()
            return await evaluate(script, *args, **kwargs)

        async def protected_work():
            async with api.llm_permit(client, {"llm": {"permit_ttl_seconds": 1}}):
                await never.wait()
            finished_normally.set()

        monkeypatch.setattr(client, "eval", unavailable_renewal)
        started = stdlib_time.monotonic()
        worker = asyncio.create_task(protected_work())
        try:
            await asyncio.wait_for(attempted.wait(), timeout=2)
            # Give the first failure time to propagate. The confirmed lease
            # still owns real Redis capacity, so its protected work must live.
            await asyncio.sleep(0.05)
            assert await client.pttl(api.LLM_PERMIT_KEY) > 0
            assert not worker.done()

            with pytest.raises(api.HTTPException) as error:
                await asyncio.wait_for(worker, timeout=2)
            elapsed = stdlib_time.monotonic() - started
            assert error.value.status_code == 503
            assert error.value.detail == "Crawl4AI LLM capacity lease was lost"
            assert 1 <= elapsed < 3
            assert not finished_normally.is_set()
            assert await client.exists(api.LLM_PERMIT_KEY) == 0
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await client.delete(api.LLM_PERMIT_KEY)
            await client.aclose()

    asyncio.run(exercise())


def test_orphan_scan_settings_defaults_and_overrides():
    assert api.llm_orphan_scan_settings({}) == (60, 100)
    assert api.llm_orphan_scan_settings({"llm": {
        "orphan_scan_seconds": 7, "orphan_scan_count": 3,
    }}) == (7, 3)


@pytest.mark.parametrize("setting", ["orphan_scan_seconds", "orphan_scan_count"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1", None])
def test_orphan_scan_settings_require_positive_integers(setting, value):
    with pytest.raises(ValueError, match=rf"llm\.{setting}"):
        api.llm_orphan_scan_settings({"llm": {setting: value}})


def test_job_cancelled_at_shutdown_still_sends_its_failure_webhook(redis_url, monkeypatch):
    """The reaper skips terminal tasks, so the shutdown cancel is the only notice."""
    async def exercise():
        client = aioredis.from_url(redis_url)
        config = reaper_config()
        cancelled = {}

        async def enqueue_then_shut_down(_background, _factory, principal=None, on_cancel=None):
            cancelled["done"] = await on_cancel()

        monkeypatch.setattr(api, "_enqueue_job", enqueue_then_shut_down)
        monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
        try:
            async with webhook_receiver(monkeypatch) as (url, receipts):
                response = await api.create_new_task(
                    client, None, "https://example.com", "extract", None, "0",
                    "https://crawl.example/", config,
                    webhook_config={"webhook_url": url, "webhook_headers": {}},
                )
                task_id = json.loads(response.body)["task_id"]
                stored = decode_redis_hash(await client.hgetall(f"task:{task_id}"))
                assert stored["status"] == TaskStatus.FAILED.value
                assert len(receipts) == 1
                assert receipts[0][2]["status"] == "failed"
                assert receipts[0][2]["error"] == "LLM extraction interrupted by server shutdown"
        finally:
            await client.aclose()

    asyncio.run(exercise())
