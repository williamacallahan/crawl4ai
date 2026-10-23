"""Real Redis/HTTP coverage for old and new workers sharing the v3 queue.

The legacy queue, worker, and webhook service execute the unchanged pre-upgrade
source frozen in fixtures/crawl_job_v3_legacy.json, including its exact Lua.
The snapshot records its Git revision; shallow CI checkouts need no Git history
or network. Other application dependencies remain shared. No Redis transaction,
HTTP delivery, or legacy attempt marker is simulated.
"""

import asyncio
import contextlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest
from redis import asyncio as aioredis

from crawl_job_queue import (
    CrawlJobPrincipalQuotaExceeded, CrawlJobQueue, CrawlJobQueueFull,
)
from crawl_job_worker import CrawlJobWorker
from test_crawl_job_durability import local_webhook, receiver  # noqa: F401
from test_crawl_job_queue_redis import redis_url  # noqa: F401
from work_queue import principal_lease_key

@pytest.fixture(scope="session")
def legacy_modules():
    """Load real old implementations without replacing current import names."""
    snapshot = json.loads(
        (Path(__file__).parent / "fixtures" / "crawl_job_v3_legacy.json").read_text()
    )
    modules = {}
    try:
        for name in ("crawl_job_queue", "webhook", "crawl_job_worker"):
            module = ModuleType(f"_mixed_version_legacy_{name}")
            modules[name] = module
            sys.modules[module.__name__] = module  # dataclasses resolves this namespace
            source = snapshot["modules"][name]
            origin = f"{snapshot['revision']}/{name}.py"
            exec(compile(source, origin, "exec"), module.__dict__)
        queue, worker = modules["crawl_job_queue"], modules["crawl_job_worker"]
        # The old worker's bare imports initially resolve to current modules.
        # Rebind only its private namespace to the matching old implementations.
        for name in ("CrawlJobQueue", "CrawlJobEntry", "CrawlJobAttempt", "CrawlJobLeaseLost"):
            setattr(worker, name, getattr(queue, name))
        worker.WebhookDeliveryService = modules["webhook"].WebhookDeliveryService
        yield SimpleNamespace(queue=queue, worker=worker)
    finally:
        for module in modules.values():
            sys.modules.pop(module.__name__, None)


@pytest.fixture(autouse=True)
def public_seeds(monkeypatch, legacy_modules):
    monkeypatch.setattr("crawl_job_queue.validate_url_destination", lambda _url: None)
    monkeypatch.setattr(legacy_modules.queue, "validate_url_destination", lambda _url: None)


@contextlib.asynccontextmanager
async def mixed_queue(redis_url, legacy_modules):
    namespace = f"mixed-version-{uuid4().hex}"
    config = {
        "redis": {"task_ttl_seconds": 60},
        "crawl_jobs": {
            "stream": namespace, "group": namespace, "lease_seconds": 2,
            "heartbeat_seconds": 1, "read_block_ms": 1, "max_attempts": 2,
            "max_pending_jobs": 4, "max_attempt_seconds": 5,
        },
        "limits": {"queue": {"per_principal": 1}},
        "webhooks": {"enabled": True},
    }
    client = aioredis.from_url(redis_url)
    current = CrawlJobQueue(client, config)
    legacy = legacy_modules.queue.CrawlJobQueue(client, config)
    versions = {"old": legacy, "new": current}
    tasks, owners = [], set()

    async def enqueue(version, owner, webhook=None):
        owner = f"{namespace}:{owner}"
        owners.add(owner)
        task_id = await versions[version].enqueue(
            ["https://example.com"], {}, {}, None, webhook, owner=owner,
        )
        tasks.append(task_id)
        return task_id

    try:
        await legacy.ensure_group()
        await current.ensure_group()
        assert legacy.settings.stream == current.settings.stream
        assert legacy.settings.group == current.settings.group
        yield client, current, versions, enqueue
    finally:
        keys = [key async for key in client.scan_iter(match=f"{namespace}*")]
        keys += [principal_lease_key(owner) for owner in owners]
        for task_id in tasks:
            keys += [current.task_key(task_id), current.payload_key(task_id)]
        if keys:
            await client.delete(*keys)
        await client.aclose()


async def make_stale(client, queue, entry, consumer):
    await client.xclaim(queue.settings.stream, queue.settings.group, consumer, 0,
                        [entry.stream_id], idle=3_000)


@pytest.mark.parametrize("producer,completer", [
    ("old", "new"), ("new", "old"), ("old", "old"), ("new", "new"),
])
def test_mixed_admission_completion_keeps_backlog_and_quota_consistent(
    redis_url, legacy_modules, producer, completer,
):
    async def exercise():
        async with mixed_queue(redis_url, legacy_modules) as (client, queue, versions, enqueue):
            webhook = {"webhook_url": "https://callback.example/done"}
            task_ids = [await enqueue(producer, 0, webhook)]
            for version in versions:
                with pytest.raises((CrawlJobPrincipalQuotaExceeded,
                                    legacy_modules.queue.CrawlJobPrincipalQuotaExceeded)):
                    await enqueue(version, 0)
            task_ids += [
                await enqueue(producer, owner, webhook)
                for owner in range(1, queue.settings.max_pending_jobs)
            ]
            assert await client.xlen(queue.settings.stream) == queue.settings.max_pending_jobs
            for version in versions:
                with pytest.raises((CrawlJobQueueFull,
                                    legacy_modules.queue.CrawlJobQueueFull)):
                    await enqueue(version, "full")

            for task_id in task_ids:
                entry = (await queue.read_new("completion"))[0]
                assert entry.task_id == task_id
                payload = await queue.load_payload(task_id)
                quota_key = principal_lease_key(payload["owner"])
                assert await client.zscore(quota_key, task_id) == float("inf")
                attempt = await versions[completer].start_attempt(entry, payload, "completion")
                await versions[completer].complete(entry, payload, attempt, result={"ok": True})
                assert await client.hget(queue.task_key(task_id), "status") == b"completed"
                assert await client.exists(queue.payload_key(task_id)) == 0
                assert await client.zcard(quota_key) == 0

            assert await client.xlen(queue.settings.stream) == 0
            assert (await client.xpending(queue.settings.stream, queue.settings.group))["pending"] == 0
            outbox = await client.xrange(queue.outbox_stream)
            assert len(outbox) == (len(task_ids) if completer == "new" else 0)
            # Old completion already owns its inline webhook; no new sender may
            # invent another notification for that terminal task.
            if completer == "old":
                assert await queue.read_outbox("new-sender") == []

            # max_pending_jobs remains the crawl XLEN contract. A webhook
            # backlog must not silently reduce admission during rolling deploys.
            replacements = [
                await enqueue(version, owner)
                for owner, version in enumerate(("old", "new", "old", "new"))
            ]
            assert len(set(replacements + task_ids)) == 8
            assert await client.xlen(queue.settings.stream) == queue.settings.max_pending_jobs
            assert await client.xrange(queue.outbox_stream) == outbox
            for version in versions:
                with pytest.raises((CrawlJobQueueFull,
                                    legacy_modules.queue.CrawlJobQueueFull)):
                    await enqueue(version, "still-full")

    asyncio.run(exercise())


@pytest.mark.parametrize("producer,completer", [
    ("old", "old"), ("old", "new"), ("new", "old"), ("new", "new"),
])
@pytest.mark.parametrize("success", [True, False], ids=["completed", "failed"])
@pytest.mark.parametrize("default_webhook", [False, True], ids=["per-job-url", "default-url"])
def test_mixed_workers_deliver_terminal_webhook_once(
    redis_url, local_webhook, legacy_modules, producer, completer, success, default_webhook,
):
    async def exercise():
        with receiver() as (url, requests):
            async with mixed_queue(redis_url, legacy_modules) as (client, queue, versions, enqueue):
                webhook = {"webhook_url": url, "webhook_data_in_payload": True}
                if default_webhook:
                    queue.config["webhooks"].update(default_url=url, data_in_payload=True)
                    webhook = None
                task_id = await enqueue(producer, "owner", webhook)
                active = versions[completer]
                entry = (await active.read_new("completer"))[0]
                payload = await active.load_payload(task_id)
                result = {"success": success, "results": [{"url": "https://example.com"}]}

                async def crawl(_payload):
                    return result

                worker_class = (legacy_modules.worker.CrawlJobWorker
                                if completer == "old" else CrawlJobWorker)
                await worker_class(active, queue.config, "completer", crawl=crawl).process(entry)
                expected_status = "completed" if success else "failed"
                assert await client.hget(queue.task_key(task_id), "status") == expected_status.encode()
                assert len(requests) == 1
                assert requests[0]["task_id"] == task_id
                assert requests[0]["status"] == expected_status
                assert requests[0]["data"] == result
                assert requests[0].get("error") == (None if success else "Every crawled URL failed")
                if completer == "new":
                    assert requests[0]["event_id"] == f"crawl-terminal:{task_id}"
                else:
                    assert "event_id" not in requests[0]
                assert await client.exists(queue.payload_key(task_id)) == 0
                assert await client.zcard(principal_lease_key(payload["owner"])) == 0
                assert await client.xlen(queue.settings.stream) == 0
                assert (await client.xpending(queue.settings.stream, queue.settings.group))["pending"] == 0
                assert await client.xlen(queue.outbox_stream) == 0
                assert (await client.xpending(queue.outbox_stream, queue.outbox_group))["pending"] == 0
                # A new sender after old direct delivery must not synthesize a
                # second notification from the surviving terminal task hash.
                restarted = CrawlJobQueue(client, queue.config)
                await restarted.ensure_group()
                assert await restarted.read_outbox("restart") == []
                assert await restarted.pending_outbox("completer") == []
                assert len(requests) == 1
                assert await enqueue("new", "owner") != task_id

    asyncio.run(exercise())


@pytest.mark.parametrize("retry", [False, True], ids=["legacy-running", "legacy-retry"])
@pytest.mark.parametrize("spent", [1, 2], ids=["one-attempt-left", "budget-exhausted"])
def test_new_reclaim_preserves_old_attempt_budget_without_poisoning(
    redis_url, legacy_modules, retry, spent,
):
    async def exercise():
        async with mixed_queue(redis_url, legacy_modules) as (client, queue, versions, enqueue):
            legacy = versions["old"]
            task_id = await enqueue("old", "owner")
            entry = (await queue.read_new("old-worker"))[0]
            payload = await queue.load_payload(task_id)
            first = await legacy.start_attempt(entry, payload, "old-worker")
            assert first.number == 1
            if spent == 2:
                await legacy.mark_retry(entry, payload, "old-worker", first, "retry old attempt")
                await make_stale(client, queue, entry, "old-worker")
                assert len(await legacy.claim_stale("old-worker")) == 1
                first = await legacy.start_attempt(entry, payload, "old-worker")
                assert first.number == 2
            if retry:
                await legacy.mark_retry(entry, payload, "old-worker", first, "clean old retry")
            assert await client.hget(queue.payload_key(task_id), "attempt_state") is None
            await make_stale(client, queue, entry, "old-worker")
            assert await queue.claim_stale("new-worker") == [entry]
            assert await client.hget(queue.payload_key(task_id), "attempt") == str(spent).encode()
            assert await client.hget(queue.payload_key(task_id), "worker_died") is None
            assert await client.zscore(principal_lease_key(payload["owner"]), task_id) == float("inf")
            with pytest.raises(legacy_modules.queue.CrawlJobLeaseLost):
                await legacy.complete(entry, payload, first, result={"stale": True})

            calls = []
            async def crawl(_payload):
                calls.append(task_id)
                return {"success": True}
            await CrawlJobWorker(queue, queue.config, "new-worker", crawl=crawl).process(entry)
            assert calls == ([task_id] if spent == 1 else [])
            terminal = await client.hgetall(queue.task_key(task_id))
            assert terminal[b"attempt"] == str(spent + 1).encode()
            assert terminal[b"status"] == (b"completed" if spent == 1 else b"failed")
            if spent == 2:
                assert b"exhausted its retry budget" in terminal[b"error"]
            assert b"worker_died" not in terminal[b"error"]
            assert await client.zcard(principal_lease_key(payload["owner"])) == 0
            assert await client.xlen(queue.settings.stream) == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("retry", [False, True], ids=["stale-running-marker", "stale-retry-marker"])
def test_legacy_start_supersedes_new_marker_and_keeps_retry_limit(redis_url, legacy_modules, retry):
    async def exercise():
        async with mixed_queue(redis_url, legacy_modules) as (client, queue, versions, enqueue):
            legacy = versions["old"]
            task_id = await enqueue("new", "owner")
            entry = (await queue.read_new("new-first"))[0]
            payload = await queue.load_payload(task_id)
            first = await queue.start_attempt(entry, payload, "new-first")
            if retry:
                await queue.mark_retry(entry, payload, "new-first", first, "clean new retry")
            await make_stale(client, queue, entry, "new-first")
            assert [item.stream_id for item in await legacy.claim_stale("old-second")] == [entry.stream_id]
            second = await legacy.start_attempt(entry, payload, "old-second")
            assert second.number == 2
            assert await client.hget(queue.payload_key(task_id), "start_operation") == first.fence_token.encode()
            assert second.fence_token != first.fence_token

            await make_stale(client, queue, entry, "old-second")
            assert await queue.claim_stale("new-third") == [entry]
            assert await client.hget(queue.payload_key(task_id), "worker_died") is None
            calls = []
            async def crawl(_payload):
                calls.append(task_id)
                return {"success": True}
            # The legacy attempt still spent the second slot. A new worker
            # terminalizes exhaustion without executing a third browser crawl.
            await CrawlJobWorker(queue, queue.config, "new-third", crawl=crawl).process(entry)
            assert calls == []
            terminal = await client.hgetall(queue.task_key(task_id))
            assert terminal[b"attempt"] == b"3"
            assert terminal[b"status"] == b"failed"
            assert b"exhausted its retry budget" in terminal[b"error"]
            assert b"worker_died" not in terminal[b"error"]
            assert await client.zcard(principal_lease_key(payload["owner"])) == 0
            assert await client.xlen(queue.settings.stream) == 0
            assert await enqueue("new", "owner") != task_id

    asyncio.run(exercise())


def test_committed_webhook_survives_old_worker_rollback_and_new_recovery(
    redis_url, local_webhook, legacy_modules,
):
    async def exercise():
        with receiver() as (url, requests):
            async with mixed_queue(redis_url, legacy_modules) as (client, queue, versions, enqueue):
                webhook = {"webhook_url": url, "webhook_data_in_payload": True}
                task_id = await enqueue("old", "owner", webhook)
                entry = (await queue.read_new("crashed-new"))[0]
                payload = await queue.load_payload(task_id)
                attempt = await queue.start_attempt(entry, payload, "crashed-new")
                result = {"success": True, "body": "frozen before rollback"}
                # Stop at the real commit boundary, before the worker's inline
                # delivery, as if that process exited immediately after Redis.
                await queue.complete(entry, payload, attempt, result=result)
                outbox = await client.xrange(queue.outbox_stream)
                assert len(outbox) == 1
                stream_id = outbox[0][0]
                pending = await client.xpending_range(queue.outbox_stream, queue.outbox_group,
                                                      "-", "+", 10)
                assert len(pending) == 1
                assert pending[0]["consumer"] == b"crashed-new"
                assert await queue.read_outbox("too-early") == []
                assert requests == []

                # During rollback, old workers only drain crawl work. Their
                # old enqueue/start/complete scripts must retain the new PEL.
                legacy = versions["old"]
                old_id = await enqueue("old", "owner", webhook)
                old_entry = (await legacy.read_new("rollback-old"))[0]
                assert old_entry.task_id == old_id
                async def old_crawl(_payload):
                    return {"old": True}
                await legacy_modules.worker.CrawlJobWorker(
                    legacy, queue.config, "rollback-old", crawl=old_crawl,
                ).process(old_entry)
                assert await client.xrange(queue.outbox_stream) == outbox
                assert await client.xlen(queue.settings.stream) == 0
                assert (await client.xpending(queue.outbox_stream, queue.outbox_group))["pending"] == 1
                assert len(requests) == 1
                assert requests[0]["task_id"] == old_id
                assert requests[0]["data"] == {"old": True}
                assert "event_id" not in requests[0]
                await client.delete(queue.task_key(task_id))

                # A restored new worker recovers the existing event after the
                # grace period, using its frozen data despite task eviction.
                restored = CrawlJobQueue(client, queue.config)
                await restored.ensure_group()
                await client.xclaim(queue.outbox_stream, queue.outbox_group, "crashed-new", 0,
                                    [stream_id], idle=3_000)
                events = await restored.read_outbox("restored-new")
                assert len(events) == 1
                event_id, event = events[0]
                assert event_id == stream_id.decode()
                assert event["task_id"] == task_id
                assert event["result"] == result
                await CrawlJobWorker(restored, queue.config, "restored-new").process_outbox(*events[0])
                assert len(requests) == 2
                assert requests[1]["task_id"] == task_id
                assert requests[1]["event_id"] == event["event_id"]
                assert requests[1]["data"] == result
                assert await client.xlen(queue.outbox_stream) == 0
                assert (await client.xpending(queue.outbox_stream, queue.outbox_group))["pending"] == 0
                assert await restored.read_outbox("another-restart") == []
                assert len(requests) == 2

    asyncio.run(exercise())
