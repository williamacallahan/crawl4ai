import asyncio
import inspect
import json
import os
import sys
from collections import defaultdict

import pytest
from fastapi import HTTPException
from redis.exceptions import ResponseError

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

from crawl_job_queue import CrawlJobEntry, CrawlJobQueue, CrawlJobQueueFull
from crawl_job_worker import CrawlJobWorker
from api import handle_crawl_job


def queue_config(**overrides):
    settings = {
        "stream": "crawl-jobs",
        "group": "crawl-workers",
        "lease_seconds": 2,
        "heartbeat_seconds": 1,
        "read_block_ms": 1,
        "max_attempts": 3,
        "max_pending_jobs": 3,
    }
    settings.update(overrides)
    return {"redis": {"task_ttl_seconds": 60}, "crawl_jobs": settings}


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.operations = []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self.operations.append((name, args, kwargs))
            return self

        return queue

    async def execute(self):
        return [await getattr(self.redis, name)(*args, **kwargs) for name, args, kwargs in self.operations]


class FakeRedis:
    def __init__(self):
        self.hashes = defaultdict(dict)
        self.streams = defaultdict(list)
        self.groups = set()
        self.pending = {}
        self.acks = []
        self.deleted = []
        self.claims = []
        self.expires = []
        self.eval_scripts = []
        self._sequence = 0

    def pipeline(self, transaction=True):
        assert transaction is True
        return FakePipeline(self)

    async def hset(self, key, key_name=None, value=None, mapping=None, **_kwargs):
        if mapping:
            self.hashes[key].update({
                str(name): str(field.value if hasattr(field, "value") else field)
                for name, field in mapping.items()
            })
        elif key_name is not None:
            self.hashes[key][str(key_name)] = str(value)
        return 1

    async def hget(self, key, field):
        return self.hashes[key].get(field)

    async def hincrby(self, key, field, amount):
        updated = int(self.hashes[key].get(field, "0")) + amount
        self.hashes[key][field] = str(updated)
        return updated

    async def expire(self, _key, _seconds):
        self.expires.append((_key, _seconds))
        return True

    async def delete(self, key):
        self.hashes.pop(key, None)
        return 1

    async def xadd(self, stream, fields):
        self._sequence += 1
        stream_id = f"{self._sequence}-0"
        self.streams[stream].append((stream_id, dict(fields)))
        return stream_id

    async def eval(self, script, numkeys, *keys_and_args):
        assert numkeys == 3
        self.eval_scripts.append(script)
        (
            stream,
            task_key,
            payload_key,
            max_pending_jobs,
            status,
            created_at,
            url,
            result,
            error,
            webhook_config,
            payload,
            task_id,
        ) = keys_and_args
        if len(self.streams[stream]) >= int(max_pending_jobs):
            return 0
        await self.hset(
            task_key,
            mapping={
                "status": status,
                "created_at": created_at,
                "url": url,
                "result": result,
                "error": error,
                "webhook_config": webhook_config,
            },
        )
        await self.hset(payload_key, mapping={"payload": payload, "attempt": "0"})
        await self.xadd(stream, {"task_id": task_id})
        return 1

    async def xgroup_create(self, stream, group, id="0-0", mkstream=False):
        key = (stream, group)
        if key in self.groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        self.groups.add(key)
        if mkstream:
            self.streams[stream]
        return True

    async def xreadgroup(self, group, consumer, streams, count=None, block=None):
        assert count == 1
        assert block is not None
        stream, marker = next(iter(streams.items()))
        assert marker == ">"
        for stream_id, fields in self.streams[stream]:
            if stream_id not in self.pending:
                self.pending[stream_id] = consumer
                return [(stream, [(stream_id, fields)])]
        return []

    async def xautoclaim(self, stream, group, consumer, min_idle_time, start_id="0-0", count=None):
        assert group == "crawl-workers"
        assert min_idle_time > 0
        assert count == 1
        for stream_id, fields in self.streams[stream]:
            if stream_id in self.pending:
                self.pending[stream_id] = consumer
                return ["0-0", [(stream_id, fields)], []]
        return ["0-0", [], []]

    async def xclaim(self, stream, group, consumer, min_idle_time, message_ids, idle=None, retrycount=None):
        assert group == "crawl-workers"
        assert min_idle_time == 0
        assert idle == 0
        assert retrycount is not None
        stream_id = message_ids[0]
        if stream_id not in self.pending:
            return []
        self.pending[stream_id] = consumer
        self.claims.append(stream_id)
        return [(stream_id, {})]

    async def xack(self, stream, group, stream_id):
        assert group == "crawl-workers"
        self.pending.pop(stream_id, None)
        self.acks.append((stream, stream_id))
        return 1

    async def xdel(self, stream, stream_id):
        self.streams[stream] = [entry for entry in self.streams[stream] if entry[0] != stream_id]
        self.deleted.append((stream, stream_id))
        return 1


class NoopWebhook:
    async def notify_job_completion(self, **_kwargs):
        return None


class TerminalWebhook:
    def __init__(self, redis, queue, entry):
        self.redis = redis
        self.queue = queue
        self.entry = entry
        self.called = False

    async def notify_job_completion(self, *, status, **_kwargs):
        assert self.entry.stream_id not in self.redis.pending
        assert self.redis.hashes[self.queue.task_key(self.entry.task_id)]["status"] == status
        self.called = True


def enqueue(redis, config):
    queue = CrawlJobQueue(redis, config)
    task_id = asyncio.run(
        queue.enqueue(
            urls=["https://example.com"],
            browser_config={"type": "BrowserConfig", "params": {}},
            crawler_config={"type": "CrawlerRunConfig", "params": {}},
            result_fields=["url", "success"],
            webhook_config=None,
        )
    )
    return queue, task_id


def test_enqueue_persists_payload_before_stream_visibility():
    redis = FakeRedis()
    queue, task_id = enqueue(redis, queue_config())

    payload = json.loads(redis.hashes[queue.payload_key(task_id)]["payload"])
    assert payload["urls"] == ["https://example.com"]
    assert redis.hashes[queue.task_key(task_id)]["status"] == "processing"
    assert redis.streams[queue.settings.stream][0][1] == {"task_id": task_id}
    assert redis.expires == []
    assert "XLEN" in redis.eval_scripts[0]
    assert "XADD" in redis.eval_scripts[0]


def test_full_queue_rejects_atomically_without_creating_task_or_payload():
    redis = FakeRedis()
    config = queue_config(max_pending_jobs=1)
    queue, _task_id = enqueue(redis, config)
    keys_before_rejection = set(redis.hashes)

    with pytest.raises(CrawlJobQueueFull):
        asyncio.run(
            queue.enqueue(
                urls=["https://second.example"],
                browser_config={},
                crawler_config={},
                result_fields=None,
                webhook_config=None,
            )
        )

    assert len(redis.streams[queue.settings.stream]) == 1
    assert set(redis.hashes) == keys_before_rejection


def test_consumer_group_creation_is_idempotent_for_two_supervised_workers():
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())

    asyncio.run(queue.ensure_group())
    asyncio.run(queue.ensure_group())

    assert redis.groups == {(queue.settings.stream, queue.settings.group)}


def test_public_crawl_job_submit_contract_has_no_background_task_dependency():
    redis = FakeRedis()
    config = queue_config(max_pending_jobs=1)

    response = asyncio.run(
        handle_crawl_job(
            redis,
            ["https://example.com"],
            {"type": "BrowserConfig", "params": {}},
            {"type": "CrawlerRunConfig", "params": {}},
            config,
        )
    )

    assert response["task_id"].startswith("crawl_")
    assert "background_tasks" not in inspect.signature(handle_crawl_job).parameters

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            handle_crawl_job(
                redis,
                ["https://second.example"],
                {"type": "BrowserConfig", "params": {}},
                {"type": "CrawlerRunConfig", "params": {}},
                config,
            )
        )
    assert error.value.status_code == 503


def test_worker_completes_and_removes_stream_entry_atomically():
    redis = FakeRedis()
    config = queue_config()
    queue, task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("worker-a"))[0]

    async def crawl(_payload):
        return {"success": True, "results": [{"url": "https://example.com"}]}

    webhook = TerminalWebhook(redis, queue, entry)
    worker = CrawlJobWorker(queue, config, "worker-a", crawl=crawl, webhook_service=webhook)
    asyncio.run(worker.process(entry))

    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "completed"
    assert json.loads(task["result"])["success"] is True
    assert queue.payload_key(task_id) not in redis.hashes
    assert redis.acks == [(queue.settings.stream, entry.stream_id)]
    assert redis.deleted == [(queue.settings.stream, entry.stream_id)]
    assert redis.expires == [(queue.task_key(task_id), 60)]
    assert webhook.called is True


def test_worker_retains_pending_entry_until_retry_budget_is_exhausted():
    redis = FakeRedis()
    config = queue_config(max_attempts=2)
    queue, task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("worker-a"))[0]

    async def crawl(_payload):
        raise RuntimeError("upstream unavailable")

    worker = CrawlJobWorker(queue, config, "worker-a", crawl=crawl, webhook_service=NoopWebhook())
    asyncio.run(worker.process(entry))
    assert entry.stream_id in redis.pending
    assert redis.hashes[queue.task_key(task_id)]["status"] == "processing"
    assert redis.acks == []

    asyncio.run(worker.process(entry))
    assert redis.hashes[queue.task_key(task_id)]["status"] == "failed"
    assert redis.acks == [(queue.settings.stream, entry.stream_id)]


def test_cancellation_leaves_the_pending_entry_unacknowledged():
    redis = FakeRedis()
    config = queue_config()
    queue, _task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("worker-a"))[0]

    async def crawl(_payload):
        raise asyncio.CancelledError

    worker = CrawlJobWorker(queue, config, "worker-a", crawl=crawl, webhook_service=NoopWebhook())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker.process(entry))

    assert entry.stream_id in redis.pending
    assert redis.acks == []


def test_stale_claim_and_heartbeat_use_redis_stream_leases():
    redis = FakeRedis()
    config = queue_config()
    queue, _task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("dead-worker"))[0]

    claimed = asyncio.run(queue.claim_stale("worker-a"))
    assert claimed == [entry]

    payload = asyncio.run(queue.load_payload(entry.task_id))
    asyncio.run(queue.heartbeat(entry, payload, "worker-a", 1))
    assert redis.claims == [entry.stream_id]


def test_worker_heartbeats_during_a_long_crawl():
    redis = FakeRedis()
    config = queue_config()
    queue, _task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("worker-a"))[0]

    async def crawl(_payload):
        await asyncio.sleep(1.1)
        return {"success": True, "results": []}

    worker = CrawlJobWorker(queue, config, "worker-a", crawl=crawl, webhook_service=NoopWebhook())
    asyncio.run(worker.process(entry))

    assert redis.claims == [entry.stream_id]
