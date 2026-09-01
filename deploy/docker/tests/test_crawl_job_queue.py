import asyncio
import inspect
import json
import os
import sys
import time
from collections import defaultdict

import pytest
from fastapi import HTTPException
from redis.exceptions import ResponseError

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawl_job_queue as crawl_job_queue_module  # noqa: E402
from api import handle_crawl_job  # noqa: E402
from crawl_job_queue import (  # noqa: E402
    CrawlJobLeaseLost,
    CrawlJobPayloadRejected,
    CrawlJobPrincipalQuotaExceeded,
    CrawlJobQueue,
    CrawlJobQueueFull,
)
from crawl_job_worker import CrawlJobWorker  # noqa: E402
from work_queue import principal_lease_key  # noqa: E402


@pytest.fixture(autouse=True)
def public_seed_validation(monkeypatch):
    monkeypatch.setattr(
        crawl_job_queue_module,
        "validate_url_destination",
        lambda _url: None,
    )


def queue_config(**overrides):
    settings = {
        "stream": "crawl-jobs",
        "group": "crawl-workers",
        "protocol_version": 2,
        "lease_seconds": 2,
        "heartbeat_seconds": 1,
        "read_block_ms": 1,
        "max_attempts": 3,
        "max_pending_jobs": 3,
        "max_payload_bytes": 262144,
    }
    settings.update(overrides)
    return {"redis": {"task_ttl_seconds": 60}, "crawl_jobs": settings}


class FakeRedis:
    def __init__(self):
        self.hashes = defaultdict(dict)
        self.zsets = defaultdict(dict)  # key -> {member: score}
        self.streams = defaultdict(list)
        self.groups = set()
        self.pending = {}
        self.acks = []
        self.deleted = []
        self.claims = []
        self.expires = []
        self.eval_scripts = []
        self._sequence = 0

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

    async def hdel(self, key, field):
        existed = field in self.hashes[key]
        self.hashes[key].pop(field, None)
        return int(existed)

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

    async def xrange(self, stream, start, end, count=None):
        return [
            (stream_id, dict(fields))
            for stream_id, fields in self.streams[stream]
            if stream_id == start == end
        ]

    def _zset_purge(self, key, cutoff):
        self.zsets[key] = {
            member: score
            for member, score in self.zsets[key].items()
            if score > float(cutoff)
        }

    def _zset_add(self, key, member, score):
        self.zsets[key][member] = float(score)

    def _zset_remove(self, key, member):
        self.zsets[key].pop(member, None)

    async def xlen(self, stream):
        return len(self.streams[stream])

    async def eval(self, script, numkeys, *keys_and_args):
        self.eval_scripts.append(script)
        keys = keys_and_args[:numkeys]
        args = keys_and_args[numkeys:]
        if "crawl4ai:enqueue" in script:
            return await self._eval_enqueue(keys, args)
        if "crawl4ai:claim-stale" in script:
            return await self._eval_claim_stale(keys, args)
        if "crawl4ai:start-attempt" in script:
            return await self._eval_start_attempt(keys, args)
        if "crawl4ai:heartbeat" in script:
            return await self._eval_heartbeat(keys, args)
        if "crawl4ai:mark-retry" in script:
            return await self._eval_mark_retry(keys, args)
        if "crawl4ai:complete" in script:
            return await self._eval_complete(keys, args)
        if "crawl4ai:discard-missing-payload" in script:
            return await self._eval_discard_missing_payload(keys, args)
        raise AssertionError("unexpected Lua script")

    async def _eval_enqueue(self, keys, args):
        (
            max_pending_jobs,
            per_principal,
            status,
            created_at,
            url,
            result,
            error,
            webhook_config,
            payload,
            task_id,
            owner,
            protocol_version,
            now,
        ) = args
        (
            stream,
            task_key,
            payload_key,
            principal_lease_key,
        ) = keys
        if len(self.streams[stream]) >= int(max_pending_jobs):
            return 0
        if owner and int(per_principal) > 0:
            self._zset_purge(principal_lease_key, now)
            if len(self.zsets[principal_lease_key]) >= int(per_principal):
                return -1
        await self.hset(
            task_key,
            mapping={
                "status": status,
                "created_at": created_at,
                "url": url,
                "result": result,
                "error": error,
                "webhook_config": webhook_config,
                "owner": owner,
                "protocol_version": protocol_version,
            },
        )
        await self.hset(
            payload_key,
            mapping={
                "payload": payload,
                "attempt": "0",
                "owner": owner,
                "lease_owner": "",
                "fence_token": "",
                "protocol_version": protocol_version,
            },
        )
        await self.xadd(
            stream,
            {
                "task_id": task_id,
                "protocol_version": str(protocol_version),
                "owner": owner,
            },
        )
        if owner and int(per_principal) > 0:
            self._zset_add(principal_lease_key, task_id, float("inf"))
        return 1

    async def _eval_claim_stale(self, keys, args):
        stream = keys[0]
        (
            _group,
            consumer,
            _min_idle,
            _cursor,
            claim_fence,
            payload_prefix,
            protocol_version,
        ) = args
        for stream_id, fields in self.streams[stream]:
            if stream_id in self.pending:
                self.pending[stream_id] = consumer
                task_id = fields["task_id"]
                payload_key = f"{payload_prefix}{task_id}"
                if (
                    fields.get("protocol_version") == str(protocol_version)
                    and payload_key in self.hashes
                    and "payload" in self.hashes[payload_key]
                ):
                    await self.hset(
                        payload_key,
                        mapping={"lease_owner": consumer, "fence_token": claim_fence},
                    )
                return ["0-0", [(stream_id, fields)], []]
        return ["0-0", [], []]

    async def _eval_start_attempt(self, keys, args):
        stream, payload_key, task_key = keys
        (
            _group,
            consumer,
            stream_id,
            fence_token,
            status,
            created_at,
            url,
            owner,
            heartbeat_at,
            protocol_version,
        ) = args
        if (
            self.pending.get(stream_id) != consumer
            or "payload" not in self.hashes[payload_key]
            or self.hashes[payload_key].get("protocol_version") != str(protocol_version)
        ):
            return [0, 0]
        attempt = await self.hincrby(payload_key, "attempt", 1)
        await self.hset(
            payload_key,
            mapping={
                "attempt": attempt,
                "lease_owner": consumer,
                "fence_token": fence_token,
                "owner": owner,
                "protocol_version": protocol_version,
            },
        )
        await self.hset(
            task_key,
            mapping={
                "status": status,
                "created_at": created_at,
                "url": url,
                "result": "",
                "error": "",
                "owner": owner,
                "attempt": attempt,
                "lease_owner": consumer,
                "lease_stream_id": stream_id,
                "lease_heartbeat_at": heartbeat_at,
                "protocol_version": protocol_version,
            },
        )
        assert any(candidate[0] == stream_id for candidate in self.streams[stream])
        return [1, attempt]

    async def _eval_heartbeat(self, keys, args):
        stream, payload_key, task_key = keys
        (
            _group,
            consumer,
            stream_id,
            fence_token,
            attempt,
            status,
            created_at,
            url,
            owner,
            heartbeat_at,
        ) = args
        payload = self.hashes[payload_key]
        if payload.get("fence_token") != fence_token or payload.get("lease_owner") != consumer:
            return 0
        if stream_id not in self.pending:
            return 0
        self.pending[stream_id] = consumer
        self.claims.append(stream_id)
        await self.hset(
            task_key,
            mapping={
                "status": status,
                "created_at": created_at,
                "url": url,
                "result": "",
                "error": "",
                "owner": owner,
                "attempt": attempt,
                "lease_owner": consumer,
                "lease_stream_id": stream_id,
                "lease_heartbeat_at": heartbeat_at,
            },
        )
        assert any(candidate[0] == stream_id for candidate in self.streams[stream])
        return 1

    async def _eval_mark_retry(self, keys, args):
        payload_key, task_key = keys
        (
            fence_token,
            consumer,
            status,
            created_at,
            url,
            owner,
            attempt,
            error,
            stream_id,
            heartbeat_at,
        ) = args
        payload = self.hashes[payload_key]
        if payload.get("fence_token") != fence_token or payload.get("lease_owner") != consumer:
            return 0
        await self.hset(
            task_key,
            mapping={
                "status": status,
                "created_at": created_at,
                "url": url,
                "result": "",
                "error": "",
                "owner": owner,
                "attempt": attempt,
                "last_error": error,
                "lease_owner": consumer,
                "lease_stream_id": stream_id,
                "lease_heartbeat_at": heartbeat_at,
            },
        )
        await self.hset(
            payload_key,
            mapping={"last_error": error, "attempt": attempt, "owner": owner},
        )
        return 1

    async def _eval_complete(self, keys, args):
        (
            payload_key,
            task_key,
            stream,
            principal_lease_key,
        ) = keys
        (
            fence_token,
            consumer,
            status,
            created_at,
            url,
            result,
            error,
            completed_at,
            owner,
            task_ttl,
            group,
            stream_id,
            task_id,
        ) = args
        payload = self.hashes[payload_key]
        if payload.get("fence_token") != fence_token or payload.get("lease_owner") != consumer:
            return 0
        await self.hset(
            task_key,
            mapping={
                "status": status,
                "created_at": created_at,
                "url": url,
                "result": result,
                "error": error,
                "completed_at": completed_at,
                "owner": owner,
                "attempt": payload["attempt"],
            },
        )
        if int(task_ttl) > 0:
            await self.expire(task_key, int(task_ttl))
        self._zset_remove(principal_lease_key, task_id)
        await self.delete(payload_key)
        await self.xack(stream, group, stream_id)
        await self.xdel(stream, stream_id)
        return 1

    async def _eval_discard_missing_payload(self, keys, args):
        (
            task_key,
            payload_key,
            stream,
            principal_lease_key,
        ) = keys
        (
            group,
            consumer,
            stream_id,
            processing_status,
            failed_status,
            error,
            completed_at,
            task_ttl,
            task_id,
            protocol_version,
        ) = args
        if self.pending.get(stream_id) != consumer:
            return 0
        if "payload" in self.hashes[payload_key]:
            return 0
        task_exists = task_key in self.hashes
        if task_exists and (
            self.hashes[task_key].get("status") != processing_status
            or self.hashes[task_key].get("protocol_version") != str(protocol_version)
        ):
            return 0
        stream_owner = next(
            (
                fields.get("owner", "")
                for candidate_id, fields in self.streams[stream]
                if candidate_id == stream_id
            ),
            "",
        )
        owner = self.hashes[task_key].get("owner", "") if task_exists else ""
        if not owner:
            owner = stream_owner
        if task_exists:
            await self.hset(
                task_key,
                mapping={
                    "status": failed_status,
                    "result": "",
                    "error": error,
                    "completed_at": completed_at,
                    "owner": owner,
                },
            )
            if int(task_ttl) > 0:
                await self.expire(task_key, int(task_ttl))
        else:
            await self.hset(
                task_key,
                mapping={
                    "status": failed_status,
                    "created_at": completed_at,
                    "url": "",
                    "result": "",
                    "error": error,
                    "completed_at": completed_at,
                    "owner": owner,
                    "protocol_version": protocol_version,
                },
            )
            if int(task_ttl) > 0:
                await self.expire(task_key, int(task_ttl))
        await self.delete(payload_key)
        self._zset_remove(principal_lease_key, task_id)
        await self.xack(stream, group, stream_id)
        await self.xdel(stream, stream_id)
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

    async def xack(self, stream, group, stream_id):
        assert group == "crawl-workers:v2"
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


def enqueue(redis, config, *, owner=None):
    queue = CrawlJobQueue(redis, config)
    task_id = asyncio.run(
        queue.enqueue(
            urls=["https://example.com"],
            browser_config={"type": "BrowserConfig", "params": {}},
            crawler_config={"type": "CrawlerRunConfig", "params": {}},
            result_fields=["url", "success"],
            webhook_config=None,
            owner=owner,
        )
    )
    return queue, task_id


def test_enqueue_persists_payload_before_stream_visibility():
    redis = FakeRedis()
    queue, task_id = enqueue(redis, queue_config())

    payload = json.loads(redis.hashes[queue.payload_key(task_id)]["payload"])
    assert payload["urls"] == ["https://example.com"]
    assert redis.hashes[queue.task_key(task_id)]["status"] == "processing"
    assert redis.streams[queue.settings.stream][0][1] == {
        "task_id": task_id,
        "protocol_version": "2",
        "owner": "",
    }
    assert redis.expires == []
    assert "XLEN" in redis.eval_scripts[0]
    assert "XADD" in redis.eval_scripts[0]


def test_protocol_v2_refuses_startup_while_legacy_stream_entries_remain():
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())
    asyncio.run(redis.xadd("crawl-jobs", {"task_id": "legacy"}))

    queue, task_id = enqueue(redis, queue_config())
    with pytest.raises(RuntimeError, match="legacy crawl jobs remain"):
        asyncio.run(queue.ensure_group())

    assert queue.settings.stream == "crawl-jobs:v2"
    assert queue.settings.group == "crawl-workers:v2"
    assert queue.payload_key(task_id).startswith("crawl-job:v2:")
    assert redis.streams["crawl-jobs"] == [("1-0", {"task_id": "legacy"})]
    assert ("crawl-jobs:v2", "crawl-workers:v2") not in redis.groups


def test_protocol_names_are_not_double_suffixed_and_other_versions_are_rejected():
    queue = CrawlJobQueue(
        FakeRedis(),
        queue_config(stream="custom:v2", group="workers:v2"),
    )
    assert queue.settings.stream == "custom:v2"
    assert queue.settings.group == "workers:v2"

    with pytest.raises(ValueError, match="protocol_version"):
        CrawlJobQueue(FakeRedis(), queue_config(protocol_version=1))


@pytest.mark.parametrize(
    ("browser_config", "crawler_config"),
    [
        ({"extra_args": ["--disable-web-security"]}, {}),
        ({}, {"deep_crawl_strategy": {"type": "BFSDeepCrawlStrategy", "params": {}}}),
    ],
)
def test_untrusted_configs_are_rejected_before_any_redis_write(
    browser_config,
    crawler_config,
):
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())

    with pytest.raises(CrawlJobPayloadRejected):
        asyncio.run(
            queue.enqueue(
                urls=["https://example.com"],
                browser_config=browser_config,
                crawler_config=crawler_config,
                result_fields=None,
                webhook_config=None,
            )
        )

    assert redis.eval_scripts == []
    assert dict(redis.hashes) == {}
    assert dict(redis.streams) == {}


@pytest.mark.parametrize(
    "browser_config",
    [
        {"headless": False, "__dict__": "ignored"},
        {
            "type": "BrowserConfig",
            "params": {"headless": False, "__dict__": "ignored"},
        },
    ],
)
def test_enqueue_persists_canonical_untrusted_config_dumps(browser_config):
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())
    task_id = asyncio.run(
        queue.enqueue(
            urls=["https://example.com"],
            browser_config=browser_config,
            crawler_config={"css_selector": "main"},
            result_fields=None,
            webhook_config=None,
        )
    )

    payload = json.loads(redis.hashes[queue.payload_key(task_id)]["payload"])
    assert payload["protocol_version"] == 2
    assert payload["browser_config"]["type"] == "BrowserConfig"
    assert payload["browser_config"]["params"]["headless"] is False
    assert set(payload["browser_config"]["params"]) == {"headless"}
    assert "extra_args" not in payload["browser_config"]["params"]
    assert payload["crawler_config"]["type"] == "CrawlerRunConfig"
    assert payload["crawler_config"]["params"]["css_selector"] == "main"


@pytest.mark.parametrize(
    "crawler_config",
    [
        {"delay_before_return_html": 0.1},
        {
            "type": "CrawlerRunConfig",
            "params": {"delay_before_return_html": 0.1},
        },
    ],
)
def test_enqueue_preserves_explicit_default_valued_crawler_fields(crawler_config):
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())

    task_id = asyncio.run(
        queue.enqueue(
            urls=["https://example.com"],
            browser_config={},
            crawler_config=crawler_config,
            result_fields=None,
            webhook_config=None,
        )
    )

    payload = json.loads(redis.hashes[queue.payload_key(task_id)]["payload"])
    assert payload["browser_config"]["params"] == {}
    assert payload["crawler_config"]["params"]["delay_before_return_html"] == 0.1


def test_oversized_canonical_payload_is_rejected_before_any_redis_write():
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config(max_payload_bytes=512))

    with pytest.raises(CrawlJobPayloadRejected, match="durable payload limit"):
        asyncio.run(
            queue.enqueue(
                urls=["https://example.com"],
                browser_config={},
                crawler_config={"css_selector": "x" * 1_000},
                result_fields=None,
                webhook_config=None,
            )
        )

    assert redis.eval_scripts == []
    assert dict(redis.hashes) == {}
    assert dict(redis.streams) == {}


def test_seed_is_normalized_and_validated_before_redis_visibility(monkeypatch):
    validated = []
    monkeypatch.setattr(
        crawl_job_queue_module,
        "validate_url_destination",
        validated.append,
    )
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())

    task_id = asyncio.run(
        queue.enqueue(
            urls=["example.com/path"],
            browser_config={},
            crawler_config={},
            result_fields=None,
            webhook_config=None,
        )
    )

    payload = json.loads(redis.hashes[queue.payload_key(task_id)]["payload"])
    assert validated == ["https://example.com/path"]
    assert payload["urls"] == ["https://example.com/path"]


def test_blocked_seed_is_rejected_before_any_redis_write(monkeypatch):
    def reject(_url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(crawl_job_queue_module, "validate_url_destination", reject)
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())

    with pytest.raises(CrawlJobPayloadRejected, match="crawl job URL"):
        asyncio.run(
            queue.enqueue(
                urls=["http://127.0.0.1/private"],
                browser_config={},
                crawler_config={},
                result_fields=None,
                webhook_config=None,
            )
        )

    assert redis.eval_scripts == []
    assert dict(redis.hashes) == {}
    assert dict(redis.streams) == {}


def test_seed_dns_validation_runs_off_the_event_loop(monkeypatch):
    def slow_validation(_url):
        time.sleep(0.05)

    monkeypatch.setattr(
        crawl_job_queue_module,
        "validate_url_destination",
        slow_validation,
    )
    redis = FakeRedis()
    queue = CrawlJobQueue(redis, queue_config())

    async def exercise():
        enqueue_task = asyncio.create_task(
            queue.enqueue(
                urls=["https://example.com"],
                browser_config={},
                crawler_config={},
                result_fields=None,
                webhook_config=None,
            )
        )
        await asyncio.sleep(0.01)
        assert not enqueue_task.done()
        await enqueue_task

    asyncio.run(exercise())


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


@pytest.mark.parametrize(
    ("browser_config", "crawler_config", "config"),
    [
        (
            {"extra_args": ["--disable-web-security"]},
            {},
            queue_config(),
        ),
        (
            {},
            {"css_selector": "x" * 1_000},
            queue_config(max_payload_bytes=512),
        ),
    ],
)
def test_public_crawl_job_rejects_invalid_or_oversized_payload_without_redis_writes(
    browser_config,
    crawler_config,
    config,
):
    redis = FakeRedis()

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            handle_crawl_job(
                redis,
                ["https://example.com"],
                browser_config,
                crawler_config,
                config,
            )
        )

    assert error.value.status_code == 400
    assert redis.eval_scripts == []
    assert dict(redis.hashes) == {}
    assert dict(redis.streams) == {}


def test_public_crawl_job_rejects_blocked_seed_without_redis_writes(monkeypatch):
    def reject(_url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(
        crawl_job_queue_module,
        "validate_url_destination",
        reject,
    )
    redis = FakeRedis()

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            handle_crawl_job(
                redis,
                ["http://127.0.0.1/private"],
                {},
                {},
                queue_config(),
            )
        )

    assert error.value.status_code == 400
    assert redis.eval_scripts == []
    assert dict(redis.hashes) == {}
    assert dict(redis.streams) == {}


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


def test_stale_claim_fences_out_the_previous_workers_heartbeat():
    redis = FakeRedis()
    config = queue_config()
    queue, _task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("dead-worker"))[0]
    payload = asyncio.run(queue.load_payload(entry.task_id))
    stale_attempt = asyncio.run(queue.start_attempt(entry, payload, "dead-worker"))

    claimed = asyncio.run(queue.claim_stale("worker-a"))
    assert claimed == [entry]

    with pytest.raises(CrawlJobLeaseLost):
        asyncio.run(queue.heartbeat(entry, payload, "dead-worker", stale_attempt))

    current_attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-a"))
    asyncio.run(queue.heartbeat(entry, payload, "worker-a", current_attempt))
    assert redis.claims == [entry.stream_id]


def test_stale_attempt_cannot_publish_terminal_state_after_takeover():
    redis = FakeRedis()
    queue, task_id = enqueue(redis, queue_config(), owner="alice")
    entry = asyncio.run(queue.read_new("worker-a"))[0]
    payload = asyncio.run(queue.load_payload(task_id))
    stale_attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-a"))

    assert asyncio.run(queue.claim_stale("worker-b")) == [entry]
    current_attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-b"))

    with pytest.raises(CrawlJobLeaseLost):
        asyncio.run(
            queue.complete(entry, payload, stale_attempt, result={"worker": "a"})
        )
    assert redis.hashes[queue.task_key(task_id)]["status"] == "processing"
    assert queue.payload_key(task_id) in redis.hashes
    assert entry.stream_id in redis.pending

    asyncio.run(
        queue.complete(entry, payload, current_attempt, result={"worker": "b"})
    )
    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "completed"
    assert json.loads(task["result"]) == {"worker": "b"}
    assert task["owner"] == "alice"


def test_owner_is_persisted_through_processing_and_terminal_updates():
    redis = FakeRedis()
    queue, task_id = enqueue(redis, queue_config(), owner="alice")
    task_key = queue.task_key(task_id)
    payload_key = queue.payload_key(task_id)

    assert redis.hashes[task_key]["owner"] == "alice"
    assert redis.hashes[payload_key]["owner"] == "alice"
    assert json.loads(redis.hashes[payload_key]["payload"])["owner"] == "alice"

    entry = asyncio.run(queue.read_new("worker-a"))[0]
    payload = asyncio.run(queue.load_payload(task_id))
    attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-a"))
    assert redis.hashes[task_key]["owner"] == "alice"

    asyncio.run(queue.mark_retry(entry, payload, "worker-a", attempt, "try again"))
    assert redis.hashes[task_key]["owner"] == "alice"

    asyncio.run(queue.complete(entry, payload, attempt, error="done"))
    assert redis.hashes[task_key]["owner"] == "alice"


def test_per_principal_pending_cap_is_atomic_and_released_on_completion():
    redis = FakeRedis()
    config = queue_config(max_pending_jobs=4)
    config["limits"] = {"queue": {"per_principal": 1}}
    queue, alice_task = enqueue(redis, config, owner="alice")

    with pytest.raises(CrawlJobPrincipalQuotaExceeded):
        enqueue(redis, config, owner="alice")

    _queue, bob_task = enqueue(redis, config, owner="bob")
    assert bob_task != alice_task

    entry = asyncio.run(queue.read_new("worker-a"))[0]
    payload = asyncio.run(queue.load_payload(alice_task))
    attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-a"))
    asyncio.run(queue.complete(entry, payload, attempt, result={"ok": True}))

    _queue, replacement = enqueue(redis, config, owner="alice")
    assert replacement not in {alice_task, bob_task}


def test_default_per_principal_cap_preserves_global_capacity_for_other_callers():
    redis = FakeRedis()
    config = queue_config(max_pending_jobs=101)
    queue = CrawlJobQueue(redis, config)
    assert queue.settings.per_principal == 100

    for _index in range(100):
        enqueue(redis, config, owner="alice")
    with pytest.raises(CrawlJobPrincipalQuotaExceeded):
        enqueue(redis, config, owner="alice")

    _queue, bob_task = enqueue(redis, config, owner="bob")
    assert bob_task.startswith("crawl_")


def test_missing_payload_terminalizes_task_and_releases_owner_quota():
    redis = FakeRedis()
    config = queue_config()
    config["limits"] = {"queue": {"per_principal": 1}}
    queue, task_id = enqueue(redis, config, owner="alice")
    entry = asyncio.run(queue.read_new("worker-a"))[0]
    asyncio.run(redis.delete(queue.payload_key(task_id)))

    asyncio.run(queue.discard_missing_payload(entry, "worker-a"))

    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "failed"
    assert task["owner"] == "alice"
    assert "payload is missing" in task["error"]
    assert task_id not in redis.zsets[principal_lease_key("alice")]


def test_stale_missing_payload_cleanup_cannot_overwrite_valid_completion_or_double_release():
    redis = FakeRedis()
    config = queue_config(max_pending_jobs=4)
    config["limits"] = {"queue": {"per_principal": 1}}
    queue, task_id = enqueue(redis, config, owner="alice")
    entry = asyncio.run(queue.read_new("worker-a"))[0]
    payload = asyncio.run(queue.load_payload(task_id))
    stale_attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-a"))
    assert asyncio.run(queue.claim_stale("worker-b")) == [entry]
    current_attempt = asyncio.run(queue.start_attempt(entry, payload, "worker-b"))
    asyncio.run(
        queue.complete(entry, payload, current_attempt, result={"worker": "b"})
    )

    with pytest.raises(CrawlJobLeaseLost):
        asyncio.run(queue.discard_missing_payload(entry, stale_attempt.consumer))

    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "completed"
    assert json.loads(task["result"]) == {"worker": "b"}
    assert task_id not in redis.zsets[principal_lease_key("alice")]

    _queue, replacement = enqueue(redis, config, owner="alice")
    with pytest.raises(CrawlJobPrincipalQuotaExceeded):
        enqueue(redis, config, owner="alice")
    assert replacement != task_id


def test_stale_claim_does_not_create_lease_only_payload_hash():
    redis = FakeRedis()
    queue, task_id = enqueue(redis, queue_config(), owner="alice")
    entry = asyncio.run(queue.read_new("worker-a"))[0]
    asyncio.run(redis.delete(queue.payload_key(task_id)))

    assert asyncio.run(queue.claim_stale("worker-b")) == [entry]

    assert queue.payload_key(task_id) not in redis.hashes


def test_missing_task_and_payload_produce_observable_terminal_record():
    redis = FakeRedis()
    queue, task_id = enqueue(redis, queue_config(), owner="alice")
    entry = asyncio.run(queue.read_new("worker-a"))[0]
    asyncio.run(redis.delete(queue.payload_key(task_id)))
    asyncio.run(redis.delete(queue.task_key(task_id)))

    asyncio.run(queue.discard_missing_payload(entry, "worker-a"))

    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "failed"
    assert task["owner"] == "alice"
    assert task["protocol_version"] == "2"


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


def test_worker_releases_an_attempt_that_outlives_its_budget():
    """A hung crawl must not hold its consumer: the heartbeat would renew its lease forever."""
    redis = FakeRedis()
    config = queue_config(max_attempt_seconds=1, max_attempts=2)
    queue, task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("worker-a"))[0]

    async def crawl(_payload):
        await asyncio.sleep(3600)

    worker = CrawlJobWorker(queue, config, "worker-a", crawl=crawl, webhook_service=NoopWebhook())
    asyncio.run(worker.process(entry))

    # Left pending and un-acked so another worker reclaims it once the lease goes stale.
    assert entry.stream_id in redis.pending
    assert redis.acks == []
    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "processing"
    assert "attempt budget" in task["last_error"]


def test_worker_fails_a_stalled_attempt_once_the_retry_budget_is_gone():
    redis = FakeRedis()
    config = queue_config(max_attempt_seconds=1, max_attempts=1)
    queue, task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new("worker-a"))[0]

    async def crawl(_payload):
        await asyncio.sleep(3600)

    worker = CrawlJobWorker(queue, config, "worker-a", crawl=crawl, webhook_service=NoopWebhook())
    asyncio.run(worker.process(entry))

    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "failed"
    assert "attempt budget" in task["error"]
    # Acked and removed, so the stream stops counting it against max_pending_jobs.
    assert redis.acks == [(queue.settings.stream, entry.stream_id)]
    assert redis.deleted == [(queue.settings.stream, entry.stream_id)]


def test_settings_reject_a_non_positive_attempt_budget():
    with pytest.raises(ValueError, match="max_attempt_seconds"):
        CrawlJobQueue(FakeRedis(), queue_config(max_attempt_seconds=0))
