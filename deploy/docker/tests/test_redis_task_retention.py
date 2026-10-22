"""
In-flight task hashes carry no TTL. A terminal write arms ``task_ttl_seconds``.
A successful read of that finished hash returns the payload and shortens its
TTL to ``delivered_task_ttl_seconds``, unless ``keep`` is set, so a re-read
(a resume probe followed by its poll) still finds it.

Outcomes are the key's TTL and whether the hash is still present after the
read. ``volatile-lru`` can evict only keys that have a TTL, so a missing TTL
on a processing hash is the property that keeps a live task out of eviction.
"""

import json
import os
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api  # noqa: E402
from utils import TaskStatus, decode_redis_hash  # noqa: E402


class _Redis:
    """Async Redis stub. HSET merges fields, the way real Redis does."""

    def __init__(self):
        self._hashes: dict[str, dict[bytes, bytes]] = {}
        self._ttl: dict[str, int] = {}

    async def hset(self, key, mapping=None):
        fields = self._hashes.setdefault(key, {})
        for field, value in mapping.items():
            raw = value.encode() if isinstance(value, str) else str(value).encode()
            fields[str(field).encode()] = raw
        return len(mapping)

    async def expire(self, key, ttl, lt=False):
        # LT: a key with no TTL counts as infinite, as in Redis.
        current = self._ttl.get(key)
        if lt and current is not None and current <= int(ttl):
            return False
        self._ttl[key] = int(ttl)
        return True

    def ttl_seconds(self, key) -> int | None:
        return self._ttl.get(key)

    async def hgetall(self, key):
        return dict(self._hashes.get(key, {}))

    async def delete(self, key):
        self._hashes.pop(key, None)
        self._ttl.pop(key, None)
        return 1


def _config(ttl: int = 3600) -> dict:
    return {"redis": {"task_ttl_seconds": ttl}}


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def _stored(redis: _Redis, key: str) -> dict:
    return decode_redis_hash(asyncio_run(redis.hgetall(key)))


def test_inflight_llm_hash_has_no_ttl_until_terminal_write(monkeypatch):
    """Create stores owner and created_at with no TTL; completion arms the TTL."""
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_enqueue_job", AsyncMock())
    redis = _Redis()
    config = _config(3600)

    response = asyncio_run(
        api.create_new_task(
            redis,
            BackgroundTasks(),
            "https://example.com",
            "extract",
            None,
            "0",
            "https://crawl.example/",
            config,
            owner="alice@x.com",
        )
    )
    task_id = json.loads(response.body)["task_id"]
    key = f"task:{task_id}"

    stored = _stored(redis, key)
    assert stored["status"] == TaskStatus.PROCESSING.value
    assert stored["owner"] == "alice@x.com"
    assert stored["created_at"]
    assert redis.ttl_seconds(key) is None

    asyncio_run(
        api.hset_with_ttl(
            redis,
            key,
            {
                "status": TaskStatus.COMPLETED,
                "result": json.dumps({"extracted_content": "body"}),
            },
            config,
        )
    )

    finished = _stored(redis, key)
    assert finished["status"] == TaskStatus.COMPLETED.value
    assert finished["created_at"] == stored["created_at"]
    assert finished["owner"] == "alice@x.com"
    assert redis.ttl_seconds(key) == 3600


def test_processing_read_leaves_the_hash_and_its_ttl_alone():
    redis = _Redis()
    key = "task:llm_abc"
    config = _config(3600)
    asyncio_run(
        api.hset_with_ttl(
            redis,
            key,
            {
                "status": TaskStatus.PROCESSING,
                "created_at": "2026-10-22T07:00:00",
                "url": "https://example.com",
                "owner": "alice@x.com",
            },
            config,
        )
    )

    response = asyncio_run(
        api.handle_task_status(
            redis,
            "llm_abc",
            "http://t/",
            collection="llm/job",
            requester="alice@x.com",
            config=config,
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["status"] == TaskStatus.PROCESSING.value
    assert redis.ttl_seconds(key) is None
    assert _stored(redis, key)["created_at"] == "2026-10-22T07:00:00"


def test_delivered_crawl_result_keeps_only_the_delivered_ttl():
    """The hash that dominates memory expires soon after delivery, yet a re-read still finds it."""
    redis = _Redis()
    key = "task:crawl_abc"
    payload = {"markdown": "page body"}
    config = _config(3600)
    asyncio_run(
        redis.hset(
            key,
            mapping={
                "status": TaskStatus.COMPLETED.value,
                "created_at": "2026-10-22T07:00:00",
                "url": "[]",
                "result": json.dumps(payload),
                "error": "",
                "owner": "alice@x.com",
            },
        )
    )
    asyncio_run(redis.expire(key, 3600))

    response = asyncio_run(
        api.handle_task_status(
            redis,
            "crawl_abc",
            "http://t/",
            collection="crawl/job",
            requester="alice@x.com",
            config=config,
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["result"] == payload
    assert redis.ttl_seconds(key) == 120

    again = asyncio_run(
        api.handle_task_status(
            redis,
            "crawl_abc",
            "http://t/",
            collection="crawl/job",
            requester="alice@x.com",
            config=config,
        )
    )
    assert json.loads(again.body)["result"] == payload
    assert redis.ttl_seconds(key) == 120


def test_keep_retains_a_delivered_result():
    redis = _Redis()
    key = "task:crawl_abc"
    config = _config(3600)
    asyncio_run(
        redis.hset(
            key,
            mapping={
                "status": TaskStatus.FAILED.value,
                "created_at": "2026-10-22T07:00:00",
                "url": "[]",
                "result": "",
                "error": "boom",
                "owner": "alice@x.com",
            },
        )
    )
    asyncio_run(redis.expire(key, 3600))

    response = asyncio_run(
        api.handle_task_status(
            redis,
            "crawl_abc",
            "http://t/",
            collection="crawl/job",
            requester="alice@x.com",
            keep=True,
            config=config,
        )
    )

    assert response.status_code == 200
    assert json.loads(response.body)["error"] == "boom"
    assert _stored(redis, key)["error"] == "boom"
    assert redis.ttl_seconds(key) == 3600


def test_other_owner_read_does_not_delete_the_hash():
    redis = _Redis()
    key = "task:crawl_abc"
    config = _config(3600)
    asyncio_run(
        redis.hset(
            key,
            mapping={
                "status": TaskStatus.COMPLETED.value,
                "created_at": "2026-10-22T07:00:00",
                "url": "[]",
                "result": json.dumps({"markdown": "secret"}),
                "owner": "alice@x.com",
            },
        )
    )

    with pytest.raises(HTTPException) as exc:
        asyncio_run(
            api.handle_task_status(
                redis,
                "crawl_abc",
                "http://t/",
                collection="crawl/job",
                requester="mallory@x.com",
                config=config,
            )
        )

    assert exc.value.status_code == 404
    assert json.loads(_stored(redis, key)["result"]) == {"markdown": "secret"}
