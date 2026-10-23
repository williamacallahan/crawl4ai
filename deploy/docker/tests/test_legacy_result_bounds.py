"""Legacy oversized results remain pollable under the current read budget."""

import asyncio
import json
import os
import sys
import zlib
from uuid import uuid4

import pytest
from redis import asyncio as aioredis

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api  # noqa: E402
from test_crawl_job_queue_redis import redis_url  # noqa: E402, F401
from utils import MAX_SERIALIZED_RESULT_BYTES  # noqa: E402


@pytest.mark.parametrize("encoding", [None, "zlib"])
def test_legacy_oversized_result_decodes_to_polling_error(encoding):
    serialized = b'"' + b"a" * (MAX_SERIALIZED_RESULT_BYTES - 1) + b'"'
    task = {
        "status": "completed",
        "created_at": "2026-10-23T00:00:00",
        "url": "https://example.com",
        "result": zlib.compress(serialized, 6) if encoding == "zlib" else serialized,
    }
    if encoding:
        task["result_encoding"] = encoding

    response = api.create_task_response(
        task, "crawl_legacy", "https://crawl.example/", "crawl/job"
    )

    assert response["status"] == "failed"
    assert response["error"] == "result_too_large"
    assert "result" not in response
    assert task["status"] == "completed"


@pytest.mark.parametrize("encoding", [None, "zlib"])
def test_legacy_oversized_result_returns_terminal_polling_error(redis_url, encoding):
    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"crawl_{uuid4().hex}"
        key = f"task:{task_id}"
        serialized = b'"' + b"a" * (MAX_SERIALIZED_RESULT_BYTES - 1) + b'"'
        assert len(serialized) == MAX_SERIALIZED_RESULT_BYTES + 1
        task = {
            "status": "completed",
            "created_at": "2026-10-23T00:00:00",
            "url": "https://example.com",
            "result": zlib.compress(serialized, 6) if encoding == "zlib" else serialized,
        }
        if encoding:
            task["result_encoding"] = encoding
        try:
            await client.hset(key, mapping=task)
            await client.expire(key, 3600)
            stored = await client.hgetall(key)
            expiry = await client.pexpiretime(key)

            response = await api.handle_task_status(
                client, task_id, "https://crawl.example/", collection="crawl/job"
            )
            body = json.loads(response.body)

            assert response.status_code == 200
            assert body == {
                "task_id": task_id,
                "status": "failed",
                "created_at": task["created_at"],
                "url": task["url"],
                "_links": {
                    "self": {"href": f"https://crawl.example/crawl/job/{task_id}"},
                    "refresh": {"href": f"https://crawl.example/crawl/job/{task_id}"},
                },
                "error": "result_too_large",
            }
            assert await client.hgetall(key) == stored
            assert await client.pexpiretime(key) == expiry
        finally:
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())
