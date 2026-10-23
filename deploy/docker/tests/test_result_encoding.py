"""Task-result storage compatibility through the real Redis status boundary."""

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
from utils import ResultEncoding, encode_task_result  # noqa: E402


@pytest.mark.parametrize("encoding", [None, "identity", "zlib"])
@pytest.mark.parametrize("collection", ["crawl/job", "llm/job"])
@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
def test_stored_results_preserve_status_response(redis_url, encoding, collection, terminal_status):
    async def exercise():
        # The API pool uses bytes responses; only hash metadata is UTF-8 decoded.
        client = aioredis.from_url(redis_url)
        task_id = f"{collection.split('/')[0]}_{uuid4().hex}"
        key = f"task:{task_id}"
        result = {"success": terminal_status == "completed", "results": [{"markdown": "café 🌍"}]}
        serialized = json.dumps(result)
        task = {
            "status": terminal_status,
            "created_at": "2026-10-23T00:00:00",
            "url": "https://example.com",
            "owner": "alice",
            "error": "Every crawled URL failed" if terminal_status == "failed" else "",
            "result": zlib.compress(serialized.encode("utf-8"), 6) if encoding == "zlib" else serialized,
        }
        if encoding is not None:
            task["result_encoding"] = encoding
        try:
            await client.hset(key, mapping=task)
            await client.expire(key, 3600)
            expiry = await client.pexpiretime(key)
            stored = await client.hgetall(key)
            response = await api.handle_task_status(
                client, task_id, "https://crawl.example/", collection=collection, requester="alice"
            )
            task_url = f"https://crawl.example/{collection}/{task_id}"
            expected = {
                "task_id": task_id,
                "status": terminal_status,
                "created_at": task["created_at"],
                "url": task["url"],
                "_links": {"self": {"href": task_url}, "refresh": {"href": task_url}},
                "result": result,
            }
            if terminal_status == "failed":
                expected["error"] = task["error"]
            assert response.status_code == 200
            assert json.loads(response.body) == expected
            assert await client.hgetall(key) == stored
            assert await client.pexpiretime(key) == expiry
            if encoding != "zlib":
                assert stored[b"result"] == encode_task_result(result, ResultEncoding.IDENTITY).encode("utf-8")
        finally:
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("result", ["{}", ""])
def test_unknown_result_encoding_raises_with_task_context(redis_url, result):
    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"crawl_{uuid4().hex}"
        key = f"task:{task_id}"
        try:
            await client.hset(key, mapping={
                "status": "completed", "created_at": "2026-10-23T00:00:00",
                "url": "[]", "result": result, "result_encoding": "unknown-format",
            })
            with pytest.raises(ValueError, match=f"task {task_id}: unknown result_encoding 'unknown-format'"):
                await api.handle_task_status(client, task_id, "https://crawl.example/", collection="crawl/job")
        finally:
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())
