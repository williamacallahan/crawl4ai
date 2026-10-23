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


@pytest.mark.parametrize("failed", [False, True])
def test_completed_crawl_stores_compressed_bytes_and_returns_original_result(redis_url, monkeypatch, failed):
    from crawl_job_queue import CrawlJobQueue
    from test_crawl_job_queue_redis import queue_config

    monkeypatch.setattr("crawl_job_queue.validate_url_destination", lambda _url: None)

    async def exercise():
        # Match the worker's text client for writes and the API's bytes client for reads.
        writer = aioredis.from_url(redis_url, decode_responses=True)
        reader = aioredis.from_url(redis_url)
        config = queue_config()
        config["crawl_jobs"]["stream"] = f"compression-{uuid4().hex}"
        config["limits"]["queue"]["per_principal"] = 0
        queue = CrawlJobQueue(writer, config)
        task_id = ""
        result = {"success": not failed, "results": [{
            "url": "https://example.com", "success": not failed,
            "html": "".join(f'<p id="{i}">Crawl result content {i}: {i * 7919:08x}</p>\n'
                            for i in range(5500)),
        }]}
        serialized = json.dumps(result).encode("utf-8")
        assert 300_000 <= len(serialized) < 350_000
        try:
            await queue.ensure_group()
            task_id = await queue.enqueue(["https://example.com"], {}, {}, None, None, owner="alice")
            entry = (await queue.read_new("compression-worker"))[0]
            payload = await queue.load_payload(task_id)
            attempt = await queue.start_attempt(entry, payload, "compression-worker")
            await queue.complete(entry, payload, attempt, result=result,
                                 error="Every crawled URL failed" if failed else None)
            stored = await reader.hgetall(queue.task_key(task_id))
            assert stored[b"result_encoding"] == b"zlib"
            assert len(stored[b"result"]) < len(serialized)
            assert zlib.decompress(stored[b"result"]) == serialized
            expiry = await reader.pexpiretime(queue.task_key(task_id))
            assert expiry > 0
            response = await api.handle_task_status(
                reader, task_id, "https://crawl.example/", collection="crawl/job", requester="alice"
            )
            body = json.loads(response.body)
            assert response.status_code == 200
            assert body["status"] == ("failed" if failed else "completed")
            assert body["result"] == result
            assert "result_encoding" not in body
            if failed:
                assert body["error"] == "Every crawled URL failed"
            assert await reader.pexpiretime(queue.task_key(task_id)) == expiry
            print(f"stored result: {len(serialized)} -> {len(stored[b'result'])} bytes")
        finally:
            await reader.delete(queue.settings.stream)
            if task_id:
                await reader.delete(queue.task_key(task_id), queue.payload_key(task_id))
            await writer.aclose()
            await reader.aclose()

    asyncio.run(exercise())
