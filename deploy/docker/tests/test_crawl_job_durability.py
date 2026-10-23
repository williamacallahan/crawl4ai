"""Redis transaction and process-death boundaries for durable crawl jobs.

The crawler callable stands in for browser work; queue Lua, Redis persistence,
worker execution, and the webhook HTTP transport are exercised without mocks.
"""

import asyncio
import base64
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest
from redis import asyncio as aioredis

from test_crawl_job_queue_redis import redis_url  # noqa: F401

DOCKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCKER_DIR))

from crawl_job_queue import CrawlJobLeaseLost, CrawlJobQueue  # noqa: E402
from crawl_job_worker import CrawlJobWorker  # noqa: E402
from work_queue import principal_lease_key  # noqa: E402


@pytest.fixture(autouse=True)
def public_seeds(monkeypatch):
    monkeypatch.setattr("crawl_job_queue.validate_url_destination", lambda _url: None)


@contextlib.asynccontextmanager
async def isolated_queue(redis_url):
    namespace = f"durability-{uuid4().hex}"
    config = {
        "redis": {"task_ttl_seconds": 60},
        "crawl_jobs": {
            "stream": namespace,
            "group": namespace,
            "lease_seconds": 2,
            "heartbeat_seconds": 1,
            "read_block_ms": 1,
            "max_attempts": 2,
            "max_pending_jobs": 4,
            "max_attempt_seconds": 5,
        },
        "limits": {"queue": {"per_principal": 1}},
        "webhooks": {"enabled": True, "retry": {"max_attempts": 2, "initial_delay_ms": 1}},
    }
    client = aioredis.from_url(redis_url)
    queue = CrawlJobQueue(client, config)
    tasks = []

    async def enqueue(webhook=None):
        task_id = await queue.enqueue(
            ["https://example.com"], {}, {}, None, webhook, owner=namespace
        )
        tasks.append(task_id)
        return task_id

    try:
        await queue.ensure_group()
        yield client, queue, enqueue, namespace
    finally:
        keys = [key async for key in client.scan_iter(match=f"{namespace}*")]
        keys += [principal_lease_key(namespace)]
        for task_id in tasks:
            keys += [queue.task_key(task_id), queue.payload_key(task_id)]
        if keys:
            await client.delete(*keys)
        await client.aclose()


async def stale(client, queue, entry, consumer):
    await client.xclaim(
        queue.settings.stream, queue.settings.group, consumer, 0,
        [entry.stream_id], idle=3_000,
    )


async def age_outbox(client, queue):
    """Age committed pending deliveries past the fallback lease."""
    pending = await client.xpending_range(
        queue.outbox_stream, queue.outbox_group, "-", "+", queue.settings.max_pending_jobs
    )
    idle = queue.settings.lease_seconds * 1000 + 1
    for entry in pending:
        await client.xclaim(
            queue.outbox_stream, queue.outbox_group, entry["consumer"], 0,
            [entry["message_id"]], idle=idle,
        )


async def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await predicate()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("process did not reach its Redis boundary before the deadline")


@contextlib.contextmanager
def receiver():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://callback.example:{server.server_port}/done", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def local_webhook(monkeypatch):
    from egress_broker import PinnedTarget
    from urllib.parse import urlsplit

    async def local_pin(url):
        parsed = urlsplit(url)
        assert parsed.hostname == "callback.example"
        return PinnedTarget(parsed.scheme, parsed.hostname, parsed.port, "127.0.0.1")

    # Only the supported DNS/pinning boundary is substituted. Delivery still
    # opens an aiohttp connection and POSTs JSON to an actual HTTP server.
    monkeypatch.setattr("egress_broker.resolve_and_pin_async", local_pin)


@pytest.mark.parametrize("size_kind", ["serialized", "stored"])
def test_oversized_result_terminalizes_releases_quota_and_delivers_failure(
    redis_url, local_webhook, size_kind
):
    async def exercise():
        with receiver() as (url, requests):
            async with isolated_queue(redis_url) as (client, queue, enqueue, owner):
                task_id = await enqueue({"webhook_url": url, "webhook_data_in_payload": True})
                entry = (await queue.read_new("producer"))[0]
                # The first exceeds 16 MiB before compression but compresses tiny.
                # The second is <16 MiB serialized and >4 MiB after compression.
                body = "x" * (16 * 1024 * 1024) if size_kind == "serialized" else (
                    base64.b64encode(os.urandom(5 * 1024 * 1024)).decode("ascii")
                )

                async def crawl(_payload):
                    return {"success": True, "body": body}

                worker = CrawlJobWorker(queue, queue.config, "producer", crawl=crawl)
                await worker.process(entry)
                task = await client.hgetall(queue.task_key(task_id))
                assert task[b"status"] == b"failed"
                assert b"result_too_large" in task[b"error"]
                assert task[b"result"] == b""
                assert await client.xlen(queue.settings.stream) == 0
                assert await client.exists(queue.payload_key(task_id)) == 0
                assert await client.zscore(principal_lease_key(owner), task_id) is None
                # Acceptance of a replacement proves the quota is usable again.
                assert await enqueue() != task_id
                assert len(requests) == 1
                assert requests[0]["status"] == "failed"
                assert "result_too_large" in requests[0]["error"]
                assert "data" not in requests[0]
                assert await client.xlen(queue.outbox_stream) == 0
                assert await queue.read_outbox("delivery") == []
    asyncio.run(exercise())


# A real process owns the attempt and is SIGKILLed by the parent. It has no
# browser or webhook stubs: complete mode stops at the real commit boundary.
CHILD = """
import asyncio, json, os, sys
from redis import asyncio as aioredis
import crawl_job_queue
from crawl_job_queue import CrawlJobQueue
from crawl_job_worker import CrawlJobWorker
crawl_job_queue.validate_url_destination = lambda _url: None
async def main():
    options = json.load(sys.stdin)
    client = aioredis.from_url(os.environ['REDIS_URL'])
    queue = CrawlJobQueue(client, options['config'])
    entry = (await queue.read_new('killed-worker'))[0]
    async def crawl(_payload):
        await client.incr(options['counter'])
        if options['mode'] == 'running':
            await asyncio.Event().wait()
        return {'success': True, 'body': 'persisted'}
    if options['mode'] == 'complete':
        payload = await queue.load_payload(entry.task_id)
        attempt = await queue.start_attempt(entry, payload, 'killed-worker')
        await queue.complete(entry, payload, attempt, result=await crawl(payload))
    else:
        worker = CrawlJobWorker(queue, options['config'], 'killed-worker', crawl=crawl)
        await worker.process(entry)
    await client.set(options['committed'], '1')
    await asyncio.Event().wait()
asyncio.run(main())
"""


@contextlib.contextmanager
def child_worker(redis_url, queue, owner, mode):
    env = dict(os.environ, REDIS_URL=redis_url)
    env["PYTHONPATH"] = str(DOCKER_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD], env=env, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    options = {"config": queue.config, "counter": owner + ":executions",
               "committed": owner + ":committed", "mode": mode}
    process.stdin.write(json.dumps(options))
    process.stdin.close()
    try:
        yield process, options
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stderr.close()


def test_killed_running_worker_is_failed_without_a_second_crawl(redis_url):
    async def exercise():
        async with isolated_queue(redis_url) as (client, queue, enqueue, owner):
            task_id = await enqueue()
            with child_worker(redis_url, queue, owner, "running") as (process, options):
                await wait_for(lambda: client.get(options["counter"]))
                assert process.poll() is None
                process.kill()
                assert process.wait(timeout=5) == -signal.SIGKILL
            entries = await client.xrange(queue.settings.stream)
            from crawl_job_queue import CrawlJobEntry
            entry = CrawlJobEntry(entries[0][0].decode(), task_id)
            await stale(client, queue, entry, "killed-worker")
            claimed = await queue.claim_stale("replacement")
            # A running attempt has unknown external effects after process death;
            # only a deliberately marked, clean retry is safe to execute again.
            assert claimed == []
            task = await client.hgetall(queue.task_key(task_id))
            assert task[b"status"] == b"failed"
            assert b"worker_died" in task[b"error"]
            assert task[b"attempt"] == b"1"
            assert await client.get(options["counter"]) == b"1"
            assert await client.xlen(queue.settings.stream) == 0
            assert await client.zscore(principal_lease_key(owner), task_id) is None
            assert await enqueue() != task_id
    asyncio.run(exercise())


def test_clean_retry_is_reclaimed_and_keeps_the_attempt_budget(redis_url):
    async def exercise():
        async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
            task_id = await enqueue()
            entry = (await queue.read_new("first"))[0]
            executions = []

            async def crawl(_payload):
                executions.append(True)
                raise RuntimeError("target unavailable")

            await CrawlJobWorker(queue, queue.config, "first", crawl=crawl).process(entry)
            assert await client.hget(queue.payload_key(task_id), "attempt") == b"1"
            await stale(client, queue, entry, "first")
            assert await queue.claim_stale("second") == [entry]
            await CrawlJobWorker(queue, queue.config, "second", crawl=crawl).process(entry)
            task = await client.hgetall(queue.task_key(task_id))
            assert task[b"status"] == b"failed"
            assert task[b"attempt"] == b"2"
            assert len(executions) == 2
            assert await client.xlen(queue.settings.stream) == 0
    asyncio.run(exercise())


def test_worker_immediately_delivers_once_without_fallback_duplicate(redis_url, local_webhook):
    async def exercise():
        with receiver() as (url, requests):
            async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
                task_id = await enqueue({"webhook_url": url, "webhook_data_in_payload": True})
                entry = (await queue.read_new("producer"))[0]

                async def crawl(_payload):
                    return {"success": True, "body": "immediate"}

                await CrawlJobWorker(queue, queue.config, "producer", crawl=crawl).process(entry)
                assert await client.hget(queue.task_key(task_id), "status") == b"completed"
                assert len(requests) == 1
                assert requests[0]["task_id"] == task_id
                assert requests[0]["data"] == {"success": True, "body": "immediate"}
                assert await client.xlen(queue.outbox_stream) == 0
                assert await queue.read_outbox("fallback") == []
                assert await queue.read_outbox("another-restart") == []
                assert len(requests) == 1
    asyncio.run(exercise())


def test_crash_between_commit_and_send_delivers_once_after_grace(redis_url, local_webhook):
    async def exercise():
        with receiver() as (url, requests):
            async with isolated_queue(redis_url) as (client, queue, enqueue, owner):
                task_id = await enqueue({"webhook_url": url, "webhook_data_in_payload": True})
                with child_worker(redis_url, queue, owner, "complete") as (process, options):
                    await wait_for(lambda: client.get(options["committed"]))
                    assert process.poll() is None
                    assert await client.hget(queue.task_key(task_id), "status") == b"completed"
                    assert requests == []
                    process.kill()
                    assert process.wait(timeout=5) == -signal.SIGKILL
                restarted = CrawlJobWorker(queue, queue.config, "restart")
                assert await queue.read_outbox("restart") == []
                pending = await client.xpending_range(
                    queue.outbox_stream, queue.outbox_group, "-", "+", 1
                )
                assert len(pending) == 1
                assert pending[0]["consumer"] == b"killed-worker"
                await age_outbox(client, queue)
                events = await queue.read_outbox("restart")
                assert len(events) == 1
                stream_id, event = events[0]
                await restarted.process_outbox(stream_id, event)
                assert len(requests) == 1
                assert requests[0]["task_id"] == task_id
                assert requests[0]["status"] == "completed"
                assert requests[0]["data"] == {"success": True, "body": "persisted"}
                # A second process/restart sees no already-delivered record.
                assert await queue.read_outbox("another-restart") == []
                assert len(requests) == 1
    asyncio.run(exercise())


class RecordingRedis:
    """Capture exact production Lua invocations without replacing Redis behavior."""
    def __init__(self, client):
        self.client = client
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def eval(self, *args):
        self.calls.append(args)
        return await self.client.eval(*args)


@pytest.mark.parametrize("replayed", ["enqueue", "start", "complete"])
def test_lua_replay_has_one_enqueue_attempt_and_terminal_event(redis_url, replayed):
    async def exercise():
        async with isolated_queue(redis_url) as (client, queue, enqueue, owner):
            recorder = RecordingRedis(client)
            queue.redis = recorder
            task_id = await enqueue({"webhook_url": "https://callback.example/done", "webhook_data_in_payload": True})
            enqueue_args = recorder.calls[-1]
            if replayed == "enqueue":
                assert await client.eval(*enqueue_args) == 1
            assert await client.xlen(queue.settings.stream) == 1
            assert await client.zcard(principal_lease_key(owner)) == 1
            entry = (await queue.read_new("first"))[0]
            payload = await queue.load_payload(task_id)
            attempt = await queue.start_attempt(entry, payload, "first")
            start_args = recorder.calls[-1]
            if replayed == "start":
                assert list(await client.eval(*start_args)) == [1, 1]
            assert await client.hget(queue.payload_key(task_id), "attempt") == b"1"
            await queue.mark_retry(entry, payload, "first", attempt, "clean failure")
            await stale(client, queue, entry, "first")
            assert await queue.claim_stale("second") == [entry]
            current = await queue.start_attempt(entry, payload, "second")
            assert current.number == 2
            with pytest.raises(CrawlJobLeaseLost):
                await queue.complete(entry, payload, attempt, result={"stale": True})
            await queue.complete(entry, payload, current, result={"ok": True})
            complete_args = recorder.calls[-1]
            if replayed == "complete":
                assert await client.eval(*complete_args) == 1
            # A separate completion operation from even the same attempt loses;
            # an exact operation replay alone is allowed to reuse its receipt.
            with pytest.raises(CrawlJobLeaseLost):
                await queue.complete(entry, payload, current, error="competing completion")
            altered_fence = list(complete_args)
            altered_fence[7] = "another-fence"  # first ARGV follows five KEYS
            assert await client.eval(*altered_fence) == 0
            assert await client.xlen(queue.settings.stream) == 0
            assert await client.zcard(principal_lease_key(owner)) == 0
            await age_outbox(client, queue)
            events = await queue.read_outbox("delivery")
            assert len(events) == 1
            assert events[0][1]["task_id"] == task_id
            assert events[0][1]["status"] == "completed"
            assert events[0][1]["result"] == {"ok": True}
            assert await queue.read_outbox("delivery") == []
    asyncio.run(exercise())
