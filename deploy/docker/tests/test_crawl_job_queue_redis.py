import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import pytest
import redis
from redis import asyncio as aioredis

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)


@pytest.fixture(autouse=True)
def public_seed_validation(monkeypatch):
    import crawl_job_queue

    monkeypatch.setattr(crawl_job_queue, "validate_url_destination", lambda _url: None)


def queue_config():
    return {
        "redis": {"task_ttl_seconds": 60},
        "crawl_jobs": {
            "stream": "crawl-jobs-integration",
            "group": "crawl-workers",
            "lease_seconds": 2,
            "heartbeat_seconds": 1,
            "read_block_ms": 1,
            "max_attempts": 3,
            "max_pending_jobs": 4,
            "max_attempt_seconds": 5,
        },
        "limits": {"queue": {"per_principal": 1}},
    }


@pytest.fixture
def redis_url():
    configured_url = os.getenv("REDIS_URL")
    if configured_url:
        yield configured_url
        return
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is not installed")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    with tempfile.TemporaryDirectory(prefix="crawl-job-redis-") as data_dir:
        process = subprocess.Popen(
            [
                executable,
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                data_dir,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"redis://127.0.0.1:{port}/0"
        client = redis.from_url(url)
        try:
            for _attempt in range(100):
                try:
                    if client.ping():
                        break
                except redis.ConnectionError:
                    time.sleep(0.01)
            else:
                pytest.fail("temporary redis-server did not become ready")
            yield url
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_real_redis_fences_takeover_and_releases_owner_quota(redis_url):
    from crawl_job_queue import (
        CrawlJobLeaseLost,
        CrawlJobPrincipalQuotaExceeded,
        CrawlJobQueue,
    )

    async def exercise():
        client = aioredis.from_url(redis_url, decode_responses=True)
        try:
            queue = CrawlJobQueue(client, queue_config())
            await queue.ensure_group()
            task_id = await queue.enqueue(
                ["https://example.com"],
                {},
                {},
                None,
                None,
                owner="alice",
            )
            with pytest.raises(CrawlJobPrincipalQuotaExceeded):
                await queue.enqueue(
                    ["https://second.example"],
                    {},
                    {},
                    None,
                    None,
                    owner="alice",
                )

            entry = (await queue.read_new("worker-a"))[0]
            payload = await queue.load_payload(task_id)
            assert payload is not None
            stale_attempt = await queue.start_attempt(entry, payload, "worker-a")
            await client.xclaim(
                queue.settings.stream,
                queue.settings.group,
                "worker-a",
                0,
                [entry.stream_id],
                idle=3_000,
            )
            assert await queue.claim_stale("worker-b") == [entry]

            with pytest.raises(CrawlJobLeaseLost):
                await queue.heartbeat(entry, payload, "worker-a", stale_attempt)

            current_attempt = await queue.start_attempt(entry, payload, "worker-b")
            with pytest.raises(CrawlJobLeaseLost):
                await queue.complete(
                    entry,
                    payload,
                    stale_attempt,
                    result={"worker": "a"},
                )

            await queue.complete(
                entry,
                payload,
                current_attempt,
                result={"worker": "b"},
            )
            task = await client.hgetall(queue.task_key(task_id))
            assert task["status"] == "completed"
            assert task["owner"] == "alice"
            assert await client.xlen(queue.settings.stream) == 0

            with pytest.raises(CrawlJobLeaseLost):
                await queue.discard_missing_payload(entry, "worker-a")
            assert (await client.hgetall(queue.task_key(task_id)))["status"] == "completed"

            replacement = await queue.enqueue(
                ["https://replacement.example"],
                {},
                {},
                None,
                None,
                owner="alice",
            )
            assert replacement != task_id

            replacement_entry = (await queue.read_new("worker-c"))[0]
            await client.delete(queue.payload_key(replacement))
            await client.xclaim(
                queue.settings.stream,
                queue.settings.group,
                "worker-c",
                0,
                [replacement_entry.stream_id],
                idle=3_000,
            )
            assert await queue.claim_stale("worker-d") == [replacement_entry]
            assert await client.exists(queue.payload_key(replacement)) == 0
            await queue.discard_missing_payload(replacement_entry, "worker-d")
            with pytest.raises(CrawlJobLeaseLost):
                await queue.discard_missing_payload(replacement_entry, "worker-d")
            missing_payload_task = await client.hgetall(queue.task_key(replacement))
            assert missing_payload_task["status"] == "failed"
            assert missing_payload_task["owner"] == "alice"

            after_recovery = await queue.enqueue(
                ["https://after-recovery.example"],
                {},
                {},
                None,
                None,
                owner="alice",
            )
            assert after_recovery not in {task_id, replacement}
            with pytest.raises(CrawlJobPrincipalQuotaExceeded):
                await queue.enqueue(
                    ["https://must-stay-capped.example"],
                    {},
                    {},
                    None,
                    None,
                    owner="alice",
                )
        finally:
            await client.aclose()

    asyncio.run(exercise())


def test_real_redis_principal_quota_is_shared_by_llm_and_durable_crawl(redis_url):
    from crawl_job_queue import CrawlJobPrincipalQuotaExceeded, CrawlJobQueue
    from work_queue import QueueFull, QuotaExceeded, WorkQueue

    async def exercise():
        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.flushdb()
        work_queue = WorkQueue(
            maxsize=4,
            workers=1,
            per_principal=1,
            redis=client,
        )
        release_llm = asyncio.Event()
        llm_started = asyncio.Event()
        try:
            await work_queue.start()

            async def held_llm_job():
                llm_started.set()
                await release_llm.wait()

            await work_queue.submit(held_llm_job, principal="alice")
            await asyncio.wait_for(llm_started.wait(), timeout=1)

            durable = CrawlJobQueue(client, queue_config())
            await durable.ensure_group()
            with pytest.raises(CrawlJobPrincipalQuotaExceeded):
                await durable.enqueue(
                    ["https://blocked-by-llm.example"],
                    {},
                    {},
                    None,
                    None,
                    owner="alice",
                )

            release_llm.set()
            assert work_queue._q is not None
            await asyncio.wait_for(work_queue._q.join(), timeout=1)
            task_id = await durable.enqueue(
                ["https://durable.example"],
                {},
                {},
                None,
                None,
                owner="alice",
            )

            async def noop():
                return None

            with pytest.raises(QuotaExceeded):
                await work_queue.submit(noop, principal="alice")

            entry = (await durable.read_new("worker-a"))[0]
            payload = await durable.load_payload(task_id)
            assert payload is not None
            attempt = await durable.start_attempt(entry, payload, "worker-a")
            await durable.complete(entry, payload, attempt, result={"ok": True})
            await work_queue.submit(noop, principal="alice")
            await asyncio.wait_for(work_queue._q.join(), timeout=1)

            bounded = WorkQueue(
                maxsize=1,
                workers=1,
                per_principal=1,
                redis=client,
            )
            bounded._q = asyncio.Queue(maxsize=1)
            await bounded.submit(noop, principal="bob")
            with pytest.raises(QueueFull):
                await bounded.submit(noop, principal="carol")
            # Queue-full rollback releases Carol's shared claim immediately.
            await durable.enqueue(
                ["https://carol.example"],
                {},
                {},
                None,
                None,
                owner="carol",
            )
            await bounded.stop()
            # Shutdown drains Bob's queued claim even though no worker ran it.
            await durable.enqueue(
                ["https://bob.example"],
                {},
                {},
                None,
                None,
                owner="bob",
            )
        finally:
            release_llm.set()
            await work_queue.stop()
            await client.aclose()

    asyncio.run(exercise())


def test_accepted_llm_task_is_failed_when_shutdown_drains_queue(redis_url, monkeypatch):
    from fastapi import BackgroundTasks

    import api
    from work_queue import WorkQueue, get_job_queue, set_job_queue

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)

    async def exercise():
        client = aioredis.from_url(redis_url, decode_responses=True)
        queue = WorkQueue(maxsize=1, workers=1, redis=client)
        previous_queue = get_job_queue()
        worker_started = asyncio.Event()
        try:
            async def block_worker():
                worker_started.set()
                await asyncio.Event().wait()

            await queue.start()
            set_job_queue(queue)
            await queue.submit(block_worker)
            await asyncio.wait_for(worker_started.wait(), timeout=1)

            response = await api.create_new_task(
                client,
                BackgroundTasks(),
                "https://example.com",
                "extract",
                None,
                "0",
                "https://crawl.example/",
                {"redis": {"task_ttl_seconds": 60}},
            )
            task_id = json.loads(response.body)["task_id"]
            assert response.status_code == 202
            assert (await client.hgetall(f"task:{task_id}"))["status"] == "processing"

            await queue.stop()

            task = await client.hgetall(f"task:{task_id}")
            assert task["status"] == "failed"
            assert task["error"] == "LLM extraction interrupted by server shutdown"
        finally:
            await queue.stop()
            set_job_queue(previous_queue)
            await client.aclose()

    asyncio.run(exercise())
