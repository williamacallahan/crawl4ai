"""Outbox consumer takeover keeps retry accounting in Redis across senders."""

import asyncio
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from redis.exceptions import ResponseError

from crawl_job_queue import CrawlJobLeaseLost
from crawl_job_worker import CrawlJobWorker
from test_crawl_job_durability import (  # noqa: F401
    RecordingRedis, age_outbox, isolated_queue, local_webhook, receiver,
)
from test_crawl_job_queue_redis import redis_url  # noqa: F401
from webhook import WebhookDeliveryService


@pytest.fixture(autouse=True)
def public_seeds(monkeypatch):
    monkeypatch.setattr("crawl_job_queue.validate_url_destination", lambda _url: None)


@contextlib.contextmanager
def unavailable_receiver(status=503):
    requests = []
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            self.end_headers()
            received.set()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://callback.example:{server.server_port}/done", requests, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_takeover_keeps_persisted_retry_budget_and_rejects_previous_owner(redis_url, local_webhook):
    async def exercise():
        with unavailable_receiver() as (url, requests, received):
            async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
                task_id = await enqueue({"webhook_url": url, "webhook_data_in_payload": True})
                crawl_entry = (await queue.read_new("crawl"))[0]

                payload = await queue.load_payload(task_id)
                attempt = await queue.start_attempt(crawl_entry, payload, "crawl")
                await queue.complete(crawl_entry, payload, attempt,
                                     result={"success": True, "body": "compressed result"})
                assert requests == []
                await age_outbox(client, queue)
                stream_id, event = (await queue.read_outbox("first"))[0]
                first_sender = WebhookDeliveryService(queue.config)
                # Make the real sender remain in backoff after its first failed
                # HTTP request; cancellation models termination during that wait.
                first_sender.initial_delay = 60
                first = CrawlJobWorker(queue, queue.config, "first", webhook_service=first_sender)
                running = asyncio.create_task(first.process_outbox(stream_id, event))
                try:
                    assert await asyncio.to_thread(received.wait, 2)
                finally:
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)

                attempts_key = f"{queue.outbox_stream}:attempts"
                assert await client.hget(attempts_key, stream_id) == b"1"
                assert len(requests) == 1
                assert await client.xlen(queue.outbox_stream) == 1

                await client.xclaim(queue.outbox_stream, queue.outbox_group, "first", 0,
                                    [stream_id], idle=3_000)
                reclaimed = await queue.read_outbox("replacement")
                assert reclaimed == [(stream_id, event)]
                assert await queue.reserve_outbox_attempt(stream_id, "first") == 0
                with pytest.raises(CrawlJobLeaseLost):
                    await queue.finish_outbox(stream_id, "first")

                replacement = CrawlJobWorker(queue, queue.config, "replacement")
                await replacement.process_outbox(*reclaimed[0])
                # The configured maximum is two attempts across both workers,
                # so replacement sends once and retains the failed delivery.
                assert len(requests) == 2
                assert requests[0]["event_id"] == requests[1]["event_id"] == event["event_id"]
                assert requests[0]["data"] == requests[1]["data"] == event["result"]
                retained = await client.xrange(queue.outbox_stream)
                assert len(retained) == 1
                assert await client.hget(attempts_key, stream_id) == b"2"
                assert await client.hget(attempts_key, stream_id + ":failed") == b"1"
                assert await queue.pending_outbox("replacement") == []
                assert 0 < await client.ttl(queue.outbox_stream) <= 86400
                assert 0 < await client.ttl(attempts_key) <= 86400
                for consumer in ("another-restart", "third-restart", "fourth-restart"):
                    await age_outbox(client, queue)
                    assert await queue.read_outbox(consumer) == []
                    assert await queue.pending_outbox(consumer) == []
                    # A sender retaining the old decoded event still cannot send.
                    await CrawlJobWorker(queue, queue.config, consumer).process_outbox(stream_id, event)
                    assert len(requests) == 2
                    assert await client.xrange(queue.outbox_stream) == retained
                    assert await client.hget(attempts_key, stream_id) == b"2"
                    pending = await client.xpending_range(queue.outbox_stream, queue.outbox_group, "-", "+", 1)
                    assert pending[0]["message_id"] == stream_id.encode()
                    assert pending[0]["consumer"] == consumer.encode()
                assert await client.hget(queue.task_key(task_id), "status") == b"completed"

    asyncio.run(exercise())


def test_client_rejection_is_retained_without_http_retry_after_reclaim(redis_url, local_webhook):
    async def exercise():
        with unavailable_receiver(400) as (url, requests, _received):
            async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
                await enqueue({"webhook_url": url})
                await complete_next(queue)
                stream_id, event = (await queue.pending_outbox("crawl"))[0]
                await CrawlJobWorker(queue, queue.config, "crawl").process_outbox(stream_id, event)
                assert len(requests) == 1
                attempts_key = f"{queue.outbox_stream}:attempts"
                assert await client.hget(attempts_key, stream_id) == b"1"
                assert await client.hget(attempts_key, stream_id + ":failed") == b"1"
                assert await queue.pending_outbox("crawl") == []
                retained = await client.xrange(queue.outbox_stream)
                assert len(retained) == 1
                for consumer in ("replacement", "another-restart"):
                    await age_outbox(client, queue)
                    assert await queue.read_outbox(consumer) == []
                    assert await queue.pending_outbox(consumer) == []
                    await CrawlJobWorker(queue, queue.config, consumer)._deliver_pending_webhooks()
                    assert len(requests) == 1
                    assert await client.hget(attempts_key, stream_id) == b"1"
                    assert await client.xrange(queue.outbox_stream) == retained
                    assert (await client.xpending(queue.outbox_stream, queue.outbox_group))["pending"] == 1
                assert 0 < await client.ttl(queue.outbox_stream) <= 86400
                assert 0 < await client.ttl(attempts_key) <= 86400

    asyncio.run(exercise())


def test_exhausted_delivery_expires_and_next_completion_recreates_group(redis_url, local_webhook):
    async def exercise():
        with unavailable_receiver() as (failed_url, failed_requests, _received), receiver() as (url, requests):
            async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
                task_id = await enqueue({"webhook_url": failed_url})
                await complete_next(queue)
                stream_id, event = (await queue.pending_outbox("crawl"))[0]
                await CrawlJobWorker(queue, queue.config, "crawl").process_outbox(stream_id, event)
                assert len(failed_requests) == 2
                attempts_key = f"{queue.outbox_stream}:attempts"
                assert await client.hget(attempts_key, stream_id) == b"2"
                assert await queue.pending_outbox("crawl") == []
                assert 0 < await client.ttl(queue.outbox_stream) <= 86400
                assert 0 < await client.ttl(attempts_key) <= 86400

                # Accelerate the same Redis expiry; reclaim must not renew a spent event.
                await client.pexpire(queue.outbox_stream, 200)
                await client.pexpire(attempts_key, 200)
                await age_outbox(client, queue)
                assert await queue.read_outbox("restart") == []
                assert await queue.pending_outbox("restart") == []
                await CrawlJobWorker(queue, queue.config, "restart").process_outbox(stream_id, event)
                assert len(failed_requests) == 2
                assert await client.xlen(queue.outbox_stream) == 1
                assert 0 < await client.pttl(queue.outbox_stream) <= 200
                assert 0 < await client.pttl(attempts_key) <= 200

                async def expired():
                    while await client.exists(queue.outbox_stream, attempts_key):
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(expired(), 1)
                with pytest.raises(ResponseError, match="NOGROUP"):
                    await client.xpending(queue.outbox_stream, queue.outbox_group)
                assert await client.hget(queue.task_key(task_id), "status") == b"completed"

                next_id = await enqueue({"webhook_url": url})
                entry = (await queue.read_new("next"))[0]

                async def crawl(_payload):
                    return {"success": True}

                await CrawlJobWorker(queue, queue.config, "next", crawl=crawl).process(entry)
                assert len(requests) == 1
                assert requests[0]["task_id"] == next_id
                assert requests[0]["status"] == "completed"
                assert len(failed_requests) == 2
                assert await client.xlen(queue.outbox_stream) == 0
                assert (await client.xpending(queue.outbox_stream, queue.outbox_group))["pending"] == 0
                assert await queue.read_outbox("fallback") == []

    asyncio.run(exercise())


@pytest.mark.parametrize("evict_before_cleanup", [False, True])
def test_missing_payload_failure_delivers_webhook_after_task_eviction(redis_url, local_webhook, evict_before_cleanup):
    async def exercise():
        with receiver() as (url, requests):
            async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
                if not evict_before_cleanup:
                    queue.config["webhooks"]["default_url"] = url
                task_id = await enqueue({"webhook_url": url} if evict_before_cleanup else None)
                entry = (await queue.read_new("cleanup"))[0]
                await client.delete(queue.payload_key(task_id))
                if evict_before_cleanup:
                    await client.delete(queue.task_key(task_id))

                await CrawlJobWorker(queue, queue.config, "cleanup").process(entry)
                assert await client.hget(queue.task_key(task_id), "status") == b"failed"
                assert await client.xlen(queue.settings.stream) == 0
                assert await client.xlen(queue.outbox_stream) == 0
                assert len(requests) == 1
                assert requests[0]["task_id"] == task_id
                assert requests[0]["status"] == "failed"
                assert requests[0]["error"] == "Durable crawl job payload is missing"
                assert requests[0]["urls"] == ["https://example.com"]
                assert requests[0]["event_id"] == f"crawl-terminal:{task_id}"
                # Eviction/TTL of the polling record cannot erase delivery's
                # frozen recipient or terminal error.
                await client.delete(queue.task_key(task_id))
                assert await client.exists(queue.task_key(task_id)) == 0
                assert await queue.read_outbox("delivery") == []

    asyncio.run(exercise())


async def complete_next(queue, consumer="crawl"):
    entry = (await queue.read_new(consumer))[0]
    payload = await queue.load_payload(entry.task_id)
    attempt = await queue.start_attempt(entry, payload, consumer)
    return await queue.complete(entry, payload, attempt, result={"success": True})


def test_outbox_trims_at_capacity_without_blocking_enqueue_and_cleans_metadata(redis_url):
    async def exercise():
        async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
            for _ in range(queue.settings.max_pending_jobs):
                await enqueue({"webhook_url": "https://callback.example/done"})
                await complete_next(queue)
            await age_outbox(client, queue)
            stream_id, _event = (await queue.read_outbox("delivery"))[0]
            before = await client.xrange(queue.outbox_stream)
            assert len(before) == queue.settings.max_pending_jobs
            assert (await client.xpending(queue.outbox_stream, queue.outbox_group))["pending"] == len(before)
            attempts_key = f"{queue.outbox_stream}:attempts"
            assert await queue.reserve_outbox_attempt(stream_id, "delivery") == 1
            assert await client.hget(attempts_key, stream_id) == b"1"
            await queue.fail_outbox(stream_id, "delivery")
            assert await client.hget(attempts_key, stream_id + ":failed") == b"1"
            assert 0 < await client.ttl(queue.outbox_stream) <= 86400
            assert 0 < await client.ttl(attempts_key) <= 86400

            replacement_id = await enqueue({"webhook_url": "https://callback.example/done"})
            assert replacement_id
            assert await client.xlen(queue.settings.stream) == 1
            await complete_next(queue)
            after = await client.xrange(queue.outbox_stream)
            assert len(after) == queue.settings.max_pending_jobs
            assert [entry[0] for entry in after[:3]] == [entry[0] for entry in before[1:]]
            assert await client.hget(attempts_key, stream_id) is None
            assert await client.hget(attempts_key, stream_id + ":operation") is None
            assert await client.hget(attempts_key, stream_id + ":failed") is None
            pending = await client.xpending_range(
                queue.outbox_stream, queue.outbox_group, "-", "+", queue.settings.max_pending_jobs
            )
            assert len(pending) == len(after)
            assert stream_id.encode() not in {entry["message_id"] for entry in pending}
            assert 0 < await client.ttl(queue.outbox_stream) <= 86400
            assert await client.exists(attempts_key) == 0

    asyncio.run(exercise())


def test_attempt_reservation_lua_replay_does_not_spend_a_second_attempt(redis_url):
    async def exercise():
        async with isolated_queue(redis_url) as (client, queue, enqueue, _owner):
            await enqueue({"webhook_url": "https://callback.example/done"})
            await complete_next(queue)
            await age_outbox(client, queue)
            stream_id, _event = (await queue.read_outbox("delivery"))[0]
            recorder = RecordingRedis(client)
            queue.redis = recorder
            assert await queue.reserve_outbox_attempt(stream_id, "delivery") == 1
            reservation = recorder.calls[-1]
            attempts_key = f"{queue.outbox_stream}:attempts"
            assert await client.eval(*reservation) == 1
            assert await client.eval(*reservation) == 1
            assert await client.hget(attempts_key, stream_id) == b"1"
            assert await queue.reserve_outbox_attempt(stream_id, "delivery") == 2
            assert await client.hget(attempts_key, stream_id) == b"2"
            await queue.finish_outbox(stream_id, "delivery")
            assert await client.hget(attempts_key, stream_id) is None
            assert await client.hget(attempts_key, stream_id + ":operation") is None
            assert await client.xlen(queue.outbox_stream) == 0

    asyncio.run(exercise())
