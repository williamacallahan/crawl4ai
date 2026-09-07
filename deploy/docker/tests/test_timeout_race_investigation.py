"""Regression tests for the crawl-job timeout/webhook race.

These tests pin the fix in ``CrawlJobWorker._run_with_lease``: when the attempt
budget (``max_attempt_seconds``) elapses, the crawl coroutine must be cancelled
(and its cancellation awaited) *before* ``_release_stalled_attempt`` issues its
own terminal Redis write. Without that ordering the crawl can dispatch a
competing ``queue.complete`` / ``queue.mark_retry`` during the release's Redis
await, racing the release's write and dropping the webhook when the crawl's
write wins.

Two tiers of evidence, mirroring the bug report:

1. **Timing-free fence behaviour** — drives the Lua scripts directly and
   sequentially to prove the queue honours a late crawl ``complete`` after a
   ``mark_retry`` (and that whichever ``complete`` lands first fences the
   other). No event-loop interleaving is assumed.

2. **Controlled dispatcher-level interleaving** — drives the full worker with
   a ``FakeRedis`` subclass whose ``eval`` yields (``await asyncio.sleep(0)``,
   standing in for the ``aioredis`` socket read that suspends the release
   coroutine) so the crawl's nested terminal ``eval`` runs during the
   release's await. Without the fix the crawl's terminal write wins and the
   webhook is dropped; with the fix the crawl is cancelled first and the
   release's write lands unopposed.
"""

import asyncio
import os
import sys

import pytest

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import crawl_job_queue as crawl_job_queue_module  # noqa: E402
from crawl_job_queue import CrawlJobLeaseLost, CrawlJobQueue  # noqa: E402
from crawl_job_worker import CrawlJobWorker  # noqa: E402

# Re-use the battle-tested fakes from the contract suite, including the
# FakeRedis script dispatcher and the queue_config / enqueue helpers.
from test_crawl_job_queue import (  # noqa: E402
    FakeRedis,
    enqueue,
    queue_config,
)


@pytest.fixture(autouse=True)
def public_seed_validation(monkeypatch):
    monkeypatch.setattr(
        crawl_job_queue_module,
        "validate_url_destination",
        lambda _url: None,
    )


class RecordingWebhook:
    """Captures every notification so a test can assert the webhook was fired."""

    def __init__(self):
        self.calls = []

    async def notify_job_completion(self, *, task_id, status, result=None, error=None, **_kwargs):
        self.calls.append({"task_id": task_id, "status": status, "result": result, "error": error})


# ────────────────────── tier 1: timing-free fence behaviour ──────────────────
# These tests drive the Lua scripts directly (via FakeRedis.eval) with no
# event-loop interleaving. They prove the queue's fence model does not stop a
# late crawl `complete` after a `mark_retry`, and that whichever `complete`
# lands first fences the other — i.e. the race is real at the queue layer.


def _claim_and_start(redis, config, consumer):
    queue, task_id = enqueue(redis, config)
    entry = asyncio.run(queue.read_new(consumer))[0]
    payload = asyncio.run(queue.load_payload(task_id))
    attempt = asyncio.run(queue.start_attempt(entry, payload, consumer))
    return queue, task_id, entry, payload, attempt


def test_mark_retry_does_not_fence_out_a_later_complete():
    """`mark_retry` is soft-state: it does not rotate the fence, so a later
    `complete` from the same attempt holder still succeeds and finalizes the job.
    This is the queue-level precondition for the retry-case webhook drop."""
    redis = FakeRedis()
    config = queue_config(max_attempts=3)
    queue, task_id, entry, payload, attempt = _claim_and_start(redis, config, "worker-a")

    asyncio.run(queue.mark_retry(entry, payload, "worker-a", attempt, "trying again"))
    # mark_retry only soft-sets retry metadata; the entry is still processing.
    assert redis.hashes[queue.task_key(task_id)]["status"] == "processing"
    assert redis.hashes[queue.task_key(task_id)]["last_error"] == "trying again"

    # The same attempt holder can still complete — the fence was not rotated.
    asyncio.run(queue.complete(entry, payload, attempt, result={"success": True}))
    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "completed"
    assert redis.acks == [(queue.settings.stream, entry.stream_id)]
    assert queue.payload_key(task_id) not in redis.hashes


def test_complete_fences_out_a_later_mark_retry():
    """Once `complete` deletes the payload hash, a later `mark_retry` from the
    same attempt holder fails its fence check (payload gone) and raises."""
    redis = FakeRedis()
    config = queue_config(max_attempts=3)
    queue, task_id, entry, payload, attempt = _claim_and_start(redis, config, "worker-a")

    asyncio.run(queue.complete(entry, payload, attempt, result={"success": True}))
    assert queue.payload_key(task_id) not in redis.hashes

    with pytest.raises(CrawlJobLeaseLost):
        asyncio.run(queue.mark_retry(entry, payload, "worker-a", attempt, "too late"))
    assert redis.hashes[queue.task_key(task_id)]["status"] == "completed"


def test_terminal_case_complete_fences_out_a_later_complete():
    """In the terminal case both the release and the crawl call `complete`;
    whichever lands first wins, and the second fails its fence check."""
    redis = FakeRedis()
    config = queue_config(max_attempts=1)
    queue, task_id, entry, payload, attempt = _claim_and_start(redis, config, "worker-a")

    # The crawl's success-complete lands first.
    asyncio.run(queue.complete(entry, payload, attempt, result={"success": True}))
    task = redis.hashes[queue.task_key(task_id)]
    assert task["status"] == "completed"
    assert "success" in task["result"]

    # The release's failure-complete lands second and is fenced out.
    with pytest.raises(CrawlJobLeaseLost):
        asyncio.run(queue.complete(entry, payload, attempt, error="attempt budget exceeded"))
    assert redis.hashes[queue.task_key(task_id)]["status"] == "completed"


# ──────────────── tier 2: controlled dispatcher-level interleaving ───────────
# These tests drive the full worker. A FakeRedis subclass yields inside the
# release's eval so the crawl's terminal write runs during the release's await,
# exactly reproducing the race window the bug report describes.

# Match the script-marker comment the queue injects at the top of each Lua
# script so we can identify which script is being evaluated.
_MARK_RETRY_MARKER = "crawl4ai:mark-retry"
_COMPLETE_MARKER = "crawl4ai:complete"


class _RaceRedis(FakeRedis):
    """FakeRedis that yields inside the release's terminal eval.

    On the first eval of the armed script (the release's mark_retry or
    complete) it sets ``crawl_event`` and ``await asyncio.sleep(0)``s, letting
    the crawl coroutine resume and dispatch its own competing terminal eval
    during the release's await — modelling the aioredis socket-read await
    that suspends the release coroutine. Only the first matching eval yields,
    so the crawl's subsequent terminal eval runs to completion without
    suspending.
    """

    def __init__(self, crawl_event, marker):
        super().__init__()
        self._crawl_event = crawl_event
        self._marker = marker
        self._yielded = False

    async def eval(self, script, numkeys, *keys_and_args):
        if self._marker in script and not self._yielded:
            self._yielded = True
            self._crawl_event.set()
            await asyncio.sleep(0)
        return await super().eval(script, numkeys, *keys_and_args)


def _setup_race(config, consumer, crawl, webhook, marker):
    """Enqueue and wire a worker whose Redis yields during the release's eval.

    A single ``_RaceRedis`` backs both the enqueue and the worker so every
    observable side effect (acks, deletes, hashes, pending) lives on one
    instance the test can assert against.
    """
    crawl_event = asyncio.Event()
    redis = _RaceRedis(crawl_event, marker)
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
    entry = asyncio.run(queue.read_new(consumer))[0]

    async def crawl_with_event(_payload):
        await crawl_event.wait()
        return await crawl(_payload)

    worker = CrawlJobWorker(queue, config, consumer, crawl=crawl_with_event, webhook_service=webhook)
    return redis, queue, task_id, entry, worker


def test_race_retry_case_crawl_completes_during_mark_retry():
    """Retry case: the crawl completes during the release's mark_retry await.

    With the fix the crawl is cancelled before mark_retry runs, so the entry is
    left pending for retry and no webhook is sent (correct: the job is not
    terminal). Without the fix the crawl's complete wins, the release's
    mark_retry raises CrawlJobLeaseLost, and the webhook is dropped despite the
    job being completed in Redis.
    """
    config = queue_config(max_attempt_seconds=1, max_attempts=2, heartbeat_seconds=60, lease_seconds=120)

    async def crawl(_payload):
        return {"success": True, "results": [{"url": "https://example.com"}]}

    webhook = RecordingWebhook()
    redis, queue, task_id, entry, worker = _setup_race(
        config, "worker-a", crawl, webhook, _MARK_RETRY_MARKER
    )
    asyncio.run(worker.process(entry))

    task = redis.hashes[queue.task_key(task_id)]
    # With the fix the crawl was cancelled before its terminal write, so the
    # release's mark_retry landed: status stays processing and last_error is
    # the budget message. The crawl's complete never ran, so the job is NOT
    # completed in Redis.
    assert task["status"] == "processing"
    assert "attempt budget" in task["last_error"]
    # The entry stays pending and un-acked for another worker to reclaim.
    assert entry.stream_id in redis.pending
    assert redis.acks == []
    assert queue.payload_key(task_id) in redis.hashes
    # Retry case never sends a webhook (the job is not terminal yet), so this
    # is NOT the dropped-webhook assertion — the dropped-webhook regression is
    # that, WITHOUT the fix, status would be "completed" here and the payload
    # hash would be gone. The fix makes the retry path behave as designed.
    assert webhook.calls == []


def test_race_terminal_case_crawl_completes_during_release_complete():
    """Terminal case (attempt budget exhausted): the crawl completes during the
    release's complete await.

    With the fix the crawl is cancelled first and the release's complete lands
    unopposed: the job is reported ``failed`` with the budget error and the
    webhook IS delivered. Without the fix the crawl's complete wins, the
    release's complete raises CrawlJobLeaseLost, ``process`` catches it and
    returns without notifying, and the webhook is dropped — while Redis shows
    the job ``completed`` with the real result.
    """
    config = queue_config(max_attempt_seconds=1, max_attempts=1, heartbeat_seconds=60, lease_seconds=120)

    async def crawl(_payload):
        return {"success": True, "results": [{"url": "https://example.com"}]}

    webhook = RecordingWebhook()
    redis, queue, task_id, entry, worker = _setup_race(
        config, "worker-a", crawl, webhook, _COMPLETE_MARKER
    )
    asyncio.run(worker.process(entry))

    task = redis.hashes[queue.task_key(task_id)]
    # With the fix the release's complete lands (the crawl was cancelled before
    # its terminal write): status is failed with the budget error, and the
    # entry is acked and deleted from the stream.
    assert task["status"] == "failed"
    assert "attempt budget" in task["error"]
    assert redis.acks == [(queue.settings.stream, entry.stream_id)]
    assert redis.deleted == [(queue.settings.stream, entry.stream_id)]
    assert queue.payload_key(task_id) not in redis.hashes
    # The webhook is delivered — NOT dropped.
    assert len(webhook.calls) == 1
    assert webhook.calls[0]["status"] == "failed"
    assert "attempt budget" in webhook.calls[0]["error"]
