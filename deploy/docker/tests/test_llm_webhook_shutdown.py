"""Tests for LLM terminal webhook delivery surviving shutdown cancellation.

Bug: ``process_llm_extraction`` flipped the task terminal in Redis *before*
awaiting the completion webhook (unshielded).  A ``WorkQueue.stop()``
cancellation landing during delivery permanently lost the webhook:
``cancel_task``'s CAS only wins from ``processing`` (so it skipped its own
notification branch on an already-terminal task), and the orphan reaper skips
terminal tasks.

Fix: ``_deliver_terminal_webhook`` shields and bounds the delivery so it
completes even when the enclosing job is cancelled, then re-raises
``CancelledError`` so the worker exits cleanly and ``cancel_task``'s CAS
loses without sending a duplicate.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import api
import egress_broker
import llm_broker
import pytest
from redis import asyncio as aioredis
from test_crawl_job_queue_redis import redis_url  # noqa: F401
from utils import TaskStatus, decode_redis_hash

CONFIG = {
    "llm": {"task_heartbeat_seconds": 30, "job_request_timeout_seconds": 120,
            "job_request_retries": 0, "job_max_output_tokens": 4096,
            "backoff_max_attempts": 1},
    "redis": {"task_ttl_seconds": 60},
    "webhooks": {"retry": {"max_attempts": 1, "timeout_ms": 2000}},
}

_RESOLVE = lambda *_a, **_kw: {"provider": "openai/test-model", "api_token": "t",
                               "temperature": None, "base_url": None, "extra_args": {}}


class _Crawler:
    """Crawler stub whose ``arun`` returns ``result`` or raises ``exc``."""
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return None

    async def arun(self, **_kw):
        if self._exc is not None:
            raise self._exc
        return self._result


def _patch_crawl(monkeypatch, result=None, exc=None, provider_valid=True):
    monkeypatch.setattr(api, "AsyncWebCrawler", lambda **_kw: _Crawler(result, exc))
    monkeypatch.setattr(api, "validate_llm_provider", lambda *_a: (provider_valid, None))


@pytest.fixture
def slow_webhook(monkeypatch):
    """Patch boundaries; delivery parks until ``allow_event`` is set.

    Yields ``(config, calls, deliver_event, allow_event)``.
    """
    from utils import load_config
    cfg = load_config()
    cfg["llm"].update(CONFIG["llm"])
    cfg["redis"].update(CONFIG["redis"])

    deliver_event, allow_event = asyncio.Event(), asyncio.Event()
    calls = []

    async def notify(**kwargs):
        deliver_event.set()
        await allow_event.wait()
        calls.append(kwargs)
        return True

    @asynccontextmanager
    async def permit(*_):
        yield

    monkeypatch.setattr(api, "llm_permit", permit)
    monkeypatch.setattr(api, "validate_url_destination", lambda _u: None)
    monkeypatch.setattr(api, "WebhookDeliveryService", lambda _c: SimpleNamespace(notify_job_completion=notify))
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    monkeypatch.setattr(llm_broker, "resolve_llm", _RESOLVE)
    return cfg, calls, deliver_event, allow_event


@pytest.fixture
def fast_webhook(monkeypatch):
    """Like ``slow_webhook`` but delivery resolves instantly (no parking)."""
    from utils import load_config
    cfg = load_config()
    cfg["llm"].update(CONFIG["llm"])
    cfg["redis"].update(CONFIG["redis"])
    calls = []

    async def fast_notify(**kwargs):
        calls.append(kwargs)
        return True

    @asynccontextmanager
    async def permit(*_):
        yield

    monkeypatch.setattr(api, "llm_permit", permit)
    monkeypatch.setattr(api, "validate_url_destination", lambda _u: None)
    monkeypatch.setattr(api, "WebhookDeliveryService",
                        lambda _c: SimpleNamespace(notify_job_completion=fast_notify))
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    monkeypatch.setattr(llm_broker, "resolve_llm", _RESOLVE)
    return cfg, calls


async def _seed(client, task_id, webhook_config=None):
    mapping = {"status": TaskStatus.PROCESSING.value,
               "created_at": datetime.now().isoformat(), "url": "https://example.com",
               "heartbeat_at": (await client.time())[0]}
    if webhook_config:
        mapping["webhook_config"] = json.dumps(webhook_config)
    await client.hset(f"task:{task_id}", mapping=mapping)


# Path table: (id, crawler result/exc, provider_valid, expected terminal status)
_PATHS = [
    ("success", SimpleNamespace(success=True, extracted_content='{"a": 1}'),
     True, TaskStatus.COMPLETED.value),
    ("crawl-failure", SimpleNamespace(success=False, error_message="nav timeout", extracted_content=""),
     True, TaskStatus.FAILED.value),
    ("exception", None, True, TaskStatus.FAILED.value),  # _exc set in helper
    ("invalid-provider", None, False, TaskStatus.FAILED.value),
]


@pytest.mark.parametrize("path, result, provider_valid, expected_status", _PATHS,
                         ids=[p[0] for p in _PATHS])
def test_terminal_webhook_survives_shutdown_cancellation(
    redis_url, slow_webhook, monkeypatch, path, result, provider_valid, expected_status,
):
    """Shutdown (Cancel) during the post-terminal webhook delivery must not
    lose it — on every terminal path in ``process_llm_extraction``."""
    exc = RuntimeError("boom") if path == "exception" else None
    _patch_crawl(monkeypatch, result=result, exc=exc, provider_valid=provider_valid)
    if path == "invalid-provider":
        monkeypatch.setattr(api, "validate_llm_provider", lambda *_a: (False, "bad provider"))
    config, calls, deliver_event, allow_event = slow_webhook
    kwargs_extra = {"provider": "bad-provider"} if path == "invalid-provider" else {}

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        worker = None
        try:
            await _seed(client, task_id)
            worker = asyncio.create_task(api.process_llm_extraction(
                client, config, task_id, "https://example.com", "extract", **kwargs_extra,
            ))
            await asyncio.wait_for(deliver_event.wait(), timeout=5)
            assert decode_redis_hash(await client.hgetall(key))["status"] == expected_status
            assert not calls  # parked mid-delivery

            worker.cancel()
            await asyncio.sleep(0)
            allow_event.set()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, timeout=15)

            assert len(calls) == 1
            assert calls[0]["status"] == expected_status
            assert calls[0]["task_id"] == task_id
            if path == "invalid-provider":
                assert calls[0]["error"] == "bad provider"
        finally:
            allow_event.set()
            if worker and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


def test_success_webhook_not_duplicated_by_cancel_task(redis_url, slow_webhook, monkeypatch):
    """``cancel_task`` CAS only wins from ``processing``; on an already-terminal
    task it must NOT send a second (failure) webhook."""
    _patch_crawl(monkeypatch, result=SimpleNamespace(success=True, extracted_content='{"x": 1}'))
    config, calls, deliver_event, allow_event = slow_webhook
    wh_cfg = {"webhook_url": "http://h/hook"}

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        worker = None
        try:
            await _seed(client, task_id, wh_cfg)
            worker = asyncio.create_task(api.process_llm_extraction(
                client, config, task_id, "https://example.com", "extract", webhook_config=wh_cfg,
            ))
            await asyncio.wait_for(deliver_event.wait(), timeout=5)

            async def cancel_task():
                won = await api._finish_llm_task(client, key, {
                    "status": TaskStatus.FAILED, "error": "shutdown"}, config)
                if won:
                    calls.append({"_duplicate": True})

            worker.cancel()
            await asyncio.sleep(0)
            allow_event.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, timeout=15)
            await cancel_task()  # WorkQueue._worker calls on_cancel after CancelledError

            assert len(calls) == 1
            assert calls[0]["status"] == "completed"
        finally:
            allow_event.set()
            if worker and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("result, provider_valid, expected_status", [
    (SimpleNamespace(success=True, extracted_content='{"a": 7}'), True, TaskStatus.COMPLETED.value),
    (SimpleNamespace(success=False, error_message="timeout", extracted_content=""), True, TaskStatus.FAILED.value),
], ids=["success", "failure"])
def test_terminal_webhook_delivered_without_cancellation(
    redis_url, fast_webhook, monkeypatch, result, provider_valid, expected_status,
):
    """Regression guard: normal (un-cancelled) delivery still works."""
    _patch_crawl(monkeypatch, result=result, provider_valid=provider_valid)
    config, calls = fast_webhook

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        try:
            await _seed(client, task_id)
            await api.process_llm_extraction(client, config, task_id, "https://example.com", "extract")
            assert len(calls) == 1
            assert calls[0]["status"] == expected_status
            assert decode_redis_hash(await client.hgetall(key))["status"] == expected_status
        finally:
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


def test_shutdown_cancellation_during_webhook_timeout_does_not_hang(redis_url, slow_webhook, monkeypatch):
    """A receiver that never responds must not hang the worker: the shielded
    delivery is bounded by the timeout and CancelledError still propagates."""
    _patch_crawl(monkeypatch, result=SimpleNamespace(success=True, extracted_content='{"x": 1}'))
    config, calls, deliver_event, allow_event = slow_webhook

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        worker = None
        try:
            await _seed(client, task_id)
            worker = asyncio.create_task(api.process_llm_extraction(
                client, config, task_id, "https://example.com", "extract"))
            await asyncio.wait_for(deliver_event.wait(), timeout=5)
            worker.cancel()
            # Never release allow_event -> delivery times out inside the shield.
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, timeout=15)
            assert calls == []
            assert decode_redis_hash(await client.hgetall(key))["status"] == TaskStatus.COMPLETED.value
        finally:
            allow_event.set()
            if worker and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())


def test_workqueue_stop_during_success_webhook_still_delivers(redis_url, slow_webhook, monkeypatch):
    """End-to-end: ``WorkQueue.stop()`` cancels a worker parked in delivery and
    the completion webhook still reaches the subscriber."""
    from work_queue import WorkQueue
    _patch_crawl(monkeypatch, result=SimpleNamespace(success=True, extracted_content='{"ready": true}'))
    config, calls, deliver_event, allow_event = slow_webhook

    async def exercise():
        client = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        queue = WorkQueue(maxsize=0, workers=1)
        try:
            await _seed(client, task_id)
            await queue.start()

            async def factory():
                await api.process_llm_extraction(client, config, task_id, "https://example.com", "extract")

            await queue.submit(factory)
            await asyncio.wait_for(deliver_event.wait(), timeout=5)
            stop = asyncio.create_task(queue.stop())
            await asyncio.sleep(0.1)
            allow_event.set()
            await asyncio.wait_for(stop, timeout=15)
            assert len(calls) == 1
            assert calls[0]["status"] == "completed"
            assert decode_redis_hash(await client.hgetall(key))["status"] == TaskStatus.COMPLETED.value
        finally:
            allow_event.set()
            if queue.started:
                await queue.stop()
            await client.delete(key)
            await client.aclose()

    asyncio.run(exercise())
