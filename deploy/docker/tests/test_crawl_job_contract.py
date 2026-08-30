import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api  # noqa: E402
import crawl_job_worker  # noqa: E402
import job  # noqa: E402
import llm_broker  # noqa: E402
from crawl_job_queue import CrawlJobAttempt, CrawlJobEntry  # noqa: E402
from crawl_job_worker import CrawlJobWorker  # noqa: E402


def test_public_submit_accepts_every_supported_result_field(monkeypatch):
    result_fields = [
        "url",
        "redirected_url",
        "success",
        "error_message",
        "status_code",
        "markdown",
        "links",
        "metadata",
    ]
    submitted = {}

    async def accept_job(
        _redis,
        _urls,
        _browser_config,
        _crawler_config,
        *,
        config,
        result_fields,
        webhook_config,
        owner=None,
    ):
        submitted["result_fields"] = result_fields
        submitted["owner"] = owner
        return {"task_id": "crawl_contract"}

    monkeypatch.setattr(job, "handle_crawl_job", accept_job)
    app = FastAPI()

    @app.middleware("http")
    async def authenticated_request(request, call_next):
        request.state.principal = {"sub": "alice", "scope": "data"}
        return await call_next(request)

    app.include_router(job.init_job_router(object(), {}, lambda: {}))

    with TestClient(app) as client:
        response = client.post(
            "/crawl/job",
            json={
                "urls": ["https://example.com"],
                "browser_config": {"headless": True},
                "crawler_config": {"page_timeout": 30000},
                "result_fields": result_fields,
            },
        )

    assert response.status_code == 202
    assert response.json() == {"task_id": "crawl_contract"}
    assert submitted["result_fields"] == result_fields
    assert submitted["owner"] == "alice"


def test_crawl_job_reuses_canonical_url_count_bounds():
    accepted = job.CrawlJobPayload(
        urls=[f"https://example.com/{index}" for index in range(100)]
    )
    assert len(accepted.urls) == 100

    with pytest.raises(ValueError, match="at most 100 items"):
        job.CrawlJobPayload(
            urls=[f"https://example.com/{index}" for index in range(101)]
        )


def test_worker_terminalizes_deterministic_input_failure_without_retry():
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=3),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )

    async def reject_input(_payload):
        raise HTTPException(status_code=400, detail="Cannot resolve URL host")

    worker = CrawlJobWorker(queue, {}, "worker-a", crawl=reject_input, webhook_service=object())
    entry = CrawlJobEntry(stream_id="1-0", task_id="crawl_terminal")
    attempt = CrawlJobAttempt(number=1, fence_token="attempt-a", consumer="worker-a")

    result = asyncio.run(worker._process_attempt(entry, {}, attempt=attempt))

    assert result == ("failed", None, "Cannot resolve URL host")
    queue.complete.assert_awaited_once_with(
        entry,
        {},
        attempt,
        error="Cannot resolve URL host",
    )
    queue.mark_retry.assert_not_awaited()


def test_worker_terminalizes_an_all_failed_crawl_and_keeps_its_results():
    """A crawl whose every URL failed ran to the end, so it is terminal rather
    than retryable, but the per-URL diagnostics must still reach the poller."""
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=3),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )
    crawl_result = {
        "success": False,
        "results": [{"url": "https://example.com", "success": False, "error_message": "Crawl failed"}],
    }

    async def all_urls_fail(_payload):
        return crawl_result

    worker = CrawlJobWorker(queue, {}, "worker-a", crawl=all_urls_fail, webhook_service=object())
    entry = CrawlJobEntry(stream_id="1-0", task_id="crawl_all_failed")
    attempt = CrawlJobAttempt(number=1, fence_token="attempt-a", consumer="worker-a")

    result = asyncio.run(worker._process_attempt(entry, {}, attempt=attempt))

    assert result == ("failed", crawl_result, "Every crawled URL failed")
    queue.complete.assert_awaited_once_with(
        entry,
        {},
        attempt,
        result=crawl_result,
        error="Every crawled URL failed",
    )
    queue.mark_retry.assert_not_awaited()


def test_worker_completes_when_any_url_succeeded():
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=3),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )
    crawl_result = {
        "success": True,
        "results": [
            {"url": "https://ok.example", "success": True},
            {"url": "https://bad.example", "success": False, "error_message": "Crawl failed"},
        ],
    }

    async def partial_success(_payload):
        return crawl_result

    worker = CrawlJobWorker(queue, {}, "worker-a", crawl=partial_success, webhook_service=object())
    entry = CrawlJobEntry(stream_id="1-0", task_id="crawl_partial")
    attempt = CrawlJobAttempt(number=1, fence_token="attempt-a", consumer="worker-a")

    result = asyncio.run(worker._process_attempt(entry, {}, attempt=attempt))

    assert result == ("completed", crawl_result, None)
    queue.complete.assert_awaited_once_with(entry, {}, attempt, result=crawl_result)


def test_task_status_of_a_failed_crawl_job_still_carries_its_results():
    response = api.create_task_response(
        {
            "status": api.TaskStatus.FAILED,
            "created_at": "2026-08-29T12:00:00",
            "url": "https://example.com",
            "result": json.dumps({"success": False, "results": [{"url": "https://example.com"}]}),
            "error": "Every crawled URL failed",
        },
        "crawl_all_failed",
        "https://crawl.example/",
        "crawl/job",
    )

    assert response["error"] == "Every crawled URL failed"
    assert response["result"]["results"] == [{"url": "https://example.com"}]


@pytest.mark.parametrize(
    "failure",
    [
        HTTPException(
            status_code=500,
            detail=json.dumps(
                {
                    "error": (
                        "Unexpected error in _crawl_web at line 806 in _crawl_web "
                        "(/app/crawl4ai/async_crawler_strategy.py): boom"
                    ),
                    "server_memory_delta_mb": 1,
                }
            ),
        ),
        OSError(28, "No space left on device", "/app/crawl4ai/cache/x"),
    ],
    ids=["internal-500-detail", "bare-oserror"],
)
def test_worker_does_not_publish_internal_detail_to_the_job_client(failure):
    """The worker is a separate process, so the API's central 500 handler never
    genericizes what lands in the task hash /crawl/job/{id} returns."""
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=1),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )

    async def blow_up(_payload):
        raise failure

    worker = CrawlJobWorker(queue, {}, "worker-a", crawl=blow_up, webhook_service=object())
    entry = CrawlJobEntry(stream_id="1-0", task_id="crawl_leak")
    attempt = CrawlJobAttempt(number=1, fence_token="attempt-a", consumer="worker-a")

    status, _result, message = asyncio.run(worker._process_attempt(entry, {}, attempt=attempt))

    assert status == "failed"
    assert message.startswith("Crawl job failed (correlation_id=")
    assert "/app/" not in message and "async_crawler_strategy" not in message
    queue.complete.assert_awaited_once_with(entry, {}, attempt, error=message)


def test_llm_job_failure_does_not_publish_internal_detail(monkeypatch):
    """process_llm_extraction runs as a background task: its catch-all wrote
    str(e) straight into the hash /llm/job/{id} returns and into the webhook."""
    stored = {}
    sent = {}

    async def record_hset(_redis, key, mapping, _config):
        stored[key] = mapping

    class FakeWebhook:
        def __init__(self, _config):
            pass

        async def notify_job_completion(self, **kwargs):
            sent.update(kwargs)

    def blow_up(*_args, **_kwargs):
        raise OSError(2, "No such file or directory", "/app/crawl4ai/provider.json")

    monkeypatch.setattr(api, "validate_llm_provider", lambda *_a, **_k: (True, None))
    monkeypatch.setattr(llm_broker, "resolve_llm", blow_up)
    monkeypatch.setattr(api, "hset_with_ttl", record_hset)
    monkeypatch.setattr(api, "WebhookDeliveryService", FakeWebhook)

    asyncio.run(
        api.process_llm_extraction(
            object(), {"llm": {}}, "llm_leak", "https://example.com", "extract"
        )
    )

    message = stored["task:llm_leak"]["error"]
    assert message.startswith("LLM extraction failed (correlation_id=")
    assert "/app/" not in message
    assert sent["error"] == message
    assert sent["status"] == "failed"


def test_worker_retries_request_timeout():
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=3),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )

    async def time_out(_payload):
        raise HTTPException(status_code=408, detail="Target request timed out")

    worker = CrawlJobWorker(queue, {}, "worker-a", crawl=time_out, webhook_service=object())
    entry = CrawlJobEntry(stream_id="1-0", task_id="crawl_retry")
    attempt = CrawlJobAttempt(number=1, fence_token="attempt-a", consumer="worker-a")

    result = asyncio.run(worker._process_attempt(entry, {}, attempt=attempt))

    assert result is None
    queue.complete.assert_not_awaited()
    queue.mark_retry.assert_awaited_once_with(
        entry,
        {},
        "worker-a",
        attempt,
        "Target request timed out",
    )


def test_llm_unexpected_failure_propagates_to_central_exception_owner(monkeypatch):
    async def fail_task_creation(*_args, **_kwargs):
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(api, "create_new_task", fail_task_creation)
    request = SimpleNamespace(
        url=SimpleNamespace(
            scheme="https",
            netloc="crawl.example",
        )
    )

    with pytest.raises(RuntimeError, match="private provider detail"):
        asyncio.run(
            api.handle_llm_request(
                redis=object(),
                background_tasks=object(),
                request=request,
                input_path="https://example.com",
                query="extract",
                config={},
            )
        )


def test_llm_job_enqueue_returns_accepted_with_canonical_links(monkeypatch):
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_enqueue_job", AsyncMock())

    response = asyncio.run(
        api.create_new_task(
            SimpleNamespace(hset=AsyncMock(), expire=AsyncMock()),
            BackgroundTasks(),
            "https://example.com",
            "extract",
            None,
            "0",
            "https://crawl.example/",
            {"redis": {"task_ttl_seconds": 3600}},
        )
    )
    body = json.loads(response.body)

    assert response.status_code == 202
    assert body["_links"]["self"]["href"].startswith(
        "https://crawl.example/llm/job/llm_"
    )
    assert body["_links"]["status"] == body["_links"]["self"]


def test_llm_job_task_ids_are_collision_resistant(monkeypatch):
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_enqueue_job", AsyncMock())
    redis = SimpleNamespace(hset=AsyncMock(), expire=AsyncMock())

    async def create():
        response = await api.create_new_task(
            redis,
            BackgroundTasks(),
            "https://example.com",
            "extract",
            None,
            "0",
            "https://crawl.example/",
            {"redis": {"task_ttl_seconds": 3600}},
        )
        return json.loads(response.body)["task_id"]

    async def create_pair():
        return await asyncio.gather(create(), create())

    first, second = asyncio.run(create_pair())
    assert first != second


@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        ("llm/job", "https://crawl.example/llm/job/llm_123"),
        ("crawl/job", "https://crawl.example/crawl/job/crawl_123"),
    ],
)
def test_task_status_links_preserve_their_job_collection(collection, expected):
    response = api.create_task_response(
        {
            "status": api.TaskStatus.PROCESSING,
            "created_at": "2026-08-25T12:00:00",
            "url": "https://example.com",
        },
        expected.rsplit("/", 1)[-1],
        "https://crawl.example/",
        collection,
    )

    assert response["_links"]["self"]["href"] == expected
    assert response["_links"]["refresh"]["href"] == expected


@pytest.mark.parametrize("worker_error", [False, True])
def test_worker_proxy_lifecycle_wraps_execution_and_cleanup(monkeypatch, worker_error):
    events = []
    proxy = object()

    class FakeRedis:
        async def aclose(self):
            events.append("redis-close")

    class FakeWorker:
        def __init__(self, *_args):
            events.append("worker-init")

        async def run(self):
            events.append("worker-run")
            if worker_error:
                raise RuntimeError("worker failed")

    async def start_proxy():
        events.append("proxy-start")
        return proxy

    async def stop_proxy(candidate):
        assert candidate is proxy
        events.append("proxy-stop")

    async def close_pool():
        events.append("pool-close")

    monkeypatch.setattr(crawl_job_worker, "load_config", lambda: {})
    monkeypatch.setattr(crawl_job_worker, "setup_logging", lambda _config: None)
    monkeypatch.setattr(crawl_job_worker, "build_redis_url", lambda _config: "redis://test")
    monkeypatch.setattr(
        crawl_job_worker.aioredis,
        "from_url",
        lambda *_args, **_kwargs: FakeRedis(),
    )
    monkeypatch.setattr(crawl_job_worker, "CrawlJobQueue", lambda *_args: object())
    monkeypatch.setattr(crawl_job_worker, "CrawlJobWorker", FakeWorker)
    monkeypatch.setattr(crawl_job_worker, "start_pinning_proxy", start_proxy)
    monkeypatch.setattr(crawl_job_worker, "stop_pinning_proxy", stop_proxy)
    monkeypatch.setattr(crawl_job_worker, "close_all", close_pool)

    if worker_error:
        with pytest.raises(RuntimeError, match="worker failed"):
            asyncio.run(crawl_job_worker.run_worker())
    else:
        asyncio.run(crawl_job_worker.run_worker())

    assert events == [
        "proxy-start",
        "worker-init",
        "worker-run",
        "pool-close",
        "proxy-stop",
        "redis-close",
    ]
