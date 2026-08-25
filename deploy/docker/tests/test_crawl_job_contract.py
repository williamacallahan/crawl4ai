import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api  # noqa: E402
import crawl_job_worker  # noqa: E402
import job  # noqa: E402
from crawl_job_queue import CrawlJobAttempt, CrawlJobEntry  # noqa: E402
from crawl_job_worker import CrawlJobWorker  # noqa: E402
from schemas import CRAWL_RESULT_FIELDS  # noqa: E402


def test_public_submit_accepts_every_supported_result_field(monkeypatch):
    result_fields = [
        "url",
        "redirected_url",
        "success",
        "error_message",
        "status_code",
        "html",
        "markdown",
        "media",
        "links",
        "metadata",
        "response_headers",
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
    assert accepted.result_fields == list(CRAWL_RESULT_FIELDS)

    with pytest.raises(ValueError, match="at most 100 items"):
        job.CrawlJobPayload(
            urls=[f"https://example.com/{index}" for index in range(101)]
        )

    with pytest.raises(ValueError, match="must include 'success' and 'error_message'"):
        job.CrawlJobPayload(
            urls=["https://example.com"],
            result_fields=["html"],
        )


def test_worker_terminalizes_deterministic_input_failure_without_retry():
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=3),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )

    async def reject_input(_payload):
        raise HTTPException(status_code=400, detail="Cannot resolve URL host")

    worker = CrawlJobWorker(
        queue, {}, "worker-a", crawl=reject_input, webhook_service=object()
    )
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


def test_worker_retries_request_timeout():
    queue = SimpleNamespace(
        settings=SimpleNamespace(max_attempts=3),
        complete=AsyncMock(),
        mark_retry=AsyncMock(),
    )

    async def time_out(_payload):
        raise HTTPException(status_code=408, detail="Target request timed out")

    worker = CrawlJobWorker(
        queue, {}, "worker-a", crawl=time_out, webhook_service=object()
    )
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
    monkeypatch.setattr(
        crawl_job_worker, "build_redis_url", lambda _config: "redis://test"
    )
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
