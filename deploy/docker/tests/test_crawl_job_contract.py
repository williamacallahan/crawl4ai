import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import job
from crawl_job_queue import CrawlJobEntry
from crawl_job_worker import CrawlJobWorker


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

    result = asyncio.run(worker._process_attempt(entry, {}, attempt=1))

    assert result == ("failed", None, "Cannot resolve URL host")
    queue.complete.assert_awaited_once_with(entry, {}, error="Cannot resolve URL host")
    queue.mark_retry.assert_not_awaited()


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

    result = asyncio.run(worker._process_attempt(entry, {}, attempt=1))

    assert result is None
    queue.complete.assert_not_awaited()
    queue.mark_retry.assert_awaited_once_with(
        entry,
        {},
        "worker-a",
        1,
        "Target request timed out",
    )
