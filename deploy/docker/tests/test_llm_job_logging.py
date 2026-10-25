"""Background LLM backpressure is diagnostic, not a server defect."""

import asyncio
import logging
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import api
import egress_broker
import llm_broker
import pytest
import utils
from redis import asyncio as aioredis
from test_crawl_job_queue_redis import redis_url  # noqa: F401


@pytest.mark.parametrize(
    ("failure", "expected_level"),
    [
        (None, logging.WARNING),
        (api.HTTPException(status_code=503, detail="Redis unavailable"), logging.ERROR),
        (OSError(2, "Missing configuration", "/app/private.json"), logging.ERROR),
    ],
    ids=["permit-busy", "unrelated-503", "unexpected-error"],
)
def test_llm_job_classifies_permit_backpressure_without_hiding_defects(
    monkeypatch, caplog, failure, expected_level, redis_url
):
    config = utils.load_config()
    webhook = SimpleNamespace(notify_job_completion=AsyncMock())
    monkeypatch.setattr(api, "validate_llm_provider", lambda *_args: (True, None))
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(utils, "load_config", lambda: config)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    monkeypatch.setattr(api, "WebhookDeliveryService", lambda _config: webhook)
    monkeypatch.setattr(
        llm_broker,
        "resolve_llm",
        lambda *_args, **_kwargs: {
            "provider": "openai/test-model",
            "api_token": "test-only",
            "temperature": None,
            "base_url": "https://gateway.example/v1",
            "extra_args": {},
        },
    )

    async def exercise():
        redis = aioredis.from_url(redis_url)
        task_id = f"llm_{uuid4().hex}"
        key = f"task:{task_id}"
        try:
            await redis.hset(key, mapping={"status": api.TaskStatus.PROCESSING.value})
            evaluate = redis.eval

            async def acquire_refused(script, *args, **kwargs):
                if script != api._ACQUIRE_LLM_PERMIT:
                    return await evaluate(script, *args, **kwargs)
                if failure is not None:
                    raise failure
                return 0

            monkeypatch.setattr(redis, "eval", acquire_refused)
            await api.process_llm_extraction(
                redis, config, task_id, "https://example.com", "extract"
            )
            return utils.decode_redis_hash(await redis.hgetall(key))
        finally:
            await redis.delete(key)
            await redis.aclose()

    with caplog.at_level(logging.WARNING):
        mapping = asyncio.run(exercise())
    assert mapping["status"] == api.TaskStatus.FAILED
    assert mapping["error"].startswith("LLM extraction failed (correlation_id=")
    assert "/app/" not in mapping["error"]
    assert webhook.notify_job_completion.await_args.kwargs["error"] == mapping["error"]
    assert webhook.notify_job_completion.await_args.kwargs["status"] == "failed"
    records = [record for record in caplog.records if record.name in {"api", "utils"}]
    assert records
    assert {record.levelno for record in records} == {expected_level}
    assert all(record.exc_info for record in records)
