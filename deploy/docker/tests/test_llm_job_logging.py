"""Background LLM backpressure is diagnostic, not a server defect."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import api
import egress_broker
import llm_broker
import pytest
import utils


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
    monkeypatch, caplog, failure, expected_level
):
    config = utils.load_config()
    redis = SimpleNamespace(
        set=AsyncMock(return_value=False, side_effect=failure),
        hset=AsyncMock(),
        expire=AsyncMock(),
    )
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

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            api.process_llm_extraction(
                redis, config, "llm_logging", "https://example.com", "extract"
            )
        )

    mapping = redis.hset.await_args_list[-1].kwargs["mapping"]
    assert mapping["status"] == api.TaskStatus.FAILED
    assert mapping["error"].startswith("LLM extraction failed (correlation_id=")
    assert "/app/" not in mapping["error"]
    assert webhook.notify_job_completion.await_args.kwargs["error"] == mapping["error"]
    assert webhook.notify_job_completion.await_args.kwargs["status"] == "failed"
    records = [record for record in caplog.records if record.name in {"api", "utils"}]
    assert records
    assert {record.levelno for record in records} == {expected_level}
    assert all(record.exc_info for record in records)
