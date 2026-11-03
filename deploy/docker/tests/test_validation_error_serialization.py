"""Regression suite for the 500-instead-of-422 bug introduced in 54d7d09a.

Commit 54d7d09a ("fix(deploy): serialize validation errors safely") added a
custom ``RequestValidationError`` handler to ``deploy/docker/server.py`` that
strips the ``input`` key from each pydantic error dict but passes ``ctx``
through verbatim.  When a ``field_validator`` raises a ``ValueError``, pydantic
populates ``ctx['error']`` with the *live exception object*; ``json.dumps``
inside ``JSONResponse`` then raises ``TypeError``, which the catch-all
``Exception`` handler turns into a generic ``500`` -- defeating the handler's
own goal of keeping validation failures on the 422 client-error path.

The fix sanitizes ``ctx`` before serialization: if any ``ctx`` value is not a
JSON-safe primitive, every ``ctx`` value is stringified.  These tests pin the
guarantees that would break if the sanitization is dropped or weakened:

* ValueError-raising validators return 422 (not 500) with the message
  preserved in stringified ``ctx['error']``.
* JSON-safe ``ctx`` (e.g. ``{"le": 2.0}``) is preserved verbatim, not
  stringified.
* Plain pydantic type errors (no ``ctx``) still return 422.
* The ``input`` key is always stripped (untrusted values never echoed back).
* The handler never raises ``TypeError`` for a missing-``ctx`` edge case.
"""
import json

from auth import create_access_token
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError


def _bearer() -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': 'test@example.com'})}"}


def test_crawl_job_crlf_in_webhook_header_returns_422(stock_client):
    """CRLF in a webhook header value must be a 422, not a 500.

    ``WebhookConfig._validate_headers`` raises ``ValueError`` -> pydantic
    stores the live exception in ``ctx['error']`` -> the handler must
    stringify it so ``json.dumps`` does not crash.
    """
    response = stock_client.post(
        "/crawl/job",
        json={
            "urls": ["https://example.com"],
            "webhook_config": {
                "webhook_url": "https://example.com/hook",
                "webhook_headers": {"X-A": "bad\r\nvalue"},
            },
        },
        headers=_bearer(),
    )
    assert response.status_code == 422, response.text
    err = response.json()["detail"][0]
    assert err["type"] == "value_error", err
    assert "webhook_headers" in err["loc"], err
    assert "input" not in err, err
    assert isinstance(err["ctx"]["error"], str), err
    assert "invalid value for webhook header X-A" in err["ctx"]["error"], err
    assert "correlation_id" not in response.json(), response.text


def test_out_of_range_temperature_preserves_json_safe_ctx(stock_client):
    """A ``ge``/``le`` bound error has JSON-safe ``ctx`` (e.g. ``{"le": 2.0}``)
    and must remain 422 with ``ctx`` preserved verbatim (not stringified)."""
    response = stock_client.post(
        "/llm/job",
        json={"url": "https://example.com", "q": "x", "temperature": 99999.0},
        headers=_bearer(),
    )
    assert response.status_code == 422, response.text
    err = response.json()["detail"][0]
    assert err["type"] == "less_than_equal", err
    assert err["ctx"] == {"le": 2.0}, err


def test_missing_required_field_still_422(stock_client):
    """A plain pydantic ``missing`` error (no ``ctx``) must still be 422."""
    response = stock_client.post(
        "/crawl/job",
        json={"urls": ["https://example.com"], "webhook_config": {"webhook_headers": {}}},
        headers=_bearer(),
    )
    assert response.status_code == 422, response.text
    locs = [tuple(e.get("loc", ())) for e in response.json()["detail"]]
    assert any("webhook_url" in loc for loc in locs), response.text


def test_handler_serializes_live_valueerror_in_ctx(server_module):
    """A live ``ValueError`` in ``ctx['error']`` must not crash the handler."""
    from schemas import WebhookConfig

    try:
        WebhookConfig(
            webhook_url="https://example.com/hook",
            webhook_headers={"X-A": "bad\r\nvalue"},
        )
    except ValidationError as ve:
        exc = RequestValidationError(ve.errors())

    import asyncio

    async def _run():
        return await server_module._validation_exception_handler(None, exc)

    response = asyncio.run(_run())
    assert response.status_code == 422
    err = json.loads(response.body)["detail"][0]
    assert "input" not in err, err
    assert isinstance(err["ctx"]["error"], str), err


def test_handler_handles_error_without_ctx(server_module):
    """An error dict with no ``ctx`` key must serialize without issue."""
    import asyncio

    exc = RequestValidationError([
        {
            "type": "missing",
            "loc": ("body", "field"),
            "msg": "Field required",
            "input": None,
            "url": "https://errors.pydantic.dev/2.11/v/missing",
        }
    ])

    async def _run():
        return await server_module._validation_exception_handler(None, exc)

    response = asyncio.run(_run())
    assert response.status_code == 422
    err = json.loads(response.body)["detail"][0]
    assert err["type"] == "missing"
    assert "input" not in err


def test_handler_strips_input_from_every_error(server_module):
    """The ``input`` key is stripped from every error dict, regardless of ctx."""
    import asyncio

    exc = RequestValidationError([
        {
            "type": "value_error",
            "loc": ("body", "field"),
            "msg": "Value error, bad",
            "input": {"secret": "leak"},
            "ctx": {"error": ValueError("bad")},
            "url": "https://errors.pydantic.dev/2.11/v/value_error",
        },
        {
            "type": "missing",
            "loc": ("body", "other"),
            "msg": "Field required",
            "input": "untrusted",
            "url": "https://errors.pydantic.dev/2.11/v/missing",
        },
    ])

    async def _run():
        return await server_module._validation_exception_handler(None, exc)

    response = asyncio.run(_run())
    assert response.status_code == 422
    for err in json.loads(response.body)["detail"]:
        assert "input" not in err, err


def test_handler_rejects_non_finite_float_in_ctx(server_module):
    """A non-finite float (``nan``/``inf``) in ``ctx`` must not crash the handler.

    ``JSONResponse`` serializes with ``allow_nan=False``, so a bare ``nan`` or
    ``inf`` would raise ``ValueError`` and the catch-all handler would turn the
    422 into a 500.  The sanitizer treats non-finite floats as non-JSON-safe and
    stringifies ``ctx`` so the response stays a clean 422.
    """
    import asyncio
    import math

    exc = RequestValidationError([
        {
            "type": "value_error",
            "loc": ("body", "field"),
            "msg": "Value error, bad",
            "input": "untrusted",
            "ctx": {"limit": math.inf},
            "url": "https://errors.pydantic.dev/2.11/v/value_error",
        },
        {
            "type": "value_error",
            "loc": ("body", "other"),
            "msg": "Value error, bad",
            "input": "untrusted",
            "ctx": {"limit": float("nan")},
            "url": "https://errors.pydantic.dev/2.11/v/value_error",
        },
    ])

    async def _run():
        return await server_module._validation_exception_handler(None, exc)

    response = asyncio.run(_run())
    assert response.status_code == 422, response.body
    for err in json.loads(response.body)["detail"]:
        assert "input" not in err, err
        assert isinstance(err["ctx"]["limit"], str), err
        assert math.isfinite(float(err["ctx"]["limit"])) is False, err