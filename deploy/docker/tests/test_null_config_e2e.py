"""Regression tests for the crawl HTTP API rejecting JSON ``null`` for
``browser_config`` / ``crawler_config``.

Bug (introduced in 8bb7990): ``CrawlRequest.browser_config`` and
``crawler_config`` were typed ``dict | None``, so a schema-valid JSON ``null``
flowed unmodified into ``BrowserConfig.load(None, ...)`` /
``CrawlerRunConfig.load(None, ...)`` (in ``deploy/docker/api.py`` and
``deploy/docker/server.py``). The loaders dereference ``None.items()`` and raise
``AttributeError``; the route/handler guards only catch
``UntrustedConfigError`` / ``HookValidationError``, so the ``AttributeError``
became an uncaught HTTP 500 with a server-side ``AttributeError`` traceback.

``/crawl/job`` was never affected because its ``CrawlJobPayload`` body model
already typed both fields as bare ``dict`` (explicit ``null`` -> pydantic 422
before any handler runs).

Fix: tighten ``CrawlRequest`` / ``CrawlRequestWithHooks`` to use bare ``dict``
for both fields, mirroring ``CrawlJobPayload``. A JSON ``null`` is now rejected
at the schema layer with 422 instead of crashing inside a handler with 500.
"""

import pytest
from auth import create_access_token
from pydantic import ValidationError


def _bearer() -> dict:
    """A valid bearer header for the in-process test principal."""
    return {"Authorization": f"Bearer {create_access_token({'sub': 'test@example.com'})}"}


# --------------------------------------------------------------------------- #
# Schema-level unit tests (direct model validation, no router)
# --------------------------------------------------------------------------- #

def test_crawl_request_rejects_null_browser_config():
    """A JSON ``null`` for ``browser_config`` must fail schema validation."""
    from schemas import CrawlRequest

    with pytest.raises(ValidationError) as exc:
        CrawlRequest.model_validate(
            {"urls": ["https://example.com"], "browser_config": None}
        )
    assert any(err["loc"][-1] == "browser_config" for err in exc.value.errors())


def test_crawl_request_rejects_null_crawler_config():
    """A JSON ``null`` for ``crawler_config`` must fail schema validation."""
    from schemas import CrawlRequest

    with pytest.raises(ValidationError) as exc:
        CrawlRequest.model_validate(
            {"urls": ["https://example.com"], "crawler_config": None}
        )
    assert any(err["loc"][-1] == "crawler_config" for err in exc.value.errors())


def test_crawl_request_with_hooks_rejects_null_browser_config():
    """The inherited model used by /crawl and /crawl/stream must also reject
    ``null`` (this is the model the affected endpoints actually receive)."""
    from schemas import CrawlRequestWithHooks

    with pytest.raises(ValidationError) as exc:
        CrawlRequestWithHooks.model_validate(
            {"urls": ["https://example.com"], "browser_config": None}
        )
    assert any(err["loc"][-1] == "browser_config" for err in exc.value.errors())


def test_crawl_request_with_hooks_rejects_null_crawler_config():
    from schemas import CrawlRequestWithHooks

    with pytest.raises(ValidationError) as exc:
        CrawlRequestWithHooks.model_validate(
            {"urls": ["https://example.com"], "crawler_config": None}
        )
    assert any(err["loc"][-1] == "crawler_config" for err in exc.value.errors())


def test_crawl_request_defaults_when_fields_omitted():
    """Omitting the fields must still default to ``{}`` (no regression)."""
    from schemas import CrawlRequest

    req = CrawlRequest(urls=["https://example.com"])
    assert req.browser_config == {}
    assert req.crawler_config == {}


def test_crawl_request_crawler_configs_field_still_optional():
    """``crawler_configs`` is a *list* field with default ``None``; it is NOT
    touched by this fix and must keep accepting ``None`` / omission."""
    from schemas import CrawlRequest

    req = CrawlRequest(urls=["https://example.com"])
    assert req.crawler_configs is None
    req2 = CrawlRequest(
        urls=["https://example.com"],
        crawler_configs=[{"url_matcher": "*x*"}, {"url_matcher": "*y*"}],
    )
    assert req2.crawler_configs == [{"url_matcher": "*x*"}, {"url_matcher": "*y*"}]


# --------------------------------------------------------------------------- #
# End-to-end: a JSON null is rejected with 422 (the fix) on every surface
# --------------------------------------------------------------------------- #

NULL_CONFIG_CASES = [
    ("/crawl", "browser_config"),
    ("/crawl", "crawler_config"),
    ("/crawl/stream", "browser_config"),
    ("/crawl/stream", "crawler_config"),
    ("/crawl/job", "browser_config"),
    ("/crawl/job", "crawler_config"),
]


@pytest.mark.parametrize("endpoint,field", NULL_CONFIG_CASES)
def test_null_config_rejected_at_schema_422(stock_client, endpoint, field):
    """A JSON ``null`` for ``browser_config`` / ``crawler_config`` must be
    rejected at the schema layer with 422, NOT crash inside a handler with 500.

    No stubs are needed: pydantic rejects ``null`` during request-body
    validation, so the route handler never runs. The ``/crawl/job`` cases
    establish parity with the already-correct ``CrawlJobPayload`` model.
    """
    payload = {"urls": ["https://example.com"], field: None}
    response = stock_client.post(endpoint, json=payload, headers=_bearer())

    # The headline assertion: 422, not 500.
    assert response.status_code == 422, (
        f"{endpoint} with {field}=null returned {response.status_code}: "
        f"{response.text}"
    )
    body = response.json()
    assert "detail" in body
    # The 422 must be a pydantic validation error that references the field,
    # not a stray 422 from some other path.
    locs = [
        tuple(err.get("loc", ()))
        for err in body["detail"]
        if isinstance(err, dict)
    ]
    assert any(field in loc for loc in locs), body
    # The underlying AttributeError must never reach the client: with the fix
    # the loader is never called with None, so the crash signature is absent.
    assert "NoneType" not in response.text
    assert "AttributeError" not in response.text


# --------------------------------------------------------------------------- #
# Happy-path: omitted configs still reach the handler (no regression)
# --------------------------------------------------------------------------- #

def test_crawl_with_omitted_configs_succeeds(stock_client, monkeypatch):
    """Omitting both config fields must still default to ``{}`` and reach the
    handler -> 200 (the common case for virtually all clients/tests)."""
    import api
    import crawler_pool
    import egress_broker

    class _OkCrawler:
        async def arun(self, url, *, config=None, dispatcher=None, **_kwargs):
            return [{"url": url, "success": True, "markdown": "ok"}]

        async def arun_many(self, urls, *, config=None, dispatcher=None, **_kwargs):
            return [{"url": u, "success": True, "markdown": "ok"} for u in urls]

    async def _get_crawler(*_args, **_kwargs):
        return _OkCrawler()

    async def _release_crawler(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", _get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", _release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)

    response = stock_client.post(
        "/crawl", json={"urls": ["https://example.com"]}, headers=_bearer()
    )
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True


# --------------------------------------------------------------------------- #
# Parity: CrawlRequest now matches CrawlJobPayload (the intended contract)
# --------------------------------------------------------------------------- #

def test_crawl_job_schema_matches_crawl_request_for_config_fields():
    """The fix makes ``CrawlRequest`` mirror ``CrawlJobPayload``: both fields
    are bare ``dict`` with ``default_factory=dict``. This asserts the
    cross-model parity the bug report identifies as the intended contract."""
    from job import CrawlJobPayload
    from schemas import CrawlRequest

    def _field_info(model, name):
        finfo = model.model_fields[name]
        return {
            "annotation": finfo.annotation,
            "default_factory": finfo.default_factory,
        }

    for field in ("browser_config", "crawler_config"):
        crawl_info = _field_info(CrawlRequest, field)
        job_info = _field_info(CrawlJobPayload, field)
        # Both must be bare ``dict`` (not ``dict | None``), both must default
        # to a fresh ``{}`` via default_factory.
        assert crawl_info["annotation"] is dict, crawl_info
        assert job_info["annotation"] is dict, job_info
        assert crawl_info["default_factory"] is dict, crawl_info
        assert job_info["default_factory"] is dict, job_info
