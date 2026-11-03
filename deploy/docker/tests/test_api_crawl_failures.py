import asyncio
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from crawl4ai.models import CrawlResult, MarkdownGenerationResult
from fastapi import HTTPException

from api import _raise_for_crawl_failure


TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
CLIENT_REQUEST_ID = "01234567-89ab-4cde-8fab-0123456789ab"


def _install_markdown_crawler(monkeypatch, result):
    class FakeCrawler:
        async def arun(self, *args, **kwargs):
            return result

    async def get_crawler(*args, **kwargs):
        return FakeCrawler()

    async def release_crawler(*args, **kwargs):
        return None

    import api
    import crawler_pool
    import egress_broker

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)


def _markdown_terminal_events(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("event=md_terminal ")
    ]


@pytest.mark.parametrize("filter_type,fit,expected", [
    ("raw", "fit text", "raw text"),
    ("fit", "fit text", "fit text"),
    ("bm25", "fit text", "fit text"),
    ("fit", "", ""),
    ("bm25", "", ""),
])
@pytest.mark.parametrize("landed", [None, "https://publisher.example/article"])
def test_markdown_preserves_requested_url_and_observed_landing(
    stock_client, server_module, monkeypatch, filter_type, fit, expected, landed
):
    from auth import create_access_token

    requested = "https://tracker.example/redirect?article=1"
    rendered_html = '<html><head><meta property="article:published_time" content="2025-03-21T10:00:00Z"></head><body>article</body></html>'
    _install_markdown_crawler(
        monkeypatch,
        CrawlResult(
            url=requested,
            html=rendered_html,
            success=True,
            redirected_url=landed,
            status_code=302 if landed else 200,
            redirected_status_code=404 if landed else 200,
            markdown=MarkdownGenerationResult(
                raw_markdown="raw text",
                markdown_with_citations="raw text",
                references_markdown="",
                fit_markdown=fit,
            ),
        ),
    )
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/md",
        json={"url": requested, "f": filter_type},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["url"] == requested
    assert payload.get("redirected_url") == landed
    assert payload["markdown"] == expected
    assert payload["raw_markdown"] == "raw text"
    assert payload["html"] == rendered_html
    assert payload["status_code"] == (302 if landed else 200)
    assert payload["redirected_status_code"] == (404 if landed else 200)
    assert payload["success"] is True


@pytest.mark.parametrize("error_message,expected_detail", [
    ("Blocked by anti-bot protection: challenge", "Blocked by anti-bot protection: challenge"),
    ("Permission denied: /app/private/cache.db", "Crawl failed"),
])
def test_crawl_failure_is_reported_as_sanitized_bad_gateway(error_message, expected_detail):
    result = SimpleNamespace(success=False, error_message=error_message)

    with pytest.raises(HTTPException) as raised:
        _raise_for_crawl_failure(result)

    assert raised.value.status_code == 502
    assert raised.value.detail == expected_detail


@pytest.mark.parametrize("error_message,expected_detail", [
    ("Blocked by anti-bot protection: challenge", "Blocked by anti-bot protection: challenge"),
    ("Permission denied: /app/private/cache.db", "Crawl failed"),
])
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/md", {"url": "https://example.com", "f": "raw"}),
        ("get", "/llm/example.com?q=summarize", None),
    ],
)
def test_single_url_crawl_failure_reaches_client(
    stock_client, server_module, monkeypatch, method, path, payload, error_message, expected_detail
):
    failed_result = CrawlResult(
        url="https://example.com", html="", success=False, error_message=error_message,
    )

    class FailedCrawler:
        async def arun(self, *args, **kwargs):
            return failed_result

    async def get_failed_crawler(*args, **kwargs):
        return FailedCrawler()

    async def release_crawler(*args, **kwargs):
        return None

    import api
    import crawler_pool
    from auth import create_access_token

    monkeypatch.setattr(api, "validate_url_destination", lambda url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_failed_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)

    token = create_access_token({"sub": "test@example.com"})
    request = getattr(stock_client, method)
    kwargs = {"json": payload} if payload is not None else {}
    response = request(
        path, headers={"Authorization": f"Bearer {token}"}, **kwargs
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    if path == "/md" and expected_detail == "Crawl failed":
        assert re.fullmatch(r"Crawl failed \(correlation_id=[0-9a-f]{12}\)", detail)
    else:
        assert detail == expected_detail


def test_markdown_success_logs_one_safe_terminal_event(
    stock_client, server_module, monkeypatch, caplog
):
    target_url = "https://private.example/secret-path?private_query=hidden"
    markdown = "private markdown must not reach the terminal event"
    _install_markdown_crawler(
        monkeypatch,
        CrawlResult(
            url=target_url,
            html="<p>article</p>",
            success=True,
            markdown=MarkdownGenerationResult(
                raw_markdown=markdown, fit_markdown=markdown,
                markdown_with_citations=markdown, references_markdown="",
            ),
            redirected_url=None,
        ),
    )
    from auth import create_access_token

    token = create_access_token({"sub": "test@example.com"})
    caplog.set_level("INFO", logger=server_module.__name__)
    response = stock_client.post(
        "/md",
        json={"url": target_url, "f": "raw"},
        headers={
            "Authorization": f"Bearer {token}",
            "traceparent": TRACEPARENT,
            "X-Client-Request-Id": CLIENT_REQUEST_ID,
        },
    )

    assert response.status_code == 200
    event, = _markdown_terminal_events(caplog)
    assert event.startswith("event=md_terminal status=200 outcome=success duration_ms=")
    assert re.search(r"duration_ms=\d+\.\d{3}(?: |$)", event)
    assert f"trace_id={TRACE_ID}" in event
    assert f"client_request_id={CLIENT_REQUEST_ID}" in event
    assert "correlation_id=" not in event
    for sensitive in (target_url, "private_query", markdown, token, TRACEPARENT):
        assert sensitive not in event


@pytest.mark.parametrize("raw_error,public,level", [
    ("Permission denied: /app/private/private-error", "Crawl failed", "ERROR"),
    ("renderer_memory_limit: rss_bytes=1053999104 rss_limit_bytes=805306368",
     "renderer_memory_limit: rss_bytes=1053999104 rss_limit_bytes=805306368", "WARNING"),
    ("renderer_cpu_limit: cpu_cores=1.0 cpu_limit_cores=0.9",
     "renderer_cpu_limit: cpu_cores=1.0 cpu_limit_cores=0.9", "WARNING"),
])
def test_markdown_failed_result_joins_response_level_and_terminal_event(
    stock_client, server_module, monkeypatch, caplog, raw_error, public, level
):
    _install_markdown_crawler(
        monkeypatch,
        CrawlResult(
            url="https://private.example/secret-path", success=False, error_message=raw_error,
            html="<p>Observed unavailable document</p>", status_code=200,
            redirected_url="https://publisher.example/unavailable", redirected_status_code=451,
        ),
    )
    from auth import create_access_token

    caplog.set_level("INFO", logger=server_module.__name__)
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/md",
        json={"url": "https://private.example/secret-path", "f": "raw"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502
    payload = response.json()
    assert payload["html"] == "<p>Observed unavailable document</p>"
    assert payload["status_code"] == 200
    assert payload["redirected_status_code"] == 451
    assert payload["redirected_url"] == "https://publisher.example/unavailable"
    assert payload["raw_markdown"] is None
    correlation = re.fullmatch(
        re.escape(public) + r" \(correlation_id=([0-9a-f]{12})\)", response.json()["detail"]
    )
    assert correlation is not None
    correlation_id = correlation.group(1)
    event, = _markdown_terminal_events(caplog)
    assert event.startswith("event=md_terminal status=502 outcome=crawl_failure duration_ms=")
    assert f"correlation_id={correlation_id}" in event
    assert raw_error not in event
    [failure] = [
        record
        for record in caplog.records
        if record.name == "utils" and f"[cid={correlation_id}]" in record.getMessage()
    ]
    assert failure.levelname == level


def test_markdown_pass_through_failure_does_not_log_correlation(
    stock_client, server_module, monkeypatch, caplog
):
    raw_error = "Blocked by anti-bot protection: challenge"
    _install_markdown_crawler(
        monkeypatch,
        CrawlResult(url="https://private.example/secret-path", html="", success=False, error_message=raw_error),
    )
    from auth import create_access_token

    caplog.set_level("INFO", logger=server_module.__name__)
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/md",
        json={"url": "https://private.example/secret-path", "f": "raw"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": raw_error, "html": "", "raw_markdown": None, "status_code": None,
        "redirected_url": None, "redirected_status_code": None,
    }
    event, = _markdown_terminal_events(caplog)
    assert event.startswith("event=md_terminal status=502 outcome=crawl_failure duration_ms=")
    assert "correlation_id=" not in event
    assert raw_error not in event
    assert not any(
        record.name == "utils" and "[cid=" in record.getMessage()
        for record in caplog.records
    )


def test_markdown_terminal_event_omits_malformed_correlation_headers(
    stock_client, server_module, monkeypatch, caplog
):
    malformed_traceparent = "00-not-a-traceparent"
    malformed_client_request_id = "Bearer private-credential"
    _install_markdown_crawler(
        monkeypatch,
        CrawlResult(
            url="https://private.example",
            html="<p>article</p>",
            success=True,
            markdown=MarkdownGenerationResult(
                raw_markdown="markdown", fit_markdown="markdown",
                markdown_with_citations="markdown", references_markdown="",
            ),
            redirected_url=None,
        ),
    )
    from auth import create_access_token

    caplog.set_level("INFO", logger=server_module.__name__)
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/md",
        json={"url": "https://private.example", "f": "raw"},
        headers={
            "Authorization": f"Bearer {token}",
            "traceparent": malformed_traceparent,
            "X-Client-Request-Id": malformed_client_request_id,
        },
    )

    assert response.status_code == 200
    event, = _markdown_terminal_events(caplog)
    assert "trace_id=" not in event
    assert "client_request_id=" not in event
    assert malformed_traceparent not in event
    assert malformed_client_request_id not in event


def test_inline_markdown_does_not_report_synthetic_http_success(
    stock_client, server_module, monkeypatch
):
    from auth import create_access_token

    requested = "raw:<p>Inline content</p>"
    _install_markdown_crawler(monkeypatch, CrawlResult(
        url=requested, html="<p>Inline content</p>", success=True, status_code=200,
        redirected_url="https://base.example/", redirected_status_code=200,
        markdown=MarkdownGenerationResult(
            raw_markdown="Inline content", markdown_with_citations="Inline content",
            references_markdown="",
        ),
    ))
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/md", json={"url": requested, "f": "raw"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"] == "Inline content"
    assert payload["status_code"] is None
    assert payload["redirected_status_code"] is None
    assert payload["redirected_url"] is None


@pytest.mark.parametrize("failure,expected_status", [
    ("permit", 503), ("empty", 502), ("timeout", 504), ("internal", 500),
])
def test_post_render_llm_failure_keeps_capture_and_safe_http_error(
    stock_client, server_module, monkeypatch, failure, expected_status
):
    import api
    import llm_broker
    from auth import create_access_token
    from test_llm_crawler_release_timing import _resolve_llm

    _install_markdown_crawler(monkeypatch, CrawlResult(
        url="https://example.com/", html="<p>Captured article</p>",
        cleaned_html="<p>Captured article</p>", success=True, status_code=302,
        redirected_url="https://example.com/article", redirected_status_code=200,
        markdown=MarkdownGenerationResult(
            raw_markdown="Captured article", markdown_with_citations="Captured article",
            references_markdown="",
        ),
    ))
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve_llm())

    @asynccontextmanager
    async def permit(*args):
        if failure == "permit":
            raise HTTPException(503, "LLM capacity busy", headers={"Retry-After": "30"})
        yield

    async def complete(**kwargs):
        if failure == "timeout":
            raise asyncio.TimeoutError()
        if failure == "internal":
            raise RuntimeError("Permission denied: /private/llm-cache")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

    monkeypatch.setattr(api, "llm_permit", permit)
    monkeypatch.setattr(api, "aperform_completion_with_backoff", complete)
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/md", json={"url": "https://example.com/", "f": "llm"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == expected_status
    payload = response.json()
    assert payload["html"] == "<p>Captured article</p>"
    assert payload["raw_markdown"] == "Captured article"
    assert payload["status_code"] == 302
    assert payload["redirected_status_code"] == 200
    assert payload["redirected_url"] == "https://example.com/article"
    assert "/private/llm-cache" not in response.text
    if failure == "internal":
        assert payload["error"] == "Internal server error"
        assert re.fullmatch(r"[0-9a-f]{12}", payload["correlation_id"])
    else:
        assert payload["detail"] == {
            "permit": "LLM capacity busy", "empty": "LLM markdown filter returned no content",
            "timeout": "Crawl exceeded the time limit",
        }[failure]
    if failure == "permit":
        assert response.headers["Retry-After"] == "30"
