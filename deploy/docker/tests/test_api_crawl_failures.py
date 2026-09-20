import re
from types import SimpleNamespace

import pytest
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
    failed_result = SimpleNamespace(success=False, error_message=error_message)

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
        SimpleNamespace(
            success=True,
            markdown=SimpleNamespace(raw_markdown=markdown, fit_markdown=markdown),
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


def test_markdown_failed_result_joins_redacted_response_and_terminal_event(
    stock_client, server_module, monkeypatch, caplog
):
    raw_error = "Permission denied: /app/private/private-error"
    _install_markdown_crawler(
        monkeypatch,
        SimpleNamespace(success=False, error_message=raw_error),
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
    correlation = re.fullmatch(
        r"Crawl failed \(correlation_id=([0-9a-f]{12})\)", response.json()["detail"]
    )
    assert correlation is not None
    correlation_id = correlation.group(1)
    event, = _markdown_terminal_events(caplog)
    assert event.startswith("event=md_terminal status=502 outcome=crawl_failure duration_ms=")
    assert f"correlation_id={correlation_id}" in event
    assert raw_error not in event
    assert any(
        f"[cid={correlation_id}]" in record.getMessage()
        for record in caplog.records
        if record.name == "utils"
    )


def test_markdown_pass_through_failure_does_not_log_correlation(
    stock_client, server_module, monkeypatch, caplog
):
    raw_error = "Blocked by anti-bot protection: challenge"
    _install_markdown_crawler(
        monkeypatch,
        SimpleNamespace(success=False, error_message=raw_error),
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
    assert response.json() == {"detail": raw_error}
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
        SimpleNamespace(
            success=True,
            markdown=SimpleNamespace(raw_markdown="markdown", fit_markdown="markdown"),
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
