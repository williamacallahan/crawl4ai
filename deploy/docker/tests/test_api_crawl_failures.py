from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import _raise_for_crawl_failure


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
    assert response.json() == {"detail": expected_detail}
