from unittest.mock import AsyncMock

import api
import crawler_pool
import pytest
from auth import create_access_token


@pytest.mark.parametrize("error_message,expected_detail", [
    ("Wait condition failed: selector not found", "Crawl request failed: Wait condition failed: selector not found"),
    ("Permission denied: /app/private/cache.db", "Crawl request failed: Crawl failed"),
])
def test_all_failed_crawl_returns_sanitized_bad_gateway(
    stock_client, monkeypatch, error_message, expected_detail
):
    class FailedCrawler:
        async def arun(self, *args, **kwargs):
            return [{
                "url": "https://example.com",
                "success": False,
                "error_message": error_message,
            }]

    monkeypatch.setattr(api, "validate_url_destination", lambda url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=FailedCrawler()))
    monkeypatch.setattr(crawler_pool, "release_crawler", AsyncMock())

    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/crawl",
        json={"urls": ["https://example.com"]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    if "/app/" in error_message:
        assert detail.startswith(expected_detail + " (correlation_id=")
        assert "/app/private/cache.db" not in response.text
    else:
        assert detail == expected_detail
