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


def _library_failure(error: BaseException) -> str:
    """The error_message arun records for an exception escaping a crawl."""
    import sys

    from crawl4ai.utils import unexpected_error_message

    try:
        raise error
    except type(error):
        return unexpected_error_message("_crawl_web", sys.exc_info())


@pytest.mark.parametrize("error,public,level", [
    (TimeoutError("Page timeout after 60000ms at async_crawler_strategy.py:1270 "
                  "(html = await self._settled_content(page))"),
     "Crawl failed: Page timeout after 60000ms", "WARNING"),
    (RuntimeError("Target page crashed: renderer exceeded its rss limit "
                  "(pid=11176 rss_bytes=1053999104 rss_limit_bytes=805306368)"),
     "Crawl failed: Target page crashed: renderer exceeded its rss limit", "WARNING"),
    (RuntimeError("Target page crashed"), "Crawl failed", "ERROR"),
])
def test_withheld_failures_report_one_line_with_url_and_target_level(error, public, level, caplog):
    from utils import public_crawl_error

    raw = _library_failure(error)
    assert "\n" not in raw
    with caplog.at_level("WARNING", logger="utils"):
        detail = public_crawl_error(raw, "https://target.example/page")
    assert detail.startswith(public + " (correlation_id=")
    [record] = [r for r in caplog.records if r.name == "utils"]
    assert record.levelname == level
    assert "url=https://target.example/page" in record.getMessage()
    assert "\n" not in record.getMessage()
