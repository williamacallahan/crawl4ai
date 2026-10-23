"""A page that never settles must not hold a browser past limits.wall_clock_s."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from auth import create_access_token


@pytest.mark.parametrize("path,body", [
    ("/screenshot", {"url": "https://example.com"}),
    ("/pdf", {"url": "https://example.com"}),
    ("/execute_js", {"url": "https://example.com", "scripts": ["1"]}),
])
def test_single_page_endpoint_returns_504_and_releases_its_browser(
    stock_client, server_module, monkeypatch, path, body,
):
    class HangingCrawler:
        async def arun(self, *args, **kwargs):
            await asyncio.sleep(3600)

    release = AsyncMock()
    monkeypatch.setattr(server_module, "get_crawler", AsyncMock(return_value=HangingCrawler()))
    monkeypatch.setattr(server_module, "release_crawler", release)
    monkeypatch.setattr(server_module, "validate_url_scheme", AsyncMock())
    monkeypatch.setattr(server_module, "validate_webhook_url", lambda _url: None)
    monkeypatch.setattr(server_module, "EXECUTE_JS_ENABLED", True)
    monkeypatch.setitem(server_module.config["limits"], "wall_clock_s", 0.2)

    response = stock_client.post(
        path, json=body,
        headers={"Authorization": "Bearer " + create_access_token({"sub": "deadline-test"})},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == "Crawl exceeded the time limit"
    release.assert_awaited_once()
