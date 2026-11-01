"""Binary mode of /archive/fetch: exact bytes plus the post-redirect URL."""
import base64

import pytest
from fastapi import HTTPException

import archive_budget
from test_archive_budget import ARCHIVE_URL, budget_config, redis_client, redis_url  # noqa: F401


@pytest.mark.asyncio
async def test_fetch_binary_returns_exact_bytes_and_final_url(redis_client, monkeypatch):
    """Binary mode returns base64 bytes with the post-redirect URL; no 415."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-binary")
    gif = b"GIF89a\x00\xff\x10"
    landed = "https://web.archive.org/web/20000917233225id_/http://handspring.com/x.gif"
    calls = []

    async def opener(url, max_body):
        calls.append(url)
        if len(calls) == 1:
            return 302, {"Location": landed}, b""
        return 200, {"Content-Type": "image/gif"}, gif

    out = await archive_budget.fetch(
        redis_client, budget_config(), ARCHIVE_URL, opener=opener, binary=True,
    )
    assert calls == [ARCHIVE_URL, landed]
    assert out == {
        "url": landed,
        "status": 200,
        "contentType": "image/gif",
        "bodyBase64": base64.b64encode(gif).decode("ascii"),
    }

    async def direct(url, max_body):
        return 200, {"Content-Type": "image/gif"}, gif

    out = await archive_budget.fetch(
        redis_client, budget_config(), ARCHIVE_URL, opener=direct, binary=True,
    )
    assert out["url"] == ARCHIVE_URL


@pytest.mark.asyncio
async def test_fetch_binary_keeps_body_cap(redis_client, monkeypatch):
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-binary-cap")
    cfg = budget_config()
    cfg["limits"]["max_body_bytes"] = 4

    async def big(url, max_body):
        return 200, {"Content-Type": "image/gif"}, b"123456"

    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=big, binary=True)
    assert exc_info.value.status_code == 502
