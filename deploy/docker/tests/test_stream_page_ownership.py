"""A streaming response owns a page only while its body is being consumed."""

import gc
import json
from unittest.mock import AsyncMock

import api
import crawler_pool
import pytest
from auth import create_access_token
from fastapi import HTTPException
from starlette.responses import StreamingResponse


@pytest.mark.asyncio
async def test_uniterated_stream_never_acquires_a_replica_page(
    server_module, monkeypatch, offline_dns, tmp_path,
):
    budget = crawler_pool.ReplicaPageBudget(max_pages=2, directory=tmp_path / "pages")
    acquired = []

    class Crawler:
        async def arun_many(self, **_kwargs):
            async def results():
                if False:
                    yield None

            return results()

    async def acquire_crawler(_browser_config):
        lease = await budget.acquire(wait_seconds=0)
        acquired.append(lease)
        return Crawler()

    async def release_crawler(_crawler):
        for lease in acquired:
            lease.close()

    monkeypatch.setattr(api, "prefetch_archive", AsyncMock(return_value=None))
    monkeypatch.setattr(crawler_pool, "get_crawler", acquire_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)

    try:
        request = server_module.CrawlRequestWithHooks(urls=["https://example.com"])
        for _ in range(2):
            response = await server_module.stream_process(crawl_request=request)
            assert isinstance(response, StreamingResponse)
            assert acquired == [], "response construction must not reserve a page"
            del response
            gc.collect()

        # Exercise the actual flock budget: every slot remains available after
        # both responses are abandoned without iterating their bodies.
        leases = [await budget.acquire(wait_seconds=0) for _ in range(2)]
        try:
            with pytest.raises(crawler_pool.BrowserCapacityError):
                await budget.acquire(wait_seconds=0)
        finally:
            for lease in leases:
                lease.close()
    finally:
        for lease in acquired:
            lease.close()


@pytest.mark.asyncio
async def test_started_stream_releases_page_when_body_closes(
    server_module, monkeypatch, offline_dns, tmp_path,
):
    budget = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path / "pages")
    acquired = []

    class Result:
        url = "https://example.com"

        def model_dump(self):
            return {"url": self.url, "success": True, "markdown": "ok"}

    class Crawler:
        async def arun_many(self, **_kwargs):
            async def results():
                yield Result()
                yield Result()

            return results()

    async def acquire_crawler(_browser_config):
        lease = await budget.acquire(wait_seconds=0)
        acquired.append(lease)
        return Crawler()

    async def release_crawler(_crawler):
        for lease in acquired:
            lease.close()

    monkeypatch.setattr(api, "prefetch_archive", AsyncMock(return_value=None))
    monkeypatch.setattr(crawler_pool, "get_crawler", acquire_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)

    response = await server_module.stream_process(
        crawl_request=server_module.CrawlRequestWithHooks(urls=["https://example.com"])
    )
    try:
        first = json.loads(await anext(response.body_iterator))
        assert first["url"] == "https://example.com"
        assert len(acquired) == 1 and not acquired[0].closed
        with pytest.raises(crawler_pool.BrowserCapacityError):
            await budget.acquire(wait_seconds=0)
    finally:
        await response.body_iterator.aclose()

    try:
        assert acquired[0].closed
        replacement = await budget.acquire(wait_seconds=0)
        replacement.close()
    finally:
        for lease in acquired:
            lease.close()


@pytest.mark.asyncio
async def test_started_stream_reports_capacity_refusal_as_ndjson(
    server_module, monkeypatch, offline_dns, tmp_path,
):
    budget = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path / "pages")
    occupied = await budget.acquire(wait_seconds=0)

    async def refuse_crawler(_config):
        return await budget.acquire(wait_seconds=0)

    acquisition = AsyncMock(side_effect=refuse_crawler)
    monkeypatch.setattr(api, "prefetch_archive", AsyncMock(return_value=None))
    monkeypatch.setattr(crawler_pool, "get_crawler", acquisition)

    try:
        response = await server_module.stream_process(
            crawl_request=server_module.CrawlRequestWithHooks(urls=["https://example.com"])
        )
        assert isinstance(response, StreamingResponse)
        assert acquisition.await_count == 0
        try:
            failure = json.loads(await anext(response.body_iterator))
            assert failure["status"] == "failed"
            assert failure["status_code"] == 503
            assert "capacity" in failure["error"].lower()
        finally:
            await response.body_iterator.aclose()
    finally:
        occupied.close()


@pytest.mark.asyncio
async def test_lazy_stream_acquisition_obeys_request_deadline(
    server_module, monkeypatch, offline_dns, tmp_path,
):
    budget = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path / "pages")
    occupied = await budget.acquire(wait_seconds=0)

    async def acquire_crawler(_config):
        return await budget.acquire(wait_seconds=1)

    acquisition = AsyncMock(side_effect=acquire_crawler)
    monkeypatch.setattr(api, "prefetch_archive", AsyncMock(return_value=None))
    monkeypatch.setattr(crawler_pool, "get_crawler", acquisition)
    monkeypatch.setitem(server_module.config["limits"], "wall_clock_s", 0.15)

    try:
        response = await server_module.stream_process(
            crawl_request=server_module.CrawlRequestWithHooks(urls=["https://example.com"])
        )
        assert acquisition.await_count == 0
        try:
            failure = json.loads(await anext(response.body_iterator))
            assert failure == {
                "status": "failed",
                "error": "Crawl exceeded the time limit",
                "status_code": 504,
            }
            acquisition.assert_awaited_once()
        finally:
            await response.body_iterator.aclose()
    finally:
        occupied.close()

    # A cancelled waiter must not keep a flock slot after the holder leaves.
    replacement = await budget.acquire(wait_seconds=0)
    replacement.close()


@pytest.mark.asyncio
async def test_lazy_hook_attachment_failure_is_sanitized_and_releases_page(
    server_module, monkeypatch, offline_dns, tmp_path,
):
    budget = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path / "pages")
    acquired = []

    class FailingStrategy:
        def set_hook(self, _hook_point, _hook):
            raise RuntimeError("internal hook state at /app/private/browser-profile")

    class DedicatedCrawler:
        crawler_strategy = FailingStrategy()
        _docker_request_owned = True

    async def get_dedicated_crawler(_config):
        acquired.append(await budget.acquire(wait_seconds=0))
        return DedicatedCrawler()

    async def release_dedicated_crawler(_crawler):
        for lease in acquired:
            lease.close()

    monkeypatch.setattr(api, "prefetch_archive", AsyncMock(return_value=None))
    monkeypatch.setattr(crawler_pool, "get_dedicated_crawler", get_dedicated_crawler)
    monkeypatch.setattr(crawler_pool, "release_dedicated_crawler", release_dedicated_crawler)
    response = await server_module.stream_process(
        crawl_request=server_module.CrawlRequestWithHooks(
            urls=["https://example.com"],
            hooks={"hooks": [{"action": "scroll_to_bottom", "params": {"max_steps": 2}}]},
        )
    )
    assert acquired == []
    try:
        failure = json.loads(await anext(response.body_iterator))
        assert failure == {
            "status": "failed", "error": "Streaming crawl failed", "status_code": 500,
        }
    finally:
        await response.body_iterator.aclose()

    assert len(acquired) == 1 and acquired[0].closed
    replacement = await budget.acquire(wait_seconds=0)
    replacement.close()


@pytest.mark.asyncio
async def test_deep_crawl_request_rejects_before_stream_acquisition(
    server_module, monkeypatch, offline_dns,
):
    acquisition = AsyncMock()
    monkeypatch.setattr(crawler_pool, "get_crawler", acquisition)
    with pytest.raises(HTTPException) as error:
        await server_module.stream_process(
            crawl_request=server_module.CrawlRequestWithHooks(
                urls=["https://example.com"],
                crawler_config={
                    "deep_crawl_strategy": {
                        "type": "BFSDeepCrawlStrategy", "params": {"max_depth": 1},
                    },
                },
            )
        )
    assert error.value.status_code == 400
    acquisition.assert_not_awaited()


def test_projected_failed_crawl_preserves_the_safe_failure_detail(
    stock_client, monkeypatch, offline_dns,
):
    class FailedCrawler:
        async def arun(self, *_args, **_kwargs):
            return [{
                "url": "https://example.com",
                "success": False,
                "error_message": "Wait condition failed: selector not found",
            }]

    monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=FailedCrawler()))
    monkeypatch.setattr(crawler_pool, "release_crawler", AsyncMock())
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/crawl",
        json={
            "urls": ["https://example.com"],
            "result_fields": ["markdown"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Crawl request failed: Wait condition failed: selector not found"
    )
