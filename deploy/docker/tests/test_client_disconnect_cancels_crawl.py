"""A client that hangs up on a non-stream browser route frees its page slot."""

import asyncio
import json

import pytest
import uvicorn

URL = "https://example.com"
ROUTES = [
    ("/crawl", {"urls": [URL]}),
    ("/md", {"url": URL, "f": "raw"}),
    ("/html", {"url": URL}),
    ("/screenshot", {"url": URL}),
    ("/pdf", {"url": URL}),
    ("/execute_js", {"url": URL, "scripts": ["1"]}),
]


@pytest.mark.parametrize("path,payload", ROUTES)
def test_hangup_cancels_the_crawl_and_frees_its_page_slot(
    server_module, monkeypatch, tmp_path, path, payload
):
    import api
    import crawler_pool
    from auth import create_access_token

    budget = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path)
    monkeypatch.setattr(crawler_pool, "PAGE_BUDGET", budget)
    for module in (api, server_module):
        monkeypatch.setattr(module, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(server_module, "validate_webhook_url", lambda _url: None)
    monkeypatch.setattr(server_module, "EXECUTE_JS_ENABLED", True)

    async def exercise():
        started, cancelled = asyncio.Event(), asyncio.Event()

        class HangingCrawler:
            async def arun(self, *_args, **_kwargs):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        async def admitted(_cfg):
            crawler = HangingCrawler()
            crawler_pool._set_active_requests(crawler, 1)
            return crawler

        monkeypatch.setattr(crawler_pool, "_get_admitted_crawler", admitted)
        server = uvicorn.Server(uvicorn.Config(
            server_module.app, host="127.0.0.1", port=0, lifespan="off",
            log_level="warning", timeout_graceful_shutdown=1,
        ))
        serving = asyncio.create_task(server.serve())
        try:
            while not server.started:
                await asyncio.sleep(0.01)
            port = server.servers[0].sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            body = json.dumps(payload).encode()
            writer.write(
                f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                f"Authorization: Bearer {create_access_token({'sub': 'test@example.com'})}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            await asyncio.wait_for(started.wait(), timeout=10)
            with pytest.raises(crawler_pool.BrowserCapacityError):
                await budget.acquire(0)  # the crawl holds the only slot

            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(cancelled.wait(), timeout=5)
            for _attempt in range(100):
                try:
                    (await budget.acquire(0)).close()
                    break
                except crawler_pool.BrowserCapacityError:
                    await asyncio.sleep(0.01)
            else:
                pytest.fail("the hung-up crawl still holds its page slot")
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=10)

    asyncio.run(exercise())
