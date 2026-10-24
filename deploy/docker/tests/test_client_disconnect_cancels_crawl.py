"""A client that hangs up on a non-stream browser route frees what it held."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import uvicorn
from redis import asyncio as aioredis

URL = "https://example.com"
ROUTES = [
    ("/crawl", {"urls": [URL]}),
    ("/md", {"url": URL, "f": "raw"}),
    ("/html", {"url": URL}),
    ("/screenshot", {"url": URL}),
    ("/pdf", {"url": URL}),
    ("/execute_js", {"url": URL, "scripts": ["1"]}),
]


@pytest.fixture
def routes(server_module, monkeypatch, tmp_path):
    """The real app with every destination check open and a one-page budget."""
    import api
    import crawler_pool

    budget = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path)
    monkeypatch.setattr(crawler_pool, "PAGE_BUDGET", budget)
    for module in (api, server_module):
        monkeypatch.setattr(module, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(server_module, "validate_webhook_url", lambda _url: None)
    monkeypatch.setattr(server_module, "EXECUTE_JS_ENABLED", True)
    return budget


def admit(monkeypatch, arun):
    import crawler_pool

    async def admitted(_cfg):
        crawler = SimpleNamespace(arun=arun)
        crawler_pool._set_active_requests(crawler, 1)
        return crawler

    monkeypatch.setattr(crawler_pool, "_get_admitted_crawler", admitted)


async def hang_up_after(server_module, path, payload, started, while_open):
    """POST over a real socket, wait for ``started``, run ``while_open``, close."""
    from auth import create_access_token

    server = uvicorn.Server(uvicorn.Config(
        server_module.app, host="127.0.0.1", port=0, lifespan="off",
        log_level="warning", timeout_graceful_shutdown=1,
    ))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        body = json.dumps(payload).encode()
        writer.write(
            f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Authorization: Bearer {create_access_token({'sub': 'test@example.com'})}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        await asyncio.wait_for(started.wait(), timeout=10)
        await while_open()
        writer.close()
        await writer.wait_closed()
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=10)


async def eventually(check, seconds):
    for _attempt in range(int(seconds * 100)):
        if await check():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.parametrize("path,payload", ROUTES)
def test_hangup_cancels_the_crawl_and_frees_its_page_slot(
    server_module, routes, monkeypatch, path, payload
):
    import crawler_pool

    async def exercise():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def arun(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def slot_is_held():
            with pytest.raises(crawler_pool.BrowserCapacityError):
                await routes.acquire(0)

        async def slot_is_free():
            try:
                (await routes.acquire(0)).close()
                return True
            except crawler_pool.BrowserCapacityError:
                return False

        admit(monkeypatch, arun)
        async for _ in hang_up_after(server_module, path, payload, started, slot_is_held):
            await asyncio.wait_for(cancelled.wait(), timeout=5)
            assert await eventually(slot_is_free, 1), "the hung-up crawl still holds its page slot"

    asyncio.run(exercise())


def test_hangup_during_llm_markdown_releases_the_cluster_llm_permit(
    server_module, routes, monkeypatch
):
    import api
    import llm_broker
    from redis_config import RESILIENT_CLIENT_KWARGS

    async def exercise():
        # Production's client health-checks a connection idle past the interval,
        # so the release after a long LLM call first awaits a PING round trip.
        redis = aioredis.from_url(os.environ["REDIS_URL"],
                                  **{**RESILIENT_CLIENT_KWARGS, "health_check_interval": 0.1})
        await redis.delete(api.LLM_PERMIT_KEY)
        monkeypatch.setattr(server_module, "redis", redis)
        started = asyncio.Event()

        async def arun(*_args, **_kwargs):
            return SimpleNamespace(success=True, cleaned_html="<p>page text</p>")

        async def hanging_completion(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        async def permit_is_held():
            assert await redis.exists(api.LLM_PERMIT_KEY) == 1
            await asyncio.sleep(0.2)  # the LLM call outlasts the health-check interval

        async def permit_is_released():
            return await redis.exists(api.LLM_PERMIT_KEY) == 0

        admit(monkeypatch, arun)
        monkeypatch.setattr(api, "validate_llm_provider", lambda _config, _provider: (True, ""))
        monkeypatch.setattr(llm_broker, "resolve_llm", lambda *_args, **_kwargs: {
            "provider": "openai/test", "temperature": None, "extra_args": {},
            "api_token": "test", "base_url": None})
        monkeypatch.setattr(api, "aperform_completion_with_backoff", hanging_completion)
        try:
            payload = {"url": URL, "f": "llm", "q": "summarize"}
            async for _ in hang_up_after(server_module, "/md", payload, started, permit_is_held):
                assert await eventually(permit_is_released, 1), "the hung-up request kept the LLM permit"
        finally:
            await redis.delete(api.LLM_PERMIT_KEY)
            await redis.aclose()

    asyncio.run(exercise())
