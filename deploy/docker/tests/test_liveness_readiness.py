"""Real HTTP/Redis boundary checks; only Chromium launch is omitted in this lane."""

import asyncio
import os
import socket
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest
from redis import asyncio as aioredis
from starlette.testclient import TestClient


@pytest.fixture
def lifespan_server(server_module, monkeypatch, tmp_path):
    import artifacts
    import crawler_pool
    import monitor

    browser_starts = []

    async def init_permanent(config):
        # Chromium is outside this non-browser lane; Redis and lifespan are real.
        browser_starts.append(config)

    monkeypatch.setattr(crawler_pool, "init_permanent", init_permanent)
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(monitor, "monitor_stats", None)
    monkeypatch.setenv("GUNICORN_BIND", "127.0.0.1:11235")
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", "liveness-test-only-token")
    monkeypatch.setenv("C4AI_GIT_SHA", "liveness-test-revision")
    monkeypatch.setenv("HOSTNAME", "liveness-test-instance")
    return server_module, browser_starts


@asynccontextmanager
async def redis_forwarder():
    """Reserve a closed port, then let the test restore real Redis on that port."""
    target = urlsplit(os.environ["REDIS_URL"])
    connections = set()

    async def forward(reader, writer):
        connections.add(asyncio.current_task())
        upstream = None
        try:
            upstream_reader, upstream = await asyncio.open_connection(
                target.hostname, target.port or 6379
            )

            async def pipe(source, destination):
                while data := await source.read(65536):
                    destination.write(data)
                    await destination.drain()

            tasks = [
                asyncio.create_task(pipe(reader, upstream)),
                asyncio.create_task(pipe(upstream_reader, writer)),
            ]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            writer.close()
            if upstream is not None:
                upstream.close()
            connections.discard(asyncio.current_task())

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))  # Bound but not listening: real ECONNREFUSED.
    port = listener.getsockname()[1]
    url = urlunsplit(target._replace(netloc=f"127.0.0.1:{port}"))
    server = None

    async def restore():
        nonlocal server
        listener.listen()
        listener.setblocking(False)
        server = await asyncio.start_server(forward, sock=listener)

    try:
        yield url, restore
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()
        listener.close()
        pending = list(connections)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def wait_until_ready(client):
    deadline = time.monotonic() + 35
    while True:
        response = await client.get("/health")
        if response.status_code == 200:
            return response
        assert time.monotonic() < deadline, response.text
        await asyncio.sleep(0.1)


def test_liveness_route_is_public_and_reports_identity(lifespan_server):
    server, _ = lifespan_server
    # Like the auth suites, this client skips startup to isolate routing/auth.
    response = TestClient(server.app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "revision": "liveness-test-revision",
        "instance": "liveness-test-instance",
    }


@pytest.mark.asyncio
async def test_lifespan_survives_redis_outage_and_recovers(lifespan_server, monkeypatch):
    server, browser_starts = lifespan_server
    async with redis_forwarder() as (url, restore):
        redis = aioredis.from_url(url, **server._REDIS_RESILIENT_KWARGS)
        monkeypatch.setattr(server, "redis", redis)
        try:
            async with server.app.router.lifespan_context(server.app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server.app), base_url="http://test"
                ) as client:
                    live = await client.get("/health/live")
                    assert live.status_code == 200
                    assert live.json()["status"] == "alive"
                    health = await client.get("/health")
                    assert health.status_code == 503
                    assert health.json()["components"] == {"api": "unavailable"}
                    assert len(browser_starts) == 1
                    assert server.app.state.job_queue.started
                    assert not server.app.state.janitor.done()
                    assert server.app.state.egress_proxy is not None
                    # Let an actual connection failure reach the retry owner.
                    await asyncio.sleep(1.1)
                    assert (await client.get("/health/live")).status_code == 200
                    assert (await client.get("/health")).status_code == 503
                    await restore()
                    health = await wait_until_ready(client)
                    assert health.json()["components"] == {"api": "ready", "redis": "ready"}
                    assert len(browser_starts) == 1  # Recovery never restarts Chromium.
            assert not server.app.state.readiness_checks_active
            assert server.app.state.readiness_initializer.done()
        finally:
            await redis.aclose()


@pytest.mark.asyncio
async def test_readiness_waits_for_monitor_load_and_cancels_retry(
    lifespan_server, monkeypatch
):
    server, _ = lifespan_server
    redis = aioredis.from_url(os.environ["REDIS_URL"], **server._REDIS_RESILIENT_KWARGS)
    monkeypatch.setattr(server, "redis", redis)
    # PING succeeds, but GET fails with WRONGTYPE until the persisted data is fixed.
    key = "monitor:endpoint_stats"
    await redis.delete(key)
    await redis.lpush(key, "wrong-type")
    try:
        async with server.app.router.lifespan_context(server.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test"
            ) as client:
                await asyncio.sleep(0.1)
                assert (await client.get("/health/live")).status_code == 200
                assert (await client.get("/health")).status_code == 503
                import monitor
                stats = monitor.get_monitor()
                await stats.track_request_start("during-outage", "/md", "https://example.com")
                await stats.track_request_end("during-outage", success=True)
                assert await redis.type(key) == b"list"  # No premature SET.
                await redis.set(key, '{"/md": {"count": 7}}')
                assert (await wait_until_ready(client)).status_code == 200
                assert stats.endpoint_stats["/md"]["count"] == 8

        # Shutdown while the retry owner is still waiting must cancel it.
        async with redis_forwarder() as (closed_url, _):
            unavailable = aioredis.from_url(closed_url, **server._REDIS_RESILIENT_KWARGS)
            monkeypatch.setattr(server, "redis", unavailable)
            try:
                async with server.app.router.lifespan_context(server.app):
                    await asyncio.sleep(0.1)
                    retry = server.app.state.readiness_initializer
                    assert not retry.done()
                assert retry.cancelled()
            finally:
                await unavailable.aclose()
    finally:
        await redis.delete(key)
        await redis.aclose()
