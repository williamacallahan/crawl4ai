"""Behavioral tests for the per-Swarm-node archive.org budget.

The reserve/pause path is one Lua script against Redis TIME, so these tests
run a real Redis: REDIS_URL when set, otherwise a temporary redis-server on a
random port (skipped when redis-server is not installed — the same pattern as
test_crawl_job_queue_redis.py). Assertions are on observable results: returned
waits, raised exception contracts, and the /archive/fetch HTTP surface.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types

import pytest
import pytest_asyncio
import redis
from redis import asyncio as aioredis

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import archive_budget
from archive_budget import ArchiveBudgetSpent
from fastapi import HTTPException

# crawler_pool calls the glibc-only mallopt() at import, which does not exist
# on macOS. Stub it only when the real module cannot import (dev boxes); on
# Linux the real module loads and nothing here is faked.
try:
    import crawler_pool  # noqa: F401
except Exception:
    _pool_stub = types.ModuleType("crawler_pool")

    async def _pool_noop(*args, **kwargs):
        return None

    for _name in (
        "close_all",
        "get_crawler",
        "get_dedicated_crawler",
        "get_pool_snapshot",
        "janitor",
        "release_crawler",
        "release_dedicated_crawler",
        "retire_pool_crawlers",
    ):
        setattr(_pool_stub, _name, _pool_noop)
    _pool_stub.LAST_JANITOR_PASS = 0.0
    sys.modules["crawler_pool"] = _pool_stub


@pytest.fixture
def redis_url():
    configured_url = os.getenv("REDIS_URL")
    if configured_url:
        client = redis.from_url(configured_url)
        try:
            for _attempt in range(100):
                try:
                    if client.ping():
                        yield configured_url
                        return
                except redis.ConnectionError:
                    time.sleep(0.01)
            pytest.fail("configured Redis did not become ready")
        finally:
            client.close()
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is not installed")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    with tempfile.TemporaryDirectory(prefix="archive-budget-redis-") as data_dir:
        process = subprocess.Popen(
            [
                executable,
                "--bind", "127.0.0.1",
                "--port", str(port),
                "--save", "",
                "--appendonly", "no",
                "--dir", data_dir,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"redis://127.0.0.1:{port}/0"
        client = redis.from_url(url)
        try:
            for _attempt in range(100):
                try:
                    if client.ping():
                        break
                except redis.ConnectionError:
                    time.sleep(0.01)
            else:
                pytest.fail("temporary redis-server did not become ready")
            yield url
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest_asyncio.fixture
async def redis_client(redis_url):
    client = aioredis.from_url(redis_url)
    try:
        yield client
    finally:
        await client.aclose()


def budget_config(**overrides):
    """Fast budget timings; production values live in config.yml."""
    archive = {
        "hosts": ["archive.org", "web.archive.org"],
        "spacing_seconds": 0.25,
        "max_wait_seconds": 1.0,
        "throttle_pause_seconds": 6.0,
    }
    archive.update(overrides)
    return {"archive": archive, "limits": {"wall_clock_s": 0}}


def set_node(monkeypatch, name):
    monkeypatch.setenv("CRAWL4AI_NODE", name)
    return name


# ───────────────────────── reserve / spacing ─────────────────────────
@pytest.mark.asyncio
async def test_reserve_spaces_starts(redis_client, monkeypatch):
    set_node(monkeypatch, "spacing-node")
    cfg = budget_config()
    first = await archive_budget.reserve(redis_client, cfg)
    assert first < 0.1
    second = await archive_budget.reserve(redis_client, cfg)
    assert second >= 0.1, f"second start not spaced: {second}"
    third = await archive_budget.reserve(redis_client, cfg)
    assert third > second


@pytest.mark.asyncio
async def test_reserve_refuses_past_max_wait(redis_client, monkeypatch):
    node = set_node(monkeypatch, "refuse-node")
    cfg = budget_config(spacing_seconds=5.0, max_wait_seconds=0.5)
    await archive_budget.reserve(redis_client, cfg)
    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.reserve(redis_client, cfg)
    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.detail == "archive.org budget for this node is spent"
    assert exc.node == node
    assert isinstance(exc.retry_after, int)
    assert 4 <= exc.retry_after <= 5
    assert exc.headers["Retry-After"] == str(exc.retry_after)


@pytest.mark.asyncio
async def test_pause_pushes_next_start(redis_client, monkeypatch):
    set_node(monkeypatch, "pause-node")
    cfg = budget_config()
    await archive_budget.reserve(redis_client, cfg)
    pushed = await archive_budget.pause(redis_client, cfg)
    assert pushed > 4
    # A later reserve refuses — the pushed-out start is the refusal's
    # retry_after, far beyond max_wait.
    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.reserve(redis_client, cfg)
    assert exc_info.value.retry_after >= 4
    # A second pause never moves the stored start earlier.
    assert await archive_budget.pause(redis_client, cfg) > 4


@pytest.mark.asyncio
async def test_same_node_replicas_share_one_budget(redis_url, monkeypatch):
    """Two replicas on one node (one public IP) reserve through one key."""
    set_node(monkeypatch, "shared-node")
    replica_a = aioredis.from_url(redis_url)
    replica_b = aioredis.from_url(redis_url)
    try:
        cfg = budget_config()
        assert await archive_budget.reserve(replica_a, cfg) < 0.1
        # Replica B is not a fresh budget: it sees A's reservation.
        assert await archive_budget.reserve(replica_b, cfg) >= 0.1
    finally:
        await replica_a.aclose()
        await replica_b.aclose()


@pytest.mark.asyncio
async def test_different_nodes_are_independent(redis_client, monkeypatch):
    cfg = budget_config()
    set_node(monkeypatch, "node-alpha")
    assert await archive_budget.reserve(redis_client, cfg) < 0.1
    set_node(monkeypatch, "node-beta")
    assert await archive_budget.reserve(redis_client, cfg) < 0.1


@pytest.mark.asyncio
async def test_missing_node_fails_loud_503(redis_client, monkeypatch):
    monkeypatch.delenv("CRAWL4AI_NODE", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.reserve(redis_client, budget_config())
    exc = exc_info.value
    assert exc.status_code == 503
    assert exc.detail == "archive node identity is not configured"


# ───────────────────────── gating / hosts ─────────────────────────
def test_is_archive_url_hosts_and_subdomains():
    cfg = budget_config()
    assert archive_budget.is_archive_url("https://archive.org/details/x", cfg)
    assert archive_budget.is_archive_url("https://web.archive.org/web/2020id_/https://a.b/", cfg)
    assert archive_budget.is_archive_url("https://wayback.archive.org/x", cfg)
    assert archive_budget.is_archive_url("https://WEB.ARCHIVE.ORG/web/x", cfg)
    assert not archive_budget.is_archive_url("https://example.com/", cfg)
    assert not archive_budget.is_archive_url("https://archive.org.evil.com/", cfg)
    assert not archive_budget.is_archive_url("https://notarchive.org/", cfg)
    assert not archive_budget.is_archive_url("https://archive.org@evil.com/", cfg)
    assert not archive_budget.is_archive_url("not a url", cfg)


@pytest.mark.asyncio
async def test_gate_skips_non_archive_urls(redis_client, monkeypatch):
    # No CRAWL4AI_NODE and redis=None: any budget touch would raise, so
    # returning silently proves the URL was never gated.
    monkeypatch.delenv("CRAWL4AI_NODE", raising=False)
    cfg = budget_config()
    await archive_budget.gate_url(None, "https://example.com/", cfg)
    await archive_budget.gate_urls(
        redis_client, ["https://example.com/", "https://a.b/c"], cfg,
    )


@pytest.mark.asyncio
async def test_crawl_request_gates_archive_seed(
    redis_client, monkeypatch, offline_dns,
):
    """/crawl refuses fast with the typed 429 once the node budget is spent."""
    import api

    set_node(monkeypatch, "crawl-node")
    cfg = budget_config(spacing_seconds=5.0, max_wait_seconds=0.5)
    await archive_budget.reserve(redis_client, cfg)  # consume the free slot
    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await api.handle_crawl_request(
            urls=["https://web.archive.org/web/2020id_/https://example.com/"],
            browser_config={},
            crawler_config={},
            config=cfg,
            redis=redis_client,
        )
    assert exc_info.value.node == "crawl-node"


@pytest.mark.asyncio
async def test_job_worker_passes_queue_redis(monkeypatch):
    """The /crawl/job worker feeds its own Redis client into the gate."""
    import api
    import crawl_job_worker

    captured = {}

    async def fake_handle(**kwargs):
        captured.update(kwargs)
        return {"success": True, "results": []}

    monkeypatch.setattr(api, "handle_crawl_request", fake_handle)
    sentinel = object()
    queue = types.SimpleNamespace(redis=sentinel)
    worker = crawl_job_worker.CrawlJobWorker(queue, {}, "consumer-test")
    await worker._crawl(
        {
            "urls": ["https://web.archive.org/web/2020/x"],
            "browser_config": {},
            "crawler_config": {},
            "result_fields": None,
        }
    )
    assert captured["redis"] is sentinel


# ───────────────────────── pause on results ─────────────────────────
@pytest.mark.asyncio
async def test_result_pause_on_throttle_status(redis_client, monkeypatch):
    set_node(monkeypatch, "result-429")
    cfg = budget_config()
    assert await archive_budget.maybe_pause_for_result(
        redis_client, cfg, "https://web.archive.org/web/x", 429, None,
    )
    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.reserve(redis_client, cfg)
    assert exc_info.value.retry_after >= 4


@pytest.mark.asyncio
async def test_result_pause_on_connection_refused(redis_client, monkeypatch):
    set_node(monkeypatch, "result-refused")
    cfg = budget_config()
    assert await archive_budget.maybe_pause_for_result(
        redis_client, cfg, "https://archive.org/x", None,
        "Playwright error: net::ERR_CONNECTION_REFUSED at https://archive.org/x",
    )
    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.reserve(redis_client, cfg)
    assert exc_info.value.retry_after >= 4


@pytest.mark.asyncio
async def test_result_no_pause_for_healthy_or_foreign(
    redis_client, monkeypatch,
):
    set_node(monkeypatch, "result-skip")
    cfg = budget_config()
    # Foreign host throttles do not spend the archive budget.
    assert not await archive_budget.maybe_pause_for_result(
        redis_client, cfg, "https://example.com/", 429, None,
    )
    # A clean archive result does not pause.
    assert not await archive_budget.maybe_pause_for_result(
        redis_client, cfg, "https://archive.org/x", 200, None,
    )
    assert await archive_budget.reserve(redis_client, cfg) < 0.1


# ───────────────────────── fetch (route logic) ─────────────────────────
ARCHIVE_URL = "https://web.archive.org/web/2020id_/https://example.com/"


@pytest.mark.asyncio
async def test_fetch_rejects_non_archive_hosts(redis_client, monkeypatch):
    set_node(monkeypatch, "fetch-host")
    cfg = budget_config()
    for url in (
        "https://example.com/",
        "https://archive.org.evil.com/",
        "https://archive.org@evil.com/x",
    ):
        with pytest.raises(HTTPException) as exc_info:
            await archive_budget.fetch(redis_client, cfg, url)
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_fetch_returns_body(redis_client, monkeypatch):
    set_node(monkeypatch, "fetch-ok")
    cfg = budget_config()

    async def opener(url):
        return 200, "text/html; charset=utf-8", "<html>snapshot</html>"

    out = await archive_budget.fetch(
        redis_client, cfg, ARCHIVE_URL, opener=opener,
    )
    assert out == {
        "url": ARCHIVE_URL,
        "status": 200,
        "contentType": "text/html; charset=utf-8",
        "body": "<html>snapshot</html>",
    }


@pytest.mark.asyncio
async def test_fetch_throttled_pauses_and_refuses(redis_client, monkeypatch):
    node = set_node(monkeypatch, "fetch-429")
    cfg = budget_config()

    async def opener(url):
        return 429, "text/plain", ""

    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.fetch(redis_client, cfg, "https://archive.org/x", opener=opener)
    assert exc_info.value.node == node
    assert exc_info.value.retry_after >= 4
    # The pause is real: the node's next start is the throttle window out.
    with pytest.raises(ArchiveBudgetSpent) as retry_info:
        await archive_budget.reserve(redis_client, cfg)
    assert retry_info.value.retry_after >= 4


@pytest.mark.asyncio
async def test_fetch_refused_pauses_and_refuses(redis_client, monkeypatch):
    set_node(monkeypatch, "fetch-refused")
    cfg = budget_config()

    async def opener(url):
        raise archive_budget._ArchiveUpstreamRefused("Connection refused")

    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.fetch(redis_client, cfg, "https://archive.org/x", opener=opener)
    assert exc_info.value.retry_after >= 4


# ───────────────────────── /archive/fetch route ─────────────────────────
def _bearer():
    from auth import create_access_token

    token = create_access_token({"sub": "tester@x.com"}, scope="data")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def app_client(redis_url, monkeypatch):
    """The real FastAPI app with the test Redis behind the archive budget."""
    import server
    from starlette.testclient import TestClient

    client = aioredis.from_url(redis_url)
    monkeypatch.setattr(server, "redis", client)
    return TestClient(server.app, raise_server_exceptions=False)


def test_fetch_route_rejects_foreign_host(app_client, monkeypatch):
    monkeypatch.setenv("CRAWL4AI_NODE", "route-host")
    r = app_client.post(
        "/archive/fetch",
        json={"url": "https://example.com/x"},
        headers=_bearer(),
    )
    assert r.status_code == 400


def test_fetch_route_requires_authentication(app_client, monkeypatch):
    monkeypatch.setenv("CRAWL4AI_NODE", "route-auth")
    r = app_client.post("/archive/fetch", json={"url": "https://archive.org/"})
    assert r.status_code == 401


def test_fetch_route_returns_body(app_client, monkeypatch):
    monkeypatch.setenv("CRAWL4AI_NODE", "route-ok")

    async def ok(url):
        return 200, "text/plain", "snapshot-body"

    monkeypatch.setattr(archive_budget, "_default_opener", ok)
    r = app_client.post(
        "/archive/fetch", json={"url": ARCHIVE_URL}, headers=_bearer(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["url"] == ARCHIVE_URL
    assert body["status"] == 200
    assert body["contentType"] == "text/plain"
    assert body["body"] == "snapshot-body"


def test_fetch_route_spent_contract(app_client, monkeypatch):
    """The typed 429 the back-end reads for its overflow lane."""
    node = "route-429"
    monkeypatch.setenv("CRAWL4AI_NODE", node)

    async def throttled(url):
        return 503, "text/plain", ""

    monkeypatch.setattr(archive_budget, "_default_opener", throttled)
    r = app_client.post(
        "/archive/fetch", json={"url": "https://archive.org/"}, headers=_bearer(),
    )
    assert r.status_code == 429
    body = r.json()
    assert body == {
        "detail": "archive.org budget for this node is spent",
        "code": "archive_budget_spent",
        "node": node,
        "retryAfterSeconds": body["retryAfterSeconds"],
    }
    assert isinstance(body["retryAfterSeconds"], int)
    assert body["retryAfterSeconds"] >= 1
    assert r.headers["Retry-After"] == str(body["retryAfterSeconds"])
