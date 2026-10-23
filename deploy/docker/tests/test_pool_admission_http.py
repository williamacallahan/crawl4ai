"""Pool-owned capacity refusals survive the real HTTP exception boundary."""

import asyncio
from types import SimpleNamespace

import pytest

import crawler_pool


@pytest.fixture
def refusing_pool(monkeypatch, tmp_path, server_module):
    # Use a busy browser under another signature so the requested config must
    # pass through real browser-instance admission, without launching Chromium.
    occupied = SimpleNamespace(active_requests=1)
    monkeypatch.setattr(crawler_pool, "PERMANENT", None)
    monkeypatch.setattr(crawler_pool, "DEFAULT_CONFIG_SIG", None)
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {"occupied": occupied})
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {})
    monkeypatch.setattr(crawler_pool, "_CLOSE_TASKS", set())
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(1))
    monkeypatch.setattr(crawler_pool, "MAX_BROWSER_INSTANCES", 1)
    monkeypatch.setattr(crawler_pool, "MEM_LIMIT", 90)
    # Keep the browser-capacity regression runnable on the previous pool too:
    # it must fail because clients receive 500, not because a new name is absent.
    if hasattr(crawler_pool, "ReplicaPageBudget"):
        monkeypatch.setattr(
            crawler_pool, "PAGE_BUDGET",
            crawler_pool.ReplicaPageBudget(1, directory=tmp_path / "pages"),
        )
    return occupied


@pytest.mark.parametrize("endpoint", ["/screenshot", "/md"])
@pytest.mark.parametrize("pressure", ["browser", "memory"])
def test_pool_capacity_reaches_http_as_retryable_service_unavailable(
    endpoint, pressure, refusing_pool, stock_client, monkeypatch, offline_dns,
):
    from auth import create_access_token

    monkeypatch.setattr(
        crawler_pool, "get_container_memory_percent",
        lambda: 100 if pressure == "memory" else 20,
    )
    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        endpoint,
        json={"url": "https://example.com", **({"f": "raw"} if endpoint == "/md" else {})},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 503, response.text
    assert int(response.headers["Retry-After"]) > 0
    detail = response.json()["detail"]
    assert "Memory" in detail if pressure == "memory" else "capacity" in detail
    assert refusing_pool.active_requests == 1
    assert crawler_pool.ADMISSION_SEM._value == 1

    # A failed acquisition must close its replica lease as well as return its
    # local semaphore permit. Prove that through the public budget operation.
    if hasattr(crawler_pool, "PAGE_BUDGET"):
        lease = asyncio.run(crawler_pool.PAGE_BUDGET.acquire(wait_seconds=0))
        lease.close()
