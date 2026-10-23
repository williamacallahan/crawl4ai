"""Exercise real pool admission and file leases without launching Chromium."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException

from crawl4ai import BrowserConfig
import crawler_pool as pool
import utils


class _Crawler:
    def __init__(self, *, config, thread_safe=False, crawler_strategy=None):
        self.active_requests = 0
        self.closed = False
        self.crawler_strategy = SimpleNamespace(browser_manager=SimpleNamespace(
            browser=SimpleNamespace(is_connected=lambda: not self.closed),
        ))

    async def start(self):
        pass

    async def close(self):
        self.closed = True


@pytest_asyncio.fixture
async def admission(monkeypatch, tmp_path):
    created = []

    def factory(**kwargs):
        crawler = _Crawler(**kwargs)
        created.append(crawler)
        return crawler

    for name, value in {
        "PERMANENT": None, "DEFAULT_CONFIG_SIG": None,
        "HOT_POOL": {}, "COLD_POOL": {}, "LAST_USED": {}, "USAGE_COUNT": {},
        "LOCK": asyncio.Lock(), "_CLOSE_TASKS": set(),
        "ADMISSION_SEM": asyncio.Semaphore(4),
        "ADMISSION": utils.PoolAdmissionSettings(replica_max_pages=1, admission_wait_seconds=0),
        "PAGE_BUDGET": pool.ReplicaPageBudget(1, tmp_path),
        "MAX_BROWSER_INSTANCES": 10, "MEM_LIMIT": 90,
        "get_container_memory_percent": lambda: 10,
        "AsyncWebCrawler": factory,
    }.items():
        monkeypatch.setattr(pool, name, value)
    state = SimpleNamespace(cfg=BrowserConfig(headless=True), created=created,
                            directory=tmp_path)
    yield state
    for crawler in created:
        pool._retire_crawler(crawler)
        pool._close_in_background(crawler)
    if pool._CLOSE_TASKS:
        await asyncio.gather(*pool._CLOSE_TASKS)


async def _assert_slot_free(admission):
    independent = pool.ReplicaPageBudget(1, admission.directory)
    lease = await independent.acquire(0)
    lease.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["permanent", "hot", "cold"])
async def test_memory_pressure_still_serves_a_running_browser(
    admission, monkeypatch, tier,
):
    """Refusing hits on running browsers would make a replica whose heap sits
    above the limit answer 503 to everything; only a new browser is refused."""
    crawler = _Crawler(config=admission.cfg)
    sig = pool._sig(admission.cfg)
    if tier == "permanent":
        monkeypatch.setattr(pool, "PERMANENT", crawler)
        monkeypatch.setattr(pool, "DEFAULT_CONFIG_SIG", sig)
    else:
        getattr(pool, f"{tier.upper()}_POOL")[sig] = crawler
    monkeypatch.setattr(pool, "get_container_memory_percent", lambda: 90)

    assert await pool.get_crawler(admission.cfg) is crawler
    assert len(crawler._docker_page_leases) == 1
    await pool.release_crawler(crawler)
    assert not crawler._docker_page_leases
    assert pool.ADMISSION_SEM._value == 4
    await _assert_slot_free(admission)


@pytest.mark.asyncio
async def test_memory_pressure_refuses_a_new_browser(admission, monkeypatch):
    monkeypatch.setattr(pool, "get_container_memory_percent", lambda: 90)
    with pytest.raises(pool.MemoryCapacityError) as refused:
        await pool.get_crawler(admission.cfg)
    assert isinstance(refused.value, HTTPException)
    assert refused.value.status_code == 503
    assert not admission.created
    assert pool.ADMISSION_SEM._value == 4
    await _assert_slot_free(admission)


@pytest.mark.asyncio
async def test_memory_refuses_dedicated_before_starting_browser(admission, monkeypatch):
    monkeypatch.setattr(pool, "get_container_memory_percent", lambda: 91)
    with pytest.raises(pool.MemoryCapacityError):
        await pool.get_dedicated_crawler(admission.cfg)
    assert not admission.created
    assert pool.ADMISSION_SEM._value == 4
    await _assert_slot_free(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_dedicated", [False, True])
async def test_pooled_and_dedicated_requests_share_replica_budget(admission, first_dedicated):
    acquire = [pool.get_crawler, pool.get_dedicated_crawler]
    release = [pool.release_crawler, pool.release_dedicated_crawler]
    first = int(first_dedicated)
    crawler = await acquire[first](admission.cfg)
    held_lease = crawler._docker_page_leases[0]
    assert not held_lease.closed
    with pytest.raises(pool.BrowserCapacityError) as refused:
        await acquire[1 - first](admission.cfg)
    assert refused.value.status_code == 503
    assert refused.value.headers["Retry-After"] == "2"
    assert len(admission.created) == 1
    assert pool.ADMISSION_SEM._value == 3

    await release[first](crawler)
    assert held_lease.closed
    other = await acquire[1 - first](admission.cfg)
    await release[1 - first](other)
    assert pool.ADMISSION_SEM._value == 4
    await _assert_slot_free(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("dedicated", [False, True])
async def test_local_contention_returns_retryable_refusal(admission, monkeypatch, dedicated):
    monkeypatch.setattr(pool, "ADMISSION_SEM", asyncio.Semaphore(0))
    monkeypatch.setattr(pool, "ADMISSION", utils.PoolAdmissionSettings(admission_wait_seconds=0.03))
    acquire = pool.get_dedicated_crawler if dedicated else pool.get_crawler
    with pytest.raises(pool.BrowserCapacityError) as refused:
        await asyncio.wait_for(acquire(admission.cfg), timeout=1)
    assert refused.value.status_code == 503
    assert refused.value.headers["Retry-After"] == "2"
    assert not admission.created
    assert pool.ADMISSION_SEM._value == 0
    await _assert_slot_free(admission)


@pytest.mark.asyncio
async def test_local_and_replica_contention_share_one_wait_deadline(admission, monkeypatch):
    monkeypatch.setattr(pool, "ADMISSION_SEM", asyncio.Semaphore(0))
    monkeypatch.setattr(pool, "ADMISSION", utils.PoolAdmissionSettings(admission_wait_seconds=0.12))
    budget = pool.PAGE_BUDGET
    external_lease = await pool.ReplicaPageBudget(1, admission.directory).acquire(0)
    replica_waits = []

    async def acquire_remaining(wait_seconds, **kwargs):
        replica_waits.append(wait_seconds)
        return await budget.acquire(wait_seconds, **kwargs)

    monkeypatch.setattr(pool, "PAGE_BUDGET", SimpleNamespace(acquire=acquire_remaining))
    pending = asyncio.create_task(pool.get_crawler(admission.cfg))
    try:
        # Ensure the request has begun waiting for its local permit.
        while not pool.ADMISSION_SEM._waiters:
            await asyncio.sleep(0)
        await asyncio.sleep(0.06)
        pool.ADMISSION_SEM.release()
        with pytest.raises(pool.BrowserCapacityError):
            await asyncio.wait_for(pending, timeout=1)
        assert len(replica_waits) == 1
        assert 0 <= replica_waits[0] < 0.085
        assert pool.ADMISSION_SEM._value == 1
        assert not admission.created
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        external_lease.close()
    await _assert_slot_free(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("dedicated", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "start_error"])
async def test_failed_browser_start_releases_both_admissions(
    admission, monkeypatch, dedicated, failure,
):
    entered = asyncio.Event()
    never = asyncio.Event()

    async def start(self):
        entered.set()
        if failure == "start_error":
            raise ValueError("browser launch failed")
        await never.wait()

    monkeypatch.setattr(_Crawler, "start", start)
    acquire = pool.get_dedicated_crawler if dedicated else pool.get_crawler
    pending = asyncio.create_task(acquire(admission.cfg))
    await asyncio.wait_for(entered.wait(), timeout=1)
    if failure == "cancel":
        pending.cancel()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else ValueError):
        await pending
    await asyncio.gather(*pool._CLOSE_TASKS)
    assert admission.created[0].closed
    assert not pool.COLD_POOL
    assert pool.ADMISSION_SEM._value == 4
    await _assert_slot_free(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize("release_first", [False, True])
async def test_retire_and_late_pooled_releases_cannot_overrelease(
    admission, monkeypatch, release_first,
):
    monkeypatch.setattr(pool, "PAGE_BUDGET", pool.ReplicaPageBudget(2, admission.directory))
    monkeypatch.setattr(pool, "ADMISSION_SEM", asyncio.Semaphore(2))
    crawler = await pool.get_crawler(admission.cfg)
    assert await pool.get_crawler(admission.cfg) is crawler
    leases = list(crawler._docker_page_leases)
    if release_first:
        await pool.release_crawler(crawler)
    assert pool._retire_crawler(crawler) == (1 if release_first else 2)
    assert pool._retire_crawler(crawler) == 0
    await pool.release_crawler(crawler)
    await pool.release_crawler(crawler)
    assert sum(not lease.closed for lease in leases) == (1 if release_first else 2)
    await pool._close_in_background(crawler)
    assert all(lease.closed for lease in leases)
    assert crawler.active_requests == 0
    assert pool.ADMISSION_SEM._value == 2
    independent = pool.ReplicaPageBudget(2, admission.directory)
    first = await independent.acquire(0)
    second = await independent.acquire(0)
    try:
        with pytest.raises(pool.BrowserCapacityError):
            await independent.acquire(0)
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_retirement_holds_replica_slot_until_browser_teardown(admission, monkeypatch):
    crawler = await pool.get_crawler(admission.cfg)
    lease = crawler._docker_page_leases[0]
    close_entered = asyncio.Event()
    finish_close = asyncio.Event()

    async def close():
        close_entered.set()
        await finish_close.wait()
        crawler.closed = True

    monkeypatch.setattr(crawler, "close", close)
    retirement = asyncio.create_task(pool.retire_pool_crawlers())
    independent = pool.ReplicaPageBudget(1, admission.directory)
    try:
        await asyncio.wait_for(close_entered.wait(), timeout=1)
        assert not pool.COLD_POOL
        assert crawler.active_requests == 0
        assert pool.ADMISSION_SEM._value == 4
        with pytest.raises(pool.BrowserCapacityError):
            await independent.acquire(0)
        await pool.release_crawler(crawler)
        await pool.release_crawler(crawler)
        assert pool.ADMISSION_SEM._value == 4
        assert not lease.closed
        with pytest.raises(pool.BrowserCapacityError):
            await independent.acquire(0)
        assert not retirement.done()
    finally:
        finish_close.set()
        await asyncio.wait_for(retirement, timeout=1)
    assert crawler.closed
    assert lease.closed
    await _assert_slot_free(admission)


@pytest.mark.asyncio
async def test_concurrent_dedicated_cleanup_and_retire_release_once(admission):
    crawler = await pool.get_dedicated_crawler(admission.cfg)
    lease = crawler._docker_page_leases[0]
    await asyncio.gather(pool.release_dedicated_crawler(crawler),
                         pool.release_dedicated_crawler(crawler))
    assert pool._retire_crawler(crawler) == 0
    await pool.release_dedicated_crawler(crawler)
    await pool.release_crawler(crawler)
    assert crawler.closed
    assert lease.closed
    assert not pool.COLD_POOL
    assert pool.ADMISSION_SEM._value == 4
    await _assert_slot_free(admission)


@pytest.mark.asyncio
async def test_worker_process_waits_past_the_http_bound(admission, monkeypatch):
    monkeypatch.setattr(pool, "ADMISSION", utils.PoolAdmissionSettings(
        replica_max_pages=1, admission_wait_seconds=0.05, worker_admission_wait_seconds=2,
    ))
    held = await pool.ReplicaPageBudget(1, admission.directory).acquire(0)
    asyncio.get_running_loop().call_later(0.3, held.close)
    with pytest.raises(pool.BrowserCapacityError):
        await pool.get_crawler(admission.cfg)
    monkeypatch.setattr(pool, "_WORKER_PROCESS", False)
    pool.use_worker_admission_wait()
    crawler = await pool.get_crawler(admission.cfg)
    await pool.release_crawler(crawler)
