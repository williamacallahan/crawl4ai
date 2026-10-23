"""Multi-URL dispatch shares the real replica admission budget, without Chromium."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

import api
import crawler_pool as pool
import utils
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, MemoryAdaptiveDispatcher
from crawl4ai.models import CrawlResult, CrawlerTaskResult


class _Pages:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.started = asyncio.Queue()
        self.finish = asyncio.Event()
        self.release_url = {}
        self.cancelled = 0

    async def run_page(self, dispatcher, url, config, session_id):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.put_nowait(url)
        try:
            await self.release_url.get(url, self.finish).wait()
            return CrawlResult(url=url, html="", success=True)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


@pytest_asyncio.fixture
async def replica(monkeypatch, tmp_path):
    pages = _Pages()
    crawlers = []

    # Stub only the browser call: the real crawl_url, rate limiter and
    # memory checks run, so the test sees exactly what the seam charges.
    async def run_page(dispatcher, *args, **kwargs):
        return await pages.run_page(dispatcher, *args, **kwargs)

    monkeypatch.setattr(MemoryAdaptiveDispatcher, "_run_page", run_page)
    monkeypatch.setattr(pool, "ADMISSION_SEM", asyncio.Semaphore(10))
    monkeypatch.setattr(pool, "_WORKER_PROCESS", False)

    def configure(max_pages, request_max_pages=4):
        monkeypatch.setattr(pool, "PAGE_BUDGET", pool.ReplicaPageBudget(max_pages, tmp_path))
        monkeypatch.setattr(pool, "ADMISSION", utils.PoolAdmissionSettings(
            replica_max_pages=max_pages, request_max_pages=request_max_pages,
            admission_wait_seconds=0.02,
        ))

    async def admit():
        # arun_many itself does not need a browser strategy. Keep the real
        # library method and the same admission ownership as get_crawler.
        crawler = object.__new__(AsyncWebCrawler)
        crawler.active_requests = 1
        pool._attach_admission(crawler, await pool._acquire_admission())
        crawlers.append(crawler)
        return crawler

    state = SimpleNamespace(pages=pages, configure=configure, admit=admit,
                            directory=tmp_path)
    yield state
    for crawler in crawlers:
        await pool.release_crawler(crawler)


def _dispatcher():
    return api._memory_dispatcher({"crawler": {
        "pool": {"max_pages": 5},
        "memory_threshold_percent": 100,
        "recovery_threshold_percent": 99,
        "rate_limiter": {"enabled": False, "base_delay": [0, 0]},
    }})


async def _run(crawler, prefix, stream):
    results = await crawler.arun_many(
        urls=[f"https://example.com/{prefix}/{index}" for index in range(5)],
        config=CrawlerRunConfig(stream=stream), dispatcher=_dispatcher(),
    )
    return [result async for result in results] if stream else results


async def _wait_started(pages, count):
    for _ in range(count):
        await asyncio.wait_for(pages.started.get(), timeout=2)


async def _assert_free_slots(replica, max_pages):
    independent = pool.ReplicaPageBudget(max_pages, replica.directory)
    leases = []
    try:
        for _ in range(max_pages):
            leases.append(await independent.acquire(0))
        with pytest.raises(pool.BrowserCapacityError):
            await independent.acquire(0)
    finally:
        for lease in leases:
            lease.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_pages", [1, 3])
async def test_multi_url_pages_share_request_admission(replica, stream, max_pages):
    replica.configure(max_pages)
    crawler = await replica.admit()
    request_lease = crawler._docker_page_leases[0]
    pending = asyncio.create_task(_run(crawler, "batch", stream))
    try:
        await _wait_started(replica.pages, max_pages)
        # Let the other queued tasks attempt admission while pages are blocked.
        await asyncio.sleep(0.08)
        assert replica.pages.peak == max_pages
        assert not request_lease.closed
        with pytest.raises(pool.BrowserCapacityError):
            await pool.ReplicaPageBudget(max_pages, replica.directory).acquire(0)
        replica.pages.finish.set()
        results = await asyncio.wait_for(pending, timeout=3)
        assert len(results) == 5
        assert all(result.success for result in results)
        assert len({result.url for result in results}) == 5
        assert replica.pages.peak == max_pages
        assert replica.pages.active == 0
        assert not request_lease.closed
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await pool.release_crawler(crawler)
    assert request_lease.closed
    await _assert_free_slots(replica, max_pages)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_cancelled_batch_releases_extra_pages_then_request_slot(replica, stream):
    replica.configure(3)
    crawler = await replica.admit()
    request_lease = crawler._docker_page_leases[0]
    pending = asyncio.create_task(_run(crawler, "cancel", stream))
    try:
        await _wait_started(replica.pages, 3)
        await asyncio.sleep(0.08)
        assert replica.pages.peak == 3
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert replica.pages.active == 0
        assert replica.pages.cancelled == 3
        assert not request_lease.closed
        # Cancellation releases the two extra page leases; request ownership
        # of the original admission continues until its finally block runs.
        independent = pool.ReplicaPageBudget(3, replica.directory)
        leases = [await independent.acquire(0), await independent.acquire(0)]
        try:
            with pytest.raises(pool.BrowserCapacityError):
                await independent.acquire(0)
        finally:
            for lease in leases:
                lease.close()
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await pool.release_crawler(crawler)
    assert request_lease.closed
    await _assert_free_slots(replica, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_fully_admitted_concurrent_batches_reuse_their_request_slots(replica, stream):
    replica.configure(2)
    crawlers = [await replica.admit(), await replica.admit()]
    pending = [asyncio.create_task(_run(crawler, str(index), stream))
               for index, crawler in enumerate(crawlers)]
    try:
        await _wait_started(replica.pages, 2)
        await asyncio.sleep(0.08)
        assert replica.pages.peak == 2
        replica.pages.finish.set()
        batches = await asyncio.wait_for(asyncio.gather(*pending), timeout=3)
        assert [len(batch) for batch in batches] == [5, 5]
        assert all(result.success for batch in batches for result in batch)
        assert replica.pages.peak == 2
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for crawler in crawlers:
            await pool.release_crawler(crawler)
    await _assert_free_slots(replica, 2)


@pytest.mark.asyncio
async def test_extra_page_admission_timeouts_preserve_waiting_urls(replica, monkeypatch):
    replica.configure(2)
    crawler = await replica.admit()
    external = await pool.ReplicaPageBudget(2, replica.directory).acquire(0)
    acquire = pool.PAGE_BUDGET.acquire
    timeouts = []

    async def record_acquire(wait_seconds):
        try:
            return await acquire(wait_seconds)
        except pool.BrowserCapacityError:
            timeouts.append(True)
            raise

    monkeypatch.setattr(pool.PAGE_BUDGET, "acquire", record_acquire)
    pending = asyncio.create_task(_run(crawler, "timeout", False))
    try:
        await _wait_started(replica.pages, 1)
        await asyncio.sleep(0.12)
        assert replica.pages.peak == 1
        assert timeouts, "extra pages must outlive the bounded admission wait"
        external.close()
        await _wait_started(replica.pages, 1)
        assert replica.pages.peak == 2
        replica.pages.finish.set()
        results = await asyncio.wait_for(pending, timeout=3)
        assert len(results) == 5
        assert all(result.success for result in results)
        assert len({result.url for result in results}) == 5
    finally:
        external.close()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await pool.release_crawler(crawler)
    await _assert_free_slots(replica, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("archive", [False, True])
async def test_api_stream_close_drains_real_dispatcher_before_releasing_admission(
    replica, monkeypatch, archive,
):
    import archive_transport

    replica.configure(3)
    acquired = []

    async def acquire(*_args, **_kwargs):
        crawler = await replica.admit()
        acquired.append((crawler, crawler._docker_page_leases[0]))
        return crawler

    config = {"crawler": {
        "pool": {"max_pages": 5},
        "memory_threshold_percent": 100,
        "recovery_threshold_percent": 99,
        "rate_limiter": {"enabled": False, "base_delay": [0, 0]},
        "browser": {"kwargs": {}}, "base_config": {},
    }, "limits": {"wall_clock_s": 0}}
    urls = [f"https://example.com/early-close/{index}" for index in range(5)]
    first_page = asyncio.Event()
    replica.pages.release_url[urls[0]] = first_page
    if archive:
        # Keep grouped_results and its owned_crawler context real; only replace
        # acquisition so no archive browser or network fetch is needed.
        monkeypatch.setattr(archive_transport, "acquire_crawler", acquire)
        responses = {url: {"url": url, "status": 200, "body": "",
                           "responseHeaders": {}} for url in urls}
        results = archive_transport.grouped_results(
            urls, BrowserConfig(), CrawlerRunConfig(stream=True), responses, {},
            lambda: api._memory_dispatcher(config),
        )
        crawler, request_id = None, None
    else:
        monkeypatch.setattr(pool, "get_crawler", acquire)
        monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
        crawler, results, _, request_id = await api.handle_stream_crawl_request(
            urls, {}, {}, config,
        )

    stream = api.stream_results(crawler, results, request_id)
    first_result = asyncio.create_task(anext(stream))
    try:
        await _wait_started(replica.pages, 3)
        assert replica.pages.active == 3
        first_page.set()
        assert json.loads(await asyncio.wait_for(first_result, timeout=2))["url"] == urls[0]
        assert replica.pages.active >= 2
        assert not replica.pages.finish.is_set()

        await asyncio.wait_for(stream.aclose(), timeout=2)

        # These checks run in the same live event loop, before fixture teardown
        # or async-generator finalization can mask detached page tasks.
        assert replica.pages.active == 0
        assert replica.pages.cancelled >= 2
        assert len(acquired) == 1
        assert acquired[0][0].active_requests == 0
        assert acquired[0][1].closed
        await _assert_free_slots(replica, 3)
    finally:
        first_result.cancel()
        await asyncio.gather(first_result, return_exceptions=True)
        await stream.aclose()


@pytest.mark.asyncio
async def test_one_batch_leaves_slots_for_other_requests(replica):
    replica.configure(8, request_max_pages=2)
    crawler = await replica.admit()
    pending = asyncio.create_task(_run(crawler, "capped", False))
    try:
        await _wait_started(replica.pages, 2)
        await asyncio.sleep(0.08)
        assert replica.pages.peak == 2
        # Six of eight slots stay free for /md while the batch runs.
        independent = pool.ReplicaPageBudget(8, replica.directory)
        leases = [await independent.acquire(0) for _ in range(6)]
        for lease in leases:
            lease.close()
        replica.pages.finish.set()
        results = await asyncio.wait_for(pending, timeout=3)
        assert len(results) == 5 and all(result.success for result in results)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_rate_limited_page_holds_no_slot_while_it_waits(replica):
    replica.configure(3)
    crawler = await replica.admit()
    gate = asyncio.Event()

    class HeldLimiter:
        async def wait_if_needed(self, url):
            await gate.wait()

        def update_delay(self, url, status_code):
            return True

    dispatcher = _dispatcher()
    dispatcher.rate_limiter = HeldLimiter()
    pending = asyncio.create_task(crawler.arun_many(
        urls=[f"https://example.com/limited/{index}" for index in range(3)],
        config=CrawlerRunConfig(), dispatcher=dispatcher,
    ))
    try:
        await asyncio.sleep(0.1)
        assert replica.pages.active == 0
        # Only the request's own crawler lease is held: two slots stay free.
        independent = pool.ReplicaPageBudget(3, replica.directory)
        leases = [await independent.acquire(0) for _ in range(2)]
        for lease in leases:
            lease.close()
        gate.set()
        replica.pages.finish.set()
        results = await asyncio.wait_for(pending, timeout=3)
        assert len(results) == 3
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
