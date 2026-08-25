import asyncio
import json

import api
import crawler_pool
import egress_broker
import pytest
from fastapi import HTTPException
from schemas import HookConfig

from crawl4ai import BrowserConfig


class FakeStrategy:
    def __init__(self):
        self.hooks = {"before_goto": self.original_hook}

    async def original_hook(self, *_args, **_kwargs):
        return None

    def set_hook(self, hook_point, hook):
        self.hooks[hook_point] = hook


class FakeResult:
    url = "https://example.com"

    def model_dump(self):
        return {"url": self.url, "success": True, "fit_html": None, "pdf": None}


class FakeCrawler:
    def __init__(self, mode="success"):
        self.crawler_strategy = FakeStrategy()
        self.mode = mode
        self.started = False
        self.closed = False
        self.run_started = asyncio.Event()
        self.continue_run = asyncio.Event()

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def arun(self, *_args, **_kwargs):
        self.run_started.set()
        if self.mode == "error":
            raise RuntimeError("crawl failed")
        if self.mode == "block":
            await self.continue_run.wait()
        if self.mode == "slow":
            await asyncio.sleep(0.05)
        return FakeResult()

    async def arun_many(self, *_args, **_kwargs):
        async def results():
            if self.mode == "slow":
                await asyncio.sleep(0.05)
            if self.mode == "error":
                raise RuntimeError("crawl failed")
            yield FakeResult()

        return results()


class FakeBrowserConfig:
    verbose = False

    @classmethod
    def load(cls, _value, **_kwargs):
        return cls()


class FakeCrawlerRunConfig:
    deep_crawl_strategy = None
    scraping_strategy = None
    stream = False

    @classmethod
    def load(cls, _value, **_kwargs):
        return cls()


def crawl_config(*, wall_clock_s=0):
    return {
        "crawler": {
            "base_config": {},
            "pool": {"max_pages": 2},
            "memory_threshold_percent": 95,
            "recovery_threshold_percent": 80,
            "rate_limiter": {"enabled": False, "base_delay": [0, 0]},
            "browser": {"kwargs": {}},
        },
        "limits": {"wall_clock_s": wall_clock_s},
    }


def install_fakes(monkeypatch, pooled, dedicated=None):
    released = []
    dedicated = dedicated or FakeCrawler()

    async def get_crawler(_config):
        return pooled

    async def release_crawler(candidate):
        released.append(candidate)

    async def get_dedicated_crawler(_config):
        await dedicated.start()
        setattr(dedicated, "_docker_request_owned", True)
        return dedicated

    async def release_dedicated_crawler(candidate):
        assert candidate is dedicated
        await dedicated.close()

    async def declarative_hook(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api, "BrowserConfig", FakeBrowserConfig)
    monkeypatch.setattr(api, "CrawlerRunConfig", FakeCrawlerRunConfig)
    monkeypatch.setattr(api, "AsyncWebCrawler", lambda **_kwargs: dedicated)
    monkeypatch.setattr(
        api, "_apply_server_browser_policy", lambda value, _config: value
    )
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(
        api,
        "build_declarative_hooks",
        lambda _specs: {"before_goto": declarative_hook},
    )
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)
    monkeypatch.setattr(
        crawler_pool,
        "get_dedicated_crawler",
        get_dedicated_crawler,
    )
    monkeypatch.setattr(
        crawler_pool,
        "release_dedicated_crawler",
        release_dedicated_crawler,
    )
    return released, dedicated


def run(coro):
    return asyncio.run(coro)


def test_declarative_hook_contract_has_no_inert_global_timeout(server_module):
    assert set(HookConfig.model_json_schema()["properties"]) == {"hooks"}
    assert HookConfig.model_validate({"timeout": 30}).hooks == []
    response = run(server_module.get_hooks_info())
    info = json.loads(response.body)
    assert "timeout_limits" not in info


def test_non_streaming_hooks_use_dedicated_crawler(monkeypatch):
    pooled = FakeCrawler()
    original = pooled.crawler_strategy.hooks.copy()
    released, dedicated = install_fakes(monkeypatch, pooled)

    response = run(
        api.handle_crawl_request(
            ["https://example.com"],
            {},
            {},
            crawl_config(),
            hooks_config={"hooks": [{"action": "test"}]},
        )
    )

    assert response["success"] is True
    assert pooled.crawler_strategy.hooks == original
    assert released == []
    assert dedicated.started and dedicated.closed


def test_non_streaming_crawl_projects_requested_result_fields(monkeypatch):
    pooled = FakeCrawler()
    released, _dedicated = install_fakes(monkeypatch, pooled, dedicated=None)

    response = run(
        api.handle_crawl_request(
            ["https://example.com"],
            {},
            {},
            crawl_config(),
            result_fields=["url", "success"],
        )
    )

    assert response["results"] == [{"url": "https://example.com", "success": True}]
    assert released == [pooled]


def test_hook_attachment_failure_closes_dedicated_crawler(monkeypatch):
    pooled = FakeCrawler()
    released, dedicated = install_fakes(monkeypatch, pooled)

    def fail_after_mutation(crawler, _config):
        crawler.crawler_strategy.hooks["before_goto"] = object()
        raise RuntimeError("invalid hook")

    monkeypatch.setattr(api, "_attach_declarative_hooks", fail_after_mutation)
    with pytest.raises(HTTPException) as error:
        run(
            api.handle_crawl_request(
                ["https://example.com"],
                {},
                {},
                crawl_config(),
                hooks_config={"hooks": [{"action": "test"}]},
            )
        )

    assert error.value.status_code == 500
    assert dedicated.closed
    assert released == []


def test_cancelled_hook_request_closes_dedicated_crawler(monkeypatch):
    pooled = FakeCrawler()
    dedicated = FakeCrawler(mode="block")
    released, dedicated = install_fakes(monkeypatch, pooled, dedicated)

    async def exercise():
        task = asyncio.create_task(
            api.handle_crawl_request(
                ["https://example.com"],
                {},
                {},
                crawl_config(),
                hooks_config={"hooks": [{"action": "test"}]},
            )
        )
        await dedicated.run_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(exercise())
    assert dedicated.closed
    assert released == []


def test_stream_close_closes_dedicated_hook_crawler(monkeypatch):
    pooled = FakeCrawler()
    original = pooled.crawler_strategy.hooks.copy()
    released, dedicated = install_fakes(monkeypatch, pooled)

    async def exercise():
        crawler, results, _hooks = await api.handle_stream_crawl_request(
            ["https://example.com"],
            {},
            {},
            crawl_config(),
            hooks_config={"hooks": [{"action": "test"}]},
        )
        output = api.stream_results(crawler, results)
        assert json.loads((await anext(output)).decode())["success"] is True
        await output.aclose()

    run(exercise())
    assert pooled.crawler_strategy.hooks == original
    assert dedicated.closed
    assert released == []


def test_non_hook_request_keeps_pooled_lifecycle(monkeypatch):
    pooled = FakeCrawler()
    released, dedicated = install_fakes(monkeypatch, pooled)
    response = run(
        api.handle_crawl_request(["https://example.com"], {}, {}, crawl_config())
    )

    assert response["success"] is True
    assert released == [pooled]
    assert not dedicated.started and not dedicated.closed


def test_non_streaming_wall_clock_deadline_returns_504(monkeypatch):
    pooled = FakeCrawler(mode="slow")
    released, _dedicated = install_fakes(monkeypatch, pooled)

    with pytest.raises(HTTPException) as error:
        run(
            api.handle_crawl_request(
                ["https://example.com"], {}, {}, crawl_config(wall_clock_s=0.01)
            )
        )

    assert error.value.status_code == 504
    assert released == [pooled]


def test_streaming_wall_clock_deadline_emits_failure_and_releases(monkeypatch):
    pooled = FakeCrawler(mode="slow")
    released, _dedicated = install_fakes(monkeypatch, pooled)

    async def exercise():
        crawler, results, _hooks = await api.handle_stream_crawl_request(
            ["https://example.com"], {}, {}, crawl_config(wall_clock_s=0.01)
        )
        output = api.stream_results(crawler, results)
        body = json.loads((await anext(output)).decode())
        assert body == {"status": "failed", "error": "Crawl exceeded the time limit"}
        with pytest.raises(StopAsyncIteration):
            await anext(output)

    run(exercise())
    assert released == [pooled]


def configure_dedicated_pool(
    monkeypatch,
    factory,
    *,
    capacity=2,
    browser_capacity=None,
):
    monkeypatch.setattr(crawler_pool, "ADMISSION_SEM", asyncio.Semaphore(capacity))
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())
    monkeypatch.setattr(crawler_pool, "MAX_ACTIVE_REQUESTS", capacity)
    monkeypatch.setattr(
        crawler_pool,
        "MAX_BROWSER_INSTANCES",
        browser_capacity if browser_capacity is not None else capacity,
    )
    monkeypatch.setattr(crawler_pool, "MEM_LIMIT", 100)
    monkeypatch.setattr(crawler_pool, "PERMANENT", None)
    monkeypatch.setattr(crawler_pool, "HOT_POOL", {})
    monkeypatch.setattr(crawler_pool, "COLD_POOL", {})
    monkeypatch.setattr(crawler_pool, "LAST_USED", {})
    monkeypatch.setattr(crawler_pool, "USAGE_COUNT", {})
    monkeypatch.setattr(crawler_pool, "get_container_memory_percent", lambda: 0)
    monkeypatch.setattr(crawler_pool, "AsyncWebCrawler", factory)


@pytest.mark.asyncio
async def test_dedicated_hook_crawlers_share_pool_admission_and_instance_cap(
    monkeypatch,
):
    state = {"open": 0, "max_open": 0}

    class CapacityCrawler(FakeCrawler):
        async def start(self):
            await super().start()
            state["open"] += 1
            state["max_open"] = max(state["max_open"], state["open"])

        async def close(self):
            if not self.closed:
                state["open"] -= 1
            await super().close()

    configure_dedicated_pool(
        monkeypatch,
        lambda **_kwargs: CapacityCrawler(),
        capacity=2,
    )

    first = await crawler_pool.get_dedicated_crawler(BrowserConfig())
    second = await crawler_pool.get_dedicated_crawler(BrowserConfig())
    third_task = asyncio.create_task(
        crawler_pool.get_dedicated_crawler(BrowserConfig())
    )
    await asyncio.sleep(0)
    assert not third_task.done()

    await crawler_pool.release_dedicated_crawler(first)
    third = await asyncio.wait_for(third_task, timeout=1)
    assert state["max_open"] == 2
    assert len(crawler_pool.COLD_POOL) == 2

    await crawler_pool.release_dedicated_crawler(second)
    await crawler_pool.release_dedicated_crawler(third)
    assert state["open"] == 0
    assert crawler_pool.COLD_POOL == {}
    assert crawler_pool.ADMISSION_SEM._value == 2

    await api._dispose_crawler(first)
    await crawler_pool.release_dedicated_crawler(first)
    assert crawler_pool.ADMISSION_SEM._value == 2


@pytest.mark.asyncio
async def test_dedicated_admission_released_when_start_fails(monkeypatch):
    created = []

    class FailingCrawler(FakeCrawler):
        async def start(self):
            self.started = True
            raise RuntimeError("browser start failed")

    def factory(**_kwargs):
        crawler = FailingCrawler()
        created.append(crawler)
        return crawler

    configure_dedicated_pool(monkeypatch, factory, capacity=1)
    with pytest.raises(RuntimeError, match="browser start failed"):
        await crawler_pool.get_dedicated_crawler(BrowserConfig())

    assert created[0].closed
    assert crawler_pool.COLD_POOL == {}
    assert crawler_pool.ADMISSION_SEM._value == 1


@pytest.mark.asyncio
async def test_dedicated_hook_crawlers_cannot_exceed_browser_instance_cap(
    monkeypatch,
):
    configure_dedicated_pool(
        monkeypatch,
        lambda **_kwargs: FakeCrawler(),
        capacity=3,
        browser_capacity=2,
    )
    first = await crawler_pool.get_dedicated_crawler(BrowserConfig())
    second = await crawler_pool.get_dedicated_crawler(BrowserConfig())

    with pytest.raises(RuntimeError, match="pool is at capacity"):
        await crawler_pool.get_dedicated_crawler(BrowserConfig())

    assert len(crawler_pool.COLD_POOL) == 2
    assert crawler_pool.ADMISSION_SEM._value == 1
    await crawler_pool.release_dedicated_crawler(first)
    await crawler_pool.release_dedicated_crawler(second)
    assert crawler_pool.ADMISSION_SEM._value == 3


@pytest.mark.asyncio
async def test_dedicated_admission_released_when_creation_is_cancelled(monkeypatch):
    created = []

    class BlockingCrawler(FakeCrawler):
        async def start(self):
            self.started = True
            self.run_started.set()
            await self.continue_run.wait()

    def factory(**_kwargs):
        crawler = BlockingCrawler()
        created.append(crawler)
        return crawler

    configure_dedicated_pool(monkeypatch, factory, capacity=1)
    task = asyncio.create_task(crawler_pool.get_dedicated_crawler(BrowserConfig()))
    while not created:
        await asyncio.sleep(0)
    await created[0].run_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert created[0].closed
    assert crawler_pool.COLD_POOL == {}
    assert crawler_pool.ADMISSION_SEM._value == 1


@pytest.mark.asyncio
async def test_dedicated_release_finishes_cleanup_when_cancelled(monkeypatch):
    created = []

    class BlockingCloseCrawler(FakeCrawler):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.continue_close = asyncio.Event()

        async def close(self):
            self.close_started.set()
            await self.continue_close.wait()
            await super().close()

    def factory(**_kwargs):
        crawler = BlockingCloseCrawler()
        created.append(crawler)
        return crawler

    configure_dedicated_pool(monkeypatch, factory, capacity=1)
    crawler = await crawler_pool.get_dedicated_crawler(BrowserConfig())
    blocking_crawler = created[0]
    assert crawler is blocking_crawler
    release_task = asyncio.create_task(crawler_pool.release_dedicated_crawler(crawler))
    await blocking_crawler.close_started.wait()
    release_task.cancel()
    await asyncio.sleep(0)
    assert not release_task.done()

    blocking_crawler.continue_close.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert blocking_crawler.closed
    assert not getattr(crawler, "_docker_request_owned")
    assert getattr(crawler, "_docker_pool_sig") is None
    assert crawler_pool.COLD_POOL == {}
    assert crawler_pool.ADMISSION_SEM._value == 1
