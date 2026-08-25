import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import api
import crawler_pool
import egress_broker
import llm_broker
import pytest

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_logger import AsyncLogger
from crawl4ai.markdown_generation_strategy import MarkdownGenerationStrategy
from crawl4ai.models import MarkdownGenerationResult
from utils import load_config


@pytest.mark.parametrize(
    ("request_config", "expected_delay"),
    [
        ({}, 1.0),
        ({"delay_before_return_html": 0.0}, 0.0),
        (
            {
                "type": "CrawlerRunConfig",
                "params": {"delay_before_return_html": 0.1},
            },
            0.1,
        ),
        ({"params": {"delay_before_return_html": 0.0}}, 1.0),
    ],
)
def test_server_defaults_preserve_explicit_wire_values(
    request_config,
    expected_delay,
):
    loaded = CrawlerRunConfig.load(request_config)

    api.apply_server_crawler_defaults(loaded, request_config, load_config())

    assert loaded.delay_before_return_html == expected_delay


@pytest.mark.parametrize(
    "request_config",
    [{}, {"type": "CrawlerRunConfig", "params": {}}],
)
def test_per_url_configs_receive_omitted_server_defaults(request_config):
    loaded = CrawlerRunConfig.load(request_config)

    api.apply_server_crawler_defaults(loaded, request_config, load_config())

    assert loaded.delay_before_return_html == 1.0


def test_markdown_uses_server_render_readiness_default(monkeypatch):
    captured = {}
    markdown_result = SimpleNamespace(
        raw_markdown="raw",
        fit_markdown="fit",
    )

    async def arun(_crawler, *args, **kwargs):
        captured["config"] = kwargs["config"]
        return SimpleNamespace(success=True, markdown=markdown_result)

    async def get_crawler(_browser_config):
        return object()

    async def release_crawler(_crawler):
        return None

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_crawler_arun", arun)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)

    markdown = asyncio.run(
        api.handle_markdown_request(
            "https://example.com",
            api.FilterType.FIT,
            config=load_config(),
        )
    )

    assert markdown == "fit"
    assert captured["config"].delay_before_return_html == 1.0


def test_quick_llm_uses_server_render_readiness_default(monkeypatch):
    captured = {}
    crawl_result = SimpleNamespace(
        success=True,
        markdown=SimpleNamespace(
            fit_markdown="context",
            raw_markdown="context",
        ),
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )

    async def arun(_crawler, *args, **kwargs):
        captured["config"] = kwargs["config"]
        return crawl_result

    async def get_crawler(_browser_config):
        return object()

    async def release_crawler(_crawler):
        return None

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(api, "_crawler_arun", arun)
    monkeypatch.setattr(
        api,
        "aperform_completion_with_backoff",
        AsyncMock(return_value=completion),
    )
    monkeypatch.setattr(
        llm_broker,
        "resolve_llm",
        lambda *_args, **_kwargs: {
            "provider": "test/provider",
            "api_token": "test-only",
            "temperature": 0.0,
            "base_url": None,
            "extra_args": {"timeout": 25, "num_retries": 1},
        },
    )
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)

    answer = asyncio.run(
        api.handle_llm_qa(
            "https://example.com",
            "question",
            {
                **load_config(),
                "llm": {
                    "provider": "test/provider",
                    "api_key": "test-only",
                },
            },
        )
    )

    assert answer == "answer"
    assert captured["config"].delay_before_return_html == 1.0
    assert api.aperform_completion_with_backoff.await_args.kwargs["max_attempts"] == 1


def test_markdown_generation_does_not_block_the_event_loop():
    class SlowMarkdownGenerator(MarkdownGenerationStrategy):
        def generate_markdown(self, **_kwargs):
            time.sleep(0.2)
            return MarkdownGenerationResult(
                raw_markdown="markdown",
                markdown_with_citations="markdown",
                references_markdown="",
            )

    async def exercise():
        crawler = object.__new__(AsyncWebCrawler)
        crawler.logger = AsyncLogger(verbose=False)
        crawler._markdown_generation_sem = asyncio.Semaphore(1)
        config = CrawlerRunConfig(markdown_generator=SlowMarkdownGenerator())
        processing = asyncio.create_task(
            crawler.aprocess_html(
                "https://example.com",
                "<html><body><h1>Example</h1></body></html>",
                None,
                config,
                None,
                None,
                False,
            )
        )
        started = time.monotonic()
        await asyncio.sleep(0.05)
        responsive_after = time.monotonic() - started
        result = await processing
        return responsive_after, result

    responsive_after, result = asyncio.run(exercise())

    assert responsive_after < 0.15
    assert result.markdown.raw_markdown == "markdown"


def test_markdown_deadline_keeps_admission_until_worker_completion():
    call_started = []

    class SlowMarkdownGenerator(MarkdownGenerationStrategy):
        def generate_markdown(self, **_kwargs):
            call_started.append(time.monotonic())
            time.sleep(0.2)
            return MarkdownGenerationResult(
                raw_markdown="markdown",
                markdown_with_citations="markdown",
                references_markdown="",
            )

    async def exercise():
        crawler = object.__new__(AsyncWebCrawler)
        crawler.logger = AsyncLogger(verbose=False)
        crawler._markdown_generation_sem = asyncio.Semaphore(1)
        config = CrawlerRunConfig(markdown_generator=SlowMarkdownGenerator())
        processing = asyncio.create_task(
            crawler.aprocess_html(
                "https://example.com",
                "<html><body><h1>Example</h1></body></html>",
                None,
                config,
                None,
                None,
                False,
            )
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(processing, timeout=0.05)
        timed_out_after = time.monotonic() - started
        replacement = asyncio.create_task(
            crawler.aprocess_html(
                "https://example.org",
                "<html><body><h1>Replacement</h1></body></html>",
                None,
                config,
                None,
                None,
                False,
            )
        )
        await asyncio.sleep(0.05)
        assert len(call_started) == 1
        await replacement
        return timed_out_after

    assert asyncio.run(exercise()) < 0.1
    assert len(call_started) == 2


def test_streaming_uses_server_render_readiness_default(monkeypatch):
    captured = {}

    class FakeCrawler:
        async def arun_many(self, *, config, **_kwargs):
            captured["config"] = config

            async def results():
                if False:
                    yield None

            return results()

    async def get_crawler(_browser_config):
        return FakeCrawler()

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)

    asyncio.run(
        api.handle_stream_crawl_request(
            ["https://example.com"],
            {},
            {},
            load_config(),
        )
    )

    assert captured["config"].delay_before_return_html == 1.0
