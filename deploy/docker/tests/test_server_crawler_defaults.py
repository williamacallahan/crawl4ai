import asyncio
from types import SimpleNamespace

import api
import crawler_pool
import egress_broker
import llm_broker
import pytest

from crawl4ai import CrawlerRunConfig
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
    monkeypatch.setattr(api, "perform_completion_with_backoff", lambda **_kwargs: completion)
    monkeypatch.setattr(
        llm_broker,
        "resolve_llm",
        lambda *_args, **_kwargs: {
            "provider": "test/provider",
            "api_token": "test-only",
            "temperature": 0.0,
            "base_url": None,
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
