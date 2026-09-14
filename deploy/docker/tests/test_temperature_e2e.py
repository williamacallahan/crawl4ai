"""End-to-end temperature-override coverage at the litellm transport boundary.

The existing suite mocks `api.aperform_completion_with_backoff` itself, which
short-circuits the real helper logic that hard-codes
`extra_args = {"temperature": 0.01, ...}` and only merges `kwargs["extra_args"]`.
Those tests therefore asserted the *intended* temperature on a mock and never
observed that the real helper dropped it.

These tests patch the true transport boundary (`litellm.acompletion` for the
async QA/`/md` paths; `litellm.completion` for the sync extraction path) and let
the real `aperform_completion_with_backoff` / `perform_completion_with_backoff`
run, so the effective temperature reaching litellm is observable.

Guarantees:
- A caller `temperature` override reaches litellm on all three LLM paths.
- An operator-configured temperature (`llm["temperature"]`) is honoured when
  the caller omits the override, on all three LLM paths.
- An explicit caller `temperature=0.0` is honoured (not collapsed to the
  operator/server default by a falsy `or`).
- When neither caller nor operator sets a temperature, the helper's own
  hard-coded default (0.01) remains in effect (no regression in default
  behaviour; `None` is not forced through extra_args).
- The sync extraction helper consumes the temperature from `extra_args`
  (the channel the fork now routes through), not from `LLMConfig.temperature`.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import api
import crawler_pool
import egress_broker
import llm_broker
import pytest
from crawl4ai import LLMConfig, LLMExtractionStrategy
from utils import load_config


class PermitRedis:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.hset = AsyncMock()
        self.expire = AsyncMock()
        self.eval = AsyncMock(return_value=1)

    async def set(self, *_args, **_kwargs):
        return self.acquired


def _resolve(sever_temperature):
    return lambda *_a, **_kw: {
        "provider": "openai/qwen3-32b",
        "api_token": "test-only",
        "temperature": sever_temperature,
        "base_url": "https://gateway.example/v1",
        "extra_args": {
            "timeout": 300,
            "num_retries": 0,
            "reasoning_effort": "low",
            "max_tokens": 4096,
            "extra_headers": {"X-User-ID": "crawl4ai"},
        },
    }


def _config():
    return {**load_config(), "llm": {"provider": "openai/qwen3-32b", "api_key": "test-only"}}


def _patch_crawl_path(monkeypatch):
    crawl_result = SimpleNamespace(
        success=True,
        markdown=SimpleNamespace(fit_markdown="ctx", raw_markdown="ctx"),
    )

    async def fake_arun(_crawler, *a, **kw):
        return crawl_result

    async def fake_get_crawler(_bc):
        return object()

    async def fake_release_crawler(_c):
        return None

    monkeypatch.setattr(api, "validate_url_destination", lambda _u: None)
    monkeypatch.setattr(api, "_crawler_arun", fake_arun)
    monkeypatch.setattr(crawler_pool, "get_crawler", fake_get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", fake_release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    return crawl_result


def _patch_acompletion(monkeypatch, captured):
    async def fake_acompletion(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)


# ---------------------------------------------------------------------------
# QA path: handle_llm_qa -> aperform_completion_with_backoff -> litellm.acompletion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("caller_temp", "server_temp", "expected"),
    [
        (0.5, 0.7, 0.5),   # caller override wins
        (None, 0.7, 0.7),  # operator config respected when caller omits
        (0.0, 0.7, 0.0),   # explicit 0.0 honoured, not collapsed to server default
        (None, None, 0.01),  # neither sets: helper hard-coded default preserved
    ],
)
def test_qa_temperature_reaches_litellm(monkeypatch, caller_temp, server_temp, expected):
    captured = {}
    _patch_crawl_path(monkeypatch)
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve(server_temp))
    _patch_acompletion(monkeypatch, captured)

    answer = asyncio.run(
        api.handle_llm_qa(
            "https://example.com",
            "question",
            _config(),
            temperature=caller_temp,
            redis=PermitRedis(),
        )
    )

    assert answer == "answer"
    assert captured["kwargs"]["temperature"] == expected


# ---------------------------------------------------------------------------
# /md path: handle_markdown_request -> aperform_completion_with_backoff -> litellm.acompletion
# ---------------------------------------------------------------------------

def _patch_md_response(monkeypatch, captured):
    crawl_result = SimpleNamespace(
        success=True,
        cleaned_html="<h1>Example</h1>",
        markdown=SimpleNamespace(raw_markdown="# Example", fit_markdown=""),
    )

    async def fake_arun(_crawler, *a, **kw):
        return crawl_result

    async def fake_get_crawler(_bc):
        return object()

    async def fake_release_crawler(_c):
        return None

    monkeypatch.setattr(api, "validate_url_destination", lambda _u: None)
    monkeypatch.setattr(api, "_crawler_arun", fake_arun)
    monkeypatch.setattr(crawler_pool, "get_crawler", fake_get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", fake_release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)

    async def fake_acompletion(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<content># Filtered</content>")
                )
            ]
        )

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)


@pytest.mark.parametrize(
    ("caller_temp", "server_temp", "expected"),
    [
        (0.5, 0.7, 0.5),   # caller override forwarded (previously dropped entirely)
        (None, 0.7, 0.7),  # operator config forwarded (previously dropped entirely)
        (0.0, 0.7, 0.0),   # explicit 0.0 honoured
        (None, None, 0.01),  # neither sets: helper default preserved
    ],
)
def test_md_temperature_reaches_litellm(monkeypatch, caller_temp, server_temp, expected):
    captured = {}
    _patch_md_response(monkeypatch, captured)
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve(server_temp))

    markdown = asyncio.run(
        api.handle_markdown_request(
            PermitRedis(),
            "https://example.com",
            api.FilterType.LLM,
            query="Keep the title",
            config=_config(),
            temperature=caller_temp,
        )
    )

    assert markdown == "# Filtered"
    assert captured["kwargs"]["temperature"] == expected


# ---------------------------------------------------------------------------
# Extraction path: process_llm_extraction -> LLMExtractionStrategy.extra_args
# (the channel the sync perform_completion_with_backoff actually merges).
# ---------------------------------------------------------------------------

class _FakeCrawler:
    def __init__(self, captured, **_kwargs):
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def arun(self, *, config, **_kwargs):
        strat = config.extraction_strategy
        self._captured["extra_args"] = dict(strat.extra_args)
        self._captured["llm_temperature"] = strat.llm_config.temperature
        return SimpleNamespace(
            success=True,
            extracted_content='[{"index": 0, "content": "extracted"}]',
        )


def _run_extraction(monkeypatch, caller_temp, server_temp):
    captured = {}
    config = load_config()
    redis = PermitRedis()
    webhook = SimpleNamespace(notify_job_completion=AsyncMock())

    monkeypatch.setattr(api, "AsyncWebCrawler", lambda **kw: _FakeCrawler(captured, **kw))
    monkeypatch.setattr(api, "validate_url_destination", lambda _u: None)
    monkeypatch.setattr(api, "WebhookDeliveryService", lambda _c: webhook)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    import utils as _utils
    monkeypatch.setattr(_utils, "load_config", lambda: config)
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve(server_temp))

    asyncio.run(
        api.process_llm_extraction(
            redis,
            config,
            "llm_test",
            "https://example.com",
            "extract",
            temperature=caller_temp,
        )
    )
    return captured


@pytest.mark.parametrize(
    ("caller_temp", "server_temp", "expected_temp"),
    [
        (0.5, 0.7, 0.5),   # caller override routed through job_extra_args
        (None, 0.7, 0.7),  # operator config routed through job_extra_args
        (0.0, 0.7, 0.0),   # explicit 0.0 honoured, not collapsed
    ],
)
def test_extraction_temperature_routed_through_job_extra_args(
    monkeypatch, caller_temp, server_temp, expected_temp
):
    captured = _run_extraction(monkeypatch, caller_temp, server_temp)

    assert captured["extra_args"]["temperature"] == expected_temp
    # LLMConfig.temperature mirrors the resolved value too (no falsy collapse).
    assert captured["llm_temperature"] == expected_temp
    # Existing job overrides must remain intact (no regression in shape).
    assert captured["extra_args"]["timeout"] == 300
    assert captured["extra_args"]["max_tokens"] == 12288


def test_extraction_no_temperature_key_when_neither_sets(monkeypatch):
    """When neither caller nor operator set a temperature, no `temperature` key
    is forced into job_extra_args, so the sync helper's hard-coded 0.01 default
    stays in effect (provider/litellm default otherwise). Regression guard for
    the "only add when not None" branch."""
    captured = _run_extraction(monkeypatch, caller_temp=None, server_temp=None)

    assert "temperature" not in captured["extra_args"]
    assert captured["llm_temperature"] is None


# ---------------------------------------------------------------------------
# Sync extraction helper: prove the temperature reaches litellm.completion
# only via extra_args (LLMConfig.temperature is NOT a channel the helper reads).
# ---------------------------------------------------------------------------

def test_extraction_sync_helper_receives_temperature_from_extra_args(monkeypatch):
    captured = {}

    async def fake_acompletion(**_kwargs):  # not used here; guard against accidental call
        raise AssertionError("sync path must use litellm.completion, not acompletion")

    def fake_completion(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            usage=SimpleNamespace(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
                completion_tokens_details=None,
                prompt_tokens_details=None,
            ),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='<blocks>[{"index": 0, "content": "x"}]</blocks>'
                    ),
                    finish_reason="stop",
                )
            ],
        )

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="openai/test-model",
            api_token="test-only",
            temperature=0.9,  # deliberately set on LLMConfig; the helper must NOT read this
            base_url="https://gateway.example/v1",
        ),
        instruction="extract main content",
        extra_args={"temperature": 0.5},  # the channel the helper actually merges
        apply_chunking=False,
    )

    blocks = strategy.extract("https://example.com", 0, "<html><body>x</body></html>")

    assert isinstance(blocks, list)
    # The temperature reaching litellm is the extra_args value (0.5), NOT the
    # LLMConfig.temperature (0.9). This is the central contract the fix relies on.
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["api_key"] == "test-only"
    assert captured["kwargs"]["base_url"] == "https://gateway.example/v1"
