"""
R3 LLM-broker tests: a request selects a provider BY NAME only; base_url and
api_token are always server-derived, so the configured provider key can never
be redirected to an attacker endpoint (the reported credential-exfil gadget).
"""

import json

import httpx
import pytest
from llm_broker import LLMProviderNotAllowed, LLMWorkload, resolve_llm

pytestmark = pytest.mark.posture

CONF = {"llm": {"provider": "openai/gpt-4o-mini"}}


class TestProviderAllowlist:
    def test_default_provider_allowed(self):
        out = resolve_llm(CONF, None, workload=LLMWorkload.PAGE_QUESTION)
        assert out["provider"] == "openai/gpt-4o-mini"

    def test_same_family_allowed(self):
        out = resolve_llm(CONF, "openai/gpt-4o", workload=LLMWorkload.PAGE_QUESTION)
        assert out["provider"] == "openai/gpt-4o"

    def test_server_bounds_slow_provider_attempts(self):
        out = resolve_llm(CONF, workload=LLMWorkload.PAGE_QUESTION)
        assert out["extra_args"] == {
            "timeout": 300,
            "num_retries": 0,
            "reasoning_effort": "low",
            "allowed_openai_params": ["reasoning_effort"],
            "max_tokens": 4096,
            "extra_headers": {
                "X-User-ID": "crawl4ai:page-question:live",
                "X-Tier": "production-a",
            },
        }

    def test_other_family_rejected(self):
        with pytest.raises(LLMProviderNotAllowed):
            resolve_llm(CONF, "anthropic/claude-3-opus", workload=LLMWorkload.PAGE_QUESTION)

    def test_model_override_keeps_its_own_reasoning_contract(self):
        conf = {"llm": {"provider": "openai/default-model", "reasoning_effort": "minimal"}}
        assert resolve_llm(conf, workload=LLMWorkload.PAGE_QUESTION)["extra_args"]["reasoning_effort"] == "minimal"
        assert resolve_llm(conf, "openai/default-model", workload=LLMWorkload.PAGE_QUESTION)["extra_args"]["reasoning_effort"] == "minimal"
        assert "reasoning_effort" not in resolve_llm(conf, "openai/another-model", workload=LLMWorkload.PAGE_QUESTION)["extra_args"]

    @pytest.mark.parametrize("workload", LLMWorkload)
    def test_every_workload_has_a_gateway_identity_and_tier(self, workload):
        headers = resolve_llm(CONF, workload=workload)["extra_args"]["extra_headers"]
        assert headers["X-User-ID"] == f"crawl4ai:{workload.value}:{'batch' if workload is LLMWorkload.JOB_EXTRACTION else 'live'}"
        assert headers["X-Tier"] == ("batch" if workload is LLMWorkload.JOB_EXTRACTION else "production-a")

    def test_explicit_allowlist_widens(self):
        conf = {"llm": {"provider": "openai/gpt-4o-mini",
                        "allowed_providers": ["anthropic/claude-3-opus"]}}
        assert resolve_llm(conf, "anthropic/claude-3-opus", workload=LLMWorkload.PAGE_QUESTION)["provider"].startswith("anthropic/")


@pytest.mark.asyncio
@pytest.mark.parametrize("workload", LLMWorkload)
@pytest.mark.parametrize("requested_provider", [None, "openai/default-model", "openai/another-model"])
async def test_reasoning_effort_reaches_serialized_http(monkeypatch, workload, requested_provider):
    # Use the installed cost map so dependency initialization stays offline too.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    from openai import AsyncOpenAI
    from crawl4ai.utils import aperform_completion_with_backoff

    monkeypatch.setattr(litellm, "drop_params", litellm.drop_params)
    conf = {"llm": {
        "provider": "openai/default-model",
        "reasoning_effort": "minimal",
        "api_key": "test-only-key",
    }}
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    resolved = resolve_llm(conf, requested_provider, workload=workload)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-completion",
            "object": "chat.completion",
            "created": 1,
            "model": resolved["provider"].removeprefix("openai/"),
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "filtered content",
            }, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        async with AsyncOpenAI(
            api_key=resolved["api_token"], base_url=resolved["base_url"],
            http_client=transport, max_retries=0,
        ) as client:
            response = await aperform_completion_with_backoff(
                provider=resolved["provider"], prompt_with_variables="Filter this page",
                api_token=resolved["api_token"], base_url=resolved["base_url"],
                max_attempts=1, extra_args={**resolved["extra_args"], "client": client},
            )

    assert response.choices[0].message.content == "filtered content"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://gateway.example/v1/chat/completions"
    payload = json.loads(request.content)
    assert payload["model"] == resolved["provider"].removeprefix("openai/")
    if requested_provider == "openai/another-model":
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == "minimal"
    assert "allowed_openai_params" not in payload
    assert request.headers["X-User-ID"] == f"crawl4ai:{workload.value}:{'batch' if workload is LLMWorkload.JOB_EXTRACTION else 'live'}"
    assert request.headers["X-Tier"] == ("batch" if workload is LLMWorkload.JOB_EXTRACTION else "production-a")


class TestBaseUrlIsServerOnly:
    def test_resolve_has_no_request_base_url_param(self):
        import inspect
        params = list(inspect.signature(resolve_llm).parameters)
        assert params == ["config", "requested_provider", "workload"]

    def test_base_url_is_canonical_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example")
        out = resolve_llm(CONF, "openai/gpt-4o-mini", workload=LLMWorkload.PAGE_QUESTION)
        assert out["base_url"] == "https://api.openai.example"

    def test_no_base_url_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        out = resolve_llm(CONF, "openai/gpt-4o-mini", workload=LLMWorkload.PAGE_QUESTION)
        assert out["base_url"] is None


class TestRequestSurfaceRemoved:
    def test_markdown_request_has_no_base_url(self):
        from schemas import MarkdownRequest
        assert "base_url" not in MarkdownRequest.model_fields

    def test_llm_job_payload_has_no_base_url(self):
        from job import LlmJobPayload
        assert "base_url" not in LlmJobPayload.model_fields

    def test_llm_endpoint_has_no_base_url_query(self, server_module):
        import inspect
        assert "base_url" not in inspect.signature(server_module.llm_endpoint).parameters

    def test_md_endpoint_rejects_disallowed_provider(self, stock_client, server_module):
        from auth import create_access_token
        h = {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'})}"}
        # LLM filter + a provider outside the allowed family -> 400 (not a 500,
        # and no LLM call to an attacker endpoint).
        r = stock_client.post(
            "/md",
            json={"url": "https://example.com", "f": "llm", "provider": "attacker/model",
                  "base_url": "http://169.254.169.254"},
            headers=h,
        )
        assert r.status_code == 400, r.status_code
