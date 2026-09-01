"""Regression coverage for the Docker client hooks payload contract.

The Docker server's ``HookConfig`` schema (since 0.9.0) accepts only a
declarative ``hooks`` list; the legacy ``code``/``timeout`` fields were
removed (replaced by fixed actions to prevent RCE) - see
``deploy/docker/MIGRATION.md``. These tests pin the client/server hook
contract so a future server schema change (or a client regression to the
legacy shape) is caught here instead of causing hooks to be silently
ignored again. No browser, network, or Redis required.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# deploy/docker holds ``schemas`` (imported as a bare module by the server
# tests); put it on sys.path so we can cross-validate the client payload
# against the real ``HookConfig`` without starting the server.
_DEPLOY_DOCKER = Path(__file__).resolve().parents[1] / "deploy" / "docker"
if str(_DEPLOY_DOCKER) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DOCKER))

from crawl4ai.docker_client import Crawl4aiDockerClient  # noqa: E402
from schemas import HookConfig  # noqa: E402


@pytest.fixture
def client():
    """A client whose HTTP transport never touches the network.

    ``_prepare_request`` is synchronous; the constructor's ``httpx.AsyncClient``
    is harmless without I/O and is closed to avoid resource warnings.
    """
    c = Crawl4aiDockerClient(base_url="http://localhost:0", verbose=False)
    try:
        yield c
    finally:
        asyncio.run(c.close())


def _spec(action="block_resources", **params):
    return {"action": action, "params": params or {}}


# --- server contract pin (would have caught this bug when the server changed) ---
def test_server_hook_config_accepts_only_the_declarative_hooks_field():
    # If the server re-adds ``code``/``timeout`` or renames ``hooks``, the client
    # payload format must be revisited. Mirrors the server-side contract test
    # in deploy/docker/tests/test_crawl_hook_lifecycle.py.
    assert set(HookConfig.model_json_schema()["properties"]) == {"hooks"}


# --- _build_hooks_payload: declarative input ---
def test_declarative_wrapper_dict_is_forwarded_unchanged():
    spec = _spec("block_resources", resource_types=["image", "font"])
    assert Crawl4aiDockerClient._build_hooks_payload({"hooks": [spec]}) == {"hooks": [spec]}


def test_declarative_bare_list_is_wrapped_under_the_hooks_key():
    spec = _spec("scroll_to_bottom", max_steps=10, delay_ms=500)
    assert Crawl4aiDockerClient._build_hooks_payload([spec]) == {"hooks": [spec]}


def test_declarative_payload_carries_no_legacy_code_or_timeout_keys():
    # Guards against regressing the bug: the client must never emit the shape
    # the server removed in 0.9.0.
    payload = Crawl4aiDockerClient._build_hooks_payload(
        {"hooks": [_spec("wait_for_timeout", timeout_ms=500)]}
    )
    assert set(payload) == {"hooks"}
    assert "code" not in payload and "timeout" not in payload


# --- _build_hooks_payload: legacy code-based input must not fail silently ---
def test_legacy_callable_hooks_emit_deprecation_warning_and_return_none():
    async def my_hook(page, context, **kwargs):
        return page

    with pytest.warns(FutureWarning, match="Code-based hooks"):
        assert Crawl4aiDockerClient._build_hooks_payload(
            {"on_page_context_created": my_hook}
        ) is None


def test_legacy_string_hooks_emit_deprecation_warning_and_return_none():
    legacy = {"before_goto": "async def hook(page, context, url, **kwargs):\n    return page\n"}
    with pytest.warns(FutureWarning, match="Code-based hooks"):
        assert Crawl4aiDockerClient._build_hooks_payload(legacy) is None


# --- _prepare_request ---
def test_no_hooks_omits_the_hooks_key(client):
    assert "hooks" not in client._prepare_request(["https://example.com"])


def test_declarative_request_body_validates_against_server_schema(client):
    # Cross-contract: the body the client builds must round-trip through the
    # server's HookConfig with the specs populated (not silently dropped).
    spec = _spec("block_resources", resource_types=["image", "font"])
    data = client._prepare_request(["https://example.com"], hooks={"hooks": [spec]})

    config = HookConfig.model_validate(data["hooks"])
    assert len(config.hooks) == 1
    assert config.hooks[0].action == "block_resources"
    assert config.hooks[0].params == {"resource_types": ["image", "font"]}


# --- public crawl() wire contract ---
@pytest.mark.asyncio
async def test_crawl_sends_declarative_hooks_payload(client, monkeypatch):
    captured = {}

    async def fake_check_server(_self):
        return None

    async def fake_request(_self, method, endpoint, **kwargs):
        captured["json"] = kwargs.get("json")
        captured["timeout"] = kwargs.get("timeout")

        class FakeResponse:
            def json(self):
                return {"success": True, "results": []}

        return FakeResponse()

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    monkeypatch.setattr(Crawl4aiDockerClient, "_request", fake_request)

    spec = _spec("block_resources", resource_types=["image"])
    await client.crawl(
        ["https://example.com"],
        hooks={"hooks": [spec]},
        hooks_timeout=42,
    )

    assert captured["json"]["hooks"] == {"hooks": [spec]}
    assert "code" not in captured["json"]["hooks"]
    assert "timeout" not in captured["json"]["hooks"]
    # hooks_timeout is now the HTTP request timeout (the per-hook timeout
    # field was removed from the server schema).
    assert captured["timeout"] == 42


@pytest.mark.asyncio
async def test_crawl_with_legacy_hooks_warns_and_sends_no_hooks(client, monkeypatch):
    captured = {}

    async def fake_check_server(_self):
        return None

    async def fake_request(_self, method, endpoint, **kwargs):
        captured["json"] = kwargs.get("json")

        class FakeResponse:
            def json(self):
                return {"success": True, "results": []}

        return FakeResponse()

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    monkeypatch.setattr(Crawl4aiDockerClient, "_request", fake_request)

    async def my_hook(page, context, **kwargs):
        return page

    with pytest.warns(FutureWarning, match="Code-based hooks"):
        await client.crawl(
            ["https://example.com"],
            hooks={"on_page_context_created": my_hook},
        )

    # Legacy hooks are dropped (not sent as a payload the server would ignore).
    assert "hooks" not in captured["json"]
