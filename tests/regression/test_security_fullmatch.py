"""Docker header validators must reject names with a trailing newline."""

from pathlib import Path

import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _docker_module_path(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "deploy" / "docker")
    )


class TestHeaderNameTrailingNewline:
    def test_hook_registry_rejects_trailing_newline(self):
        from hook_registry import SetHeadersParams

        with pytest.raises(ValidationError, match="invalid header name"):
            SetHeadersParams(headers={"X-Injected\n": "v"})

    def test_hook_registry_accepts_normal_name(self):
        from hook_registry import SetHeadersParams

        params = SetHeadersParams(headers={"X-Injected": "v"})
        assert params.headers == {"X-Injected": "v"}

    def test_webhook_rejects_trailing_newline(self):
        from webhook import sanitize_webhook_headers

        with pytest.raises(ValueError, match="invalid webhook header name"):
            sanitize_webhook_headers({"X-Custom\n": "v"})

    def test_webhook_accepts_normal_name(self):
        from webhook import sanitize_webhook_headers

        clean = sanitize_webhook_headers({"X-Custom": "v"})
        assert clean == {"X-Custom": "v"}
