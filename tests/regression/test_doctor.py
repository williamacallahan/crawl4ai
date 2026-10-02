"""Tests for crawl4ai.install.run_doctor (the ``crawl4ai-doctor`` health check).

Pins two contracts:

1. The root-context ``--no-sandbox`` behavior. ``BrowserManager`` treats
   ``--no-sandbox`` as an operator policy the caller must supply via
   ``BrowserConfig.extra_args`` when the runtime cannot use Chromium's sandbox;
   running as root (e.g. the Docker build before the ``USER appuser`` switch) is
   the canonical case where the sandbox is unavailable.
2. The ``doctor()`` entry-point exit code, which scripted callers (notably the
   ``Dockerfile`` ``&&`` chain) gate on: ``0`` on pass, non-zero on fail.

These mock-based tests run in the default CI lane (``-m "not network and not
browser"``); no real browser launches and no network is touched.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crawl4ai.install import doctor, run_doctor


@pytest.fixture
def captured_extra_args(monkeypatch):
    """Record the raw ``extra_args`` kwarg passed to ``BrowserConfig``.

    The root ``conftest.py`` auto-appends ``--no-sandbox`` to every
    ``BrowserConfig`` when the test process runs as root. To assert what
    ``run_doctor`` itself sets (independent of that global patch), the kwarg
    is captured before delegating to the real ``__init__``; the conftest's
    later mutation of the instance list cannot corrupt that copy.
    """
    from crawl4ai.async_configs import BrowserConfig

    real_init = BrowserConfig.__init__
    captured = []

    def spy(self, *args, **kwargs):
        raw = kwargs.get("extra_args")
        captured.append(list(raw) if raw is not None else None)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(BrowserConfig, "__init__", spy)
    return captured


def _mock_crawler(markdown="# content"):
    """A mock usable as ``async with AsyncWebCrawler(...) as c`` returning markdown."""
    instance = AsyncMock()
    instance.__aenter__.return_value = instance
    instance.__aexit__.return_value = None
    result = MagicMock()
    result.markdown = markdown
    instance.arun = AsyncMock(return_value=result)
    return instance


class TestNoSandboxGating:
    @pytest.mark.asyncio
    async def test_adds_no_sandbox_when_root(self, captured_extra_args, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        with patch("crawl4ai.async_webcrawler.AsyncWebCrawler", return_value=_mock_crawler()):
            await run_doctor()
        assert captured_extra_args == [["--no-sandbox"]]

    @pytest.mark.asyncio
    async def test_omits_no_sandbox_when_non_root(self, captured_extra_args, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        with patch("crawl4ai.async_webcrawler.AsyncWebCrawler", return_value=_mock_crawler()):
            await run_doctor()
        assert captured_extra_args == [[]]
        assert "--no-sandbox" not in captured_extra_args[0]

    @pytest.mark.asyncio
    async def test_empty_extra_args_when_geteuid_unavailable(self, captured_extra_args, monkeypatch):
        monkeypatch.delattr(os, "geteuid", raising=False)
        with patch("crawl4ai.async_webcrawler.AsyncWebCrawler", return_value=_mock_crawler()):
            await run_doctor()
        assert captured_extra_args == [[]]
        assert "--no-sandbox" not in captured_extra_args[0]


def _doctor_invocation():
    """Return the command used to invoke the real ``crawl4ai-doctor`` entry point.

    Prefers the installed console script (``<venv>/bin/crawl4ai-doctor``) so the
    exact setuptools-generated shim (``from crawl4ai.install import doctor;
    sys.exit(doctor())``) is exercised; falls back to a ``python -c`` line that
    reproduces that shim verbatim when the script file is absent.
    """
    script = Path(sys.executable).parent / "crawl4ai-doctor"
    if script.exists():
        return [str(script)]
    return [
        sys.executable,
        "-c",
        "import sys; from crawl4ai.install import doctor; sys.exit(doctor())",
    ]


def _run_doctor_stub(tmp_path, return_value):
    """Invoke the real ``doctor()`` entry point with ``run_doctor`` stubbed.

    A ``sitecustomize.py`` written into ``tmp_path`` (placed on ``PYTHONPATH``)
    patches ``crawl4ai.install.run_doctor`` with an async function returning
    ``return_value`` before the entry point runs, so no real browser launches
    and no network is touched. Returns the ``subprocess.CompletedProcess``.
    """
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import crawl4ai.install as _inst\n"
        "async def _fake_run_doctor():\n"
        f"    return {bool(return_value)!r}\n"
        "_inst.run_doctor = _fake_run_doctor\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        _doctor_invocation(),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestDoctorExitCode:
    """The ``doctor()`` wrapper must propagate the health-check result to the
    process exit code (regression test for the unconditional ``sys.exit(0)``).
    """

    def test_doctor_exits_zero_when_run_doctor_returns_true(self, monkeypatch):
        async def fake_run_doctor():
            return True

        monkeypatch.setattr("crawl4ai.install.run_doctor", fake_run_doctor)
        with pytest.raises(SystemExit) as exc_info:
            doctor()
        assert exc_info.value.code == 0

    def test_doctor_exits_nonzero_when_run_doctor_returns_false(self, monkeypatch):
        async def fake_run_doctor():
            return False

        monkeypatch.setattr("crawl4ai.install.run_doctor", fake_run_doctor)
        with pytest.raises(SystemExit) as exc_info:
            doctor()
        assert exc_info.value.code != 0
        assert exc_info.value.code == 1

    def test_console_script_exits_zero_on_success(self, tmp_path):
        """The installed console script must exit 0 when the health check passes."""
        result = _run_doctor_stub(tmp_path, return_value=True)
        assert result.returncode == 0, (
            f"expected exit 0 on success, got {result.returncode}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

    def test_console_script_exits_nonzero_on_failure(self, tmp_path):
        """The installed console script must exit non-zero when the health check fails.

        This is the regression for the reported bug: a forced failure previously
        produced exit code 0 from the real entry point, defeating scripted
        callers that gate on the exit code (e.g. the Dockerfile ``&&`` chain).
        """
        result = _run_doctor_stub(tmp_path, return_value=False)
        assert result.returncode != 0, (
            f"expected non-zero exit on failure, got {result.returncode}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        assert result.returncode == 1
