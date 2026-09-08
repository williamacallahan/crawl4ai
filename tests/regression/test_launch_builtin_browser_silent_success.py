"""Regression tests for ``launch_builtin_browser`` success/failure contract.

``BrowserProfiler.launch_builtin_browser`` polls ``/json/version`` up to 10
times to "verify the browser is responsive" but previously discarded the
result: when the browser never answered (or died during launch) it still wrote
``browser_config.json`` and returned the CDP URL, so ``crwl browser start``
printed a green success panel for a dead/unreachable browser. These tests pin
the docstring contract — ``str: CDP URL for the browser, or None if launch
failed`` — for both failure gate and the happy path, with no real browser.
"""

import asyncio
import json
import os

import aiohttp
import pytest

from crawl4ai.async_configs import BrowserConfig
from crawl4ai.browser_profiler import BrowserProfiler


class FakeProc:
    """Stand-in for ``subprocess.Popen`` exposing only ``pid`` and ``poll()``."""

    def __init__(self, pid, poll_return=None):
        self.pid = pid
        self._poll = poll_return

    def poll(self):
        return self._poll


class FakeManagedBrowser:
    """``ManagedBrowser`` double: records construction, owns a fake process."""

    proc_factory = staticmethod(lambda: FakeProc(os.getpid(), poll_return=None))
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.browser_process = FakeManagedBrowser.proc_factory()
        self.cleanup_calls = 0
        FakeManagedBrowser.instances.append(self)

    async def start(self):
        return None

    async def cleanup(self):
        self.cleanup_calls += 1


class FakeResp:
    """Async-context-manager response double with a configurable status."""

    def __init__(self, status, payload=None):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


class FakeSession:
    """``aiohttp.ClientSession`` double driven by a per-call response factory."""

    def __init__(self, response_factory):
        self._factory = response_factory
        self.get_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, *args, **kwargs):
        self.get_calls += 1
        return self._factory(self.get_calls)

    async def close(self):
        pass


def _install_fast_sleep(monkeypatch):
    """Replace ``asyncio.sleep`` seen by ``browser_profiler`` with a 0-delay."""
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr("crawl4ai.browser_profiler.asyncio.sleep", fast_sleep)


def _make_profiler(tmp_path, monkeypatch, proc_factory):
    """Build a profiler isolated in ``tmp_path`` with the given process double."""
    FakeManagedBrowser.proc_factory = staticmethod(proc_factory)
    FakeManagedBrowser.instances = []
    monkeypatch.setattr("crawl4ai.browser_profiler.ManagedBrowser", FakeManagedBrowser)

    profiler = BrowserProfiler()
    profiler.builtin_browser_dir = str(tmp_path)
    profiler.builtin_config_file = str(tmp_path / "browser_config.json")

    errors = []
    successes = []

    def record_error(message, tag="ERROR", **kwargs):
        errors.append(message)

    def record_success(message, tag="SUCCESS", **kwargs):
        successes.append(message)

    monkeypatch.setattr(profiler.logger, "error", record_error)
    monkeypatch.setattr(profiler.logger, "success", record_success)
    _install_fast_sleep(monkeypatch)
    return profiler, errors, successes


def _patch_client_session(monkeypatch, response_factory):
    captured = {"session": None}

    def factory(*args, **kwargs):
        session = FakeSession(response_factory)
        captured["session"] = session
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    return captured


@pytest.mark.asyncio
async def test_returns_none_when_cdp_never_answers(tmp_path, monkeypatch):
    """10x non-200 /json/version -> launch fails, no config file is written."""
    captured = _patch_client_session(monkeypatch, lambda n: FakeResp(404))
    profiler, errors, successes = _make_profiler(
        tmp_path, monkeypatch, lambda: FakeProc(os.getpid(), poll_return=None)
    )

    result = await profiler.launch_builtin_browser(
        browser_type="chromium", debugging_port=9455, headless=True
    )

    assert result is None
    assert captured["session"].get_calls == 10
    assert not os.path.exists(profiler.builtin_config_file)
    assert any("/json/version" in m for m in errors)
    assert successes == []
    assert FakeManagedBrowser.instances[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_returns_none_when_version_response_is_not_cdp(tmp_path, monkeypatch):
    """A 200 response without the CDP WebSocket endpoint is not launch success."""
    _patch_client_session(
        monkeypatch,
        lambda n: FakeResp(200, {"Browser": "other", "webSocketDebuggerUrl": ""}),
    )
    profiler, errors, successes = _make_profiler(
        tmp_path, monkeypatch, lambda: FakeProc(os.getpid(), poll_return=None)
    )

    result = await profiler.launch_builtin_browser(
        browser_type="chromium", debugging_port=9455, headless=True
    )

    assert result is None
    assert not os.path.exists(profiler.builtin_config_file)
    assert any("CDP" in message for message in errors)
    assert successes == []
    assert FakeManagedBrowser.instances[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_returns_none_when_process_exits_after_cdp_answered(tmp_path, monkeypatch):
    """CDP answered but the process is dead -> launch fails (second gate)."""
    payload = {"Browser": "Chrome/1.0", "webSocketDebuggerUrl": "ws://localhost"}
    _patch_client_session(monkeypatch, lambda n: FakeResp(200, payload))
    profiler, errors, successes = _make_profiler(
        tmp_path, monkeypatch, lambda: FakeProc(os.getpid(), poll_return=0)
    )

    result = await profiler.launch_builtin_browser(
        browser_type="chromium", debugging_port=9455, headless=True
    )

    assert result is None
    assert not os.path.exists(profiler.builtin_config_file)
    assert any("process exited" in m for m in errors)
    assert successes == []
    assert FakeManagedBrowser.instances[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_happy_path_returns_url_and_writes_config(tmp_path, monkeypatch):
    """CDP answers and process alive -> URL returned, config persisted."""
    payload = {"Browser": "Chrome/1.0", "webSocketDebuggerUrl": "ws://localhost"}
    captured = _patch_client_session(monkeypatch, lambda n: FakeResp(200, payload))
    profiler, errors, successes = _make_profiler(
        tmp_path, monkeypatch, lambda: FakeProc(os.getpid(), poll_return=None)
    )

    result = await profiler.launch_builtin_browser(
        browser_type="chromium", debugging_port=9455, headless=True
    )

    assert result == "http://localhost:9455"
    assert captured["session"].get_calls == 1
    assert any("launched" in m for m in successes)
    assert errors == []
    assert FakeManagedBrowser.instances[0].cleanup_calls == 0
    assert FakeManagedBrowser.instances[0].browser_process is None

    with open(profiler.builtin_config_file) as f:
        saved = json.load(f)
    assert saved["cdp_url"] == "http://localhost:9455"
    assert saved["config"] == payload
