"""Mock-safe regression coverage for persisted builtin-browser identity."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import crawl4ai.browser_profiler as browser_profiler_module
from crawl4ai.browser_profiler import BrowserProfiler


class _Process:
    def __init__(self, cmdline):
        self._cmdline = cmdline
        self.running = True
        self.terminate_calls = 0
        self.kill_calls = 0

    def cmdline(self):
        return self._cmdline

    def is_running(self):
        return self.running

    def terminate(self):
        self.terminate_calls += 1
        self.running = False

    def kill(self):
        self.kill_calls += 1
        self.running = False


@pytest.fixture
def profiler(tmp_path):
    profiler = BrowserProfiler()
    profiler.builtin_browser_dir = str(tmp_path)
    profiler.builtin_config_file = str(tmp_path / "browser_config.json")
    return profiler


def _browser_info(tmp_path, browser_type="chromium", **overrides):
    info = {
        "pid": 9455,
        "cdp_url": "http://localhost:9455",
        "debugging_port": 9455,
        "user_data_dir": str(tmp_path / "profile"),
        "browser_type": browser_type,
    }
    info.update(overrides)
    return info


def _patch_process(monkeypatch, process):
    monkeypatch.setattr(
        browser_profiler_module,
        "psutil",
        SimpleNamespace(Process=MagicMock(return_value=process), Error=Exception),
    )


def test_identity_accepts_exact_chromium_arguments(profiler, tmp_path, monkeypatch):
    info = _browser_info(tmp_path)
    process = _Process(
        [
            "chromium",
            "--remote-debugging-port=9455",
            f"--user-data-dir={info['user_data_dir']}",
        ]
    )
    _patch_process(monkeypatch, process)

    assert profiler._get_verified_browser_process(info["pid"], info) is not None


def test_identity_rejects_prefix_only_chromium_arguments(
    profiler, tmp_path, monkeypatch
):
    info = _browser_info(tmp_path)
    process = _Process(
        [
            "chromium",
            "--remote-debugging-port=94550",
            f"--user-data-dir={info['user_data_dir']}-other",
        ]
    )
    _patch_process(monkeypatch, process)

    assert profiler._get_verified_browser_process(info["pid"], info) is None


def test_identity_accepts_exact_firefox_arguments(profiler, tmp_path, monkeypatch):
    info = _browser_info(tmp_path, browser_type="firefox")
    process = _Process(
        [
            "firefox",
            "--remote-debugging-port",
            "9455",
            "--profile",
            info["user_data_dir"],
        ]
    )
    _patch_process(monkeypatch, process)

    assert profiler._get_verified_browser_process(info["pid"], info) is not None


def test_identity_rejects_missing_metadata(profiler, tmp_path, monkeypatch):
    info = _browser_info(tmp_path)
    info.pop("debugging_port")
    _patch_process(monkeypatch, _Process(["unrelated-process"]))

    assert profiler._get_verified_browser_process(info["pid"], info) is None


def test_stale_config_is_removed_without_signaling(profiler, tmp_path, monkeypatch):
    info = _browser_info(tmp_path)
    info.pop("debugging_port")
    config_file = tmp_path / "browser_config.json"
    config_file.write_text(json.dumps(info))
    _patch_process(monkeypatch, _Process(["unrelated-process"]))
    signal_process = MagicMock()
    monkeypatch.setattr(browser_profiler_module.os, "kill", signal_process)

    assert profiler.get_builtin_browser_info() is None
    assert not config_file.exists()
    signal_process.assert_not_called()


@pytest.mark.asyncio
async def test_kill_refuses_unverified_process(profiler, tmp_path, monkeypatch):
    info = _browser_info(tmp_path)
    info.pop("debugging_port")
    config_file = tmp_path / "browser_config.json"
    config_file.write_text(json.dumps(info))
    _patch_process(monkeypatch, _Process(["unrelated-process"]))
    signal_process = MagicMock()
    monkeypatch.setattr(browser_profiler_module.os, "kill", signal_process)

    assert await profiler.kill_builtin_browser() is False
    assert not config_file.exists()
    signal_process.assert_not_called()


@pytest.mark.asyncio
async def test_matching_process_is_terminated_and_config_is_removed(
    profiler, tmp_path, monkeypatch
):
    info = _browser_info(tmp_path)
    config_file = tmp_path / "browser_config.json"
    config_file.write_text(json.dumps(info))
    process = _Process(
        [
            "chromium",
            "--remote-debugging-port=9455",
            f"--user-data-dir={info['user_data_dir']}",
        ]
    )
    _patch_process(monkeypatch, process)

    assert await profiler.kill_builtin_browser() is True
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert not config_file.exists()


@pytest.mark.asyncio
async def test_matching_process_short_circuits_builtin_launch(
    profiler, tmp_path, monkeypatch
):
    info = _browser_info(tmp_path)
    (tmp_path / "browser_config.json").write_text(json.dumps(info))
    process = _Process(
        [
            "chromium",
            "--remote-debugging-port=9455",
            f"--user-data-dir={info['user_data_dir']}",
        ]
    )
    _patch_process(monkeypatch, process)

    assert await profiler.launch_builtin_browser(debugging_port=9455) == info["cdp_url"]
