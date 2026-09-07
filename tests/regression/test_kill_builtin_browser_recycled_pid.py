"""Regression test for the PID-recycle / wrong-process-kill defect in
`BrowserProfiler.kill_builtin_browser`.

The bug: `BrowserProfiler._is_browser_running`, `get_builtin_browser_info`,
and `kill_builtin_browser` trusted the persisted `pid` field after a bare
existence check (`os.kill(pid, 0)` on Unix; `tasklist /FI "PID eq <pid>"` on
Windows). When the original Chromium died abnormally and the kernel reused
its PID for an unrelated same-UID process, `crwl browser stop` / `restart`
would SIGTERM/SIGKILL that unrelated process and report success.

The fix:
- `_is_browser_running` now also verifies the live process's cmdline
  contains the recorded `--remote-debugging-port` and `--user-data-dir`
  flags (psutil-based; the same pattern `ManagedBrowser.start` already
  used on Windows).
- `get_builtin_browser_info` rejects recycled PIDs (returns None) AND
  unlinks the stale config so a later PID reuse cannot strike.
- `kill_builtin_browser` re-verifies identity immediately before signaling,
  refuses to signal on mismatch, unlinks the stale config, and uses
  identity checking during the SIGTERM poll so a PID recycled during the
  wait does not trigger a SIGKILL of an unrelated process.
- `launch_builtin_browser` short-circuit passes `browser_info`, so a
  recycled-PID config no longer falsely short-circuits with a stale
  `cdp_url`.

These tests confirm:
  (1) the identity check rejects recycled PIDs,
  (2) the kill path no longer signals the wrong process,
  (3) the happy path still kills the matching browser,
  (4) the `crwl browser` CLI commands inherit the fix end-to-end, and
  (5) the SIGKILL escalation path still targets the matching browser.
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
from click.testing import CliRunner

from crawl4ai.browser_profiler import BrowserProfiler
from crawl4ai.cli import cli


# ───────────────────────────────────────────────────────────────────────────
# Fake-process helpers
# ───────────────────────────────────────────────────────────────────────────

# A short-lived marker child that:
#   - writes its pid to MARKER_READY on start,
#   - on SIGTERM writes its pid to MARKER_KILLED and exits (unless
#     MARKER_IGNORE_SIGTERM=1, in which case SIGTERM is ignored).
# SIGKILL is uncatchable; for the SIGKILL-escalation test we only assert
# proc.wait() returns and the killed marker is absent.
_MARKER_SCRIPT = r"""
import os, signal, sys, time
ready = os.environ['MARKER_READY']
killed = os.environ['MARKER_KILLED']
open(ready, 'w').write(str(os.getpid()))
if os.environ.get('MARKER_IGNORE_SIGTERM') == '1':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    def on_term(*_):
        try:
            open(killed, 'w').write(str(os.getpid()))
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, signal.SIG_IGN)
# Loop until a signal arrives. SIGKILL is delivered by the kernel.
while True:
    time.sleep(3600)
"""


def _wait_for_file(path, timeout=5.0):
    """Poll until `path` exists; return its contents."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for marker: {path}")


def _spawn_marker_proc(tmp_path, argv_extra, prefix, ignore_sigterm=False):
    """Spawn the marker child with `argv_extra` appended to its cmdline.

    The cmdline tokens in `argv_extra` are what psutil's
    `Process(pid).cmdline()` will see; the identity check looks for
    `--remote-debugging-port=...` and `--user-data-dir=...` there.
    """
    ready = tmp_path / f"{prefix}_ready.marker"
    killed = tmp_path / f"{prefix}_killed.marker"
    env = os.environ.copy()
    env["MARKER_READY"] = str(ready)
    env["MARKER_KILLED"] = str(killed)
    if ignore_sigterm:
        env["MARKER_IGNORE_SIGTERM"] = "1"
    argv = [sys.executable, "-c", _MARKER_SCRIPT] + list(argv_extra)
    proc = subprocess.Popen(
        argv,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_file(ready)
    return proc, ready, killed


def _spawn_fake_browser(tmp_path, port, user_data_dir, prefix="browser"):
    """Spawn a process whose cmdline contains the recorded browser tokens."""
    return _spawn_marker_proc(
        tmp_path,
        [f"--remote-debugging-port={port}", f"--user-data-dir={user_data_dir}"],
        prefix,
    )


def _spawn_unrelated_victim(tmp_path, prefix="victim"):
    """Spawn an unrelated process whose cmdline lacks the browser tokens."""
    return _spawn_marker_proc(tmp_path, ["unrelated-arg"], prefix)


def _cleanup_proc(proc):
    """Best-effort cleanup of a subprocess; safe to call on a dead proc."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        pass


def _write_config(tmp_path, pid, port, user_data_dir, **extra):
    """Write a browser_config.json under tmp_path and return its path."""
    config_path = tmp_path / "browser_config.json"
    info = {
        "pid": pid,
        "cdp_url": f"http://localhost:{port}",
        "user_data_dir": str(user_data_dir),
        "browser_type": "chromium",
        "debugging_port": port,
        "start_time": time.time(),
        "config": None,
    }
    info.update(extra)
    config_path.write_text(json.dumps(info))
    return config_path


@pytest.fixture
def profiler(tmp_path):
    """BrowserProfiler with builtin-browser state redirected to tmp_path."""
    p = BrowserProfiler()
    p.builtin_browser_dir = str(tmp_path)
    p.builtin_config_file = str(tmp_path / "browser_config.json")
    return p


# ───────────────────────────────────────────────────────────────────────────
# Unit tests: _is_browser_running / _verify_browser_identity
# ───────────────────────────────────────────────────────────────────────────


class TestIsBrowserRunningIdentity:
    """Identity-based checks in `_is_browser_running`."""

    def test_dead_pid_returns_false_without_browser_info(self, profiler):
        # 2_000_000 is effectively never assigned on a default Linux kernel.
        assert profiler._is_browser_running(2_000_000) is False

    def test_dead_pid_returns_false_with_browser_info(self, profiler, tmp_path):
        info = {"debugging_port": 9455, "user_data_dir": str(tmp_path / "u")}
        assert profiler._is_browser_running(2_000_000, info) is False

    def test_none_pid_returns_false(self, profiler, tmp_path):
        info = {"debugging_port": 9455, "user_data_dir": str(tmp_path / "u")}
        assert profiler._is_browser_running(None) is False
        assert profiler._is_browser_running(0) is False
        assert profiler._is_browser_running(None, info) is False
        assert profiler._is_browser_running(0, info) is False

    def test_live_unrelated_pid_returns_false_with_identity(self, profiler, tmp_path):
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            info = {
                "debugging_port": 9455,
                "user_data_dir": str(tmp_path / "no_such_profile"),
            }
            assert profiler._is_browser_running(proc.pid, info) is False
        finally:
            _cleanup_proc(proc)

    def test_live_matching_browser_returns_true_with_identity(
        self, profiler, tmp_path
    ):
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, _killed = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            info = {
                "debugging_port": 9455,
                "user_data_dir": str(user_data_dir),
            }
            assert profiler._is_browser_running(proc.pid, info) is True
        finally:
            _cleanup_proc(proc)

    def test_partial_cmdline_match_returns_false(self, profiler, tmp_path):
        # Only one of the two tokens must NOT count as a match.
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, _killed = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            # Port matches, dir does not — must reject.
            info = {
                "debugging_port": 9455,
                "user_data_dir": str(tmp_path / "different_dir"),
            }
            assert profiler._is_browser_running(proc.pid, info) is False
        finally:
            _cleanup_proc(proc)

    def test_port_token_only_mismatch_returns_false(self, profiler, tmp_path):
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, _killed = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            # Dir matches, port does not — must reject.
            info = {
                "debugging_port": 9999,
                "user_data_dir": str(user_data_dir),
            }
            assert profiler._is_browser_running(proc.pid, info) is False
        finally:
            _cleanup_proc(proc)

    def test_missing_identifying_fields_falls_back_to_existence_only(
        self, profiler, tmp_path
    ):
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            info = {}  # no debugging_port / user_data_dir
            assert profiler._is_browser_running(proc.pid, info) is True
        finally:
            _cleanup_proc(proc)

    def test_no_browser_info_falls_back_to_existence_only(self, profiler, tmp_path):
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            # No browser_info at all — pure existence check (backward compat).
            assert profiler._is_browser_running(proc.pid) is True
        finally:
            _cleanup_proc(proc)


class TestVerifyBrowserIdentityFallback:
    """When psutil is unavailable, identity verification degrades gracefully."""

    def test_returns_true_when_psutil_is_none(
        self, profiler, tmp_path, monkeypatch
    ):
        # Simulate psutil being unavailable — must NOT block the kill path
        # (degrades to previously documented existence-only behavior).
        monkeypatch.setattr("crawl4ai.browser_profiler.psutil", None)
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            info = {
                "debugging_port": 9455,
                "user_data_dir": str(tmp_path / "user_data"),
            }
            assert profiler._verify_browser_identity(proc.pid, info) is True
            assert profiler._is_browser_running(proc.pid, info) is True
        finally:
            _cleanup_proc(proc)


# ───────────────────────────────────────────────────────────────────────────
# get_builtin_browser_info: identity gate + stale-config cleanup
# ───────────────────────────────────────────────────────────────────────────


class TestGetBuiltinBrowserInfo:
    def test_dead_pid_unlinks_stale_config(self, profiler, tmp_path):
        config = _write_config(
            tmp_path, pid=2_000_000, port=9455,
            user_data_dir=str(tmp_path / "user_data"),
        )
        assert config.exists()
        assert profiler.get_builtin_browser_info() is None
        # Stale config removed — the dangerous window is closed.
        assert not config.exists()

    def test_recycled_pid_unlinks_stale_config(self, profiler, tmp_path):
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            config = _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(tmp_path / "user_data"),
            )
            assert profiler.get_builtin_browser_info() is None
            # The unrelated victim is NOT killed; the stale config is gone.
            assert not config.exists()
            assert proc.poll() is None
        finally:
            _cleanup_proc(proc)

    def test_matching_browser_returns_info(self, profiler, tmp_path):
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, _killed = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            config = _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(user_data_dir),
            )
            info = profiler.get_builtin_browser_info()
            assert info is not None
            assert info["pid"] == proc.pid
            assert info["debugging_port"] == 9455
            assert config.exists()  # not unlinked while browser is alive
            assert proc.poll() is None
        finally:
            _cleanup_proc(proc)

    def test_missing_config_file_returns_none(self, profiler, tmp_path):
        # No config file at all.
        assert profiler.get_builtin_browser_info() is None

    def test_corrupt_config_returns_none(self, profiler, tmp_path):
        config = tmp_path / "browser_config.json"
        config.write_text("not valid json {{{")
        assert profiler.get_builtin_browser_info() is None


# ───────────────────────────────────────────────────────────────────────────
# kill_builtin_browser: the centre of the bug
# ───────────────────────────────────────────────────────────────────────────


class TestKillBuiltinBrowser:
    @pytest.mark.asyncio
    async def test_recycled_pid_refuses_to_kill_and_unlinks(
        self, profiler, tmp_path
    ):
        # Bug-report Test A reversed: an unrelated live PID is in the
        # config. Before the fix this returned True and SIGTERM'd the
        # victim. After the fix: returns False, victim untouched, stale
        # config removed.
        proc, _ready, killed_marker = _spawn_unrelated_victim(tmp_path)
        try:
            config = _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(tmp_path / "user_data"),
            )
            result = await profiler.kill_builtin_browser()
            assert result is False
            assert proc.poll() is None  # still alive
            assert not killed_marker.exists()  # did NOT receive SIGTERM
            assert not config.exists()
        finally:
            _cleanup_proc(proc)

    @pytest.mark.asyncio
    async def test_matching_browser_is_killed_and_unlinked(
        self, profiler, tmp_path
    ):
        # Happy path: the recorded PID still points at the browser we
        # launched. The kill path must work and clean up the config.
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, killed_marker = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            config = _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(user_data_dir),
            )
            result = await profiler.kill_builtin_browser()
            assert result is True
            assert killed_marker.exists()
            assert proc.wait(timeout=5) is not None
            assert not config.exists()
        finally:
            _cleanup_proc(proc)

    @pytest.mark.asyncio
    async def test_dead_pid_returns_false_and_unlinks(self, profiler, tmp_path):
        # Bug-report causal-chain link 1+2: PID is dead, so the profiler
        # used to return False but LEAVE the stale config on disk. After
        # the fix: returns False AND unlinks the stale config.
        config = _write_config(
            tmp_path, pid=2_000_000, port=9455,
            user_data_dir=str(tmp_path / "user_data"),
        )
        result = await profiler.kill_builtin_browser()
        assert result is False
        assert not config.exists()

    @pytest.mark.asyncio
    async def test_missing_config_returns_false(self, profiler, tmp_path):
        # No config; nothing to kill.
        result = await profiler.kill_builtin_browser()
        assert result is False

    @pytest.mark.asyncio
    async def test_sigkill_escalation_targets_matching_browser(
        self, profiler, tmp_path
    ):
        # When SIGTERM does not cause the matching browser to exit within
        # the polling window, kill_builtin_browser must escalate to
        # SIGKILL on the SAME (still-matching) PID. The poll loop uses
        # identity checking, so a PID recycled during the wait would NOT
        # trigger a SIGKILL of an unrelated process.
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, killed_marker = _spawn_marker_proc(
            tmp_path,
            [f"--remote-debugging-port=9455", f"--user-data-dir={user_data_dir}"],
            "stubborn",
            ignore_sigterm=True,
        )
        try:
            config = _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(user_data_dir),
            )
            result = await profiler.kill_builtin_browser()
            assert result is True
            # SIGTERM was ignored by the child; SIGKILL must have been
            # delivered. killed_marker will NOT be set (SIGKILL is
            # uncatchable) but proc.wait() will return.
            assert proc.wait(timeout=5) is not None
            assert not killed_marker.exists()
            assert not config.exists()
        finally:
            _cleanup_proc(proc)


# ───────────────────────────────────────────────────────────────────────────
# launch_builtin_browser short-circuit
# ───────────────────────────────────────────────────────────────────────────


class TestLaunchBuiltinBrowserShortCircuit:
    @pytest.mark.asyncio
    async def test_recycled_pid_does_not_short_circuit(
        self, profiler, tmp_path, monkeypatch
    ):
        # Bug-report Test E reversed: against a live-recycled PID the
        # launcher used to short-circuit and return a stale cdp_url. Now
        # `get_builtin_browser_info` returns None (rejecting the recycled
        # PID), so the launcher must proceed to spawn a new browser.
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(tmp_path / "user_data"),
            )

            captured = {}

            class FakeManagedBrowser:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.browser_process = None

                async def start(self):
                    return None

            monkeypatch.setattr(
                "crawl4ai.browser_profiler.ManagedBrowser", FakeManagedBrowser
            )

            result = await profiler.launch_builtin_browser(
                browser_type="chromium", debugging_port=9455, headless=True
            )
            # FakeManagedBrowser.browser_process is None → launch reports
            # failure (returns None). Crucially: it must NOT have
            # short-circuited by returning the stale recorded cdp_url.
            assert result is None
            # It must have attempted a fresh ManagedBrowser construction.
            assert "browser_config" in captured
            # The victim was not signaled by launch (only kill signals).
            assert proc.poll() is None
        finally:
            _cleanup_proc(proc)

    @pytest.mark.asyncio
    async def test_matching_browser_short_circuits_with_stale_url(
        self, profiler, tmp_path
    ):
        # Happy path: an actually-running, matching browser must still
        # short-circuit and return its recorded cdp_url.
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, _killed = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            _write_config(
                tmp_path, pid=proc.pid, port=9455,
                user_data_dir=str(user_data_dir),
                cdp_url="http://localhost:9455",
            )
            result = await profiler.launch_builtin_browser(
                browser_type="chromium", debugging_port=9455, headless=True
            )
            assert result == "http://localhost:9455"
            assert proc.poll() is None
        finally:
            _cleanup_proc(proc)


# ───────────────────────────────────────────────────────────────────────────
# End-to-end CLI tests — `crwl browser stop` / `restart` / `start`
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def home_env(monkeypatch, tmp_path):
    """Redirect CRAWL4_AI_BASE_DIRECTORY so every `BrowserProfiler()`
    constructed inside the CLI uses the test tree only."""
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))
    builtin_dir = tmp_path / ".crawl4ai" / "builtin-browser"
    return tmp_path, builtin_dir


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def write_builtin_config(home_env):
    """Closure that writes a config under the redirected home."""
    _tmp, builtin_dir = home_env

    def _write(pid, port=9455, user_data_dir=None, **extra):
        builtin_dir.mkdir(parents=True, exist_ok=True)
        cfg = builtin_dir / "browser_config.json"
        info = {
            "pid": pid,
            "cdp_url": f"http://localhost:{port}",
            "user_data_dir": user_data_dir or str(builtin_dir / "user_data"),
            "browser_type": "chromium",
            "debugging_port": port,
            "start_time": time.time(),
            "config": None,
        }
        info.update(extra)
        cfg.write_text(json.dumps(info))
        return cfg

    return _write


class TestCliBrowserStop:
    def test_stop_with_recycled_pid_does_not_kill_victim(
        self, home_env, runner, write_builtin_config, tmp_path
    ):
        # Bug-report Test C reversed: the recorded PID is live but
        # belongs to an unrelated victim. Before the fix the CLI printed
        # "stopped successfully" and SIGTERM'd the victim. After the fix:
        # the CLI refuses to kill, reports "not running", unlinks config.
        proc, _ready, killed_marker = _spawn_unrelated_victim(tmp_path)
        try:
            cfg = write_builtin_config(
                pid=proc.pid, port=9455,
                user_data_dir=str(tmp_path / "user_data"),
            )
            result = runner.invoke(cli, ["browser", "stop"])
            assert result.exit_code == 0, result.output
            assert "stopped successfully" not in result.output
            assert "no builtin browser is currently running" in result.output.lower()
            # The victim was NOT signaled.
            assert not killed_marker.exists()
            assert proc.poll() is None
            # The stale config is removed so the dangerous window is closed.
            assert not cfg.exists()
        finally:
            _cleanup_proc(proc)

    def test_stop_with_matching_browser_succeeds(
        self, home_env, runner, write_builtin_config, tmp_path
    ):
        # Happy path: real matching browser in config; CLI prints
        # "stopped successfully", sends SIGTERM, unlinks config.
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, killed_marker = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            cfg = write_builtin_config(
                pid=proc.pid, port=9455,
                user_data_dir=str(user_data_dir),
            )
            result = runner.invoke(cli, ["browser", "stop"])
            assert result.exit_code == 0, result.output
            assert "stopped successfully" in result.output
            assert killed_marker.exists()
            assert proc.wait(timeout=5) is not None
            assert not cfg.exists()
        finally:
            _cleanup_proc(proc)

    def test_stop_with_no_config_reports_not_running(self, home_env, runner):
        # No config file — CLI must report "not running" and exit 0.
        result = runner.invoke(cli, ["browser", "stop"])
        assert result.exit_code == 0, result.output
        assert "no builtin browser is currently running" in result.output.lower()


class TestCliBrowserRestart:
    def test_restart_with_recycled_pid_does_not_kill_victim(
        self, home_env, runner, write_builtin_config, tmp_path, monkeypatch
    ):
        # Recycled-PID case for restart: CLI must NOT kill the victim, and
        # MUST still attempt to launch a new browser. We mock
        # ManagedBrowser so no real Chromium spawns.
        proc, _ready, killed_marker = _spawn_unrelated_victim(tmp_path)
        try:
            cfg = write_builtin_config(
                pid=proc.pid, port=9455,
                user_data_dir=str(tmp_path / "user_data"),
            )

            class FakeManagedBrowser:
                def __init__(self, **kwargs):
                    self.browser_process = None

                async def start(self):
                    return None

            monkeypatch.setattr(
                "crawl4ai.browser_profiler.ManagedBrowser", FakeManagedBrowser
            )

            result = runner.invoke(cli, ["browser", "restart"])
            # restart proceeds to launch_builtin_browser (which reports
            # failure because FakeManagedBrowser has no browser_process)
            # → CLI exits 1. The crucial assertions are that the victim
            # was NOT signaled and the stale config was unlinked.
            assert result.exit_code == 1, result.output
            assert not killed_marker.exists()
            assert proc.poll() is None
            assert not cfg.exists()
        finally:
            _cleanup_proc(proc)


class TestCliBrowserStart:
    def test_start_with_live_recycled_config_proceeds_to_launch(
        self, home_env, runner, write_builtin_config, tmp_path, monkeypatch
    ):
        # Bug-report Test E reversed: against a live-recycled PID the
        # launcher used to short-circuit ("already running") and preserve
        # the stale config. After the fix, get_builtin_browser_info
        # returns None (rejects recycled PID) so `start` proceeds to
        # launch a new browser instead of falsely reporting "already
        # running".
        proc, _ready, _killed = _spawn_unrelated_victim(tmp_path)
        try:
            cfg = write_builtin_config(
                pid=proc.pid, port=9455,
                user_data_dir=str(tmp_path / "user_data"),
            )

            class FakeManagedBrowser:
                def __init__(self, **kwargs):
                    self.browser_process = None

                async def start(self):
                    return None

            monkeypatch.setattr(
                "crawl4ai.browser_profiler.ManagedBrowser", FakeManagedBrowser
            )

            result = runner.invoke(cli, ["browser", "start"])
            # FakeManagedBrowser exposes no browser_process → launch fails
            # → CLI exits 1. The crucial assertions are: it did NOT
            # short-circuit ("already running") and the stale config is
            # gone.
            assert result.exit_code == 1, result.output
            assert "already running" not in result.output.lower()
            assert not cfg.exists()
            assert proc.poll() is None
        finally:
            _cleanup_proc(proc)

    def test_start_with_matching_browser_short_circuits(
        self, home_env, runner, write_builtin_config, tmp_path
    ):
        # Happy path: a real matching browser in the config — `start`
        # must short-circuit and print "already running" with the cdp_url.
        user_data_dir = tmp_path / "user_data"
        user_data_dir.mkdir()
        proc, _ready, _killed = _spawn_fake_browser(
            tmp_path, port=9455, user_data_dir=str(user_data_dir)
        )
        try:
            write_builtin_config(
                pid=proc.pid, port=9455,
                user_data_dir=str(user_data_dir),
                cdp_url="http://localhost:9455",
            )
            result = runner.invoke(cli, ["browser", "start"])
            assert result.exit_code == 0, result.output
            assert "already running" in result.output.lower()
            assert "http://localhost:9455" in result.output
            assert proc.poll() is None
        finally:
            _cleanup_proc(proc)
