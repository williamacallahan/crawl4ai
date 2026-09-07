"""Behavioral regression tests for deploy/docker/utils.get_container_memory_percent.

Stubs the cgroup fs reads (Path.exists/Path.read_text) and psutil so the
cgroup v1/v2/"max"/fallback branches of get_container_memory_percent run
deterministically without a real container.

The unlimited cgroup v2 case (memory.max == "max") is the reported bug: the
original code called int(limit_path.read_text()) before checking the "max"
token, so int("max") raised ValueError and the bare except returned host
psutil.virtual_memory().percent instead of (container_usage / host_total) * 100
as the in-source comment promised. These tests pin the documented behavior so
that regression cannot return.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import utils  # noqa: E402

_V2_USAGE = "/sys/fs/cgroup/memory.current"
_V2_LIMIT = "/sys/fs/cgroup/memory.max"
_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_INTERCEPT = {_V2_USAGE, _V2_LIMIT, _V1_USAGE, _V1_LIMIT}


def _patch_fs(monkeypatch, files, exists_paths=None):
    real_exists = Path.exists
    real_read_text = Path.read_text
    exist_set = set(exists_paths) if exists_paths is not None else set(files)

    def exists(self, *args, **kwargs):
        if str(self) in _INTERCEPT:
            return str(self) in exist_set
        return real_exists(self, *args, **kwargs)

    def read_text(self, *args, **kwargs):
        if str(self) in _INTERCEPT:
            try:
                return files[str(self)]
            except KeyError as error:
                raise FileNotFoundError(str(self)) from error
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "read_text", read_text)


def _patch_vm(monkeypatch, total, percent, used=None):
    if used is None:
        used = int(total * percent / 100)
    vm = SimpleNamespace(total=total, percent=percent, used=used)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)


def test_cgroup_v2_capped_returns_usage_over_limit(monkeypatch):
    _patch_fs(monkeypatch, {_V2_USAGE: "800", _V2_LIMIT: "1000"})
    assert utils.get_container_memory_percent() == pytest.approx(80.0)


def test_cgroup_v2_unlimited_uses_host_total_not_host_pressure(monkeypatch):
    host_total = 12 * 1024**3
    container_usage = 1024**3
    expected = (container_usage / host_total) * 100
    _patch_fs(monkeypatch, {_V2_USAGE: str(container_usage), _V2_LIMIT: "max\n"})
    _patch_vm(monkeypatch, total=host_total, percent=50.0)
    result = utils.get_container_memory_percent()
    assert result == pytest.approx(expected)
    assert result != pytest.approx(50.0)


def test_cgroup_v2_unlimited_strips_whitespace(monkeypatch):
    _patch_fs(monkeypatch, {_V2_USAGE: "0\n", _V2_LIMIT: "  max  \n"})
    _patch_vm(monkeypatch, total=8 * 1024**3, percent=99.0)
    assert utils.get_container_memory_percent() == pytest.approx(0.0)


def test_cgroup_v1_huge_sentinel_uses_host_total_not_host_pressure(monkeypatch):
    host_total = 8 * 1024**3
    container_usage = 1024**3
    expected = (container_usage / host_total) * 100
    _patch_fs(
        monkeypatch,
        {_V1_USAGE: str(container_usage), _V1_LIMIT: str(2**63)},
    )
    _patch_vm(monkeypatch, total=host_total, percent=50.0)
    result = utils.get_container_memory_percent()
    assert result == pytest.approx(expected)
    assert result != pytest.approx(50.0)


def test_non_container_falls_back_to_host_percent(monkeypatch):
    _patch_fs(monkeypatch, {})
    _patch_vm(monkeypatch, total=16 * 1024**3, percent=42.5)
    assert utils.get_container_memory_percent() == pytest.approx(42.5)

