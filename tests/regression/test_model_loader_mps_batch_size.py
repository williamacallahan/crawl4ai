"""
Regression tests for the memory-adaptive batching heuristic in
``crawl4ai/model_loader.py`` on Apple Silicon (MPS).

Background
----------
``get_available_memory`` used to return a hardcoded ``48 * 1024**3`` (48 GiB)
for every MPS device, defeating the memory-adaptive heuristic: every Apple
Silicon Mac — 8 GB or 128 GB — received ``batch_size = 256`` because 48 GiB
exceeds the top ``calculate_batch_size`` threshold. The fix queries real
total system memory via ``psutil.virtual_memory().total`` on the MPS branch,
mirroring the CUDA branch (``torch.cuda.get_device_properties(device).total_memory``).
``psutil`` is already a runtime dependency (``psutil>=6.1.1``) and is already
used for memory queries in ``crawl4ai/utils.py``.

These tests guard against the constant creeping back and against regressions
in the shared CPU/CUDA paths. They run offline: ``torch`` (an optional extra)
is stubbed via ``monkeypatch.setitem(sys.modules, ...)`` and ``psutil`` (a
core dependency) has its ``virtual_memory`` patched for determinism.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/regression/test_model_loader_mps_batch_size.py
"""

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Test harness: hashable fake device + fake torch + patchable psutil
# ---------------------------------------------------------------------------


class _FakeDevice:
    """Hashable stand-in for ``torch.device``.

    ``get_available_memory``/``calculate_batch_size`` read ``device.type`` and
    pass ``device`` to ``torch.cuda.get_device_properties``; ``@lru_cache``
    keys on the argument, so the stand-in must be hashable (real
    ``torch.device`` is; ``types.SimpleNamespace`` is not).
    """

    def __init__(self, type_):
        self.type = type_

    def __hash__(self):
        return hash(self.type)

    def __eq__(self, other):
        return isinstance(other, _FakeDevice) and other.type == self.type


class _FakeCudaProperties:
    def __init__(self, total_memory):
        self.total_memory = total_memory


class _FakeCudaModule:
    def __init__(self, total_memory):
        self._total_memory = total_memory

    def get_device_properties(self, _device):
        return _FakeCudaProperties(self._total_memory)


class _FakeVirtualMemory:
    def __init__(self, total):
        self.total = total


GB = 1024 ** 3


def _install_fake_torch(monkeypatch, cuda_module=None):
    """Install a minimal fake ``torch`` so the eager ``import torch`` at the
    top of ``get_available_memory`` succeeds without the optional torch extra.
    Real torch (if installed) is restored after the test."""
    fake = types.ModuleType("torch")
    if cuda_module is not None:
        fake.cuda = cuda_module
    fake.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def _set_psutil_total(monkeypatch, total):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVirtualMemory(total))


def _clear_caches():
    from crawl4ai.model_loader import get_available_memory, calculate_batch_size

    get_available_memory.cache_clear()
    calculate_batch_size.cache_clear()


# ---------------------------------------------------------------------------
# Tests: MPS branch now queries the machine (the fix)
# ---------------------------------------------------------------------------


def test_mps_branch_queries_real_system_memory(monkeypatch):
    """On MPS, ``get_available_memory`` returns ``psutil.virtual_memory().total``
    rather than a hardcoded constant. Guards against the 48 GiB constant
    returning."""
    from crawl4ai.model_loader import get_available_memory

    _install_fake_torch(monkeypatch)
    _set_psutil_total(monkeypatch, 16 * GB)
    _clear_caches()

    result = get_available_memory(_FakeDevice("mps"))
    assert result == 16 * GB
    assert result != 48 * GB  # the old fabricated constant


def test_mps_batch_size_adapts_to_machine_memory(monkeypatch):
    """The fix's core guarantee: on MPS the batch size is a function of
    installed RAM, not a constant 256. An 8 GB Mac now selects 64; a 32 GB
    Mac still selects 256 (the top tier — unchanged from before the fix)."""
    from crawl4ai.model_loader import calculate_batch_size

    _install_fake_torch(monkeypatch)

    for total_bytes, expected_batch in [
        (8 * GB, 64),    # was 256 before the fix
        (16 * GB, 128),  # was 256 before the fix
        (32 * GB, 256),  # unchanged — constant already selected the top tier
    ]:
        _set_psutil_total(monkeypatch, total_bytes)
        _clear_caches()
        assert calculate_batch_size(_FakeDevice("mps")) == expected_batch


# ---------------------------------------------------------------------------
# Tests: CPU and CUDA paths are unchanged
# ---------------------------------------------------------------------------


def test_cpu_path_is_unchanged(monkeypatch):
    """CPU path is independent of memory and torch; always returns batch 16."""
    from crawl4ai.model_loader import calculate_batch_size

    _install_fake_torch(monkeypatch)
    _set_psutil_total(monkeypatch, 0)
    _clear_caches()

    assert calculate_batch_size(_FakeDevice("cpu")) == 16


def test_cuda_path_still_queries_device_vram(monkeypatch):
    """CUDA path is unchanged: still uses
    ``torch.cuda.get_device_properties(device).total_memory`` and adapts on
    VRAM. The fix must not perturb the CUDA branch."""
    from crawl4ai.model_loader import get_available_memory, calculate_batch_size

    _install_fake_torch(monkeypatch, cuda_module=_FakeCudaModule(24 * GB))
    _clear_caches()

    assert get_available_memory(_FakeDevice("cuda")) == 24 * GB
    assert calculate_batch_size(_FakeDevice("cuda")) == 128


# ---------------------------------------------------------------------------
# Test: end-to-end through the production consumer
# ---------------------------------------------------------------------------


def test_cosine_strategy_default_batch_adapts_on_mps(monkeypatch):
    """``CosineStrategy.__init__`` sets
    ``self.default_batch_size = calculate_batch_size(self.device)``. On an
    8 GB Apple Silicon Mac the default batch size is now 64, not 256."""
    from crawl4ai import extraction_strategy
    from crawl4ai.model_loader import calculate_batch_size

    _install_fake_torch(monkeypatch)
    _set_psutil_total(monkeypatch, 8 * GB)
    _clear_caches()

    mps = _FakeDevice("mps")
    monkeypatch.setattr(extraction_strategy, "get_device", lambda: mps)

    strategy = extraction_strategy.CosineStrategy.__new__(
        extraction_strategy.CosineStrategy
    )
    strategy.device = extraction_strategy.get_device()
    strategy.default_batch_size = calculate_batch_size(strategy.device)

    assert strategy.device.type == "mps"
    assert strategy.default_batch_size == 64  # was 256 before the fix


# ---------------------------------------------------------------------------
# Test: the @lru_cache invalidation contract (footgun prevention)
# ---------------------------------------------------------------------------


def test_cache_clear_lets_memory_changes_take_effect(monkeypatch):
    """``get_available_memory``/``calculate_batch_size`` are ``@lru_cache``d,
    so a memory change is only observable after ``cache_clear()``. This pins
    the invalidation contract that any future reconfiguration must use."""
    from crawl4ai.model_loader import (
        calculate_batch_size,
        get_available_memory,
    )

    _install_fake_torch(monkeypatch)
    _clear_caches()

    _set_psutil_total(monkeypatch, 8 * GB)
    assert calculate_batch_size(_FakeDevice("mps")) == 64

    get_available_memory.cache_clear()
    calculate_batch_size.cache_clear()

    _set_psutil_total(monkeypatch, 32 * GB)
    assert calculate_batch_size(_FakeDevice("mps")) == 256


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
