"""Rollout admission at the CLI and native service-spec boundaries."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import rollout_policy
from conftest import BASELINE, service_spec


def test_whitespace_delay_matches_shell_empty_fallback():
    for delay in ("   ", "\t", "  \t"):
        shell = subprocess.run(["bash", "-c", 'sleep "${CRAWL4AI_DRAIN_DELAY_SECONDS:-2}"'],
                               env={**os.environ, "CRAWL4AI_DRAIN_DELAY_SECONDS": delay},
                               capture_output=True, text=True, timeout=2)
        assert shell.returncode == 1
        with pytest.raises(ValueError, match="CRAWL4AI_DRAIN_DELAY_SECONDS"):
            rollout_policy._drain_budget_ns([f"CRAWL4AI_DRAIN_DELAY_SECONDS={delay}"])
    assert rollout_policy._drain_budget_ns([]) == 362_000_000_000
    assert rollout_policy._drain_budget_ns(["CRAWL4AI_DRAIN_DELAY_SECONDS="]) == 362_000_000_000
    # No whole-second truncation, and strtod's exponent/hex forms work.
    assert rollout_policy._drain_budget_ns(["CRAWL4AI_DRAIN_DELAY_SECONDS=2.5"]) == 362_500_000_000
    for delay in ("0e2", "0x0p0", " -0"):
        shell = subprocess.run(["sleep", delay], capture_output=True, timeout=2)
        assert shell.returncode == 0
        assert rollout_policy._drain_budget_ns([f"CRAWL4AI_DRAIN_DELAY_SECONDS={delay}"]) == 360_000_000_000


def test_no_rollout_subcommand_is_usage_error():
    command = Path(rollout_policy.__file__).with_name("verify_rollout.py")
    env = {key: value for key, value in os.environ.items() if not key.startswith("DOKPLOY_")}
    result = subprocess.run([sys.executable, str(command)], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "usage:" in result.stderr and "command" in result.stderr
    assert "rollout failure" not in result.stderr
    assert "Traceback" not in result.stderr

