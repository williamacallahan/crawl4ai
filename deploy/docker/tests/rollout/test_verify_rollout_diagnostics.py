"""Subprocess failures retain their diagnosis only in the runner's private log."""

import subprocess
import sys

import pytest

import verify_rollout as rollout


def test_subprocess_failure_keeps_private_output_and_fails_closed(monkeypatch, capsys):
    diagnosis = "daemon could not retrieve logs from private-node"

    def failed_deploy():
        subprocess.run(
            [sys.executable, "-c", "import sys; print('partial journal'); "
             f"print({diagnosis!r}, file=sys.stderr); sys.exit(17)"],
            check=True, capture_output=True, text=True,
        )

    monkeypatch.setattr(rollout, "deploy", failed_deploy)
    monkeypatch.setattr(sys, "argv", ["verify_rollout.py", "deploy"])
    with pytest.raises(subprocess.CalledProcessError) as failure:
        rollout.main()

    assert failure.value.returncode == 17
    private_log = capsys.readouterr().err
    assert diagnosis in private_log
    assert "partial journal" in private_log
    public_lines = [line for line in private_log.splitlines() if line.startswith("rollout failure: ")]
    assert len(public_lines) == 1
    assert "error=CalledProcessError" in public_lines[0]
    assert "exit=17" in public_lines[0]
    assert diagnosis not in public_lines[0]
    assert "partial journal" not in public_lines[0]
