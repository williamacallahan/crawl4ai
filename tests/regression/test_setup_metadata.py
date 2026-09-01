"""Regression coverage for side-effect-free package metadata evaluation."""

import os
import shutil
import subprocess
from pathlib import Path


def test_setup_build_does_not_create_crawl4ai_cache(tmp_path: Path) -> None:
    env = os.environ | {"CRAWL4_AI_BASE_DIRECTORY": str(tmp_path)}
    repo_root = Path(__file__).resolve().parents[2]
    uv = shutil.which("uv")
    assert uv is not None

    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path / "dist")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list((tmp_path / "dist").glob("*.whl"))
    assert not (tmp_path / ".crawl4ai").exists()
