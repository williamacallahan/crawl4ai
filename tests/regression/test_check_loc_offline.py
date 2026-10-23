"""LOC ownership uses pinned tree names, never a network Git fetch."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def check_loc():
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_loc.py"
    spec = importlib.util.spec_from_file_location("check_loc_offline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_loc_passes_using_local_tree_without_fetch(check_loc, tmp_path, monkeypatch, capsys):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "--quiet")
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy/upstream.py").write_text("# upstream\n" * 501)
    git("add", "deploy/upstream.py")
    tree = git("write-tree")
    (tmp_path / "deploy/own.py").write_text("# fork\n")
    git("add", "deploy/own.py")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    monkeypatch.setattr(check_loc, "ROOT", tmp_path)
    monkeypatch.setattr(check_loc, "BASELINE", baseline)
    monkeypatch.setattr(check_loc, "UPSTREAM_SHA", tree, raising=False)
    real_run = subprocess.run

    def no_network(command, **kwargs):
        assert command[0] != "gh", "local tree must not call GitHub"
        assert "fetch" not in command, "LOC checks must not fetch objects"
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_network)
    assert check_loc.main() == 0
    assert capsys.readouterr().out.strip() == "every own Python file is within its line cap (500)"


def test_check_loc_uses_read_only_trees_api_when_object_missing(check_loc, monkeypatch):
    def missing_object(*args):
        raise subprocess.CalledProcessError(128, ["git"], stderr="missing tree object")

    monkeypatch.setattr(check_loc, "_git", missing_object)

    def api(command, **kwargs):
        assert command == [
            "gh", "api",
            f"repos/unclecode/crawl4ai/git/trees/{check_loc.UPSTREAM_SHA}?recursive=1",
        ]
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
            "truncated": False,
            "tree": [
                {"type": "blob", "path": "tests/upstream.py"},
                {"type": "tree", "path": "tests/subdir"},
                {"type": "blob", "path": "docs/example.py"},
            ],
        }))

    monkeypatch.setattr(subprocess, "run", api)
    assert check_loc.upstream_paths() == {"tests/upstream.py"}


@pytest.mark.parametrize("api_failure", ["offline", "truncated"])
def test_check_loc_names_both_failed_tree_sources(check_loc, monkeypatch, api_failure):
    def missing_object(*args):
        raise subprocess.CalledProcessError(128, ["git"], stderr="missing tree object")

    def failed_api(command, **kwargs):
        if api_failure == "offline":
            raise subprocess.CalledProcessError(1, command, stderr="GitHub offline")
        return subprocess.CompletedProcess(command, 0, stdout='{"truncated": true, "tree": []}')

    monkeypatch.setattr(check_loc, "_git", missing_object)
    monkeypatch.setattr(subprocess, "run", failed_api)
    with pytest.raises(RuntimeError) as raised:
        check_loc.upstream_paths()
    message = str(raised.value)
    assert "missing tree object" in message
    assert ("GitHub offline" if api_failure == "offline" else "truncated") in message
    assert check_loc.UPSTREAM_SHA in message
