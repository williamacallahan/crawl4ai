"""Tests for crawl4ai.install.setup_home_directory (the ``crawl4ai-setup`` seed step).

Pins the location of the seeded ``global.yml`` config file. The seed-write
must land *inside* the ``.crawl4ai/`` directory tree it is part of setting up,
at ``<base>/.crawl4ai/global.yml`` — matching the path every reader/writer in
the codebase uses (``crawl4ai/cli.py``, ``crawl4ai/cloud/cli.py``, docs, and
user-facing strings). Previously the config path was derived from
``crawl4ai_folder`` *before* the ``.crawl4ai`` segment was appended, so the
empty seed file was dropped at ``<base>/global.yml`` (home root or
``CRAWL4_AI_BASE_DIRECTORY`` root) outside the documented tree, and no file
was created at the canonical location.

These tests call ``setup_home_directory()`` in-process; no network and no
browser is touched, so they run in the default CI lane
(``-m "not network and not browser"``).
"""

from crawl4ai.install import setup_home_directory


def test_seeds_config_inside_crawl4ai_dir(tmp_path, monkeypatch):
    """Default-$HOME case: seed lands at ``~/.crawl4ai/global.yml``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CRAWL4_AI_BASE_DIRECTORY", raising=False)

    setup_home_directory()

    # The presence check below catches the bug: on the pre-fix code this file
    # did NOT exist (the seed went to <home>/global.yml).
    assert (tmp_path / ".crawl4ai" / "global.yml").exists()
    # No stray file should be left at the base root.
    assert not (tmp_path / "global.yml").exists()


def test_seeds_config_inside_crawl4ai_dir_with_base_directory(tmp_path, monkeypatch):
    """``CRAWL4_AI_BASE_DIRECTORY`` case: seed lands at ``<base>/.crawl4ai/global.yml``."""
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))

    setup_home_directory()

    assert (tmp_path / ".crawl4ai" / "global.yml").exists()
    assert not (tmp_path / "global.yml").exists()


def test_does_not_overwrite_existing_config(tmp_path, monkeypatch):
    """A pre-existing config at the canonical path must not be clobbered by the seed."""
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))
    base = tmp_path / ".crawl4ai"
    base.mkdir()
    (base / "global.yml").write_text("DEFAULT_LLM_PROVIDER: ollama/llama3.3\n")

    setup_home_directory()

    assert (base / "global.yml").read_text() == "DEFAULT_LLM_PROVIDER: ollama/llama3.3\n"


def test_no_files_outside_crawl4ai_tree(tmp_path, monkeypatch):
    """No regular file is created at the base root outside the ``.crawl4ai/`` tree.

    Regression guard against the original defect: a stray ``global.yml`` at
    ``<base>/`` outside the ``.crawl4ai/`` tree. This is strictly stronger than
    checking for ``global.yml`` alone — it future-proofs against any seed path
    regression that would drop files outside the tree.
    """
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))

    setup_home_directory()

    top_level = [p for p in tmp_path.iterdir() if p.is_file()]
    assert top_level == [], f"unexpected files at base root: {top_level}"
