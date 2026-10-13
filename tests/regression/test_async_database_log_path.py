"""AsyncDatabaseManager must write its DB log to ``~/.crawl4ai/crawler_db.log``.

Regression coverage for the double-nested DB-log path bug introduced in
merge commit b6af94cb. That merge combined two parallel branches: one that
added the ``AsyncLogger`` keyed off a *raw* ``base_directory = Path.home()``
(``log_file=os.path.join(base_directory, ".crawl4ai", "crawler_db.log")``) and
one that turned ``base_directory``/``DB_PATH`` into the *already-nested*
``os.path.join(os.getenv("CRAWL4_AI_BASE_DIRECTORY", Path.home()), ".crawl4ai")``.
The merge resolution pre-nested ``base_directory`` but left the
``log_file`` join untouched, so the redundant ``.crawl4ai`` segment became a
double-nest: DB logs landed at ``~/.crawl4ai/.crawl4ai/crawler_db.log`` and a
stray ``.crawl4ai`` directory was created at import time.

The fix drops the redundant ``.crawl4ai`` segment so the logger path is
built from ``base_directory`` the same way ``DB_PATH`` is (single nest),
matching the sibling module ``async_webcrawler.py`` (which builds
``~/.crawl4ai/crawler.log`` from a raw ``base_directory``).

The first two tests assert against the canonical ``async_db_manager``
singleton's resolved path (no env manipulation). The on-disk tests load an
isolated copy of ``async_database.py`` via
``importlib.util.spec_from_file_location`` (the pattern established in
``test_async_logger_stderr.py``) so ``CRAWL4_AI_BASE_DIRECTORY`` is
re-evaluated for the copy without disturbing the canonical module's
singleton or other tests in the session.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_async_database_log_path.py
"""

import importlib.util
import os
from pathlib import Path

import crawl4ai.async_database as _canonical  # noqa: F401

_REPO_ROOT = Path(__file__).parents[2]


def _load_isolated_async_database(base_directory_env: str):
    """Load a fresh, isolated copy of ``async_database.py`` under a controlled
    ``CRAWL4_AI_BASE_DIRECTORY`` and return the resulting module object.

    The copy is registered under a unique name whose parent package is
    ``crawl4ai`` so its relative imports (``.models``, ``.utils``,
    ``.async_logger``) resolve against the already-imported ``crawl4ai``
    package. It is never placed in ``sys.modules`` under the canonical name,
    so the real ``async_db_manager`` singleton used by the rest of the test
    session is left untouched.
    """
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(base_directory_env)
    spec = importlib.util.spec_from_file_location(
        "crawl4ai._isolated_async_database_test_copy",
        _REPO_ROOT / "crawl4ai" / "async_database.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_db_logger_path_matches_db_path_construction():
    """``log_file`` and ``db_path`` use the same parent (``base_directory``).

    Pins the intra-file consistency the bug violated: six lines apart,
    ``DB_PATH = os.path.join(base_directory, "crawl4ai.db")`` (single nest)
    and the logger ``log_file`` must *also* be a single nest of
    ``base_directory``. Before the fix the logger re-appended ``.crawl4ai``.
    """
    base_directory = _canonical.base_directory
    db_manager = _canonical.async_db_manager

    assert db_manager.db_path == os.path.join(base_directory, "crawl4ai.db")
    assert db_manager.logger.log_file == os.path.join(base_directory, "crawler_db.log")
    assert os.path.dirname(db_manager.db_path) == base_directory
    assert os.path.dirname(db_manager.logger.log_file) == base_directory


def test_db_logger_path_contains_crawl4ai_exactly_once():
    """The resolved log path must contain exactly one ``.crawl4ai`` segment.

    The canonical (single-nested) form is ``<base>/.crawl4ai/crawler_db.log``;
    the buggy form was ``<base>/.crawl4ai/.crawl4ai/crawler_db.log``. Counting
    occurrences of the ``.crawl4ai`` path component is a direct, intent-free
    check that the double-nest does not regress.
    """
    parts = _canonical.async_db_manager.logger.log_file.split(os.sep)
    assert parts.count(".crawl4ai") == 1, (
        f"expected exactly one '.crawl4ai' segment, got path="
        f"{_canonical.async_db_manager.logger.log_file}"
    )
    assert parts[-1] == "crawler_db.log"


def test_no_stray_nested_crawl4ai_dir_created(tmp_path, monkeypatch):
    """Import-time side effect must NOT create ``<base>/.crawl4ai/.crawl4ai/``.

    Before the fix, ``AsyncLogger.__init__`` proactively
    ``os.makedirs``-d the parent of the (double-nested) ``log_file``, so a
    stray ``.crawl4ai`` directory appeared alongside the legitimate content
    directories at import time. Mirrors the "strictly stronger" guard in
    ``test_setup_home_directory.py::test_no_files_outside_crawl4ai_tree``.
    """
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))
    mod = _load_isolated_async_database(tmp_path)

    base = os.path.join(str(tmp_path), ".crawl4ai")
    assert os.path.isdir(base)
    nested = os.path.join(base, ".crawl4ai")
    assert not os.path.isdir(nested), f"stray double-nested directory created: {nested}"
    entries = sorted(os.listdir(base))
    assert ".crawl4ai" not in entries, (
        f"stray '.crawl4ai' entry present in base tree: {entries}"
    )


def test_db_logger_writes_to_single_nested_path(tmp_path, monkeypatch):
    """A log line actually lands at ``<base>/.crawl4ai/crawler_db.log``.

    Exercises the real ``AsyncLogger`` file sink (``_write_to_file`` is called
    unconditionally from ``_log`` regardless of ``verbose=False``) against the
    resolved path, and confirms the double-nested path is never created.
    """
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", str(tmp_path))
    mod = _load_isolated_async_database(tmp_path)

    expected_log = os.path.join(str(tmp_path), ".crawl4ai", "crawler_db.log")
    nested_log = os.path.join(str(tmp_path), ".crawl4ai", ".crawl4ai", "crawler_db.log")

    mod.async_db_manager.logger.info("regression probe", tag="TEST")

    assert os.path.isfile(expected_log), (
        f"single-nested log file was not created at {expected_log}"
    )
    content = Path(expected_log).read_text(encoding="utf-8")
    assert "regression probe" in content
    assert not os.path.exists(nested_log), (
        f"double-nested log path unexpectedly created at {nested_log}"
    )
