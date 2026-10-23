"""Cap this fork's own Python files under deploy/ and tests/ at 500 lines.

A file is the fork's own when the pinned upstream tree has no file at that path, so
upstream code and upstream merges never trip the cap and no hand-kept list of
owned paths exists. An own file already over the cap is listed in
loc-baseline.txt at its length when the cap was introduced: it may shrink but
never grow, and it leaves the list once it is back under the cap.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

LIMIT = 500
ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("loc-baseline.txt")
# Upstream 0.9.3 integrated by 16dfdaa376; update with the next upstream merge.
UPSTREAM_SHA: Final[str] = "862f6bccb9c063f49b9d42701baa0eea17a4993f"
UPSTREAM_URL: Final[str] = "https://github.com/unclecode/crawl4ai.git"
SCOPE = ("deploy/*.py", "tests/*.py")


def _git(*args: str) -> list[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    ).stdout.split("\0")


def _failure_reason(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        return (error.stderr or error.stdout or str(error)).strip()
    return str(error)


def _pinned_tree() -> set[str]:
    return set(_git(
        "ls-tree", "-r", "-z", "--name-only", UPSTREAM_SHA, "--", "deploy", "tests"
    ))


def upstream_paths() -> set[str]:
    """Pinned upstream file names: local objects, then a tree-only fetch, then the API.

    CI's shallow checkout lacks the pinned tree but can fetch it; read-only agent
    lanes cannot write fetched objects but can read the GitHub trees API.
    """
    try:
        return _pinned_tree()
    except (OSError, subprocess.CalledProcessError) as error:
        local_reason = _failure_reason(error)
    try:
        _git("fetch", "--quiet", "--depth=1", "--filter=blob:none", UPSTREAM_URL, UPSTREAM_SHA)
        return _pinned_tree()
    except (OSError, subprocess.CalledProcessError) as error:
        local_reason += f"; fetch: {_failure_reason(error)}"
    try:
        response = subprocess.run(
            ["gh", "api", f"repos/unclecode/crawl4ai/git/trees/{UPSTREAM_SHA}?recursive=1"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        tree = json.loads(response.stdout)
        if tree.get("truncated") is not False:
            raise ValueError("GitHub tree is truncated or missing its completeness flag")
        return {
            entry["path"] for entry in tree["tree"]
            if entry["type"] == "blob" and entry["path"].startswith(("deploy/", "tests/"))
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Cannot read pinned upstream tree {UPSTREAM_SHA}; "
            f"local git: {local_reason}; GitHub trees API: {_failure_reason(error)}"
        ) from error


def main() -> int:
    caps = {
        path: int(lines)
        for lines, path in (row.split() for row in BASELINE.read_text().splitlines())
    }
    try:
        upstream = upstream_paths()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    errors = []
    for path in filter(None, _git("ls-files", "-z", "--", *SCOPE)):
        if path in upstream:
            continue
        lines = (ROOT / path).read_bytes().count(b"\n")
        cap = caps.pop(path, LIMIT)
        if lines > cap:
            errors.append(f"{path}: {lines} lines exceeds its cap of {cap}")
        elif cap > LIMIT and lines <= LIMIT:
            errors.append(f"{path}: now {lines} lines; remove it from {BASELINE.name}")
    errors += [f"{path}: listed in {BASELINE.name} but not an own tracked file" for path in caps]
    print("\n".join(errors) or f"every own Python file is within its line cap ({LIMIT})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
