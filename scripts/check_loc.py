"""Cap this fork's own Python files under deploy/ and tests/ at 500 lines.

A file is the fork's own when upstream's main tree has no file at that path, so
upstream code and upstream merges never trip the cap and no hand-kept list of
owned paths exists. An own file already over the cap is listed in
loc-baseline.txt at its length when the cap was introduced: it may shrink but
never grow, and it leaves the list once it is back under the cap.
"""

import subprocess
import sys
from pathlib import Path

LIMIT = 500
ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("loc-baseline.txt")
UPSTREAM = "https://github.com/unclecode/crawl4ai.git"
SCOPE = ("deploy/*.py", "tests/*.py")


def _git(*args: str) -> list[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.split("\0")


def main() -> int:
    caps = {
        path: int(lines)
        for lines, path in (row.split() for row in BASELINE.read_text().splitlines())
    }
    # Tree names only: the blob filter keeps this fetch to about two seconds.
    _git("fetch", "--quiet", "--depth=1", "--filter=blob:none", UPSTREAM, "main")
    upstream = set(_git("ls-tree", "-r", "-z", "--name-only", "FETCH_HEAD", "--", "deploy", "tests"))
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
