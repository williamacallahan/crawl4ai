"""Cap this fork's own Python files under deploy/ and tests/ at 500 lines.

The upstream crawl4ai/ library is exempt. A file already over the cap is listed
in loc-baseline.txt at its length when the cap was introduced: it may shrink but
never grow, and it leaves the list once it is back under the cap.
"""

import subprocess
import sys
from pathlib import Path

LIMIT = 500
ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("loc-baseline.txt")


def main() -> int:
    caps = {
        path: int(lines)
        for lines, path in (row.split() for row in BASELINE.read_text().splitlines())
    }
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "deploy/*.py", "tests/*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    errors = []
    for path in filter(None, tracked):
        lines = (ROOT / path).read_bytes().count(b"\n")
        cap = caps.pop(path, LIMIT)
        if lines > cap:
            errors.append(f"{path}: {lines} lines exceeds its cap of {cap}")
        elif cap > LIMIT and lines <= LIMIT:
            errors.append(f"{path}: now {lines} lines; remove it from {BASELINE.name}")
    errors += [f"{path}: listed in {BASELINE.name} but no longer tracked" for path in caps]
    print("\n".join(errors) or f"every own Python file is within its line cap ({LIMIT})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
