import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parent.parent
INSTALLER = DOCKER_DIR / "install_swarm_observer.sh"


def test_observer_release_imports_from_exactly_the_installed_files(tmp_path):
    # The installer's list is the only owner of the release file set. A module
    # the observer imports but the installer omits fails here, not on the host.
    installed = re.findall(r'install -m \d+ "\$source_dir/([\w.-]+)"', INSTALLER.read_text())
    assert "swarm_observer.py" in installed
    for name in installed:
        shutil.copy(DOCKER_DIR / name, tmp_path / name)

    imported = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import swarm_observer, swarm_observer_server",
         str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
