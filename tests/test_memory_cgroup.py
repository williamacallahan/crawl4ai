from pathlib import Path
from unittest.mock import patch

from crawl4ai.utils import get_cgroup_memory_usage_percent, get_true_memory_usage_percent


def _read(values, path):
    try:
        return values[str(path)]
    except KeyError as error:
        raise FileNotFoundError(path) from error


def test_cgroup_v2_memory_usage_precedes_host_memory():
    values = {
        "/sys/fs/cgroup/memory.current": "800",
        "/sys/fs/cgroup/memory.max": "1000",
    }

    with patch.object(Path, "read_text", autospec=True) as read_text:
        read_text.side_effect = lambda path: _read(values, path)

        assert get_cgroup_memory_usage_percent() == 80.0
        assert get_true_memory_usage_percent() == 80.0


def test_unlimited_cgroup_falls_through():
    values = {
        "/sys/fs/cgroup/memory.current": "800",
        "/sys/fs/cgroup/memory.max": "max",
    }

    with patch.object(Path, "read_text", autospec=True) as read_text:
        read_text.side_effect = lambda path: _read(values, path)

        assert get_cgroup_memory_usage_percent() is None
