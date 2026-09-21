"""Regression tests for ``CrawlerMonitor`` memory-percentage units.

``CrawlerMonitor`` surfaces two memory-percentage annotations on its
dashboard. Both historically mixed a megabyte numerator with a bytes-valued
denominator (``psutil.virtual_memory().total``), making the displayed
values smaller by a factor of ``1024 * 1024`` and collapsing the live
``Memory: X.X%`` header and the ``Peak Mem: X.X%`` line / public
``get_summary()['peak_memory_percent']`` field to ``0.0%`` under all
realistic inputs.

The dispatcher's memory-pressure control flow was *not* affected — it reads
``get_true_memory_usage_percent()`` from ``crawl4ai/utils.py`` (already
consistent GB/GB units) and never reads the patched fields. These tests pin
the corrected, unit-consistent behavior at both sites and guard against an
over-eager fix that also re-scales the stored MB fields.
"""

import psutil

from crawl4ai.components.crawler_monitor import CrawlerMonitor, TerminalUI
from crawl4ai.models import CrawlStatus


def _make_monitor(n: int = 1) -> CrawlerMonitor:
    """A UI-less monitor. ``enable_ui=False`` avoids spawning the terminal
    thread so the tests stay deterministic and don't clobber the runner's
    tty."""
    return CrawlerMonitor(urls_total=n, enable_ui=False)


def _total_mb() -> float:
    return psutil.virtual_memory().total / (1024 * 1024)


class TestCrawlerMonitorMemoryPercent:
    def test_peak_memory_percent_uses_consistent_units(self):
        """A ``memory_usage`` equal to 25% of total RAM (mb, mirroring how
        ``async_dispatcher`` supplies the value) must store exactly ~25% in
        ``peak_memory_percent`` — not ``25 / 1024 / 1024`` ≈ ``2.4e-5`` (the
        buggy reading). Guards the ``update_task`` site against any future
        reintroduction of the MB/bytes unit mismatch."""
        total_mb = _total_mb()
        memory_usage_mb = total_mb / 4  # 25% of total RAM

        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            monitor.add_task("t1", "https://example.com")
            monitor.update_task("t1", status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(
                "t1", memory_usage=memory_usage_mb, peak_memory=memory_usage_mb
            )

            expected = (memory_usage_mb / total_mb) * 100
            assert abs(monitor.peak_memory_percent - expected) < 0.01
        finally:
            monitor.stop()

    def test_status_panel_renders_correct_memory_and_peak_percentages(self):
        """The header ``Memory:`` line (process RSS share of total RAM) and
        the ``Peak Mem:`` line (per-task RSS-delta share of total RAM) must
        render the corrected percentages, not ``0.0%``. Exercises both fixed
        sites through the real ``TerminalUI._create_status_panel``."""
        import psutil as _psutil
        from unittest.mock import MagicMock, patch

        vm_total_bytes = _psutil.virtual_memory().total
        total_mb = vm_total_bytes / (1024 * 1024)
        memory_usage_mb = total_mb / 4  # 25% of total RAM

        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            monitor.add_task("t1", "https://example.com")
            monitor.update_task("t1", status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(
                "t1", memory_usage=memory_usage_mb, peak_memory=memory_usage_mb
            )

            # Make the header deterministic: set process RSS to 50% of RAM so
            # the corrected value (50.0%) is unambiguously non-zero.
            fake_mem_info = MagicMock()
            fake_mem_info.rss = int(vm_total_bytes / 2)
            fake_process = MagicMock()
            fake_process.memory_info.return_value = fake_mem_info

            ui = TerminalUI()
            ui.monitor = monitor
            with patch(
                "crawl4ai.components.crawler_monitor.psutil.Process",
                return_value=fake_process,
            ):
                text = str(ui._create_status_panel().renderable)

            expected_memory = (fake_mem_info.rss / vm_total_bytes) * 100
            expected_peak = ((memory_usage_mb * 1024 * 1024) / vm_total_bytes) * 100

            assert f"Memory: {expected_memory:.1f}%" in text
            assert f"Peak Mem: {expected_peak:.1f}%" in text
        finally:
            monitor.stop()

    def test_task_stats_memory_usage_stored_in_mb_unchanged(self):
        """The fix only changes the *percentage* computation; ``task_stats``
        must still store the raw ``memory_usage``/``peak_memory`` values in MB
        (as the dispatcher supplies them and the task-details panel renders
        with ``:.1f`` MB). Guards against an over-eager fix that also
        re-scales the stored fields."""
        total_mb = _total_mb()
        memory_usage_mb = total_mb / 4

        monitor = _make_monitor(n=1)
        monitor.start()
        try:
            monitor.add_task("t1", "https://example.com")
            monitor.update_task("t1", status=CrawlStatus.IN_PROGRESS)
            monitor.update_task(
                "t1", memory_usage=memory_usage_mb, peak_memory=memory_usage_mb
            )

            stats = monitor.get_task_stats("t1")
            assert stats["memory_usage"] == memory_usage_mb
            assert stats["peak_memory"] == memory_usage_mb
        finally:
            monitor.stop()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
