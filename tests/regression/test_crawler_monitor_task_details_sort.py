"""Regression tests for ``TerminalUI._create_task_details_panel`` row selection.

Background
----------
``CrawlerMonitor`` registers every task up-front in URL/enqueue order via
``add_task`` (which appends to the plain ``self.stats`` dict, so iteration is
insertion-ordered). ``MemoryAdaptiveDispatcher.run_urls`` then drains the
``asyncio.PriorityQueue`` FIFO at uniform priority, so once the first
``max_session_permit`` (default 20) tasks complete, every currently
``IN_PROGRESS`` task sits at an insertion position >= 20.

The "Task Details" panel ranks rows by status priority (``IN_PROGRESS`` first,
then ``QUEUED``, then ``COMPLETED``/``FAILED`` by recency) and caps the table
at 20 rows. The original implementation sliced the insertion-ordered
``task_stats.items()`` down to the first 20 *before* sorting, so the sort
could never promote tasks registered after the 20th URL — including the
currently running tasks in the steady state. The panel's own SUMMARY row
(still computed over the full dict) reported a non-zero ``Active: N`` while
the per-task rows below it showed zero ``IN_PROGRESS`` tasks.

These tests pin the corrected behavior: sort the *full* task dict first, then
take the top ``display_count``. They exercise the exact bug scenario, the
status-priority ordering, the recency tiebreak within the terminal bucket,
the 20-row cap, and small-N (no-truncation) behavior, and they confirm the
SUMMARY row and the per-task rows agree on the active count.
"""

import uuid

from crawl4ai.components.crawler_monitor import CrawlerMonitor, TerminalUI
from crawl4ai.models import CrawlStatus

_STATUS_NAMES = ("IN_PROGRESS", "QUEUED", "COMPLETED", "FAILED")


def _make_monitor(n: int = 0) -> CrawlerMonitor:
    """A UI-less monitor. ``enable_ui=False`` avoids spawning the terminal
    thread so the tests stay deterministic and don't clobber the runner's
    tty."""
    return CrawlerMonitor(urls_total=n, enable_ui=False)


def _ui_for(monitor: CrawlerMonitor) -> TerminalUI:
    """Wire a standalone ``TerminalUI`` to ``monitor`` without starting the
    refresh thread, mirroring the pattern in
    ``test_crawler_monitor_memory_percent.py``."""
    ui = TerminalUI()
    ui.monitor = monitor
    return ui


def _render_task_details(ui: TerminalUI) -> str:
    """Render the Task Details panel to plain text (no ANSI) so the tests can
    assert against the exact rows the user would see."""
    from rich.console import Console

    panel = ui._create_task_details_panel()
    console = Console(
        width=120, record=True, force_terminal=False, color_system=None
    )
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


def _ordered_statuses(rendered: str) -> list:
    """Return the status name of each per-task data row in display order.

    Only lines that contain one of the canonical status names are counted;
    the SUMMARY row (``Active: N``), the column header (``Status``), and the
    em-dash separator/borders never contain these tokens, so the result is
    exactly the sequence of task rows top-to-bottom."""
    statuses = []
    for line in rendered.splitlines():
        match = next((n for n in _STATUS_NAMES if n in line), None)
        if match is not None:
            statuses.append(match)
    return statuses


def _short_id_in_render(rendered: str, short_id: str) -> bool:
    """True if ``short_id`` (``task_id[:8]``) appears on its own task row."""
    return short_id in rendered


def _line_index_of(rendered: str, short_id: str) -> int:
    """1-based-ish line index of the row carrying ``short_id``; used to assert
    relative ordering of two specific tasks. Returns -1 if absent."""
    for i, line in enumerate(rendered.splitlines()):
        if short_id in line:
            return i
    return -1


def _make_task(monitor: CrawlerMonitor, url: str) -> str:
    """Register a task and return its id."""
    tid = str(uuid.uuid4())
    monitor.add_task(tid, url)
    return tid


def _set_status(
    monitor: CrawlerMonitor,
    task_id: str,
    status: CrawlStatus,
    *,
    end_time: float = None,
) -> None:
    """Move a task to ``status`` using the same transition sequence the
    dispatcher produces. ``QUEUED`` is the add_task default (no-op here).
    Terminal statuses carry an ``end_time`` used by the recency sort."""
    if status is CrawlStatus.QUEUED:
        return
    monitor.update_task(task_id, status=CrawlStatus.IN_PROGRESS, start_time=0.0)
    monitor.update_task(task_id, status=status, end_time=end_time)


class TestTaskDetailsSortBeforeSlice:
    """The core regression: with >20 registered tasks where the active window
    has advanced past insertion position 20, currently IN_PROGRESS tasks must
    still appear in the per-task table."""

    def test_active_tasks_past_position_20_are_displayed(self):
        """The exact bug scenario: 25 tasks, first 20 COMPLETED (with monotonic
        end_times), last 5 IN_PROGRESS. Pre-fix the slice dropped positions
        20..24 before sorting, so zero IN_PROGRESS rows were shown while the
        SUMMARY row reported ``Active: 5``."""
        monitor = _make_monitor(n=25)
        ui = _ui_for(monitor)
        in_progress_tids = []
        for i in range(25):
            tid = _make_task(monitor, f"http://example.com/{i}")
            if i < 20:
                _set_status(
                    monitor, tid, CrawlStatus.COMPLETED, end_time=1_000_000.0 + i
                )
            else:
                monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS, start_time=0.0)
                in_progress_tids.append(tid)

        all_stats = monitor.get_all_task_stats()
        summary_active = sum(
            1
            for s in all_stats.values()
            if s["status"] == CrawlStatus.IN_PROGRESS.name
        )
        rendered = _render_task_details(ui)
        statuses = _ordered_statuses(rendered)

        assert summary_active == 5, summary_active
        assert len(statuses) == 20, len(statuses)
        assert statuses.count("IN_PROGRESS") == 5, statuses
        assert statuses.count("COMPLETED") == 15, statuses
        # All five active (post-position-20) tasks must be present.
        for tid in in_progress_tids:
            assert _short_id_in_render(rendered, tid[:8]), tid
        # Status-priority: IN_PROGRESS block precedes the COMPLETED block.
        first_completed = next(
            i for i, s in enumerate(statuses) if s == "COMPLETED"
        )
        assert all(s == "IN_PROGRESS" for s in statuses[:first_completed]), statuses
        assert first_completed == 5, statuses

    def test_summary_active_agrees_with_displayed_rows(self):
        """The user-visible symptom of the bug was the SUMMARY row (``Active:
        N``) disagreeing with the per-task rows. When the active count is
        <=20 (so all active tasks fit in the table) the per-task rows must
        report exactly the same active count as the SUMMARY row."""
        monitor = _make_monitor(n=25)
        ui = _ui_for(monitor)
        for i in range(20):
            tid = _make_task(monitor, f"http://example.com/c{i}")
            _set_status(monitor, tid, CrawlStatus.COMPLETED, end_time=float(i))
        for i in range(5):
            tid = _make_task(monitor, f"http://example.com/a{i}")
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS, start_time=0.0)

        all_stats = monitor.get_all_task_stats()
        summary_active = sum(
            1
            for s in all_stats.values()
            if s["status"] == CrawlStatus.IN_PROGRESS.name
        )
        statuses = _ordered_statuses(_render_task_details(ui))

        assert summary_active == statuses.count("IN_PROGRESS"), (
            summary_active,
            statuses,
        )


class TestTaskDetailsSortOrder:
    """The sort is: 1. IN_PROGRESS, 2. QUEUED, 3. COMPLETED/FAILED by
    recency (most recent end_time first)."""

    def test_status_priority_order(self):
        """With at most one task per status, the single IN_PROGRESS row
        precedes the QUEUED row, which precedes the terminal bucket
        (COMPLETED/FAILED). Within the terminal bucket the recency tiebreak
        decides order, so COMPLETED is given a more recent ``end_time`` than
        FAILED to make the expected sequence deterministic."""
        monitor = _make_monitor(n=4)
        ui = _ui_for(monitor)

        tid_q = _make_task(monitor, "http://example.com/q")  # stays QUEUED
        tid_c = _make_task(monitor, "http://example.com/c")
        tid_f = _make_task(monitor, "http://example.com/f")
        tid_i = _make_task(monitor, "http://example.com/i")

        _set_status(monitor, tid_c, CrawlStatus.COMPLETED, end_time=900.0)
        _set_status(monitor, tid_f, CrawlStatus.FAILED, end_time=600.0)
        monitor.update_task(tid_i, status=CrawlStatus.IN_PROGRESS, start_time=0.0)

        statuses = _ordered_statuses(_render_task_details(ui))
        assert statuses == ["IN_PROGRESS", "QUEUED", "COMPLETED", "FAILED"], statuses

    def test_completed_recency_most_recent_first(self):
        """Among COMPLETED tasks (terminal bucket) the one with the highest
        ``end_time`` must render first."""
        monitor = _make_monitor(n=3)
        ui = _ui_for(monitor)

        older = _make_task(monitor, "http://example.com/old")
        newer = _make_task(monitor, "http://example.com/new")
        _set_status(monitor, older, CrawlStatus.COMPLETED, end_time=100.0)
        _set_status(monitor, newer, CrawlStatus.COMPLETED, end_time=900.0)

        rendered = _render_task_details(ui)
        assert _line_index_of(rendered, newer[:8]) < _line_index_of(
            rendered, older[:8]
        ), (newer, older)

    def test_failed_uses_same_bucket_as_completed(self):
        """FAILED lives in priority bucket 2 alongside COMPLETED, so a FAILED
        task with a more recent ``end_time`` must precede an older COMPLETED
        task (and vice-versa). Guards against a fix that hard-separates
        COMPLETED before FAILED regardless of recency."""
        monitor = _make_monitor(n=2)
        ui = _ui_for(monitor)

        completed_old = _make_task(monitor, "http://example.com/co")
        failed_new = _make_task(monitor, "http://example.com/fn")
        _set_status(monitor, completed_old, CrawlStatus.COMPLETED, end_time=100.0)
        _set_status(monitor, failed_new, CrawlStatus.FAILED, end_time=900.0)

        rendered = _render_task_details(ui)
        assert _line_index_of(rendered, failed_new[:8]) < _line_index_of(
            rendered, completed_old[:8]
        )

    def test_in_progress_precedes_completed_regardless_of_end_time(self):
        """An IN_PROGRESS task (end_time is None, sort key 0) must always sort
        ahead of any COMPLETED task, even though the COMPLETED task's
        ``end_time`` is large. The recency tiebreak only applies *within* a
        priority bucket."""
        monitor = _make_monitor(n=2)
        ui = _ui_for(monitor)

        active = _make_task(monitor, "http://example.com/active")
        done_recent = _make_task(monitor, "http://example.com/done-recent")
        monitor.update_task(active, status=CrawlStatus.IN_PROGRESS, start_time=0.0)
        _set_status(monitor, done_recent, CrawlStatus.COMPLETED, end_time=10_000_000.0)

        statuses = _ordered_statuses(_render_task_details(ui))
        assert statuses[0] == "IN_PROGRESS", statuses
        assert statuses[1] == "COMPLETED", statuses


class TestTaskDetailsDisplayCount:
    """The cap (``min(len(task_stats), 20)``) must apply *after* the sort, so
    the 20 most-relevant rows are shown — and small N must show every task."""

    def test_cap_at_twenty_when_more_in_progress_than_cap(self):
        """30 IN_PROGRESS tasks: exactly 20 rows render (the cap), all of them
        IN_PROGRESS. Pre-fix the slice happened before the sort but here, since
        all 30 share the same status, the bug was invisible — this test guards
        the cap itself against an over-fix that drops the limit."""
        monitor = _make_monitor(n=30)
        ui = _ui_for(monitor)
        for i in range(30):
            tid = _make_task(monitor, f"http://example.com/{i}")
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS, start_time=0.0)

        statuses = _ordered_statuses(_render_task_details(ui))
        assert len(statuses) == 20, len(statuses)
        assert all(s == "IN_PROGRESS" for s in statuses), statuses

    def test_cap_drops_least_relevant_completed_when_active_present(self):
        """With 25 tasks where 5 are IN_PROGRESS and 20 are COMPLETED, the cap
        keeps all 5 active plus the 15 most-recent COMPLETED; the 5 *oldest*
        COMPLETED tasks are the ones dropped. Guards that the post-sort slice
        drops the lowest-ranked rows, not arbitrary insertion-ordered ones."""
        monitor = _make_monitor(n=25)
        ui = _ui_for(monitor)
        completed_tids = []
        for i in range(20):
            tid = _make_task(monitor, f"http://example.com/c{i}")
            _set_status(
                monitor, tid, CrawlStatus.COMPLETED, end_time=float(i)
            )  # i=0 oldest .. i=19 newest
            completed_tids.append(tid)
        # The 5 active tasks are added *after* the 20 completed — the bug's
        # insertion-order trigger (active window advanced past position 20).
        active_tids = []
        for i in range(5):
            tid = _make_task(monitor, f"http://example.com/a{i}")
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS, start_time=0.0)
            active_tids.append(tid)

        rendered = _render_task_details(ui)
        statuses = _ordered_statuses(rendered)
        assert len(statuses) == 20, len(statuses)
        assert statuses.count("IN_PROGRESS") == 5, statuses
        # The newest 15 COMPLETED (i=5..19) are kept; the oldest 5 (i=0..4) are
        # dropped because they sort lowest.
        for i in range(5, 20):
            assert _short_id_in_render(rendered, completed_tids[i][:8]), i
        for i in range(5):
            assert not _short_id_in_render(rendered, completed_tids[i][:8]), i
        for tid in active_tids:
            assert _short_id_in_render(rendered, tid[:8]), tid

    def test_fewer_than_twenty_shows_all(self):
        """With 5 tasks there is no truncation; all 5 must render."""
        monitor = _make_monitor(n=5)
        ui = _ui_for(monitor)
        for i in range(5):
            tid = _make_task(monitor, f"http://example.com/{i}")
            _set_status(monitor, tid, CrawlStatus.COMPLETED, end_time=float(i))

        statuses = _ordered_statuses(_render_task_details(ui))
        assert len(statuses) == 5, len(statuses)

    def test_exactly_twenty_shows_all(self):
        """At the boundary of 20 tasks the cap equals the population, so every
        task renders regardless of status mix."""
        monitor = _make_monitor(n=20)
        ui = _ui_for(monitor)
        for i in range(10):
            tid = _make_task(monitor, f"http://example.com/c{i}")
            _set_status(monitor, tid, CrawlStatus.COMPLETED, end_time=float(i))
        for i in range(10):
            tid = _make_task(monitor, f"http://example.com/a{i}")
            monitor.update_task(tid, status=CrawlStatus.IN_PROGRESS, start_time=0.0)

        statuses = _ordered_statuses(_render_task_details(ui))
        assert len(statuses) == 20, len(statuses)
        assert statuses.count("IN_PROGRESS") == 10, statuses
        assert statuses.count("COMPLETED") == 10, statuses
        # All 10 IN_PROGRESS precede the 10 COMPLETED.
        assert statuses[:10] == ["IN_PROGRESS"] * 10, statuses
        assert statuses[10:] == ["COMPLETED"] * 10, statuses


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v", "--asyncio-mode=auto"]))
