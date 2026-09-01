"""Regression tests for the optional-monitor null guard in ``MemoryAdaptiveDispatcher``.

``CrawlerMonitor`` is optional (``monitor=None`` is the default), so every call to
``self.monitor.update_memory_status(...)`` must be guarded with ``if self.monitor:``.
These tests pin that contract on the queue-drain exception path, which previously
raised a masking ``AttributeError`` when no monitor was attached.
"""
import time
from unittest.mock import patch

import pytest
from crawl4ai import CrawlerMonitor, MemoryAdaptiveDispatcher


class _BoomError(RuntimeError):
    """A non-timeout exception that escapes the inner drain ``try`` block."""


async def _boom_get(*args, **kwargs):
    raise _BoomError("simulated drain failure")


@pytest.mark.asyncio
class TestUpdateQueuePrioritiesMonitorGuard:
    async def test_no_crash_when_monitor_none_and_drain_fails(self):
        """With ``monitor=None`` (the default), an exception while draining the
        priority queue must not crash with ``AttributeError`` and must leave the
        unconsumed queued item in place."""
        dispatcher = MemoryAdaptiveDispatcher(monitor=None, max_session_permit=2)
        assert dispatcher.monitor is None

        queued_item = (0.0, ("http://x", "t1", 0, time.time()))
        await dispatcher.task_queue.put(queued_item)

        with patch.object(dispatcher.task_queue, "get", _boom_get):
            await dispatcher._update_queue_priorities()  # must not raise

        assert dispatcher.task_queue.qsize() == 1
        assert dispatcher.task_queue.get_nowait() == queued_item

    async def test_monitor_updated_when_present_and_drain_fails(self):
        """When a monitor IS attached, the QUEUE_ERROR status must still be
        reported — the null guard must not suppress legitimate monitor updates."""
        monitor = CrawlerMonitor(urls_total=1, enable_ui=False)
        dispatcher = MemoryAdaptiveDispatcher(monitor=monitor, max_session_permit=2)
        assert dispatcher.monitor is monitor

        await dispatcher.task_queue.put((0.0, ("http://x", "t1", 0, time.time())))

        with patch.object(dispatcher.task_queue, "get", _boom_get):
            await dispatcher._update_queue_priorities()  # must not raise

        assert monitor.get_memory_status().startswith("QUEUE_ERROR")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
