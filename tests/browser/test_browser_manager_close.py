import pytest

from crawl4ai.browser_manager import BrowserManager
from crawl4ai import BrowserConfig


class _Context:
    def __init__(self, manager=None):
        self.manager = manager
        self.closed = False

    async def close(self):
        self.closed = True
        if self.manager:
            self.manager.contexts_by_config.pop("second", None)


@pytest.mark.asyncio
async def test_close_iterates_over_context_snapshot():
    manager = BrowserManager(BrowserConfig(headless=True))
    first, second = _Context(manager), _Context()
    manager.contexts_by_config = {"first": first, "second": second}

    await manager.close()

    assert first.closed
    assert second.closed
    assert manager.contexts_by_config == {}
