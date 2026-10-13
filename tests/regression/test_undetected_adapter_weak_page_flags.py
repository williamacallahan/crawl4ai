"""Regression for the closed-page retention bug in UndetectedAdapter.

`UndetectedAdapter` tracks whether its `add_init_script` console/error-capture
scripts have been injected per `Page` via `_console_script_injected` /
`_error_script_injected`. Those maps used to be plain dicts keyed by the
`Page` object, and were only ever read and written -- never pruned -- so every
page ever crawled stayed referenced by the adapter forever, pinning the
already-closed patchright `Page` wrapper for the lifetime of the strategy.
The non-session crawl path creates a fresh `Page` per crawl, so the map grew
by one entry per crawl; killed sessions leaked one entry per killed session;
`max_pages_before_recycle` did not help because the adapter outlives the
browser/manager it is plugged into.

The fix: weak-keyed maps (`weakref.WeakKeyDictionary`). An entry disappears as
soon as the `Page` is no longer referenced elsewhere, while a live pooled /
session page (kept alive by the browser manager) stays in the map so init
scripts are not re-injected on reuse.
"""

import gc
import weakref

import pytest

from crawl4ai.browser_adapter import UndetectedAdapter


class _RecordingPage:
    """Minimal page double recording `add_init_script` calls.

    Mirrors the surface `UndetectedAdapter` touches: `add_init_script(script)`.
    Hashable/usable as a dict key by identity (the real Playwright/patchright
    `Page` is also keyed by identity in the adapter's per-page flag maps). Has
    no `__slots__`, so it supports weak references -- the real `Page` does too
    -- making it a faithful model for the WeakKeyDictionary fix.
    """

    def __init__(self):
        self.init_scripts = []

    async def add_init_script(self, script):
        self.init_scripts.append(script)


@pytest.mark.asyncio
async def test_undetected_adapter_flag_dicts_are_weak_keyed():
    """Guard against regressions to a plain dict: the per-page flag maps must
    be WeakKeyDictionary instances so closed/dropped pages auto-evict."""
    adapter = UndetectedAdapter()
    assert isinstance(adapter._console_script_injected, weakref.WeakKeyDictionary)
    assert isinstance(adapter._error_script_injected, weakref.WeakKeyDictionary)


@pytest.mark.asyncio
async def test_undetected_adapter_evicts_dropped_page_from_flag_dicts():
    """The core fix: once the only strong refs to a page are gone, both flag
    dicts release it -- a closed page is no longer pinned for the lifetime of
    the adapter."""
    adapter = UndetectedAdapter()
    page = _RecordingPage()
    captured_console = []

    await adapter.setup_console_capture(page, captured_console)
    await adapter.setup_error_capture(page, captured_console)

    # While live, both flags record the page and both init scripts injected.
    assert page in adapter._console_script_injected
    assert page in adapter._error_script_injected
    assert len(page.init_scripts) == 2

    # Drop every strong ref to the page and force collection. The flag dicts
    # must release it (pre-fix: the page stayed pinned here forever).
    del page
    gc.collect()

    assert len(list(adapter._console_script_injected)) == 0
    assert len(list(adapter._error_script_injected)) == 0


@pytest.mark.asyncio
async def test_undetected_adapter_retains_live_page_and_dedupes_init_scripts():
    """A live page still referenced elsewhere (e.g. held by the browser
    manager) must stay in the flag maps so init scripts are not re-injected on
    reuse -- the reuse-dedup optimization is preserved by the weak map while
    the page is kept alive by its real owner."""
    adapter = UndetectedAdapter()
    page = _RecordingPage()
    captured_console = []

    await adapter.setup_console_capture(page, captured_console)
    await adapter.setup_error_capture(page, captured_console)

    # Page is still held by `page` -- both flags keep it.
    assert page in adapter._console_script_injected
    assert page in adapter._error_script_injected

    # Re-run on the same live page: must not re-inject either init script.
    await adapter.setup_console_capture(page, captured_console)
    await adapter.setup_error_capture(page, captured_console)

    assert len(page.init_scripts) == 2
    assert adapter._console_script_injected[page] is True
    assert adapter._error_script_injected[page] is True
