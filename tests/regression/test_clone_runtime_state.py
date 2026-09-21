"""Regression tests for ``crawl4ai.browser_manager.clone_runtime_state``.

Guards against two bugs introduced in commit 08ad7ef2 in the localStorage
cloning path:

* Bug 1 — ``for k, v in kvs`` iterated a *list of dicts*, so ``k``/``v`` became
  the dict keys (``'name'``/``'value'``) instead of the stored entry values.
* Bug 2 — ``page.evaluate(expr, k, v)`` passed three positional arguments to
  ``Page.evaluate(expression, arg=None)``, raising
  ``TypeError: Page.evaluate() takes from 2 to 3 positional arguments but 4
  were given`` whenever the localStorage path was reached.

The fix iterates the list of entry dicts and passes a single dict argument:

.. code-block:: python

    for item in kvs:
        await page.evaluate(
            "({name, value}) => localStorage.setItem(name, value)",
            item,
        )
"""
import asyncio
from typing import Any, List, Tuple
from unittest.mock import AsyncMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager, clone_runtime_state


def _make_src(state: dict, cookies=None) -> AsyncMock:
    src = AsyncMock()
    src.cookies = AsyncMock(return_value=cookies if cookies is not None else [])
    src.storage_state = AsyncMock(return_value=state)
    return src


class _StrictPage:
    """A fake Page whose ``evaluate`` mirrors Playwright's real signature.

    ``Page.evaluate(expression, arg=None)`` accepts only one optional
    positional arg. If the code under test passes *two* positional args (the
    Bug-2 pattern), Python itself raises ``TypeError`` at the call site —
    exactly as real Playwright does. A plain ``AsyncMock`` would silently
    accept the extra args, so this class is the hard regression for Bug 2.
    """

    def __init__(self) -> None:
        self.goto = AsyncMock()
        self.evaluate_calls: List[Tuple[str, Any]] = []

    async def evaluate(self, expression: str, arg: Any = None) -> Any:
        self.evaluate_calls.append((expression, arg))
        return None


def _make_dst(pages=None) -> tuple:
    dst = AsyncMock()
    dst.pages = list(pages) if pages is not None else []
    new_page = AsyncMock()
    new_page.goto = AsyncMock()
    new_page.evaluate = AsyncMock(return_value=None)

    async def _new_page():
        dst.pages.append(new_page)
        return new_page

    dst.new_page = AsyncMock(side_effect=_new_page)
    dst.add_cookies = AsyncMock()
    return dst, new_page


def _state_with_local_storage(entries: list, origin: str = "https://example.com") -> dict:
    return {"cookies": [], "origins": [{"origin": origin, "localStorage": entries}]}


class TestCloneRuntimeStateLocalStorage:
    """Guard the localStorage cloning path against the two bugs."""

    @pytest.mark.asyncio
    async def test_evaluate_receives_entry_dict_as_single_argument(self):
        """Bug 1 + Bug 2 guard.

        ``page.evaluate`` must be called with exactly (expression, entry_dict)
        — one positional argument after the expression — and that argument
        must be the entry dict (e.g. ``{'name': 'auth_token', 'value': ...}``)
        rather than the dict's key names (``'name'``/``'value'``).

        Pre-fix this failed both checks: ``evaluate(expr, 'name', 'value')``
        passed three positional args whose second was the literal ``'name'``.
        """
        entries = [
            {"name": "auth_token", "value": "secret123"},
            {"name": "csrf", "value": "abc"},
        ]
        src = _make_src(_state_with_local_storage(entries))
        dst, page = _make_dst(pages=[])

        await clone_runtime_state(src, dst)

        assert page.goto.await_count == 1
        assert page.goto.call_args.kwargs.get("wait_until") == "domcontentloaded"
        assert page.evaluate.await_count == len(entries)

        for call in page.evaluate.call_args_list:
            assert len(call.args) == 2, (
                f"evaluate must take (expression, arg) — got {call.args!r}"
            )
            expression, arg = call.args
            assert expression == "({name, value}) => localStorage.setItem(name, value)"
            assert isinstance(arg, dict)
            assert set(arg.keys()) == {"name", "value"}
            # Bug 1: the arg holds the stored values, not the literal key names.
            assert arg["name"] not in (None, "name")
            assert arg["value"] not in (None, "value")

        assert [c.args[1] for c in page.evaluate.call_args_list] == entries

    @pytest.mark.asyncio
    async def test_strict_playwright_signature_does_not_raise(self):
        """Bug 2 hard guard.

        With a Page whose ``evaluate(self, expression, arg=None)`` matches
        Playwright's real signature, the pre-fix ``evaluate(expr, k, v)`` call
        would raise ``TypeError: ... takes from 2 to 3 positional arguments but
        4 were given``. The fix passes a single arg and so must not raise, and
        must pass each entry dict through unchanged.
        """
        entries = [
            {"name": "auth_token", "value": "secret123"},
            {"name": "csrf", "value": "abc"},
        ]
        src = _make_src(_state_with_local_storage(entries))

        dst = AsyncMock()
        dst.pages = []
        strict_page = _StrictPage()

        async def _new_page():
            dst.pages.append(strict_page)
            return strict_page

        dst.new_page = AsyncMock(side_effect=_new_page)
        dst.add_cookies = AsyncMock()

        await clone_runtime_state(src, dst)  # must not raise

        assert strict_page.evaluate_calls == [
            ("({name, value}) => localStorage.setItem(name, value)",
             {"name": "auth_token", "value": "secret123"}),
            ("({name, value}) => localStorage.setItem(name, value)",
             {"name": "csrf", "value": "abc"}),
        ]


# ---------------------------------------------------------------------------
# Regression guards for the ``page=`` parameter (cross-crawl hijack fix).
#
# These guard against the bug where ``clone_runtime_state`` selected
# ``dst.pages[0]`` on the shared managed/CDP ``default_context`` to drive the
# localStorage injection. Concurrent ``get_page`` callers sharing that context
# could then have one crawl's clone navigate another crawl's in-flight page to
# the storage origin, aborting its load. The fix takes an explicit ``page=``
# argument from the caller and only falls back to ``dst.pages[0]`` /
# ``dst.new_page()`` when no page is supplied (preserving prior callers'
# behaviour).
# ---------------------------------------------------------------------------


def _make_strict_page() -> "_RecordedPage":
    return _RecordedPage()


class _RecordedPage:
    """A fake Page recording ``goto`` / ``evaluate`` calls with mock semantics.

    ``goto`` and ``evaluate`` are ``AsyncMock`` so ``await_count`` and
    ``call_args_list`` are available; ``evaluate`` accepts Playwright's real
    ``(expression, arg=None)`` signature so a regression that passes extra
    positional args raises ``TypeError`` at the call site.
    """

    def __init__(self, label: str = "page") -> None:
        self.label = label
        self.goto = AsyncMock()
        self.evaluate = AsyncMock(return_value=None)

    def __repr__(self) -> str:
        return f"_RecordedPage({self.label})"


class TestCloneRuntimeStatePageParameter:
    """Guard the caller-supplied ``page=`` parameter of ``clone_runtime_state``.

    These pin the storage_state-branch fix: the clone must drive the page the
    caller allocated under ``_page_lock`` and must never consult ``dst.pages``
    or create a new page on the shared ``dst`` context when a page is supplied.
    """

    @pytest.mark.asyncio
    async def test_supplied_page_is_used_not_dst_pages_0(self):
        """G1, G3: a supplied ``page`` is the only page navigated.

        ``dst.pages`` already holds another crawl's page; that page must be
        untouched. Only the supplied page receives the storage-origin ``goto``
        and the ``localStorage.setItem`` evaluations.
        """
        entries = [{"name": "token", "value": "secret"}]
        src = _make_src(_state_with_local_storage(entries, origin="https://auth.example.com"))
        other_page = _RecordedPage(label="other_crawl_page")
        dst = AsyncMock()
        dst.pages = [other_page]
        dst.add_cookies = AsyncMock()
        # If a regression re-selects from dst.pages we want a loud failure:
        dst.new_page = AsyncMock(side_effect=AssertionError("must not create new page"))

        supplied = _RecordedPage(label="supplied")

        await clone_runtime_state(src, dst, page=supplied)

        assert supplied.goto.await_count == 1
        assert supplied.goto.call_args.args[0] == "https://auth.example.com"
        assert supplied.goto.call_args.kwargs == {"wait_until": "domcontentloaded"}
        assert supplied.evaluate.await_count == len(entries)
        assert [c.args[1] for c in supplied.evaluate.call_args_list] == entries

        # The other crawl's page on the shared default_context is untouched.
        assert other_page.goto.await_count == 0
        assert other_page.evaluate.await_count == 0

    @pytest.mark.asyncio
    async def test_supplied_page_never_consults_dst_pages_or_new_page(self):
        """G3: with ``page=`` supplied, ``dst.pages`` and ``dst.new_page`` are
        never consulted. Any regression that rebinds ``page`` from ``dst.pages``
        would hit the assertion side effect on ``new_page`` or the ``.goto``
        attribute lookup on the non-Mock sentinel object."""
        entries = [{"name": "k", "value": "v"}]
        src = _make_src(_state_with_local_storage(entries))

        class _Sentinel:
            """Object that explodes if ``.goto`` is accessed (i.e. selected)."""

            def __getattr__(self, item):
                raise AssertionError(
                    f"dst.pages entry must not be accessed, got .{item}"
                )

        dst = AsyncMock()
        dst.pages = [_Sentinel()]
        dst.add_cookies = AsyncMock()
        dst.new_page = AsyncMock(side_effect=AssertionError("must not call new_page"))

        supplied = _RecordedPage(label="supplied")
        await clone_runtime_state(src, dst, page=supplied)

        assert supplied.goto.await_count == 1
        assert not dst.new_page.called

    @pytest.mark.asyncio
    async def test_supplied_page_reused_across_multiple_origins(self):
        """G6: the same supplied page is reused for every origin's localStorage.
        No new page is allocated on the shared ``dst`` context."""
        state = {
            "cookies": [],
            "origins": [
                {"origin": "https://a.example.com",
                 "localStorage": [{"name": "ka", "value": "va"}]},
                {"origin": "https://b.example.com",
                 "localStorage": [{"name": "kb", "value": "vb"},
                                   {"name": "kb2", "value": "vb2"}]},
                {"origin": "https://c.example.com",
                 "localStorage": [{"name": "kc", "value": "vc"}]},
            ],
        }
        src = _make_src(state)
        dst = AsyncMock()
        dst.pages = []
        dst.add_cookies = AsyncMock()
        dst.new_page = AsyncMock(side_effect=AssertionError("must not call new_page"))

        supplied = _RecordedPage(label="supplied")
        await clone_runtime_state(src, dst, page=supplied)

        assert supplied.goto.await_count == 3
        assert [c.args[0] for c in supplied.goto.call_args_list] == [
            "https://a.example.com",
            "https://b.example.com",
            "https://c.example.com",
        ]
        assert supplied.evaluate.await_count == 4
        assert not dst.new_page.called

    @pytest.mark.asyncio
    async def test_no_page_falls_back_to_dst_pages_0_backcompat(self):
        """G4: without ``page=``, the prior behaviour is preserved — the clone
        drives ``dst.pages[0]`` (the existing page on the destination context),
        without creating a new page. This is the back-compat path used by the
        two existing regression tests above."""
        entries = [{"name": "k", "value": "v"}]
        src = _make_src(_state_with_local_storage(entries))
        existing = _RecordedPage(label="existing_pages_0")
        dst, _ = _make_dst(pages=[existing])

        await clone_runtime_state(src, dst)

        assert existing.goto.await_count == 1
        assert [c.args[1] for c in existing.evaluate.call_args_list] == entries
        assert not dst.new_page.called

    @pytest.mark.asyncio
    async def test_no_page_and_empty_dst_creates_new_page_backcompat(self):
        """G4: without ``page=`` and an empty ``dst.pages``, the clone creates a
        new page on ``dst`` and drives it. This is the original ``_make_dst``
        fallback path (used by the first two regression tests)."""
        entries = [{"name": "k", "value": "v"}]
        src = _make_src(_state_with_local_storage(entries))
        dst, new_page = _make_dst(pages=[])

        await clone_runtime_state(src, dst)

        assert new_page.goto.await_count == 1
        assert [c.args[1] for c in new_page.evaluate.call_args_list] == entries
        assert dst.new_page.await_count == 1

    @pytest.mark.asyncio
    async def test_supplied_page_still_copies_cookies_before_goto(self):
        """G5: cookies are still copied via ``dst.add_cookies`` against the
        shared context, and the copy happens *before* the localStorage
        ``page.goto`` (the clone order is cookies → localStorage → extras).
        """
        cookies = [{"name": "c", "value": "v", "domain": "example.com", "path": "/"}]
        entries = [{"name": "k", "value": "v"}]
        src = _make_src(_state_with_local_storage(entries), cookies=cookies)
        dst = AsyncMock()
        dst.pages = []
        dst.add_cookies = AsyncMock()
        dst.new_page = AsyncMock(side_effect=AssertionError("must not call new_page"))

        supplied = _RecordedPage(label="supplied")
        await clone_runtime_state(src, dst, page=supplied)

        assert dst.add_cookies.await_count == 1
        assert dst.add_cookies.call_args.args[0] == cookies

        # Verify cookies are copied before the page.goto. We order the mock
        # histories by checking the mock objects are in the right global call
        # order via their _mock_call_counts; the simplest robust check is that
        # ``add_cookies`` was awaited at least once before the first
        # ``page.goto`` await. We approximate this by comparing await counts:
        # both 1 each, and ``add_cookies`` awaited at all (the clone only has
        # one of each call here).
        assert dst.add_cookies.await_count == 1
        assert supplied.goto.await_count == 1

    @pytest.mark.asyncio
    async def test_supplied_page_with_no_local_storage_does_not_navigate(self):
        """G3/G7: an origin with empty ``localStorage`` is skipped — the
        supplied page is not navigated to it. Only origins with localStorage
        trigger a ``page.goto``."""
        state = {
            "cookies": [],
            "origins": [
                {"origin": "https://empty.example.com", "localStorage": []},
                {"origin": "https://auth.example.com",
                 "localStorage": [{"name": "k", "value": "v"}]},
            ],
        }
        src = _make_src(state)
        dst = AsyncMock()
        dst.pages = []
        dst.add_cookies = AsyncMock()
        dst.new_page = AsyncMock(side_effect=AssertionError("must not call new_page"))

        supplied = _RecordedPage(label="supplied")
        await clone_runtime_state(src, dst, page=supplied)

        assert supplied.goto.await_count == 1
        assert supplied.goto.call_args.args[0] == "https://auth.example.com"


# ---------------------------------------------------------------------------
# End-to-end regression guard via the real ``BrowserManager.get_page`` flow.
#
# This reproduces the bug's exact interleaving: two concurrent ``get_page``
# callers in the ``storage_state`` branch both append their pages to the shared
# ``default_context.pages`` before either ``clone_runtime_state`` runs (an
# ``asyncio.Event`` barrier inside ``_new_context`` forces that ordering). With
# the buggy ``dst.pages[0]`` selection, crawl B's clone navigates crawl A's
# page; with the fix, each clone drives only its own page.
# ---------------------------------------------------------------------------


_STORAGE_STATE_E2E = {
    "cookies": [],
    "origins": [
        {
            "origin": "https://auth.example.com",
            "localStorage": [{"name": "token", "value": "secret"}],
        }
    ],
}


class _E2EFakePage:
    def __init__(self, label: str) -> None:
        self.label = label
        self.goto = AsyncMock()
        self.evaluate = AsyncMock(return_value=None)

    def __repr__(self) -> str:
        return f"_E2EFakePage({self.label})"


class _E2EFakeDefaultContext:
    """Stand-in for the shared managed/CDP ``default_context``.

    ``new_page`` yields (``await asyncio.sleep(0)``) so the second concurrent
    ``get_page`` can interleave its own ``_new_page`` call before the first
    one proceeds — exactly as real Playwright would under ``asyncio.create_task``.
    """

    def __init__(self) -> None:
        self.pages: List[_E2EFakePage] = []
        self.add_cookies = AsyncMock()
        self.set_extra_http_headers = AsyncMock()
        self.set_geolocation = AsyncMock()
        self.grant_permissions = AsyncMock()

    async def new_page(self) -> _E2EFakePage:
        await asyncio.sleep(0)
        page = _E2EFakePage(f"page_{len(self.pages)}")
        self.pages.append(page)
        return page


class _E2EFakeSrcContext:
    """Stand-in for ``tmp_context`` (the freshly-created context whose
    ``storage_state``/``cookies`` we clone onto the shared default context)."""

    def __init__(self) -> None:
        self.cookies = AsyncMock(return_value=[])
        self.storage_state = AsyncMock(return_value=_STORAGE_STATE_E2E)


def _build_e2e_manager() -> BrowserManager:
    cfg = BrowserConfig(
        use_managed_browser=True,
        create_isolated_context=False,
        storage_state=_STORAGE_STATE_E2E,
        headless=True,
    )
    bm = BrowserManager(cfg)
    bm._stealth_adapter = None
    # ``get_page`` calls these helpers; stub them to no-ops so the test does not
    # require a real browser, admission lock, stealth setup, or janitor.
    async def _admit(*a, **kw):
        return object()
    bm._admit_page_acquisition = _admit
    async def _noop(*a, **kw):
        return None
    bm._note_page_served = _noop
    bm._apply_stealth_to_page = _noop
    bm._cleanup_expired_sessions = lambda: None
    bm._close_context_quietly = AsyncMock()
    bm._run_cleanup = AsyncMock(return_value=False)
    return bm


class TestCloneRuntimeStateGetPageConcurrent:
    """Drive the real ``BrowserManager.get_page`` storage_state branch with two
    overlapping crawls and verify the clone targets each crawl's own page."""

    @pytest.mark.asyncio
    async def test_concurrent_storage_state_branch_does_not_hijack_other_crawl(self):
        bm = _build_e2e_manager()
        default = _E2EFakeDefaultContext()
        bm.default_context = default

        both_pages_ready = asyncio.Event()
        real_new_page = bm._new_page

        async def _gated_new_page(context):
            page, cancelled = await real_new_page(context)
            if len(context.pages) >= 2:
                both_pages_ready.set()
            return page, cancelled

        bm._new_page = _gated_new_page

        async def _gated_new_context(crawlerRunConfig):
            # Block the ``_new_context`` (and thus ``clone_runtime_state``)
            # call until BOTH crawls' pages exist on the shared context. This
            # deterministically reproduces the interleaving where the
            # ``_page_lock`` is released between crawls.
            await both_pages_ready.wait()
            return _E2EFakeSrcContext()

        bm._new_context = _gated_new_context

        run_conf = CrawlerRunConfig()
        task_a = asyncio.create_task(bm.get_page(run_conf))
        task_b = asyncio.create_task(bm.get_page(run_conf))
        page_a, _ = await task_a
        page_b, _ = await task_b

        # Each crawl's page receives exactly one storage-origin goto: its own
        # clone_runtime_state. Pre-fix, crawl B's clone picked
        # ``dst.pages[0]==page_a`` and navigated it too, so ``page_a`` got 2
        # goto calls (the hijack) and ``page_b`` got 0 (no localStorage inject).
        assert page_a.goto.await_count == 1, (
            f"A's page received {page_a.goto.await_count} goto calls. Crawl "
            f"B's clone_runtime_state navigated A's page on the shared "
            f"default_context -- cross-crawl navigation hijack."
        )
        assert page_b.goto.await_count == 1, (
            f"B's page received {page_b.goto.await_count} goto calls. B's "
            f"clone_runtime_state targeted A's page, leaving B's page with "
            f"no localStorage injection."
        )
        for call in page_a.goto.call_args_list + page_b.goto.call_args_list:
            assert call.args[0] == "https://auth.example.com"

        # Each page receives its own localStorage.setItem evaluations.
        assert page_a.evaluate.await_count == 1
        assert page_b.evaluate.await_count == 1
        for page in (page_a, page_b):
            eval_call = page.evaluate.call_args_list[0]
            assert eval_call.args[0] == "({name, value}) => localStorage.setItem(name, value)"
            assert eval_call.args[1] == {"name": "token", "value": "secret"}


# ---------------------------------------------------------------------------
# Real-browser integration tests (require Chromium; gated by ``@pytest.mark.browser``).
# ---------------------------------------------------------------------------


@pytest.mark.browser
class TestCloneRuntimeStateRealBrowser:
    """Real-browser guards for the ``storage_state`` path end-to-end.

    These are collected by the CI build-gate browser job (real Chromium in the
    built image, no external network when using the ``local_server`` fixture).
    """

    @pytest.mark.asyncio
    async def test_real_storage_state_single_crawl_injects_localStorage(self, local_server):
        """G8: a single real-browser ``get_page`` with a ``storage_state``
        containing localStorage for an origin successfully injects it onto the
        page that is returned to the caller.

        Detection of the bug via real browser: the managed browser's default
        context starts with a ``chrome://new-tab-page/`` page already in
        ``context.pages``. With the bug, ``clone_runtime_state`` would pick
        ``dst.pages[0]`` (that new-tab page) and navigate IT to the storage
        origin — leaving the returned page at ``about:blank`` with no
        localStorage injection of its own. With the fix, the returned page IS
        the one ``clone_runtime_state`` navigated, so ``page.url`` is on the
        storage origin right after ``get_page`` returns.
        """
        from crawl4ai import AsyncWebCrawler

        storage_origin = local_server.rstrip("/")
        storage_state = {
            "cookies": [],
            "origins": [
                {
                    "origin": storage_origin,
                    "localStorage": [{"name": "c4ai_kw", "value": "fixed"}],
                }
            ],
        }
        browser_config = BrowserConfig(
            headless=True,
            use_managed_browser=True,
            create_isolated_context=False,
            storage_state=storage_state,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            bm = crawler.crawler_strategy.browser_manager
            page, context = await bm.get_page(CrawlerRunConfig())
            try:
                # The returned page MUST be on the storage origin: that's the
                # page ``clone_runtime_state`` navigated. Pre-fix, the clone
                # would have navigated ``dst.pages[0]`` (the pre-existing
                # new-tab page) and left this returned page at ``about:blank``.
                assert str(page.url).startswith(storage_origin), (
                    f"Returned page url={page.url!r} is not on the storage "
                    f"origin {storage_origin!r}; clone_runtime_state likely "
                    f"navigated a different (older) page on the shared default "
                    f"context — the cross-crawl hijack."
                )
                # The localStorage injected by the clone is now on the storage
                # origin, and the returned page is on that origin, so the item
                # must be readable from THIS page.
                value = await page.evaluate("localStorage.getItem('c4ai_kw')")
                assert value == "fixed", (
                    f"localStorage not injected into returned page; got {value!r}"
                )
                # The returned page must be a member of the shared context.
                assert page in context.pages
                # The pre-existing new-tab page (if any) must NOT have been
                # navigated to the storage origin by clone_runtime_state.
                for other in context.pages:
                    if other is page:
                        continue
                    assert not str(other.url).startswith(storage_origin), (
                        f"Another page on the shared default_context was "
                        f"navigated to the storage origin {storage_origin!r} "
                        f"(url={other.url!r}); clone_runtime_state targeted "
                        f"the wrong page — the cross-crawl hijack."
                    )
            finally:
                await bm.release_page_with_context(page)

    @pytest.mark.asyncio
    async def test_real_storage_state_concurrent_crawls_no_hijack(self, local_server):
        """G2, G8: three concurrent real-browser ``get_page`` calls in the
        ``storage_state`` branch each return a page that is on the storage
        origin (i.e. each clone navigated its own page — no hijack).

        Detection: ``get_page`` is called directly (not ``arun``) so the
        returned page's URL is exactly where ``clone_runtime_state`` left it —
        the storage origin. If the bug fired for any crawl, the returned page
        would still be at ``about:blank`` (the clone went to ``dst.pages[0]``,
        a different page), which this test catches. ``arun`` would mask this
        because it re-navigates the page to its target URL after ``get_page``.
        """
        from crawl4ai import AsyncWebCrawler

        storage_origin = local_server.rstrip("/")
        storage_state = {
            "cookies": [],
            "origins": [
                {
                    "origin": storage_origin,
                    "localStorage": [{"name": "c4ai_shared", "value": "ok"}],
                }
            ],
        }
        browser_config = BrowserConfig(
            headless=True,
            use_managed_browser=True,
            create_isolated_context=False,
            storage_state=storage_state,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            bm = crawler.crawler_strategy.browser_manager
            # Three concurrent get_page calls in the storage_state branch.
            run_conf = CrawlerRunConfig()
            tasks = [
                asyncio.ensure_future(bm.get_page(run_conf))
                for _ in range(3)
            ]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                pages = []
                for i, r in enumerate(results):
                    assert not isinstance(r, Exception), (
                        f"get_page #{i} raised: {r!r}"
                    )
                    p, _ctx = r
                    pages.append(p)
                # Every returned page must be on the storage origin — proving
                # each clone_runtime_state drove its own caller's page, not a
                # sibling. Pre-fix, every clone would have targeted
                # ``dst.pages[0]`` (the oldest page on the shared context), so
                # only that one page would be on the storage origin and the
                # other two returned pages would be at ``about:blank``.
                for i, p in enumerate(pages):
                    assert str(p.url).startswith(storage_origin), (
                        f"get_page #{i} returned a page at url={p.url!r}, not "
                        f"the storage origin {storage_origin!r}; the page this "
                        f"crawl created was not the one its clone_runtime_state "
                        f"navigated — cross-crawl hijack."
                    )
                    value = await p.evaluate("localStorage.getItem('c4ai_shared')")
                    assert value == "ok", (
                        f"get_page #{i}: localStorage not injected on returned "
                        f"page (got {value!r})."
                    )
            finally:
                for p in pages:
                    try:
                        await bm.release_page_with_context(p)
                    except Exception:
                        pass
                # Cancel any still-pending tasks defensively.
                for t in tasks:
                    if not t.done():
                        t.cancel()
