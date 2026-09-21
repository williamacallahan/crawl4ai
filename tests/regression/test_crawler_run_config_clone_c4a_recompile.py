"""Regression tests: ``CrawlerRunConfig.clone(c4a_script=...)`` must recompile.

``CrawlerRunConfig`` compiles a user-supplied C4A script (``c4a_script``) into
JavaScript (``js_code``) at construction time; the crawler then executes
``config.js_code`` on the page (the ``c4a_script`` itself is never re-consulted
at runtime; see ``async_crawler_strategy.py``'s
``robust_execute_user_script(page, config.js_code)``).

``clone(**kwargs)`` builds a dict from ``to_dict()`` (which emits BOTH the
source ``c4a_script`` AND the compiled ``js_code``), applies the caller's
overrides, and rebuilds via ``CrawlerRunConfig.from_kwargs(dict)``.

When the caller overrode ``c4a_script`` via ``clone(c4a_script=<new>)`` without
also overriding ``js_code``, the stale compiled ``js_code`` from the original
``c4a_script`` survived into the new ``__init__`` call. ``__init__``'s compile
gate ``if self.c4a_script and not self.js_code:`` then skipped recompilation
because ``js_code`` was truthy — so the cloned config reported the new
``c4a_script`` on its attribute while the *old* JavaScript was what would have
been executed on the page.

The fix drops ``js_code`` from the cloned dict when the caller overrides
``c4a_script`` *without* also passing an explicit ``js_code``::

    if "c4a_script" in kwargs and "js_code" not in kwargs:
        config_dict.pop("js_code", None)

This forces ``__init__``'s compile gate to recompile from the new
``c4a_script``. When the caller does pass an explicit ``js_code`` (with or
without ``c4a_script``), the ``pop`` is skipped and the caller's ``js_code``
wins, preserving the existing "explicit ``js_code`` wins" precedence rule.

The equivalent ``CrawlerRunConfig.from_kwargs(dict)`` surface — where a
caller hand-edits ``c4a_script`` in a ``to_dict()`` output without dropping
``js_code`` — is NOT closed by this ``clone()``-side guard, because
``from_kwargs`` is a generic forwarder with no signal to distinguish
caller-supplied ``js_code`` from stale serialized ``js_code``. That residual
exposure is documented by an ``xfail(strict=True)`` test below; if a future
``__init__``-side remedy closes it, the strict xfail will surface as a hard
failure so the marker can be removed.
"""

import pytest

from crawl4ai.async_configs import CrawlerRunConfig


@pytest.fixture(autouse=True)
def _reset_crawler_run_config_defaults():
    """Ensure class-level set_defaults() state from other tests does not leak in."""
    CrawlerRunConfig.reset_defaults()
    yield
    CrawlerRunConfig.reset_defaults()


class TestCloneOverridesC4AScriptRecompiles:
    """``clone(c4a_script=<new>)`` must produce a config whose ``js_code`` is
    the freshly-compiled form of the new ``c4a_script``."""

    def test_clone_with_new_c4a_script_recompiles_js_code(self):
        """The reported bug, failing pre-fix: the override's text appears in
        the executable ``js_code`` while the original's text does not."""
        c1 = CrawlerRunConfig(c4a_script='WAIT "hello" 1')
        c2 = c1.clone(c4a_script='WAIT "world" 1')
        assert c2.c4a_script == 'WAIT "world" 1'
        assert 'world' in str(c2.js_code), "expected new script compiled into js_code"
        assert 'hello' not in str(c2.js_code), "stale js_code from original c4a_script survived clone"

    def test_clone_c4a_override_matches_fresh_construction(self):
        """``clone(c4a_script=X).js_code == CrawlerRunConfig(c4a_script=X).js_code``.

        Equivalence to a fresh construction is the strongest guarantee that the
        override's compile-and-run semantics are correct."""
        c1 = CrawlerRunConfig(c4a_script='WAIT "hello" 1', page_timeout=5000)
        c2 = c1.clone(c4a_script='WAIT "world" 2')
        fresh = CrawlerRunConfig(c4a_script='WAIT "world" 2', page_timeout=5000)
        assert c2.js_code == fresh.js_code


class TestCloneC4ARemoval:
    """Overriding ``c4a_script`` to a falsy value clears the compiled ``js_code``."""

    def test_clone_c4a_script_none_clears_compiled_js_code(self):
        """Without the fix the stale ``js_code`` survived even when the caller
        cleared ``c4a_script``; the override to ``None`` must clear it."""
        c1 = CrawlerRunConfig(c4a_script='WAIT "hello" 1')
        c2 = c1.clone(c4a_script=None)
        assert c2.c4a_script is None
        assert c2.js_code is None


class TestCloneExplicitJsCodePrecedence:
    """The "explicit ``js_code`` wins" precedence rule is preserved when the
    fix pops ``js_code`` from the cloned dict — a too-aggressive pop would
    invert this rule."""

    def test_clone_with_explicit_js_code_and_c4a_script_override_explicit_js_wins(self):
        """When the caller overrides BOTH ``c4a_script`` and ``js_code``, the
        explicit ``js_code`` wins and the new ``c4a_script`` is recorded but
        NOT compiled (precedence rule)."""
        c1 = CrawlerRunConfig(c4a_script='WAIT "hello" 1')
        c2 = c1.clone(c4a_script='WAIT "world" 1', js_code=['my-explicit-js'])
        assert c2.c4a_script == 'WAIT "world" 1'
        assert c2.js_code == ['my-explicit-js']
        assert 'world' not in str(c2.js_code), "explicit js_code should win, not recompile"


class TestCloneDeepCrawlStyleNoRegression:
    """The deep-crawl clone pattern (``clone(deep_crawl_strategy=None,
    stream=...)``) does NOT pass ``c4a_script`` in kwargs, so the fix's
    ``pop("js_code", None)`` must not trigger: the compiled ``js_code`` is
    preserved across these clones. A too-aggressive fix that always pops
    ``js_code`` would break the deep-crawl batch path."""

    def test_clone_deep_crawl_style_preserves_c4a_script_and_js_code(self):
        """The BFS/DFS/BFF pattern (``bfs_strategy.py:253,356``,
        ``dfs_strategy.py:84,201``, ``bff_strategy.py:282``) clones a
        c4a-bearing parent this way for each sub-batch."""
        c1 = CrawlerRunConfig(c4a_script='WAIT "hello" 1')
        batch = c1.clone(deep_crawl_strategy=None, stream=False)
        assert batch.c4a_script == 'WAIT "hello" 1'
        assert batch.stream is False
        assert batch.deep_crawl_strategy is None
        assert 'hello' in str(batch.js_code), "deep-crawl clone must preserve compiled js"


class TestFromKwargsHandEditedDictResidualExposure:
    """Surface 2 (the documented residual exposure): ``from_kwargs(dict)``
    after a caller hand-edits ``c4a_script`` in a ``to_dict()`` output is NOT
    closed by the ``clone()``-side fix. ``from_kwargs`` is a generic forwarder
    with no signal to distinguish caller-supplied ``js_code`` from the stale
    serialized ``js_code`` carried by ``to_dict()``; the existing ``__init__``
    compile gate treats any populated ``js_code`` as "explicit," so the new
    ``c4a_script`` is recorded but the stale ``js_code`` is what would run.

    The test is ``xfail(strict=True)`` so that any future change that DOES
    close surface 2 will surface as a hard failure here, prompting removal of
    the residual marker.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Surface 2 residual exposure: from_kwargs(dict) cannot distinguish "
            "caller-supplied js_code from stale js_code carried by to_dict(); the "
            "clone()-side fix does not close this surface. A future __init__-side "
            "remedy may do so, at which point this xfail should be removed."
        ),
    )
    def test_from_kwargs_hand_edited_c4a_script_recompiles(self):
        d = CrawlerRunConfig(c4a_script='WAIT "hello" 1').to_dict()
        d['c4a_script'] = 'WAIT "world" 1'
        c = CrawlerRunConfig.from_kwargs(d)
        assert c.c4a_script == 'WAIT "world" 1'
        assert 'world' in str(c.js_code), "expected new script compiled into js_code"
        assert 'hello' not in str(c.js_code), "stale js_code from original c4a_script survived from_kwargs"


@pytest.mark.browser
class TestCloneC4AScriptOverrideE2E:
    """End-to-end verification, in a real browser, that the crawler executes
    the NEW c4a script after a ``clone(c4a_script=...)`` override.

    Differential signal: ``robust_execute_user_script`` returns a result dict
    whose per-script ``results`` list contains ``{success: false, error:
    'WAIT text timeout'}`` iff a ``WAIT "<text>" <t>`` promise rejects (text
    not present on the page). With the fix, the override script (which waits
    for text present on the local home page) resolves immediately; under the
    bug, the stale original script (which waits for absent text) would
    reject and surface the marker.

    Preconditions: Chromium installed (``playwright install chromium``) and
    the session-scoped ``local_server`` fixture from
    ``tests/regression/conftest.py``.
    """

    @pytest.mark.asyncio
    async def test_clone_c4a_script_override_runs_new_script_in_browser(self, local_server):
        """The override's WAIT-text promise resolves (no timeout marker)."""
        from crawl4ai import AsyncWebCrawler
        from crawl4ai.async_configs import BrowserConfig

        base = CrawlerRunConfig(
            c4a_script='WAIT "ZZZ_marker_not_in_home_page" 1',
            page_timeout=5000,
        )
        cloned = base.clone(c4a_script='WAIT "Welcome" 3')

        # Sanity gate: the cloned config carries the new script, not the old.
        assert 'Welcome' in str(cloned.js_code)
        assert 'ZZZ_marker_not_in_home_page' not in str(cloned.js_code)

        async with AsyncWebCrawler(
            config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])
        ) as crawler:
            result = await crawler.arun(local_server + "/", config=cloned)

        assert result.success, f"Crawl failed: {result.error_message}"
        assert "Welcome to the Crawl4AI Test Site" in result.markdown
        # The override 'Welcome' resolves immediately (text is present on the
        # home page), so no WAIT text timeout is surfaced. If the stale original
        # script had run instead, "WAIT text timeout" would appear here.
        assert "WAIT text timeout" not in str(result.js_execution_result), (
            "stale js_code from the original c4a_script was executed on the page — "
            "clone(c4a_script=...) recompile regressed"
        )
