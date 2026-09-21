#!/usr/bin/env python3
"""
Regression tests for the C4A-Script conditional ``REPEAT`` looping fix.

``REPEAT (<command>, <condition>)`` (documented at
``docs/md_v2/api/c4a-script-reference.md``) is meant to "repeat a command while
condition is true" with the "condition checked before each iteration".

The original emitter (introduced in commit 3f6f2e9) compiled a backticked JS
condition into a one-shot ``else if (_count) { body }`` branch: the condition
expression was evaluated exactly once and the body could only run zero or one
times -- never a real loop. Any conditional-repeat script silently did the
wrong thing (a single scroll/keypress instead of a loop).

The fix replaces the one-shot ``else if`` with a ``while (count_expr)`` loop
that re-evaluates the condition expression before every iteration while keeping
the numeric-count path (plain ``for`` loop, and the ``typeof === 'number'``
sub-branch for backticked numeric expressions) unchanged.

These tests guard:
  * Compiler output: a ``while`` loop is emitted for conditional REPEAT and the
    buggy ``else if (_count)`` branch is gone; the numeric ``for`` loop is
    unchanged.
  * Runtime behaviour in a real (Chromium) browser: the conditional body runs
    multiple times, the condition is re-evaluated each iteration, the loop
    terminates when the condition becomes false, and the numeric form still
    runs exactly N times.

Preconditions: Chromium installed (``playwright install chromium``). The
compiler-output tests need no browser.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from crawl4ai.script import compile as c4a_compile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile(src: str) -> str:
    """Compile a single c4a statement and return the generated JS block."""
    result = c4a_compile(src)
    assert result.success, f"Compilation failed: {result.errors}"
    assert len(result.js_code) == 1, f"Expected exactly 1 statement, got {len(result.js_code)}"
    return result.js_code[0]


def _runtime_wrap(script: str, harness_timeout_ms: int | None = None) -> str:
    """Wrap the compiled JS the way ``robust_execute_user_script``
    (async_crawler_strategy.py) does, returning a structured result so the
    promise outcome is observable in Python.

    When ``harness_timeout_ms`` is set, the inner promise is raced against a
    JS-side watchdog so a never-settling promise fails the test cleanly
    instead of hanging the suite.
    """
    inner = f"""
            (async () => {{
                {script}
            }})()"""
    if harness_timeout_ms is not None:
        inner = f"""
            Promise.race([
                {inner},
                new Promise((_, rej) => setTimeout(() => rej('TEST_HARNESS_TIMEOUT'), {harness_timeout_ms}))
            ])"""
    return f"""
    (async () => {{
        try {{
            const r = await {inner};
            return {{ success: true, result: r === undefined ? null : r }};
        }} catch (err) {{
            return {{ success: false, error: err.toString(), stack: err.stack }};
        }}
    }})();
    """


# ===========================================================================
# Compiler output (deterministic, no browser)
# ===========================================================================


class TestRepeatConditionalCompilerOutput:
    """Guard against re-introducing the one-shot ``else if (_count)`` branch."""

    def test_conditional_scroll_emits_while_loop(self):
        js = _compile('REPEAT (SCROLL DOWN 500, `document.querySelector(".load-more")`)')
        assert 'while (document.querySelector(".load-more"))' in js
        assert "else if (_count)" not in js
        assert "window.scrollBy(0,500);" in js

    def test_conditional_scroll_emits_while_for_boolean_expression(self):
        js = _compile('REPEAT (SCROLL DOWN 500, `window.scrollY < document.body.scrollHeight`)')
        assert "while (window.scrollY < document.body.scrollHeight)" in js
        assert "else if (_count)" not in js

    def test_conditional_press_body_inside_while(self):
        js = _compile('REPEAT (PRESS ArrowDown, `window.scrollY < document.body.scrollHeight`)')
        assert "while (window.scrollY < document.body.scrollHeight)" in js
        assert "KeyboardEvent" in js
        assert "else if (_count)" not in js

    def test_conditional_re_evaluates_raw_expression_not_snapshot(self):
        """The while condition must be the raw ``{count_expr}`` (re-evaluated
        each iteration), not the one-time ``const _count`` snapshot."""
        js = _compile('REPEAT (SCROLL DOWN 100, `window.scrollY < 1000`)')
        assert "while (window.scrollY < 1000)" in js
        # The snapshot is still computed once for the numeric typeof check;
        # but the loop must NOT use the snapshot as its condition.
        assert "while (_count)" not in js

    def test_numeric_count_unchanged_pure_for_loop(self):
        """Regression guard: plain numeric REPEAT emits only a bounded for loop
        (no IIFE, no typeof check, no while)."""
        js = _compile('REPEAT (SCROLL DOWN 500, 5)')
        assert "for (let _i = 0; _i < 5; _i++)" in js
        assert "while" not in js
        assert "typeof _count" not in js

    def test_numeric_backtick_routes_to_for_loop_branch(self):
        """A backticked numeric expression is dispatched via the
        ``typeof === 'number'`` check to a bounded for loop."""
        js = _compile('REPEAT (SCROLL DOWN 500, `5`)')
        assert "typeof _count === 'number'" in js
        assert "for (let _i = 0; _i < _count; _i++)" in js
        assert "const _count = 5;" in js

    def test_conditional_body_lives_inside_while_block(self):
        """The body must live inside the while block (closed before the outer
        IIFE), not dangle after a one-shot ``else if``."""
        js = _compile('REPEAT (SCROLL DOWN 500, `document.querySelector(".load-more")`)')
        while_open = js.index("while (")
        iife_close = js.rindex("})();")
        while_block = js[while_open:iife_close]
        assert "window.scrollBy(0,500);" in while_block


# ===========================================================================
# End-to-end JS execution in a real browser (Playwright/Chromium)
# ===========================================================================


@pytest.fixture
def browser_page():
    """Launch Chromium and provide a fresh page (and fresh ``window``
    built-ins) for each test, so spies on ``window.scrollBy`` /
    ``document.dispatchEvent`` cannot leak between tests."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context()
            page = context.new_page()
            yield page
        finally:
            browser.close()


@pytest.mark.browser
class TestRepeatConditionalRuntime:
    """The core looping regressions, verified in a real browser."""

    def test_conditional_repeat_loops_until_condition_false(self, browser_page):
        """The conditional REPEAT body runs multiple times (a real loop),
        re-checking the condition before each iteration, and stops once the
        condition becomes false.

        Fixture: ``.load-more`` is present; the ``scrollBy`` spy removes it
        after the 3rd call. A correct ``while``-loop implementation runs the
        body exactly 3 times (checks: present, present, present -> remove on
        the 3rd body call, then the 4th check is false). Before the fix only
        1 body execution occurred.
        """
        page = browser_page
        page.set_content(
            '<!doctype html><html><body>'
            '<div class="load-more">more</div>'
            '</body></html>'
        )
        page.evaluate("""() => {
            window.__scrollCount = 0;
            const orig = window.scrollBy.bind(window);
            window.scrollBy = function (x, y) {
                window.__scrollCount++;
                if (window.__scrollCount >= 3) {
                    const el = document.querySelector('.load-more');
                    if (el) el.remove();
                }
                return orig(x, y);
            };
        }""")
        js = _compile('REPEAT (SCROLL DOWN 500, `document.querySelector(".load-more")`)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"] is True, f"script failed: {result}"
        count = page.evaluate("() => window.__scrollCount")
        present = page.evaluate("() => !!document.querySelector('.load-more')")
        assert count == 3, f"body ran {count} times, expected 3 (one-shot bug regressed)"
        assert present is False, ".load-more should have been removed during the loop"

    def test_conditional_repeat_falsy_condition_runs_zero_times(self, browser_page):
        """When the condition is initially false, the body never executes."""
        page = browser_page
        page.set_content("<html><body><p>no load-more here</p></body></html>")
        page.evaluate("""() => {
            window.__scrollCount = 0;
            window.scrollBy = function () { window.__scrollCount++; };
        }""")
        js = _compile('REPEAT (SCROLL DOWN 500, `document.querySelector(".load-more")`)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"] is True, f"script failed: {result}"
        assert page.evaluate("() => window.__scrollCount") == 0

    def test_conditional_repeat_rechecks_condition_each_iteration(self, browser_page):
        """A synthetic condition that flips after the body runs N times proves
        the condition is re-evaluated before every iteration (a one-shot truthy
        check would run the body exactly once)."""
        page = browser_page
        page.set_content("<html><body></body></html>")
        page.evaluate("""() => {
            window.__n = 0;       // body executions
            window.__condCalls = 0; // condition evaluations
            window.__cond = () => { window.__condCalls++; return window.__n < 3; };
            window.__body = () => { window.__n++; };
        }""")
        js = _compile('REPEAT (EVAL `window.__body()`, `window.__cond()`)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"] is True, f"script failed: {result}"
        n = page.evaluate("() => window.__n")
        cond_calls = page.evaluate("() => window.__condCalls")
        # Body runs 3 times; the while check fires once more (false) to exit.
        # A one-shot buggy implementation would leave n == 1.
        assert n == 3, f"body ran {n} times, expected 3 (one-shot bug regressed)"
        assert cond_calls >= 4, (
            f"condition evaluated only {cond_calls} times; re-evaluation regressed"
        )

    def test_conditional_repeat_press_loops(self, browser_page):
        """The second documented example (PRESS while a condition holds) also
        loops: the keypress body runs more than once.

        A ``dispatchEvent`` spy counts KeyboardEvents so the body execution
        count is observable; the condition is bounded by that count to
        guarantee termination.
        """
        page = browser_page
        page.set_content(
            '<!doctype html><html><body style="margin:0">'
            + '<div style="height:5000px">tall</div>'
            + '</body></html>'
        )
        page.evaluate("""() => {
            window.__keyCount = 0;
            const orig = document.dispatchEvent.bind(document);
            document.dispatchEvent = function (ev) {
                if (ev instanceof KeyboardEvent) window.__keyCount++;
                return orig(ev);
            };
        }""")
        # PRESS emits one keydown + one keyup (2 KeyboardEvents) per iteration.
        # Loop while __keyCount < 6 -> 3 iterations (6 events), then it stops.
        js = _compile('REPEAT (PRESS ArrowDown, `window.__keyCount < 6`)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"] is True, f"script failed: {result}"
        # Before the fix only 1 PRESS (2 KeyboardEvents) fired.
        assert page.evaluate("() => window.__keyCount") == 6

    def test_numeric_repeat_unchanged_at_runtime(self, browser_page):
        """Regression guard: numeric REPEAT still runs the body exactly N times."""
        page = browser_page
        page.set_content("<html><body></body></html>")
        page.evaluate("""() => {
            window.__scrollCount = 0;
            const o = window.scrollBy.bind(window);
            window.scrollBy = function () { window.__scrollCount++; return o(0, 0); };
        }""")
        js = _compile('REPEAT (SCROLL DOWN 500, 4)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"] is True, f"script failed: {result}"
        assert page.evaluate("() => window.__scrollCount") == 4

    def test_numeric_backtick_repeat_runs_n_times(self, browser_page):
        """Regression guard: a backticked numeric count routes via the typeof
        check to the bounded for loop (4 iterations), not the while branch."""
        page = browser_page
        page.set_content("<html><body></body></html>")
        page.evaluate("""() => {
            window.__scrollCount = 0;
            const o = window.scrollBy.bind(window);
            window.scrollBy = function () { window.__scrollCount++; return o(0, 0); };
        }""")
        js = _compile('REPEAT (SCROLL DOWN 500, `4`)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"] is True, f"script failed: {result}"
        assert page.evaluate("() => window.__scrollCount") == 4
