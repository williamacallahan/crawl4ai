#!/usr/bin/env python3
"""
Regression tests for the C4A-Script WAIT text injection fix.

The ``WAIT "<text>" <timeout>`` command documented at
``docs/md_v2/api/c4a-script-reference.md`` promises plain "Case-sensitive text
matching". The original emitter interpolated user text into a JS template
literal (backticks) with only backtick escaping, so any ``${...}`` sequence was
evaluated as JavaScript in the page context. This produced silent code
execution (e.g. ``WAIT "${alert(1)}" 5``) and unbounded page hangs when the
interpolated expression threw (e.g. ``WAIT "price is ${productPrice}" 5``).

The fix emits the text inside a single-quoted JS string, escaping ``\\`` and
``'`` (matching the existing WAIT selector branch), so ``${...}`` is literal
text. These tests guard against re-introducing the template literal or the
missing escapes.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from crawl4ai.script import compile as c4a_compile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_text_js(src: str) -> str:
    """Compile a single c4a statement and return the generated JS block."""
    result = c4a_compile(src)
    assert result.success, f"Compilation failed: {result.errors}"
    assert len(result.js_code) == 1, f"Expected exactly 1 statement, got {len(result.js_code)}"
    return result.js_code[0]


def _includes_segment(js: str) -> str:
    """Return the ``.includes(...)`` call so assertions can target the quoted argument."""
    i = js.index(".includes(")
    j = i + len(".includes(")
    assert js[j] == "'", f"includes() argument is not single-quoted: {js[j:]!r}"
    j += 1
    while j < len(js):
        c = js[j]
        if c == "\\" and j + 1 < len(js):
            j += 2
            continue
        if c == "'":
            assert js[j + 1] == ")", f"unexpected char after closing quote: {js[j:]!r}"
            return js[i:j + 2]
        j += 1
    raise AssertionError("unterminated includes() argument")


# ===========================================================================
# Compiler output (deterministic, no browser)
# ===========================================================================


class TestWaitTextCompilerOutput:
    """Guard against re-introducing a JS template literal for the WAIT text."""

    def test_uses_single_quoted_string_not_template_literal(self):
        js = _wait_text_js('WAIT "hello" 5')
        assert ".includes('hello')" in js
        assert ".includes(`hello`)" not in js
        assert "`" not in js

    def test_dollar_curly_not_interpolated(self):
        js = _wait_text_js('WAIT "${alert(1)}" 5')
        assert ".includes('${alert(1)}')" in js
        assert ".includes(`${alert(1)}`)" not in js
        assert "`" not in js


class TestWaitTextEscaping:
    """Guard against removing the backslash/single-quote escaping."""

    def test_single_quote_escaped(self):
        js = _wait_text_js("WAIT \"it's a test\" 5")
        assert _includes_segment(js) == ".includes('it\\'s a test')"

    def test_backslash_escaped(self):
        # One literal backslash in the text must be doubled so JS sees a literal
        # backslash rather than an escape sequence.
        js = _wait_text_js('WAIT "a\\b" 5')
        assert _includes_segment(js) == ".includes('a\\\\b')"


# ===========================================================================
# End-to-end JS execution in a real browser (Playwright/Chromium)
# ===========================================================================

# These tests require Chromium (playwright install chromium). They wrap the
# compiled JS in the exact async-IIFE used by eval_js_code at runtime
# (async_crawler_strategy.py) so behaviour matches production.


def _runtime_wrap(script: str, harness_timeout_ms: int | None = None) -> str:
    """Wrap the compiled JS the way eval_js_code does, returning a structured
    result so the promise outcome is observable in Python.

    When ``harness_timeout_ms`` is set, the inner promise is raced against a
    JS-side watchdog so a never-settling promise (the original bug) fails the
    test cleanly instead of hanging the suite.
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


@pytest.fixture(scope="module")
def browser_page():
    """Launch Chromium and provide a fresh page for each test."""
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
class TestWaitTextSecurity:
    """The core security and hang regressions, verified in a real browser."""

    def test_dollar_curly_not_executed_as_javascript(self, browser_page):
        page = browser_page
        page.set_content("<html><body>Hello world</body></html>")
        page.evaluate(
            "() => { window.__alert_calls = 0; window.alert = () => { window.__alert_calls++; }; }"
        )
        js = _wait_text_js('WAIT "${alert(1)}" 1')
        result = page.evaluate(_runtime_wrap(js))
        calls = page.evaluate("() => window.__alert_calls")
        assert calls == 0, f"alert() was executed {calls} times — code injection regressed"
        assert result["success"] is False
        assert "WAIT text timeout" in result["error"]

    def test_undefined_variable_rejects_without_hang(self, browser_page):
        page = browser_page
        page.set_content("<html><body>Hello world</body></html>")
        js = _wait_text_js('WAIT "price is ${productPrice}" 1')
        t0 = time.monotonic()
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"Promise did not settle within 3s (elapsed={elapsed:.2f}s) — hang regressed"
        assert result["success"] is False
        assert "WAIT text timeout" in result["error"]

    def test_text_present_resolves_quickly(self, browser_page):
        page = browser_page
        page.set_content("<html><body>Loading complete</body></html>")
        js = _wait_text_js('WAIT "Loading complete" 5')
        t0 = time.monotonic()
        result = page.evaluate(_runtime_wrap(js))
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"Took too long to find present text: {elapsed:.2f}s"
        assert result["success"] is True
