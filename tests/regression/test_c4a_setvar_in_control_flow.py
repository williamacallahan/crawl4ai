#!/usr/bin/env python3
"""
Regression tests for the C4A-Script ``$var`` substitution inside IF/REPEAT
bodies fix.

The C4A-Script variable mechanism is a *compile-time, Python-side* string
substitution pass (``Compiler._apply_set_vars`` in
``crawl4ai/script/c4ai_script.py``). It records each ``SETVAR name = "value"``
into ``self.vars`` (in source order) and replaces ``$name`` tokens inside the
args of ``TYPE``/``EVAL``/``SET`` commands with the stored values before JS
emission. There is no runtime variable store.

The original pass (introduced in commit 3f6f2e9) walked only the flat
top-level command list and substituted only the args of top-level
``TYPE``/``EVAL``/``SET`` commands. ``IF`` and ``REPEAT`` carry their body
commands as nested ``Cmd`` objects inside ``c.args`` (e.g.
``Cmd("IF", [condition, then_cmd, else_cmd])``,
``Cmd("REPEAT", [cmd, count])``). The pass never descended into those nested
``Cmd``s, so a ``$var`` that substituted correctly at the top level was left
as the literal text ``$var`` when the same ``TYPE``/``EVAL``/``SET`` command
was placed textually inside an ``IF`` or ``REPEAT`` body.

The fix recurses into the nested ``Cmd``s carried by ``IF`` (then/else) and
``REPEAT`` (body) during ``_apply_set_vars``, applying the same
``TYPE``/``EVAL``/``SET`` arg substitution. The substitution *scope* is
unchanged: only ``TYPE``/``EVAL``/``SET`` args are substituted (as before); a
``$var`` in a ``CLICK`` selector, an ``IF`` condition, or a ``REPEAT`` count is
still left literal, exactly as it was at the top level. Undefined variables
still fall back to the literal ``$name`` token.

Preconditions: Chromium installed (``playwright install chromium``) for the
``@pytest.mark.browser`` tests. The compiler-output tests need no browser.
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
    """Compile a c4a script that emits exactly one JS statement (a ``SETVAR``
    definition followed by a single ``IF``/``REPEAT``/``TYPE`` command) and
    return that statement.

    ``SETVAR`` is consumed by the substitution pass and not emitted, so the
    single emitted statement is the control-flow / leaf command.
    """
    result = c4a_compile(src)
    assert result.success, f"Compilation failed: {result.errors}"
    assert len(result.js_code) == 1, (
        f"Expected exactly 1 emitted statement, got {len(result.js_code)}: {result.js_code}"
    )
    return result.js_code[0]


def _runtime_wrap(script: str, harness_timeout_ms: int | None = None) -> str:
    """Wrap the compiled JS the way ``robust_execute_user_script``
    (async_crawler_strategy.py) does, returning a structured result so the
    promise outcome is observable in Python.
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


class TestSetvarSubstitutionInControlFlow:
    """A top-level ``$var`` must substitute inside ``IF``/``REPEAT`` bodies,
    including at arbitrary nesting depth, while the substitution scope and
    fallback behaviour are unchanged."""

    def test_substitutes_in_if_then_type(self):
        js = _compile('SETVAR username = "bob"\nIF (EXISTS `#login`) THEN TYPE $username')
        assert "'bob'" in js, "then-branch TYPE arg not substituted"
        assert "$username" not in js, "literal $username survived in then branch"

    def test_substitutes_in_if_then_set_value_slot(self):
        js = _compile('SETVAR v = "x"\nIF (EXISTS `#i`) THEN SET `#i` $v')
        assert "'x'" in js, "then-branch SET value slot not substituted"
        assert "$v" not in js, "literal $v survived in then branch"

    def test_substitutes_in_if_then_eval(self):
        js = _compile('SETVAR msg = "hi"\nIF (EXISTS `#i`) THEN EVAL `console.log("$msg")`')
        assert '"hi"' in js, "then-branch EVAL arg not substituted"
        assert "$msg" not in js, "literal $msg survived in then branch"

    def test_substitutes_in_both_if_branches(self):
        js = _compile(
            'SETVAR greeting = "hello"\n'
            'IF (EXISTS `#a`) THEN TYPE $greeting ELSE TYPE $greeting'
        )
        assert js.count("'hello'") == 2, "both branches must substitute"
        assert "$greeting" not in js, "literal $greeting survived"

    def test_substitutes_in_repeat_type(self):
        js = _compile('SETVAR greeting = "hello"\nREPEAT (TYPE $greeting, 3)')
        assert "'hello'" in js, "REPEAT body TYPE arg not substituted"
        assert "$greeting" not in js, "literal $greeting survived in REPEAT body"

    def test_repeat_numeric_count_is_unchanged(self):
        """The count slot and the emitted loop shape must not be disturbed by
        the substitution fix."""
        js = _compile('SETVAR greeting = "hello"\nREPEAT (TYPE $greeting, 3)')
        assert "for (let _i = 0; _i < 3; _i++)" in js, "numeric count changed"
        assert "while" not in js

    def test_substitutes_in_nested_if_inside_if(self):
        js = _compile(
            'SETVAR a = "A"\n'
            'IF (EXISTS `#x`) THEN IF (EXISTS `#y`) THEN TYPE $a'
        )
        assert "'A'" in js, "doubly-nested IF body not substituted"
        assert "$a" not in js

    def test_substitutes_in_if_inside_repeat(self):
        js = _compile(
            'SETVAR g = "hi"\n'
            'REPEAT (IF (EXISTS `#x`) THEN TYPE $g, 3)'
        )
        assert "'hi'" in js, "IF-in-REPEAT body not substituted"
        assert "$g" not in js
        assert "for (let _i = 0; _i < 3; _i++)" in js

    def test_substitutes_in_repeat_inside_if(self):
        js = _compile(
            'SETVAR g = "hi"\n'
            'IF (EXISTS `#x`) THEN REPEAT (TYPE $g, 2)'
        )
        assert "'hi'" in js, "REPEAT-in-IF body not substituted"
        assert "$g" not in js
        assert "for (let _i = 0; _i < 2; _i++)" in js

    def test_toplevel_setvar_still_substitutes(self):
        """Regression guard: the pre-existing top-level substitution path is
        unchanged by the recursion fix."""
        js = _compile('SETVAR greeting = "hello"\nTYPE $greeting')
        assert "'hello'" in js
        assert "$greeting" not in js

    def test_undefined_var_inside_if_stays_literal(self):
        """No SETVAR defined this var; the substitution fallback leaves the
        literal ``$name`` (no crash, no KeyError). This preserves the existing
        fallback behaviour of ``self.vars.get(name, m.group(0))``."""
        js = _compile('IF (EXISTS `#x`) THEN TYPE $undefined_var')
        assert "$undefined_var" in js, "undefined var was dropped or mangled"

    def test_substitution_not_applied_to_click_selector_inside_if(self):
        """``CLICK`` is not in (TYPE, EVAL, SET); a ``$var`` in its selector
        must NOT be substituted. Consistent with the top-level behaviour;
        guards against the fix over-reaching."""
        js = _compile('SETVAR sel = "x"\nIF (EXISTS `#y`) THEN CLICK `$sel`')
        assert "$sel" in js, "CLICK selector was wrongly substituted"
        assert "'x'" not in js, "var value leaked into the CLICK selector"

    def test_if_condition_selector_not_substituted(self):
        """The ``EXISTS`` condition selector is an ``IF`` arg (``args[0]``),
        not a TYPE/EVAL/SET arg; a ``$var`` there must stay literal."""
        js = _compile('SETVAR s = "#x"\nIF (EXISTS `$s`) THEN TYPE "ok"')
        assert "$s" in js, "IF condition selector was wrongly substituted"
        assert "'#x'" not in js, "var value leaked into the IF condition"

    def test_repeat_count_not_substituted(self):
        """The ``REPEAT`` count (``args[1]``) is not a substituted slot; a
        backticked ``$var`` count stays literal (routed to the while branch),
        exactly as at the top level. Guards against the fix over-reaching."""
        js = _compile('SETVAR n = "3"\nREPEAT (TYPE "x", `$n`)')
        assert "$n" in js, "REPEAT count was wrongly substituted"
        # The numeric var value '3' must not appear as the loop bound.
        assert "for (let _i = 0; _i < 3; _i++)" not in js, (
            "var value leaked into the REPEAT count slot"
        )

    def test_later_setvar_updates_value_used_in_if_body(self):
        """Variable resolution follows source order, including across the
        top-level / control-flow boundary."""
        js = _compile(
            'SETVAR v = "first"\n'
            'SETVAR v = "second"\n'
            'IF (EXISTS `#x`) THEN TYPE $v'
        )
        assert "'second'" in js, "later SETVAR did not update the value used inside IF"
        assert "'first'" not in js
        assert "$v" not in js


# ===========================================================================
# End-to-end JS execution in a real browser (Playwright/Chromium)
# ===========================================================================


@pytest.fixture
def browser_page():
    """Launch Chromium and provide a fresh page for each test, so input
    value mutations (TYPE does ``el.value += '...'``) cannot leak between
    tests."""
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
class TestSetvarInControlFlowRuntime:
    """The core substitution regressions, verified in a real browser: the
    *value* (not the literal ``$var``) reaches the active input, and control
    flow is intact."""

    def test_if_then_type_substitutes_value_on_page(self, browser_page):
        r"""``IF (EXISTS `#present`) THEN TYPE $v`` must type the value of
        ``v`` into the focused input, not the literal ``$v``. Before the fix
        the input received ``$v``."""
        page = browser_page
        page.set_content(
            '<!doctype html><html><body>'
            '<div id="present"></div>'
            '<input id="target">'
            '</body></html>'
        )
        page.evaluate("() => document.querySelector('#target').focus()")
        js = _compile('SETVAR v = "hello"\nIF (EXISTS `#present`) THEN TYPE $v')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"], f"script failed: {result}"
        value = page.evaluate("() => document.querySelector('#target').value")
        assert value == "hello", f"input got {value!r}, expected 'hello' (literal $v survived)"

    def test_if_false_condition_skips_substituted_body(self, browser_page):
        """The substitution fix must not change control flow: when the IF
        condition is false, the (substituted) body still does not run, so the
        input stays empty."""
        page = browser_page
        page.set_content('<!doctype html><html><body><input id="target"></body></html>')
        page.evaluate("() => document.querySelector('#target').focus()")
        js = _compile('SETVAR v = "hello"\nIF (EXISTS `#absent`) THEN TYPE $v')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"], f"script failed: {result}"
        assert page.evaluate("() => document.querySelector('#target').value") == ""

    def test_repeat_type_substitutes_value_each_iteration(self, browser_page):
        """``REPEAT (TYPE $g, 3)`` must type the value ``ab`` three times
        (``el.value += 'ab'`` x3 => ``'ababab'``), not the literal ``$g``.
        Before the fix the input received ``$g$g$g``."""
        page = browser_page
        page.set_content('<!doctype html><html><body><input id="target"></body></html>')
        page.evaluate("() => document.querySelector('#target').focus()")
        js = _compile('SETVAR g = "ab"\nREPEAT (TYPE $g, 3)')
        result = page.evaluate(_runtime_wrap(js, harness_timeout_ms=5000))
        assert result["success"], f"script failed: {result}"
        value = page.evaluate("() => document.querySelector('#target').value")
        assert value == "ababab", f"input got {value!r}, expected 'ababab' (literal $g survived)"
