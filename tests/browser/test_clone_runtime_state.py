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
from typing import Any, List, Tuple
from unittest.mock import AsyncMock

import pytest

from crawl4ai.browser_manager import clone_runtime_state


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
