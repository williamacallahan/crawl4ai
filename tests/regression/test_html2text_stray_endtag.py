"""Regression tests for stray (unbalanced) ``</script>``, ``</style>``, and
``</head>`` end tags in the vendored html2text converter.

Background
----------
``crawl4ai/html2text/__init__.py`` keeps a ``self.quiet`` suppression-depth
counter for ``head``/``style``/``script`` content. Before the fix, a stray
end tag (e.g. ``</script>`` with no matching ``<script>``) decremented
``quiet`` from ``0`` to ``-1``. Because ``-1`` is truthy, the
``if not self.quiet:`` gate in ``o()`` then suppressed *all* subsequent
output for the rest of the document, silently truncating the markdown.

These tests confirm the fix (guard the decrement so ``quiet`` never drops
below zero) and assert there are no regressions for balanced inputs.
"""

import os
import subprocess
import sys

import pytest

from crawl4ai.html2text import HTML2Text, html2text

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CLI_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH", "")])
    ),
}


def _convert(html: str) -> str:
    converter = HTML2Text()
    converter.body_width = 0
    return converter.handle(html)


# ---------------------------------------------------------------------------
# Core bug: stray end tags must not truncate subsequent output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "end_tag",
    ["script", "style", "head"],
    ids=["stray-script", "stray-style", "stray-head"],
)
def test_stray_end_tag_does_not_truncate_output(end_tag):
    html = (
        "<p>Before</p>"
        f"<p>Use </{end_tag}> here</p>"
        f"<p>After {end_tag} should appear</p>"
    )
    result = _convert(html)
    assert "Before" in result, f"stray </{end_tag}> dropped preceding content"
    assert (
        f"After {end_tag} should appear" in result
    ), f"stray </{end_tag}> truncated the document tail"
    assert "here" in result


def test_stray_script_end_tag_via_html2text_function():
    """The public ``html2text()`` convenience function must also be fixed."""
    result = html2text(
        '<p>Before</p><p>Use </script> here</p><p>After this should appear</p>'
    )
    assert "After this should appear" in result
    assert "Before" in result


def test_stray_style_end_tag_via_html2text_function():
    result = html2text(
        '<p>Before</p><p>Use </style> here</p><p>After should appear</p>'
    )
    assert "After should appear" in result


def test_stray_head_end_tag_via_html2text_function():
    result = html2text(
        '<p>Before</p><p>Use </head> here</p><p>After should appear</p>'
    )
    assert "After should appear" in result


def test_multiple_stray_end_tags_in_sequence():
    """Several stray end tags must not compound into suppression."""
    html = (
        "<p>start</p>"
        "</script>"
        "</style>"
        "</head>"
        "<p>middle</p>"
        "</script></script>"
        "<p>end</p>"
    )
    result = _convert(html)
    assert "start" in result
    assert "middle" in result
    assert "end" in result


def test_stray_end_tag_at_start_of_document():
    """A stray end tag as the very first node must not truncate anything."""
    html = "</script><p>visible content</p>"
    result = _convert(html)
    assert "visible content" in result


def test_stray_end_tag_at_end_of_document():
    """A stray end tag as the very last node must not break final text."""
    html = "<p>visible content</p></script>"
    result = _convert(html)
    assert "visible content" in result


def test_mixed_stray_end_tags():
    html = "<p>a</p></style><p>b</p></script><p>c</p></head><p>d</p>"
    result = _convert(html)
    for expected in ("a", "b", "c", "d"):
        assert expected in result, f"missing {expected!r}: {result!r}"


# ---------------------------------------------------------------------------
# No regression: balanced head/style/script still suppress their content
# ---------------------------------------------------------------------------


def test_balanced_script_content_is_still_suppressed():
    result = _convert("<p>before</p><script>var x = 1;</script><p>after</p>")
    assert "before" in result
    assert "after" in result
    assert "var x = 1;" not in result, "script body leaked into markdown"


def test_balanced_style_content_is_still_suppressed():
    result = _convert(
        "<p>before</p><style>body { color: red; }</style><p>after</p>"
    )
    assert "before" in result
    assert "after" in result
    assert "color: red" not in result
    assert "color:red" not in result


def test_balanced_head_content_is_still_suppressed():
    result = _convert(
        "<head><title>Page Title</title><meta charset=\"utf-8\"></head>"
        "<body><p>visible</p></body>"
    )
    assert "visible" in result
    assert "Page Title" not in result


def test_two_consecutive_balanced_scripts():
    """Two script blocks one after another still both get suppressed."""
    result = _convert(
        "<p>a</p><script>x</script><p>b</p><script>y</script><p>c</p>"
    )
    assert "a" in result
    assert "b" in result
    assert "c" in result
    assert "x" not in result
    assert "y" not in result


# ---------------------------------------------------------------------------
# Interaction cases: script CDATA + stray end tags
# ---------------------------------------------------------------------------


def test_script_with_embedded_close_string_then_stray_close():
    """A ``</script>`` inside a JS string closes the block (HTMLParser
    semantics), then a trailing stray ``</script>`` must be ignored."""
    html = (
        "<p>text</p>"
        '<script>var x = "</script>";alert(1)</script>'
        "<p>after should appear</p>"
    )
    result = _convert(html)
    assert "text" in result
    assert "after should appear" in result
    assert "var x =" not in result


def test_stray_end_tag_then_balanced_block():
    """Stray end tag followed by a legitimately balanced block."""
    html = (
        "<p>before</p>"
        "</script>"
        "<style>body { color: red; }</style>"
        "<p>after</p>"
    )
    result = _convert(html)
    assert "before" in result
    assert "after" in result
    assert "color: red" not in result


def test_balanced_block_then_stray_then_balanced_block():
    html = (
        "<style>a{}</style>"
        "<p>between</p>"
        "</style>"
        "<style>b{}</style>"
        "<p>after</p>"
    )
    result = _convert(html)
    assert "between" in result
    assert "after" in result
    assert "a{}" not in result
    assert "b{}" not in result


# ---------------------------------------------------------------------------
# quiet counter invariants
# ---------------------------------------------------------------------------


def _make_tracing_converter():
    class Tracing(HTML2Text):
        def __init__(self):
            super().__init__()
            self.min_quiet = 0

        def handle_tag(self, tag, attrs, start):
            super().handle_tag(tag, attrs, start)
            if self.quiet < self.min_quiet:
                self.min_quiet = self.quiet

    return Tracing()


def test_quiet_never_goes_negative_with_stray_end_tags():
    converter = _make_tracing_converter()
    converter.body_width = 0
    converter.handle(
        "<p>Before</p><p>Use </script> here</p>"
        "<p>Use </style> here</p><p>Use </head> here</p>"
    )
    assert converter.min_quiet >= 0, "quiet went negative"
    assert converter.quiet == 0, "quiet not restored to 0"


def test_quiet_restored_to_zero_after_balanced_blocks():
    converter = _make_tracing_converter()
    converter.body_width = 0
    converter.handle(
        "<head><title>x</title></head>"
        "<style>a{}</style>"
        "<script>b</script>"
        "<p>body</p>"
    )
    assert converter.quiet == 0
    assert converter.min_quiet >= 0


# ---------------------------------------------------------------------------
# CLI (python -m crawl4ai.html2text) - third public entry point
# ---------------------------------------------------------------------------


def test_cli_stray_script_end_tag_does_not_truncate():
    html = (
        "<p>Before</p><p>Use </script> here</p>"
        "<p>After this should appear</p>"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "crawl4ai.html2text", "-b", "0"],
        input=html,
        capture_output=True,
        text=True,
        check=True,
        env=_CLI_ENV,
    )
    out = proc.stdout
    assert "Before" in out
    assert "After this should appear" in out, (
        f"CLI truncated output: {out!r}"
    )


def test_cli_well_formed_html_unchanged():
    html = "<p>Hello world</p><p>Second paragraph</p>"
    proc = subprocess.run(
        [sys.executable, "-m", "crawl4ai.html2text", "-b", "0"],
        input=html,
        capture_output=True,
        text=True,
        check=True,
        env=_CLI_ENV,
    )
    assert "Hello world" in proc.stdout
    assert "Second paragraph" in proc.stdout


# ---------------------------------------------------------------------------
# General no-regression on typical well-formed documents
# ---------------------------------------------------------------------------


def test_well_formed_document_with_full_head_and_body():
    html = (
        "<html><head><title>Title</title>"
        "<style>body{color:red}</style>"
        "<script>alert('hi')</script>"
        "</head><body>"
        "<h1>Heading</h1><p>Paragraph one.</p>"
        "<p>Paragraph two.</p>"
        "</body></html>"
    )
    result = _convert(html)
    assert "Heading" in result
    assert "Paragraph one." in result
    assert "Paragraph two." in result
    assert "Title" not in result
    assert "color:red" not in result
    assert "alert('hi')" not in result


def test_empty_input_still_works():
    assert _convert("").strip() == ""


def test_plain_text_no_tags_still_works():
    assert _convert("just plain text").strip() == "just plain text"
