"""Fast (no-browser) regression coverage for the `raw:` / `file://` console
capture fix in `UndetectedAdapter.bootstrap_capture_html` and the
`_inject_script_at_head_top` helper.

The bug (introduced by commit 2550f3d2's browser-pipeline path for
`raw:` / `file://` combined with `UndetectedAdapter`'s scheme-restricted
`add_init_script` capture) silently dropped all `console.log` output and
uncaught errors for crawls of local/raw HTML. The fix prepends the same
capture bootstrap that init_script would have installed as a `<script>`
tag at the top of `<head>`, so it runs synchronously *before* any user
inline script during `page.set_content` parsing. End-to-end coverage is
in `test_undetected_adapter_raw_console.py` (browser-marked); this file
covers the pure-Python side of the fix with no Chromium.
"""

import pytest

from crawl4ai.browser_adapter import (
    BrowserAdapter,
    PlaywrightAdapter,
    StealthAdapter,
    UndetectedAdapter,
    _inject_script_at_head_top,
)


# ---------------------------------------------------------------------------
# `BrowserAdapter.bootstrap_capture_html` default no-op contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_adapter_default_bootstrap_capture_html_is_noop():
    """The base class default must be a no-op so that adapters whose
    standard capture mechanism already works for every scheme
    (PlaywrightAdapter / StealthAdapter — `page.on("console")` /
    `page.on("pageerror")` events) can simply inherit it and pay no extra
    cost. The strategy calls `bootstrap_capture_html` unconditionally in
    the local-content branch when `capture_console_messages` is on, so a
    misbehaving default (e.g. one that always prepended an empty
    `<script></script>`) would corrupt every default-adapter crawl's
    HTML."""
    class _Probe(BrowserAdapter):
        async def evaluate(self, page, expression, arg=None): pass
        async def setup_console_capture(self, page, captured_console): pass
        async def setup_error_capture(self, page, captured_console): pass
        async def retrieve_console_messages(self, page): pass
        async def cleanup_console_capture(self, page, h_c, h_e): pass

    adapter = _Probe()
    original = "<html><head><title>X</title></head><body></body></html>"
    assert adapter.bootstrap_capture_html(original) is not original or True  # bytes-equal OR same identity is fine
    assert adapter.bootstrap_capture_html(original) == original, (
        "default bootstrap_capture_html must return the HTML unchanged"
    )


@pytest.mark.asyncio
async def test_playwright_and_stealth_adapter_inherit_noop_bootstrap():
    """`PlaywrightAdapter` and `StealthAdapter` must inherit the no-op
    default — their event-based / standard-Playwright CDP `add_init_script`
    capture already works for every scheme, so they must not prepend
    anything to the local-content HTML. A naive future override that
    prepended an init-script-based bootstrap to the HTML on every default
    crawl would double-inject the bootstrap (init script + prepend) on
    http pages where the page is later reused, and would corrupt HTML
    output via the prepended wrapper."""
    for adapter_cls in (PlaywrightAdapter, StealthAdapter):
        adapter = adapter_cls()
        original = "<html><head></head><body>hello</body></html>"
        assert adapter.bootstrap_capture_html(original) == original, (
            f"{adapter_cls.__name__}.bootstrap_capture_html must be a no-op"
        )


# ---------------------------------------------------------------------------
# `UndetectedAdapter.bootstrap_capture_html` prepends the bootstrap.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undetected_adapter_bootstrap_prepends_combined_capture_script():
    """The fix: `UndetectedAdapter.bootstrap_capture_html` must prepend a
    `<script>` containing the *combined* console-AND-error capture
    bootstrap, as the first child of `<head>`, so both `console.log`
    overrides AND `error` / `unhandledrejection` listeners are installed
    in time for inline scripts in `set_content` documents."""
    adapter = UndetectedAdapter()
    html = (
        "<!DOCTYPE html><html><head><title>doc</title></head>"
        "<body><script>console.log('RAW_LOG');</script></body></html>"
    )
    wrapped = adapter.bootstrap_capture_html(html)

    # The wrapped HTML must contain the capture bootstrap as a <script>.
    assert "<script>" in wrapped, "missing <script> wrapper for the bootstrap"
    # ... and that script must include both capture scripts' contents.
    assert "window.__capturedConsole" in wrapped, "console capture missing"
    assert "console[method] = function" in wrapped, "console override missing"
    assert "window.addEventListener('error'" in wrapped, "error listener missing"
    assert "window.addEventListener('unhandledrejection'" in wrapped, (
        "unhandledrejection listener missing"
    )
    # The bootstrap must be at the top of <head>, i.e. the first child of
    # <head>, so it runs before the user's inline <script>consum.log('RAW_LOG')>.
    head_idx = wrapped.lower().find("<head>")
    user_log_idx = wrapped.find("RAW_LOG")
    script_open_idx = wrapped.find("<script>")
    assert head_idx != -1 and script_open_idx > head_idx, (
        "bootstrap <script> must come after <head> opening tag"
    )
    assert user_log_idx > script_open_idx, (
        "bootstrap <script> must come before the user's inline RAW_LOG script"
    )
    # The original HTML's body must still be present (no content loss).
    assert "RAW_LOG" in wrapped
    assert "<title>doc</title>" in wrapped


@pytest.mark.asyncio
async def test_undetected_adapter_bootstrap_uses_same_bodies_as_init_scripts():
    """The prepended bootstrap must use the SAME script bodies that the
    init scripts install via `add_init_script`, so the in-page buffer /
    override / listener state is byte-identical between the http(s)
    init-script path and the set_content prepend path. Downstream
    `retrieve_console_messages` cannot tell the two paths apart — that
    property is the only reason the fix is safe without touching
    retrieval.

    The bootstrap wraps each body in its own IIFE so the top-level
    ``const originalConsole`` does not leak into the page's global
    lexical environment when the bootstrap runs as a classic
    ``<script>`` (``add_init_script`` already scopes its body inside a
    function, so the init-script path is unaffected). State still flows
    through ``window.__capturedConsole`` / ``window.__capturedErrors``,
    so the in-page state the two paths leave is identical even though
    the literal source differs by the IIFE wrappers."""
    adapter = UndetectedAdapter()
    assert adapter._CONSOLE_CAPTURE_JS in adapter._CAPTURE_BOOTSTRAP_JS, (
        "_CAPTURE_BOOTSTRAP_JS must contain the console-capture init-script "
        "body so the prepend and init-script paths leave identical state"
    )
    assert adapter._ERROR_CAPTURE_JS in adapter._CAPTURE_BOOTSTRAP_JS, (
        "_CAPTURE_BOOTSTRAP_JS must contain the error-capture init-script "
        "body so the prepend and init-script paths leave identical state"
    )


@pytest.mark.asyncio
async def test_undetected_adapter_bootstrap_does_not_contain_closing_script_tag():
    """The bootstrap MUST NOT contain the literal `</script>` substring,
    because `_inject_script_at_head_top` wraps it inside a single
    `<script>{body}</script>` tag — a stray `</script>` inside the body
    would terminate the wrapper early and break the in-page JS. This
    pins the contract the helper relies on (the bootstrap JS is constant
    and reviewed)."""
    adapter = UndetectedAdapter()
    assert "</script>" not in adapter._CAPTURE_BOOTSTRAP_JS, (
        "_CAPTURE_BOOTSTRAP_JS must not contain the literal `</script>` — "
        "doing so would break _inject_script_at_head_top's <script>{body}</script> wrapper"
    )
    assert "</script>" not in adapter._CONSOLE_CAPTURE_JS
    assert "</script>" not in adapter._ERROR_CAPTURE_JS


# ---------------------------------------------------------------------------
# `_inject_script_at_head_top` — pure-Python HTML insertion edge cases.
# ---------------------------------------------------------------------------


def test_inject_into_html_with_existing_head():
    """A document with a `<head>` opening tag gets the bootstrap inserted
    as the first child of that head — the existing head's other children
    (e.g. `<title>`) remain after it."""
    html = "<!DOCTYPE html><html><head><title>X</title></head><body></body></html>"
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out == (
        "<!DOCTYPE html><html><head><script>/*BS*/</script><title>X</title>"
        "</head><body></body></html>"
    ), out


def test_inject_into_html_with_head_attributes():
    """A `<head>` tag with attributes (`<head class="...">`) is matched
    and the bootstrap is inserted as its first child."""
    html = '<html><head class="main"><title>Y</title></head></html>'
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out == (
        '<html><head class="main"><script>/*BS*/</script><title>Y</title></head></html>'
    ), out


def test_inject_into_html_with_only_html_no_head():
    """A document with `<html>` but no `<head>` gets a synthetic
    `<head><script>...</script></head>` inserted right after `<html>`.
    The HTML5 parser merges any subsequent `<head>` opening tag in the
    original HTML into this synthetic one."""
    html = "<html><body><p>hi</p></body></html>"
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out == (
        "<html><head><script>/*BS*/</script></head><body><p>hi</p></body></html>"
    ), out


def test_inject_into_html_with_no_html_or_head_wraps_fragment():
    """A bare fragment with no `<html>` or `<head>` is not wrapped in
    synthetic ``<html>``/``<head>``/``<body>`` tags — the HTML5 parser
    places a leading ``<script>`` into the implied ``<head>``, so it
    still runs before any subsequent inline script. Wrapping would
    insert a synthetic ``<head>`` *after* a leading ``<!DOCTYPE html>``
    (when present), which the parser treats as a parse error and
    drops the doctype, flipping the document into quirks mode; leaving
    the fragment unwrapped preserves any leading doctype / XML decl /
    comments so standards mode survives."""
    html = "<p>just a paragraph</p>"
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out == "<script>/*BS*/</script><p>just a paragraph</p>", out


def test_inject_is_case_insensitive_on_head_and_html_tags():
    """HTML5 tag names are case-insensitive (`<HTML>`, `<HEAD>` are
    legal). The matcher must handle uppercase / mixedcase tags so it
    does not silently leave the bootstrap unwrapped for documents that
    use uppercase tags."""
    html = "<HTML><HEAD><TITLE>UP</TITLE></HEAD><BODY></BODY></HTML>"
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out == (
        "<HTML><HEAD><script>/*BS*/</script><TITLE>UP</TITLE></HEAD><BODY></BODY></HTML>"
    ), out


def test_inject_preserves_content_outside_head():
    """The function must never drop or mutate the original HTML outside
    the insertion point — the strategy feeds the wrapped HTML directly
    to `page.set_content`, so any content loss would change the crawl
    result. Insert text/markers/whitespace and assert they survive."""
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n  <meta charset='utf-8'>\n"
        "  <title>Doc</title>\n</head>\n<body>\n"
        "  <h1>Title</h1>\n  <p>Body content with\n newlines and <b>markup</b>.</p>\n"
        "</body>\n</html>\n"
    )
    out = _inject_script_at_head_top(html, "/*BS*/")

    # Every line of original content must still be present.
    for needle in [
        "<!DOCTYPE html>",
        "<meta charset='utf-8'>",
        "<title>Doc</title>",
        "<h1>Title</h1>",
        "<b>markup</b>",
        "</body>",
        "</html>",
    ]:
        assert needle in out, f"content lost after injection: {needle!r} missing from {out!r}"
    # The <script> wrapper must be present exactly once and at the top
    # of <head> (before <meta> and <title>).
    assert out.count("<script>/*BS*/</script>") == 1
    assert out.index("<script>/*BS*/</script>") < out.index("<meta charset='utf-8'>")
    assert out.index("<script>/*BS*/</script>") < out.index("<title>Doc</title>")


def test_inject_handles_html_with_xml_declaration_and_comment():
    """Documents that begin with an XML declaration (`<?xml ?>`) or a
    comment (`<!-- -->`) before `<html>` are still processed by
    inserting the bootstrap as the first child of the first `<head>`
    found, OR by inserting a synthetic `<head>` after `<html>` if no
    head exists. The XML decl / comment survives unchanged."""
    html = "<?xml version='1.0'?><html><body>hi</body></html>"
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out.startswith("<?xml version='1.0'?>"), (
        "XML declaration must survive at the very start: " + out
    )
    assert "<head><script>/*BS*/</script></head>" in out


def test_inject_into_empty_string_does_not_crash():
    """An empty HTML string still produces a runnable script; the
    function must not raise on degenerate input the strategy might
    pass if a caller handed `raw:` an empty string. No synthetic
    ``<html>``/``<head>``/``<body>`` wrapper is emitted — the bootstrap
    script alone is enough for ``set_content`` to load it."""
    out = _inject_script_at_head_top("", "/*BS*/")
    assert out == "<script>/*BS*/</script>", out


def test_inject_skips_head_tag_inside_html_comment():
    """A ``<head>``-like substring inside an HTML comment (e.g.
    ``<!-- <head> -->``) must NOT be treated as the head opening tag —
    otherwise the bootstrap would be inserted before ``-->`` and the
    HTML5 parser would treat it as comment text, leaving the real
    ``<head>`` unbootstrapped so the following page script runs without
    capture. The comment-aware search must skip the comment and find the
    real ``<head>`` after it."""
    html = (
        "<!-- <head> --><html><head><title>real</title></head>"
        "<body><script>console.log('x')</script></body></html>"
    )
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out.startswith("<!-- <head> --><html><head>"), (
        "the comment must survive intact and the real <head> must be found: "
        + out
    )
    assert "<script>/*BS*/</script><title>real</title>" in out, (
        "bootstrap must be the first child of the real <head>, not inserted "
        "into the comment: " + out
    )


def test_inject_skips_head_tag_inside_script_raw_text():
    """A ``<head>``-like substring inside an earlier ``<script>``'s raw
    text (e.g. ``<script>const s = "<head>";</script>``) must NOT be
    treated as the head opening tag — otherwise the bootstrap is injected
    inside the user script, the generated ``</script>`` closes the user
    script early (truncating it), and the real ``<head>`` after the
    script stays unbootstrapped so the following page script runs without
    capture. The script-aware search must skip the entire
    ``<script>...</script>`` region and find the real ``<head>`` after
    it."""
    html = (
        '<!DOCTYPE html><script>const s = "<head>";</script>'
        "<html><head><title>real</title></head>"
        "<body><script>console.log('x')</script></body></html>"
    )
    out = _inject_script_at_head_top(html, "/*BS*/")
    # The user script must survive intact (not truncated by an injected
    # </script>).
    assert 'const s = "<head>";' in out, (
        "user script must not be truncated by the bootstrap insertion: " + out
    )
    # The bootstrap must land inside the real <head>, after the user
    # script, not inside it.
    real_head_idx = out.find("<html><head>")
    assert real_head_idx != -1, "real <head> after the user script must be found: " + out
    script_bs_idx = out.find("<script>/*BS*/</script>")
    assert script_bs_idx > real_head_idx, (
        "bootstrap must be the first child of the real <head>, not inserted "
        "into the earlier script: " + out
    )
    assert "<script>/*BS*/</script><title>real</title>" in out, (
        "bootstrap must precede <title> inside the real <head>: " + out
    )


def test_inject_into_fragment_with_leading_doctype_preserves_standards_mode():
    """A document that omits both ``<html>`` and ``<head>`` but keeps a
    leading ``<!DOCTYPE html>`` (a legal HTML5 document, e.g.
    ``<!DOCTYPE html><title>T</title><p>hi</p>``) must keep that doctype
    first in the output. The previous fallback wrapped the document as
    ``<html><head>...</head><body><!DOCTYPE html>...``, which the HTML5
    parser treats as a parse error — the doctype after ``<html>`` is
    ignored and the document renders in quirks mode, changing layout,
    screenshots, and dimension checks. The bootstrap is inserted right
    after the leading doctype so the parser still places it in the
    implied ``<head>`` and runs it before any subsequent inline script."""
    html = "<!DOCTYPE html><title>T</title><p>hi</p>"
    out = _inject_script_at_head_top(html, "/*BS*/")
    assert out.startswith("<!DOCTYPE html><script>/*BS*/</script>"), (
        "doctype must stay first and the bootstrap must follow it directly, "
        "without a synthetic <html>/<head> wrapper: " + out
    )
    assert "<title>T</title>" in out and "<p>hi</p>" in out, (
        "the original fragment content must survive unchanged: " + out
    )
    assert "<html>" not in out and "<head>" not in out, (
        "no synthetic wrapper must be added when the input has no <html>: " + out
    )
