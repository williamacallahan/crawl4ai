# browser_adapter.py
"""
Browser adapter for Crawl4AI to support both Playwright and undetected browsers
with minimal changes to existing codebase.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import re
import time
import json
import weakref

from .browser_budget import within_page_budget

# Import both, but use conditionally
try:
    from playwright.async_api import Page
except ImportError:
    Page = Any

try:
    from patchright.async_api import Page as UndetectedPage
except ImportError:
    UndetectedPage = Any


class BrowserAdapter(ABC):
    """Abstract adapter for browser-specific operations"""
    
    @abstractmethod
    async def evaluate(self, page: Page, expression: str, arg: Any = None) -> Any:
        """Execute JavaScript in the page"""
        pass
    
    @abstractmethod
    async def setup_console_capture(self, page: Page, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup console message capturing, returns handler function if needed"""
        pass
    
    @abstractmethod
    async def setup_error_capture(self, page: Page, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup error capturing, returns handler function if needed"""
        pass
    
    @abstractmethod
    async def retrieve_console_messages(self, page: Page) -> List[Dict]:
        """Retrieve captured console messages (for undetected browsers)"""
        pass
    
    @abstractmethod
    async def cleanup_console_capture(self, page: Page, handle_console: Optional[Callable], handle_error: Optional[Callable]):
        """Clean up console event listeners"""
        pass

    def bootstrap_capture_html(self, html: str) -> str:
        """Return an HTML document with console/error capture bootstrapped
        for documents that bypass the adapter's standard
        ``setup_console_capture`` / ``setup_error_capture`` mechanism.

        The default implementation is a no-op: most adapters arm capture
        through scheme-agnostic mechanisms (``page.on("console")`` /
        ``page.on("pageerror")``, or ``add_init_script`` via the standard
        Playwright CDP path that runs for every new document including
        ``about:blank``/``set_content``/``file://``). Those adapters do not
        need to override this method.

        Adapters whose standard capture mechanism is scheme-restricted —
        ``UndetectedAdapter``, where patchright 1.56's ``add_init_script``
        only rewrites the response of ``http(s)``-scheme network ``document``
        requests — must override this method to prepend a capture bootstrap
        that runs synchronously when the local-content document is parsed
        via ``page.set_content``. Without that, console messages and JS
        errors from inline scripts in ``raw:`` / ``raw://`` / ``file://``
        crawls are silently dropped.

        Called by ``AsyncPlaywrightCrawlerStrategy._crawl_web`` from the
        ``is_local_content`` branch when ``capture_console_messages`` is
        enabled, immediately before ``page.set_content(html, ...)``.
        """
        return html


class PlaywrightAdapter(BrowserAdapter):
    """Adapter for standard Playwright"""
    
    async def evaluate(self, page: Page, expression: str, arg: Any = None) -> Any:
        """Standard Playwright evaluate"""
        if arg is not None:
            return await within_page_budget(page.evaluate(expression, arg))
        return await within_page_budget(page.evaluate(expression))
    
    async def setup_console_capture(self, page: Page, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup console capture using Playwright's event system"""
        def handle_console_capture(msg):
            try:
                message_type = "unknown"
                try:
                    message_type = msg.type
                except:
                    pass
                    
                message_text = "unknown"
                try:
                    message_text = msg.text
                except:
                    pass
                    
                entry = {
                    "type": message_type,
                    "text": message_text,
                    "timestamp": time.time()
                }
                
                captured_console.append(entry)
                
            except Exception as e:
                captured_console.append({
                    "type": "console_capture_error", 
                    "error": str(e), 
                    "timestamp": time.time()
                })
        
        page.on("console", handle_console_capture)
        return handle_console_capture
    
    async def setup_error_capture(self, page: Page, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup error capture using Playwright's event system"""
        def handle_pageerror_capture(err):
            try:
                error_message = "Unknown error"
                try:
                    error_message = err.message
                except:
                    pass
                    
                error_stack = ""
                try:
                    error_stack = err.stack
                except:
                    pass
                    
                captured_console.append({
                    "type": "error",
                    "text": error_message,
                    "stack": error_stack,
                    "timestamp": time.time()
                })
            except Exception as e:
                captured_console.append({
                    "type": "pageerror_capture_error", 
                    "error": str(e), 
                    "timestamp": time.time()
                })
        
        page.on("pageerror", handle_pageerror_capture)
        return handle_pageerror_capture
    
    async def retrieve_console_messages(self, page: Page) -> List[Dict]:
        """Not needed for Playwright - messages are captured via events"""
        return []
    
    async def cleanup_console_capture(self, page: Page, handle_console: Optional[Callable], handle_error: Optional[Callable]):
        """Remove event listeners"""
        if handle_console:
            page.remove_listener("console", handle_console)
        if handle_error:
            page.remove_listener("pageerror", handle_error)


class StealthAdapter(BrowserAdapter):
    """Adapter for Playwright with stealth features using playwright_stealth"""

    def __init__(self):
        self._console_script_injected = {}
        self._stealth = None
        self._stealth_available = self._check_stealth_availability()

    def _check_stealth_availability(self) -> bool:
        """Check if playwright_stealth is importable and instantiate the Stealth helper."""
        try:
            from playwright_stealth import Stealth
        except ImportError:
            return False
        self._stealth = Stealth()
        return True

    async def apply_stealth(self, page: Page):
        """Apply stealth to a page if available"""
        if not (self._stealth_available and self._stealth):
            return
        try:
            await self._stealth.apply_stealth_async(page)
        except Exception:
            # Fail silently or log error depending on requirements
            pass

    async def evaluate(self, page: Page, expression: str, arg: Any = None) -> Any:
        """Standard Playwright evaluate with stealth applied"""
        if arg is not None:
            return await within_page_budget(page.evaluate(expression, arg))
        return await within_page_budget(page.evaluate(expression))

    async def setup_console_capture(self, page: Page, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup console capture using Playwright's event system with stealth"""
        # Apply stealth to the page first
        await self.apply_stealth(page)

        def handle_console_capture(msg):
            try:
                message_type = "unknown"
                try:
                    message_type = msg.type
                except:
                    pass

                message_text = "unknown"
                try:
                    message_text = msg.text
                except:
                    pass

                entry = {
                    "type": message_type,
                    "text": message_text,
                    "timestamp": time.time()
                }

                captured_console.append(entry)

            except Exception as e:
                captured_console.append({
                    "type": "console_capture_error",
                    "error": str(e),
                    "timestamp": time.time()
                })

        page.on("console", handle_console_capture)
        return handle_console_capture

    async def setup_error_capture(self, page: Page, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup error capture using Playwright's event system"""
        def handle_pageerror_capture(err):
            try:
                error_message = "Unknown error"
                try:
                    error_message = err.message
                except:
                    pass

                error_stack = ""
                try:
                    error_stack = err.stack
                except:
                    pass

                captured_console.append({
                    "type": "error",
                    "text": error_message,
                    "stack": error_stack,
                    "timestamp": time.time()
                })
            except Exception as e:
                captured_console.append({
                    "type": "pageerror_capture_error",
                    "error": str(e),
                    "timestamp": time.time()
                })

        page.on("pageerror", handle_pageerror_capture)
        return handle_pageerror_capture

    async def retrieve_console_messages(self, page: Page) -> List[Dict]:
        """Not needed for Playwright - messages are captured via events"""
        return []

    async def cleanup_console_capture(self, page: Page, handle_console: Optional[Callable], handle_error: Optional[Callable]):
        """Remove event listeners"""
        if handle_console:
            page.remove_listener("console", handle_console)
        if handle_error:
            page.remove_listener("pageerror", handle_error)


class UndetectedAdapter(BrowserAdapter):
    """Adapter for undetected browser automation with stealth features.

    Console / JS-error capture is armed through two complementary
    mechanisms so it works regardless of how the document was loaded:

    1. ``page.add_init_script(...)`` — the primary mechanism. Patchright
       1.56 implements ``add_init_script`` by rewriting the response of
       *network* ``document`` requests whose URL starts with ``http``, so
       the init script runs in every ``http(s)``-scheme document (including
       same-origin iframes) but NOT for ``page.set_content`` (no network
       document request) or ``file://`` (``file`` does not start with
       ``http``).

    2. ``bootstrap_capture_html(html)`` — the fallback for the schemes
       that #1 misses. Prepends the same capture bootstrap as a
       ``<script>`` tag at the top of ``<head>``, so it runs
       synchronously during ``page.set_content`` parsing, *before* any
       user inline script that emits a ``console.log`` or throws. The
       strategy calls this from the ``raw:`` / ``raw://`` / ``file://``
       branch of ``_crawl_web`` when ``capture_console_messages`` is on.

    Without #2, ``raw:`` / ``file://`` crawls silently returned
    ``console_messages=[]`` — a documented feature combination with no
    error, warning, or log to indicate the data loss (regression
    introduced in commit 2550f3d2 part of v0.8.0). Retrieval
    (``retrieve_console_messages``) is unchanged: it drains
    ``window.__capturedConsole`` / ``window.__capturedErrors`` per frame;
    the buffers are populated by whichever mechanism actually fired on
    the document — init_script for ``http(s)``, prepend for
    ``set_content`` / ``file://``.
    """

    # Console-capture bootstrap: seeds ``window.__capturedConsole`` (and
    # ``window.__capturedErrors``) and overrides ``console.<method>`` to
    # push to the console buffer while still calling the original method.
    _CONSOLE_CAPTURE_JS = """
        // Initialize console capture
        window.__capturedConsole = [];
        window.__capturedErrors = [];
        
        // Store original console methods
        const originalConsole = {};
        ['log', 'info', 'warn', 'error', 'debug'].forEach(method => {
            originalConsole[method] = console[method];
            console[method] = function(...args) {
                try {
                    window.__capturedConsole.push({
                        type: method,
                        text: args.map(arg => {
                            try {
                                if (typeof arg === 'object') {
                                    return JSON.stringify(arg);
                                }
                                return String(arg);
                            } catch (e) {
                                return '[Object]';
                            }
                        }).join(' '),
                        timestamp: Date.now()
                    });
                } catch (e) {
                    // Fail silently to avoid detection
                }
                
                // Call original method
                originalConsole[method].apply(console, args);
            };
        });
    """

    # Error-capture bootstrap: registers ``error`` /
    # ``unhandledrejection`` listeners that push to
    # ``window.__capturedErrors``.
    _ERROR_CAPTURE_JS = """
        // Capture errors
        window.addEventListener('error', (event) => {
            try {
                window.__capturedErrors.push({
                    type: 'error',
                    text: event.message,
                    stack: event.error ? event.error.stack : '',
                    filename: event.filename,
                    lineno: event.lineno,
                    colno: event.colno,
                    timestamp: Date.now()
                });
            } catch (e) {
                // Fail silently
            }
        });
        
        // Capture unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            try {
                window.__capturedErrors.push({
                    type: 'unhandledrejection',
                    text: event.reason ? String(event.reason) : 'Unhandled Promise Rejection',
                    stack: event.reason && event.reason.stack ? event.reason.stack : '',
                    timestamp: Date.now()
                });
            } catch (e) {
                // Fail silently
            }
        });
    """

    # The combined console-AND-error bootstrap that ``bootstrap_capture_html``
    # prepends to ``set_content`` / ``file://`` HTML. Built from the same
    # script bodies that ``setup_console_capture`` / ``setup_error_capture``
    # inject as init scripts, so the in-page state left behind (buffers +
    # console override + listeners) is byte-identical between the http(s)
    # init-script path and the set_content prepend path — downstream
    # retrieval / parsing cannot tell them apart.
    #
    # Each body is wrapped in its own IIFE so the top-level ``const
    # originalConsole`` (and any other lexical declarations) do not leak
    # into the page's global lexical environment when the bootstrap runs as
    # a normal ``<script>``. ``add_init_script`` already scopes its body
    # inside a function, so the init-script path is unaffected; the prepend
    # path runs the script as a classic document script where a top-level
    # ``const`` would collide with a page script that declares the same
    # name. State still flows through ``window.__capturedConsole`` /
    # ``window.__capturedErrors`` (set on ``window`` inside the IIFEs).
    _CAPTURE_BOOTSTRAP_JS = (
        "(() => {" + _CONSOLE_CAPTURE_JS + "})();\n"
        "(() => {" + _ERROR_CAPTURE_JS + "})();"
    )

    def __init__(self):
        self._console_script_injected = weakref.WeakKeyDictionary()
        self._error_script_injected = weakref.WeakKeyDictionary()
    
    async def evaluate(self, page: UndetectedPage, expression: str, arg: Any = None) -> Any:
        """Undetected browser evaluate with isolated context"""
        # For most evaluations, use isolated context for stealth
        # Only use non-isolated when we need to access our injected console capture
        isolated = not (
            "__console" in expression or 
            "__captured" in expression or
            "__error" in expression or
            "window.__" in expression
        )
        
        if arg is not None:
            return await within_page_budget(page.evaluate(expression, arg, isolated_context=isolated))
        return await within_page_budget(page.evaluate(expression, isolated_context=isolated))
    
    async def setup_console_capture(self, page: UndetectedPage, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup console capture using JavaScript injection for undetected browsers.

        Installs the console-capture init script for http(s)-scheme documents
        (the primary mechanism). For ``raw:`` / ``file://`` crawls, the
        strategy additionally calls ``bootstrap_capture_html`` to prepend the
        same bootstrap so it runs under ``set_content`` — see the class
        docstring.
        """
        if not self._console_script_injected.get(page, False):
            await page.add_init_script(self._CONSOLE_CAPTURE_JS)
            self._console_script_injected[page] = True
        
        return None  # No handler function needed for undetected browser
    
    async def setup_error_capture(self, page: UndetectedPage, captured_console: List[Dict]) -> Optional[Callable]:
        """Setup error capture using JavaScript injection for undetected browsers.

        Installs the error/unhandledrejection-listener init script for
        http(s)-scheme documents (the primary mechanism). For
        ``raw:`` / ``file://`` crawls, the strategy additionally calls
        ``bootstrap_capture_html`` to prepend the same bootstrap so the
        listeners are installed under ``set_content`` too.
        """
        if not self._error_script_injected.get(page, False):
            await page.add_init_script(self._ERROR_CAPTURE_JS)
            self._error_script_injected[page] = True
        
        return None  # No handler function needed for undetected browser

    def bootstrap_capture_html(self, html: str) -> str:
        """Prepend the console/error capture bootstrap as a ``<script>`` at
        the top of ``<head>`` so it runs synchronously *before* any user
        inline script during ``page.set_content`` parsing.

        This is the fallback for the schemes patchright 1.56's
        ``add_init_script`` does not cover: ``set_content`` (no network
        document request) and ``file://`` (non-http scheme). Without it,
        inline ``console.log`` calls and uncaught errors emitted during
        ``set_content`` parsing — the most common case for the
        ``raw:`` / ``raw://`` browser pipeline — were silently dropped
        because the in-page buffers were never created.

        Called by the ``is_local_content`` branch of
        ``AsyncPlaywrightCrawlerStrategy._crawl_web`` only when
        ``config.capture_console_messages`` is enabled, so crawls that
        do not request console capture pay no extra cost and see no
        console-override side effects.

        The injected ``<script>`` removes itself from the DOM at the end
        of its synchronous run (``document.currentScript.remove()``) so
        the ``result.html`` produced by ``page.content()`` for
        ``raw:`` / ``file://`` crawls matches the http(s) init-script
        path (which adds nothing to the DOM). The console override and
        ``error`` / ``unhandledrejection`` listeners stay active after
        the element is gone — they live on ``window`` / ``console``,
        not on the script element.
        """
        return _inject_script_at_head_top(
            html,
            self._CAPTURE_BOOTSTRAP_JS
            + "\ntry { document.currentScript && document.currentScript.remove(); } catch (e) {}\n",
        )
    
    async def retrieve_console_messages(self, page: UndetectedPage) -> List[Dict]:
        """Retrieve captured console messages and errors from the page.

        Drains the per-frame `window.__capturedConsole` / `window.__capturedErrors`
        buffers installed by `setup_console_capture` / `setup_error_capture`
        (or by `bootstrap_capture_html`'s prepended script for set_content /
        file:// documents). `add_init_script` runs in every http(s)-scheme
        document (including iframes), and `bootstrap_capture_html`'s prepended
        script runs in the set_content document, so each frame has its own
        buffer; iterating `page.frames` ensures subframe messages are
        surfaced alongside the main frame's. `page.frames` always includes
        the main frame as its first element, so single-document pages behave
        exactly as before.
        """
        messages = []

        try:
            for frame in page.frames:
                # Get console messages
                console_messages = await frame.evaluate(
                    "() => { const msgs = window.__capturedConsole || []; window.__capturedConsole = []; return msgs; }",
                    isolated_context=False
                )
                # Convert per-batch so a later frame's raise cannot skip the
                # ms->s conversion already owed to collected messages.
                for msg in (console_messages or []):
                    if 'timestamp' in msg and isinstance(msg['timestamp'], (int, float)):
                        msg['timestamp'] = msg['timestamp'] / 1000.0  # Convert from ms to seconds
                    messages.append(msg)

                # Get errors
                errors = await frame.evaluate(
                    "() => { const errs = window.__capturedErrors || []; window.__capturedErrors = []; return errs; }",
                    isolated_context=False
                )
                for msg in (errors or []):
                    if 'timestamp' in msg and isinstance(msg['timestamp'], (int, float)):
                        msg['timestamp'] = msg['timestamp'] / 1000.0  # Convert from ms to seconds
                    messages.append(msg)

        except Exception:
            # If retrieval fails, return whatever was collected so far
            pass

        return messages
    
    async def cleanup_console_capture(self, page: UndetectedPage, handle_console: Optional[Callable], handle_error: Optional[Callable]):
        """Clean up for undetected browser - retrieve final messages"""
        # For undetected browser, we don't have event listeners to remove
        # but we should retrieve any final messages
        final_messages = await self.retrieve_console_messages(page)
        return final_messages


# Matches the first ``<head ...>`` opening tag, case-insensitive, including
# any attributes (e.g. ``<head class="...">``). Used by
# ``_inject_script_at_head_top`` to find the insertion point for the
# capture-bootstrap script.
_HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)
# Fallback: matches the first ``<html ...>`` opening tag, case-insensitive,
# when the document has no ``<head>`` (e.g. ``<html><body>...</body></html>``).
_HTML_OPEN_RE = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
# Opening ``<script ...>`` tag, used to skip script raw text when locating
# the insertion point for the bootstrap — a ``<head>`` substring inside a
# script string (e.g. ``<script>const s = "<head>";</script>``) must not be
# treated as the real ``<head>`` opening tag, or the bootstrap would be
# injected inside the script and the user script truncated.
_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>]*>", re.IGNORECASE)
_SCRIPT_CLOSE_RE = re.compile(r"</script\s*>", re.IGNORECASE)
# Opening ``<style ...>`` tag, used to skip style raw text when locating the
# injection point — a ``<head>`` substring inside a ``<style>`` block
# (e.g. ``<style>.x > head { ... }</style>``, or a literal ``<head>``
# in a style string) must not be treated as the real ``<head>`` opening
# tag, or the bootstrap would be injected inside the style block and the
# real ``<head>`` would stay unbootstrapped so the following page script
# runs without capture.
_STYLE_OPEN_RE = re.compile(r"<style\b[^>]*>", re.IGNORECASE)
_STYLE_CLOSE_RE = re.compile(r"</style\s*>", re.IGNORECASE)
# Leading prologue (doctype, XML declaration, HTML comments) that must stay
# before the bootstrap script when no ``<html>`` / ``<head>`` tag is present,
# so the document does not flip into quirks mode (a doctype after ``<html>``
# is a parse error and ignored) and XML processing instructions are not
# swallowed by the implied ``<head>``.
_PROLOG_RE = re.compile(
    r"\s*(?:(?:<\?xml[^>]*>|<!doctype[^>]*>|<!--.*?-->)\s*)*",
    re.IGNORECASE | re.DOTALL,
)


def _search_outside_html_comments_and_scripts(pattern, html: str):
    """Return the leftmost ``pattern`` match in ``html`` that is not inside
    an HTML comment or a ``<script>...</script>`` raw-text region. Without
    this:

    * a tag-like substring inside ``<!-- ... -->`` (e.g. ``<!-- <head> -->``)
      would match ``_HEAD_OPEN_RE`` and the bootstrap would be inserted
      before ``-->`` — the HTML5 parser then treats it as comment text, so
      the real ``<head>`` never gets the bootstrap and the following page
      script runs without capture;
    * a tag-like substring inside a script string (e.g.
      ``<script>const s = "<head>";</script>``) would match
      ``_HEAD_OPEN_RE`` and the bootstrap would be inserted inside the
      script; the generated ``</script>`` then closes the user script
      early, truncating it and leaving the real ``<head>`` unbootstrapped.
    * a tag-like substring inside a ``<style>`` block (e.g.
      ``<style>.x > head { ... }</style>``) would match ``_HEAD_OPEN_RE``
      and the bootstrap would be injected inside the style block; the
      real ``<head>`` then stays unbootstrapped so the following page
      script runs without capture.

    For each cursor position the function picks the leftmost of: a
    ``pattern`` match, an HTML comment start, a ``<script>`` open tag, or
    a ``<style>`` open tag. A pattern match wins (it is returned). A
    comment is skipped to its ``-->`` terminator. A script open tag is
    skipped to its matching ``</script>`` close. A style open tag is
    skipped to its matching ``</style>`` close. This mirrors the HTML5
    tokenizer's raw-text handling for ``<script>`` and ``<style>``
    (content ends at the first matching close tag)."""
    cursor = 0
    n = len(html)
    while cursor < n:
        match = pattern.search(html, cursor)
        comment_start = html.find("<!--", cursor)
        script_open = _SCRIPT_OPEN_RE.search(html, cursor)
        style_open = _STYLE_OPEN_RE.search(html, cursor)
        # Leftmost special region vs. the pattern match.
        match_pos = match.start() if match is not None else n
        comment_pos = comment_start if comment_start != -1 else n
        script_pos = script_open.start() if script_open is not None else n
        style_pos = style_open.start() if style_open is not None else n
        earliest = min(match_pos, comment_pos, script_pos, style_pos)
        if earliest == n:
            return None
        if earliest == match_pos:
            return match
        if earliest == comment_pos:
            comment_end = html.find("-->", comment_start + 4)
            if comment_end == -1:
                return None
            cursor = comment_end + 3
            continue
        if earliest == style_pos:
            style_close = _STYLE_CLOSE_RE.search(html, style_open.end())
            if style_close is None:
                return None
            cursor = style_close.end()
            continue
        # earliest == script_pos: skip the script raw-text region.
        script_close = _SCRIPT_CLOSE_RE.search(html, script_open.end())
        if script_close is None:
            return None
        cursor = script_close.end()
    return None


def _inject_script_at_head_top(html: str, script_body: str) -> str:
    """Insert ``<script>{script_body}</script>`` as the first child of
    ``<head>`` in ``html`` so it runs synchronously *before* any
    subsequent inline script when the document is parsed via
    ``page.set_content``.

    The HTML5 parser treats scripts inside ``<head>`` as part of document
    loading and runs them in order, so a script at the top of ``<head>``
    runs before any user inline script in ``<head>`` or ``<body>`` — exactly
    the property needed to bootstrap console/error capture in time for
    inline-script ``console.log`` calls.

    Handles malformed / partial HTML by inserting the script after any
    leading doctype / XML declaration / comments when no ``<head>`` or
    ``<html>`` opening tag is present, leaving the rest of the input
    unchanged. The HTML5 parser places a leading ``<script>`` into the
    implied ``<head>`` so it still runs before any subsequent inline
    script, and a leading ``<!DOCTYPE html>`` stays first so the document
    does not flip into quirks mode.

    ``script_body`` MUST NOT contain the literal substring ``</script>``
    (none of the capture bootstraps do). The function does not escape it
    because the bootstrap JS is constant and that escape would change the
    in-page semantics — but the contract is asserted at import time by the
    use of fixed, reviewed constants (``_CONSOLE_CAPTURE_JS`` /
    ``_ERROR_CAPTURE_JS``).
    """
    head_match = _search_outside_html_comments_and_scripts(_HEAD_OPEN_RE, html)
    if head_match is not None:
        at = head_match.end()
        return html[:at] + f"<script>{script_body}</script>" + html[at:]

    html_match = _search_outside_html_comments_and_scripts(_HTML_OPEN_RE, html)
    if html_match is not None:
        at = html_match.end()
        return (
            html[:at]
            + f"<head><script>{script_body}</script></head>"
            + html[at:]
        )

    # No <html> or <head> at all (e.g. fragment like "<p>hi</p>" or a raw
    # HTML page that omits both tags). Keep any leading doctype / XML decl /
    # comments first so the document does not switch to quirks mode (a
    # doctype after <html> is a parse error and ignored); the HTML5 parser
    # places a leading <script> into the implied <head>, so it still runs
    # before any subsequent inline script without wrapping the rest of the
    # input in synthetic <html>/<head>/<body> tags.
    prolog = _PROLOG_RE.match(html)
    at = prolog.end() if prolog else 0
    return html[:at] + f"<script>{script_body}</script>" + html[at:]
