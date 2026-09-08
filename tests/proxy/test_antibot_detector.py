"""
Unit tests for antibot_detector.is_blocked().

Tests are organized into:
  - TRUE POSITIVES:  Real block pages that MUST be detected
  - TRUE NEGATIVES:  Legitimate pages that MUST NOT be flagged
  - EDGE CASES:      Boundary conditions
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from crawl4ai.antibot_detector import is_blocked

PASS = 0
FAIL = 0

def check(name, result, expected_blocked, expected_substr=None):
    global PASS, FAIL
    blocked, reason = result
    ok = blocked == expected_blocked
    if expected_substr and blocked:
        ok = ok and expected_substr.lower() in reason.lower()
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  {status}: {name}")
        print(f"         got blocked={blocked}, reason={reason!r}")
        print(f"         expected blocked={expected_blocked}" +
              (f", substr={expected_substr!r}" if expected_substr else ""))
    else:
        PASS += 1
        if blocked:
            print(f"  {status}: {name} -> {reason}")
        else:
            print(f"  {status}: {name} -> not blocked")


# =========================================================================
# TRUE POSITIVES — real block pages that MUST be detected
# =========================================================================
print("\n=== TRUE POSITIVES (must detect as blocked) ===\n")

# --- Akamai ---
check("Akamai Reference #",
    is_blocked(403, '<html><body>Access Denied\nYour request was blocked.\nReference #18.2d351ab8.1557333295.a4e16ab</body></html>'),
    True, "Akamai")

check("Akamai Pardon Our Interruption",
    is_blocked(403, '<html><head><title>Pardon Our Interruption</title></head><body><p>Please verify you are human</p></body></html>'),
    True, "Pardon")

check("Akamai 403 short Access Denied",
    is_blocked(403, '<html><body><h1>Access Denied</h1></body></html>'),
    True)  # Detected via near-empty 403 or Access Denied pattern

# --- Cloudflare ---
check("Cloudflare challenge form",
    is_blocked(403, '''<html><body>
        <form id="challenge-form" action="/cdn-cgi/l/chk_jschl?__cf_chl_f_tk=abc123">
        <input type="hidden" name="jschl_vc" value="test"/>
        </form></body></html>'''),
    True, "Cloudflare challenge")

check("Cloudflare error 1020",
    is_blocked(403, '''<html><body>
        <div class="cf-wrapper"><span class="cf-error-code">1020</span></div>
        <p>Access denied</p></body></html>'''),
    True, "Cloudflare firewall")

check("Cloudflare IUAM script",
    is_blocked(403, '<html><script src="/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1"></script></html>'),
    True, "Cloudflare JS challenge")

check("Cloudflare Just a moment",
    is_blocked(403, '<html><head><title>Just a moment...</title></head><body>Checking your browser</body></html>'),
    True)  # Detected via near-empty 403 or Cloudflare pattern

check("Cloudflare Checking your browser (short 503)",
    is_blocked(503, '<html><body>Checking your browser before accessing the site.</body></html>'),
    True, "503")

# --- PerimeterX ---
check("PerimeterX block page",
    is_blocked(403, '''<html><head><title>Access to This Page Has Been Blocked</title></head>
        <body><div id="px-captcha"></div>
        <script>window._pxAppId = 'PX12345';</script></body></html>'''),
    True, "PerimeterX")

check("PerimeterX captcha CDN",
    is_blocked(403, '<html><body><script src="https://captcha.px-cdn.net/PX12345/captcha.js"></script></body></html>'),
    True, "PerimeterX captcha")

# --- DataDome ---
check("DataDome captcha delivery",
    is_blocked(403, '''<html><body><script>
        var dd = {'rt':'i','cid':'AHrlq...','host':'geo.captcha-delivery.com'};
        </script></body></html>'''),
    True, "DataDome")

# --- Imperva/Incapsula ---
check("Imperva Incapsula Resource",
    is_blocked(403, '<html><body><iframe src="/_Incapsula_Resource?incident_id=123&sess_id=abc"></iframe></body></html>'),
    True, "Imperva")

check("Imperva incident ID",
    is_blocked(200, '<html><body>Request unsuccessful. Incapsula incident ID: 12345-67890</body></html>'),
    True, "Incapsula incident")

# --- Sucuri ---
check("Sucuri firewall",
    is_blocked(403, '<html><body><h1>Sucuri WebSite Firewall - Access Denied</h1></body></html>'),
    True, "Sucuri")

# --- Kasada ---
check("Kasada challenge",
    is_blocked(403, '<html><script>KPSDK.scriptStart = KPSDK.now();</script></html>'),
    True, "Kasada")

# --- Reddit / Network Security ---
check("Reddit blocked by network security (small)",
    is_blocked(403, '<html><body>You\'ve been blocked by network security.</body></html>'),
    True, "Network security block")

check("Reddit blocked by network security (190KB SPA shell)",
    is_blocked(403, '<html><body><style>' + 'x' * 180000 + '</style>' +
        'You\'ve been blocked by network security. Log in to continue.</body></html>'),
    True, "Network security block")

check("Network security block on HTTP 200 (buried in large page)",
    is_blocked(200, '<html><body><style>' + 'a:b;' * 30000 + '</style>' +
        '<p>blocked by network security</p></body></html>'),
    True, "Network security block")

# --- HTTP 429 ---
check("HTTP 429 rate limit",
    is_blocked(429, '<html><body>Rate limit exceeded</body></html>'),
    True, "429")

check("HTTP 429 empty body",
    is_blocked(429, ''),
    True, "429")

# --- Empty 200 ---
check("HTTP 200 empty page",
    is_blocked(200, ''),
    True, "empty")

check("HTTP 200 whitespace only",
    is_blocked(200, '   \n\n   '),
    True, "empty")

# --- 403 near-empty ---
check("HTTP 403 near-empty (10 bytes)",
    is_blocked(403, '<html></html>'),
    True, "403")

# --- Structural bypass via CSS-hidden text (regression) ---
# HTTP 200 block page with short visible text and a <p> (so the
# no_content_elements signal does NOT fire). Without stripping CSS-hidden
# content, the hidden text inflates visible_text past 50 chars and the page
# bypasses the minimal_text signal. The fix strips hidden elements first.
# The HTML comment keeps html.strip() above the near-empty 100-byte threshold
# so the structural path (not near-empty) is what catches these.
_HIDDEN_BLOCK_PROSE = "Checking your browser."
_HIDDEN_BLOCK_TEMPLATE = (
    '<html><!-- padding padding padding padding padding padding padding padding -->'
    '<body><p>' + _HIDDEN_BLOCK_PROSE + '</p>'
    '<div style={q}{css}{q}>' + ('pad text ' * 12) + '</div></body></html>'
)

def _hidden_block(css, quote='"'):
    return _HIDDEN_BLOCK_TEMPLATE.format(q=quote, css=css)

check("Structural: hidden display:none text cannot pad past minimal_text",
    is_blocked(200, _hidden_block('display:none')),
    True, "minimal_text")

check("Structural: hidden visibility:hidden variant",
    is_blocked(200, _hidden_block('visibility:hidden')),
    True, "minimal_text")

check("Structural: hidden with single-quoted attribute",
    is_blocked(200, _hidden_block('display:none', quote="'")),
    True, "minimal_text")

check("Structural: hidden style value embeds the other quote char",
    is_blocked(200, _hidden_block("font-family:'Arial';display:none")),
    True, "minimal_text")

check("Structural: hidden multi-line block (DOTALL)",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding padding -->'
        '<body><p>' + _HIDDEN_BLOCK_PROSE + '</p>'
        '<div style="display:none">' + ('pad text\n' * 30) + '</div></body></html>'),
    True, "minimal_text")

# Shapes a regex-based strip cannot handle (the parser-based strip must):
check("Structural: hidden block with nested same-tag child",
    is_blocked(200, _hidden_block('display:none').replace(
        '<div style="display:none">', '<div style="display:none"><div>x</div>')),
    True, "minimal_text")

check("Structural: hidden with unquoted style attribute",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding padding -->'
        '<body><p>' + _HIDDEN_BLOCK_PROSE + '</p>'
        '<div style=display:none>' + ('pad text ' * 12) + '</div></body></html>'),
    True, "minimal_text")

check("Structural: hidden with whitespace before = in style attribute",
    is_blocked(200, _hidden_block('display:none').replace('style=', 'style =')),
    True, "minimal_text")

check("Structural: HTML5 hidden attribute",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding padding -->'
        '<body><p>' + _HIDDEN_BLOCK_PROSE + '</p>'
        '<div hidden>' + ('pad text ' * 12) + '</div></body></html>'),
    True, "minimal_text")

check("Structural: unclosed hidden element still stripped",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding padding -->'
        '<body><p>' + _HIDDEN_BLOCK_PROSE + '</p>'
        '<div style="display:none">' + ('pad text ' * 12) + '</body></html>'),
    True, "minimal_text")

# --- Structural bypass via foster-parented stray text outside <body> ---
# HTTP 200 silent block whose block message lives physically OUTSIDE <body>
# (text before <body>, raw text in <head>, or text between </head> and <body>)
# while the <body> itself holds a content element (<p>/<a>/...). lxml's HTML5
# parser foster-parents that stray text into the <body> subtree, so the
# pre-fix _visible_text_len counted it as visible body content, inflated past
# 50 chars, and suppressed the minimal_text signal — letting a soft-200 block
# page through as a successful crawl. Each body here has a <p> so the
# no_content_elements signal does NOT fire; minimal_text is the only catch.
_FOSTER = ' filler ' * 120

check("Structural: foster-parented stray text before <body> (with </body>)",
    is_blocked(200, '<html>' + _FOSTER + '<body><p>nice content</p></body></html>'),
    True, "minimal_text")

check("Structural: foster-parented raw text in <head> (with </body>)",
    is_blocked(200, '<html><head>' + _FOSTER + '</head><body><p>x</p></body></html>'),
    True, "minimal_text")

check("Structural: foster-parented text between </head> and <body>",
    is_blocked(200, '<html><head></head>' + _FOSTER + '<body><p>ok</p></body></html>'),
    True, "minimal_text")

# No-</body> variant: the strict </body>-substring fallback would wrongly
# return None and re-inflate via the regex fallback; scoping to the <body
# open tag onward (lxml closes implicitly) handles both closed and unclosed.
check("Structural: foster-parented stray text before <body> (no </body>)",
    is_blocked(200, '<html>' + _FOSTER + '<body><p>ok</p>'),
    True, "minimal_text")

# Headline repro from the report: a full block sentence + padding before <body>.
check("Structural: full block sentence before <body> bypasses minimal_text",
    is_blocked(200,
        '<html>Blocked - automated traffic detected. Please contact the site administrator.'
        + _FOSTER + '<body><p>ok</p></body></html>'),
    True, "minimal_text")

# Adversarial input that made the earlier regex approach backtrack
# quadratically (~8s at 44KB). The parser path is linear; the loose wall-clock
# bound only trips on a reintroduced blow-up, not normal machine variance.
import time as _time
_adv = ('<html><body><p>hello world content here</p>'
        + '<a style="x' * 4000 + '</body></html>')
_t0 = _time.perf_counter()
_adv_result = is_blocked(200, _adv)
_adv_elapsed = _time.perf_counter() - _t0
check("Structural: adversarial unterminated-style page is not misclassified",
    _adv_result, False)
check("Structural: adversarial 44KB page processed within 5s bound",
    (_adv_elapsed < 5.0, f"took {_adv_elapsed:.3f}s"), True)


# =========================================================================
# TRUE NEGATIVES — legitimate pages that MUST NOT be flagged
# =========================================================================
print("\n=== TRUE NEGATIVES (must NOT detect as blocked) ===\n")

# --- Normal pages ---
check("Normal 200 page (example.com size)",
    is_blocked(200, '<html><head><title>Example</title></head><body><p>' + 'x' * 500 + '</p></body></html>'),
    False)

check("Normal 200 large page",
    is_blocked(200, '<html><body>' + '<p>Some content here.</p>\n' * 5000 + '</body></html>'),
    False)

# --- Security articles (false positive trap!) ---
check("Article about bot detection (large page)",
    is_blocked(200, '<html><head><title>How to Detect Bots</title></head><body>' +
        '<h1>How to Detect Bots on Your Website</h1>' +
        '<p>Anti-bot solutions like DataDome, PerimeterX, and Cloudflare ' +
        'help detect and block bot traffic. When a bot is detected, ' +
        'services show a CAPTCHA or Access Denied page. ' +
        'Common signals include blocked by security warnings.</p>' +
        '<p>The g-recaptcha and h-captcha widgets are used for challenges.</p>' +
        '<p>' + 'More article content. ' * 500 + '</p>' +
        '</body></html>'),
    False)

check("DataDome marketing page (large)",
    is_blocked(200, '<html><body><h1>DataDome Bot Protection</h1>' +
        '<p>DataDome protects websites from bot attacks. ' +
        'Our solution detects automated traffic using advanced fingerprinting. ' +
        'Competitors like PerimeterX use window._pxAppId for tracking.</p>' +
        '<p>' + 'Marketing content. ' * 1000 + '</p>' +
        '</body></html>'),
    False)


# --- Login pages with CAPTCHA (not a block!) ---
check("Login page with reCAPTCHA (large page)",
    is_blocked(200, '<html><head><title>Sign In</title></head><body>' +
        '<nav>Home | Products | Contact</nav>' +
        '<form action="/login" method="POST">' +
        '<input name="email" type="email"/>' +
        '<input name="password" type="password"/>' +
        '<div class="g-recaptcha" data-sitekey="abc123"></div>' +
        '<button type="submit">Sign In</button>' +
        '</form>' +
        '<footer>Copyright 2024</footer>' +
        '<p>' + 'Page content. ' * 500 + '</p>' +
        '</body></html>'),
    False)

check("Signup page with hCaptcha (large page)",
    is_blocked(200, '<html><body>' +
        '<h1>Create Account</h1>' +
        '<form><div class="h-captcha" data-sitekey="xyz"></div></form>' +
        '<p>' + 'Registration info. ' * 500 + '</p>' +
        '</body></html>'),
    False)

# --- 403 pages — ALL non-data 403 HTML is now treated as blocked ---
# Rationale: 403 is never the content the user wants. Even for legitimate
# auth errors (Apache/Nginx), the fallback will also get 403 and we report
# failure correctly. False positives are cheap; false negatives are catastrophic.
check("Apache directory listing denied (403, large-ish)",
    is_blocked(403, '<html><head><title>403 Forbidden</title></head><body>' +
        '<h1>Forbidden</h1>' +
        '<p>You don\'t have permission to access this resource on this server.</p>' +
        '<hr><address>Apache/2.4.41 (Ubuntu) Server at example.com Port 80</address>' +
        '<p>' + 'Server info. ' * 500 + '</p>' +
        '</body></html>'),
    True, "403")

check("Nginx 403 (large page)",
    is_blocked(403, '<html><head><title>403 Forbidden</title></head><body>' +
        '<center><h1>403 Forbidden</h1></center>' +
        '<hr><center>nginx/1.18.0</center>' +
        '<p>' + 'Content. ' * 500 + '</p>' +
        '</body></html>'),
    True, "403")

check("API 403 auth required (JSON)",
    is_blocked(403, '{"error": "Forbidden", "message": "Invalid API key", "code": 403}'),
    False)

# --- Cloudflare-served normal pages (not blocked!) ---
check("Cloudflare-served normal page with footer",
    is_blocked(200, '<html><body>' +
        '<h1>Welcome to Our Site</h1>' +
        '<p>This is a normal page served through Cloudflare CDN.</p>' +
        '<footer>Performance & security by Cloudflare</footer>' +
        '<p>' + 'Normal content. ' * 500 + '</p>' +
        '</body></html>'),
    False)

# --- Small but legitimate pages ---
check("Small valid 200 page (with content element)",
    is_blocked(200, '<html><head><title>OK</title></head><body><p>Your request was processed successfully. Everything is fine.</p></body></html>'),
    False)

check("Small JSON 200 response",
    is_blocked(200, '{"status": "ok", "data": {"id": 123, "name": "test"}, "timestamp": "2024-01-01T00:00:00Z"}'),
    False)

check("Redirect page 200",
    is_blocked(200, '<html><head><meta http-equiv="refresh" content="0;url=/dashboard"></head><body><p>Redirecting to your dashboard. Please wait while we prepare your personalized experience.</p></body></html>'),
    False)

# --- 503 pages — ALL non-data 503 HTML is now treated as blocked ---
# Same rationale as 403: 503 is never desired content. Fallback rescues false positives.
check("503 maintenance page (treated as blocked)",
    is_blocked(503, '<html><body><h1>Service Temporarily Unavailable</h1>' +
        '<p>We are performing scheduled maintenance. Please try again later.</p>' +
        '<p>' + 'Maintenance info. ' * 500 + '</p>' +
        '</body></html>'),
    True, "503")

# --- 200 with short but real content ---
check("Short thank you page (200, 120 bytes)",
    is_blocked(200, '<html><body><h1>Thank You!</h1><p>Your order has been placed. Confirmation email sent.</p></body></html>'),
    False)

# --- Legitimate pages with CSS-hidden content (false-positive guards) ---
# Real pages hide content for accessibility (skip links) and SEO. Stripping
# that hidden content must NOT push their visible text below the minimal_text
# threshold. Also, data-style= is not a real hiding attribute; its content
# must remain visible (the (?<=\s)style= guard in the regex enforces this).
check("Landing page with hidden a11y skip link",
    is_blocked(200,
        '<html><body>'
        '<a href="#main" style="display:none">Skip to main content</a>'
        '<h1>Welcome to our site</h1>'
        '<p>This is the landing page with plenty of visible prose describing the product.</p>'
        '<p>More paragraphs of genuine marketing copy for the homepage.</p>'
        '</body></html>'),
    False)

check("data-style= must not be treated as CSS-hidden",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding -->'
        '<body><p style="color:red">Checking your browser.</p>'
        '<span data-style="display:none">' + ('pad text ' * 12) + '</span></body></html>'),
    False)


# =========================================================================
# EDGE CASES
# =========================================================================
print("\n=== EDGE CASES ===\n")

check("None status code + empty html",
    is_blocked(None, ''),
    True, "no <body>")

check("None status code + block content",
    is_blocked(None, '<html><body>Reference #18.2d351ab8.1557333295.a4e16ab</body></html>'),
    True, "Akamai")

check("200 + tier1 pattern (Imperva deceptive 200)",
    is_blocked(200, '<html><body>Request unsuccessful. Incapsula incident ID: 555-999</body></html>'),
    True, "Incapsula")

check("403 + 4999 bytes (just under threshold)",
    is_blocked(403, '<html><body>Access Denied' + 'x' * 4950 + '</body></html>'),
    True, "Access Denied")

check("403 + 5001 bytes (over old threshold, now blocked)",
    is_blocked(403, '<html><body>Some error page' + 'x' * 4960 + '</body></html>'),
    True, "403")

check("403 + 9999 bytes with generic block text",
    is_blocked(403, '<html><body>blocked by security' + 'x' * 9950 + '</body></html>'),
    True, "Blocked by security")

check("403 + 10001 bytes with generic block text (now detected regardless of size)",
    is_blocked(403, '<html><body>blocked by security' + 'x' * 9970 + '</body></html>'),
    True, "Blocked by security")

check("200 + whitespace-padded but 89 bytes content (above threshold for meaningful)",
    is_blocked(200, ' ' * 10 + 'x' * 89 + ' ' * 10),
    True, "empty")

check("200 + exactly 100 bytes stripped (at threshold, no body = structural fail)",
    is_blocked(200, 'x' * 100),
    True, "no <body>")

# CSS-hidden stripping must not over-match non-hidden display/visibility values.
check("display:block is not treated as hidden (still detected via minimal_text)",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding padding -->'
        '<body><p>Checking your browser.</p>'
        '<div style="display:block">' + ('pad text ' * 12) + '</div></body></html>'),
    False)

check("visibility:visible is not treated as hidden",
    is_blocked(200,
        '<html><!-- padding padding padding padding padding padding padding padding -->'
        '<body><p>Checking your browser.</p>'
        '<div style="visibility:visible">' + ('pad text ' * 12) + '</div></body></html>'),
    False)


# =========================================================================
# css_selector fragment regressions
# =========================================================================
# When a css_selector is set, the captured HTML is wrapped in
# "<div class='crawl4ai-result'>...</div>" (a <div>-prefixed fragment), not an
# <html>-rooted document. The bug-report fix ensures 403/503 block pages
# captured this way are still flagged as blocked, while legitimate 200
# content fragments and XML feed responses remain exempt.
print("\n=== css_selector FRAGMENT REGRESSIONS ===\n")

check("403 css_selector empty fragment is blocked",
    is_blocked(403, "<div class='crawl4ai-result'>\n\n</div>"),
    True, "403")

check("503 css_selector empty fragment is blocked",
    is_blocked(503, "<div class='crawl4ai-result'>\n\n</div>"),
    True, "503")

check("403 <body>-prefixed fragment is blocked",
    is_blocked(403, "<body><h1>Forbidden</h1>nginx</body>"),
    True, "403")

# Legitimate content captured via css_selector on HTTP 200 must NOT be flagged
# (guards the Tier 3 "no <body>" signal suppression for fragments).
_substantial_fragment = (
    "<div class='crawl4ai-result'>"
    + '<div class="product"><a href="/p/1">Wireless Mouse</a>'
    + '<p>Ergonomic wireless mouse with precision tracking</p></div>' * 5
    + "</div>"
)
check("200 substantial css_selector fragment is not blocked",
    is_blocked(200, _substantial_fragment),
    False)

# XML feed and sitemap roots on 403 must remain exempt (no over-blocking of the
# data responses crawl4ai fetches for sitemap/RSS/Atom discovery).
check("403 XML feed (rss) is not blocked (data exemption)",
    is_blocked(403, '<rss version="2.0"><channel><item>x</item></channel></rss>'),
    False)
check("403 XML feed (urlset) is not blocked (data exemption)",
    is_blocked(403, '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'),
    False)
check("403 XML feed (Atom) is not blocked (data exemption)",
    is_blocked(403, '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'),
    False)
check("403 XML sitemap index is not blocked (data exemption)",
    is_blocked(403, '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>'),
    False)
check("403 namespaced RDF is not blocked (data exemption)",
    is_blocked(403, '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>'),
    False)

check("403 feedback-panel lookalike is blocked",
    is_blocked(403, '<feedback-panel>Temporarily unavailable</feedback-panel>'),
    True)
check("503 rss-widget lookalike is blocked",
    is_blocked(503, '<rss-widget>Temporarily unavailable</rss-widget>'),
    True)


# =========================================================================
# SUMMARY
# =========================================================================
print(f"\n{'=' * 60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print(f"{'=' * 60}")
if FAIL > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")


# =========================================================================
# pytest-style regression tests for the foster-parenting fix.
#
# The script harness above is this file's existing convention. These
# functions add unit-level coverage of `_visible_text_len` (the helper that
# regressed) so the file is also collectable under the repo's documented
# runner (`pytest`) — without them `pytest <file>` reports "no tests ran".
# The `is_blocked`-level shapes are already covered by the `check()` cases
# above, so the functions below focus on the helper's own contract and the
# two guards that are not otherwise exercised here.
# =========================================================================
from crawl4ai.antibot_detector import _body_start, _visible_text_len



def test_visible_text_len_excludes_foster_parented_stray_text_outside_body():
    # lxml's HTML5 parser relocates stray text outside <body> into the body
    # subtree; the helper must scope to the <body open tag onward before
    # parsing so that text cannot inflate the count. Three foster-parenting
    # shapes (text before <body>, raw text in <head>, text between </head>
    # and <body>) plus the no-</body> edge case.
    assert _visible_text_len(
        '<html>' + _FOSTER + '<body><p>nice content</p></body></html>') == 12
    assert _visible_text_len(
        '<html><head>' + _FOSTER + '</head><body><p>x</p></body></html>') == 1
    assert _visible_text_len(
        '<html><head></head>' + _FOSTER + '<body><p>ok</p></body></html>') == 2
    # No </body> close: strict </body>-substring scoping would wrongly return
    # None and re-inflate via the regex fallback; the <body-open-onward scope
    # (lxml closes implicitly) handles it.
    assert _visible_text_len(
        '<html>' + _FOSTER + '<body><p>ok</p>') == 2


def test_visible_text_len_excludes_head_title_from_body_count():
    # <title> lives in <head>, before <body>; it must not be counted as body
    # text. A body with real prose must report its own length only.
    html = ('<html><head><title>This title is long and must not be body text'
            '</title></head><body><p>real body prose here</p></body></html>')
    assert _visible_text_len(html) == len("real body prose here")


def test_visible_text_len_parses_fragments_with_hidden_subtree_removal():
    html = '<div><p hidden>' + ('padding ' * 100) + '</p><div></div></div>'

    assert _visible_text_len(html) == 0
    assert is_blocked(200, html)[0] is True


def test_comment_body_token_does_not_bypass_minimal_text():
    html = (
        '<html><head><!-- <body>'
        + ('padding ' * 100)
        + '--></head><body><p></p></body></html>'
    )

    assert _visible_text_len(html) == 0
    assert is_blocked(200, html)[0] is True


def test_body_start_handles_carriage_returns_before_a_newline():
    html = (
        '<html><head>prefix\r'
        + ('padding ' * 100)
        + '</head>\n<body><p></p></body></html>'
    )

    assert _body_start(html) == html.index('<body>')
    assert _visible_text_len(html) == 0
    assert is_blocked(200, html)[0] is True


def test_content_less_foster_parented_body_still_blocked_via_no_content_elements():
    # When the body has NO content element, the no_content_elements signal
    # catches the page regardless; the foster-parenting fix must not regress
    # the content-less variant.
    html = '<html><head>' + _FOSTER + '</head><body>x</body></html>'
    blocked, reason = is_blocked(200, html)
    assert blocked is True
    assert "no_content_elements" in reason


def test_legit_page_with_head_title_no_false_positive():
    # A page whose real content lives inside <body> (with a <head><title>) must
    # not be flagged; the foster-parenting scope must not perturb normal pages.
    html = ('<html><head><title>How to Detect Bots</title></head><body>'
            '<h1>How to Detect Bots</h1>'
            '<p>Anti-bot solutions help detect and block bot traffic.</p>'
            + ('<p>More article content. </p>' * 50) + '</body></html>')
    assert _visible_text_len(html) > 50
    assert is_blocked(200, html) == (False, "")
