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
