"""
R6 headers / CSP / CORS behavioral tests.

Security headers are emitted unconditionally (independent of security.enabled).
The strict CSP applies to the API / error surface; the still-inline dashboard /
playground keep the baseline headers but not the strict CSP until they are
externalized (CSP-compat refactor - tracked separately). CORS is deny-by-default.
"""

import re

import pytest

pytestmark = pytest.mark.posture


class TestBaselineHeaders:
    def test_present_on_api_response(self, stock_client):
        r = stock_client.get("/health")
        h = {k.lower(): v for k, v in r.headers.items()}
        assert h.get("x-content-type-options") == "nosniff"
        assert h.get("x-frame-options", "").upper() == "DENY"
        assert h.get("referrer-policy") == "no-referrer"
        assert h.get("cross-origin-opener-policy") == "same-origin"

    def test_present_even_with_security_disabled(self, stock_client, server_module, monkeypatch):
        # Headers must be emitted independent of the security.enabled flag (the
        # old middleware only added them when enabled). Force it off and verify.
        monkeypatch.setitem(server_module.config["security"], "enabled", False)
        r = stock_client.get("/health")
        h = {k.lower() for k in r.headers}
        assert "content-security-policy" in h
        assert "x-content-type-options" in h

    def test_baseline_on_dashboard_mount(self, stock_client, server_module):
        from auth import create_access_token
        h = {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'}, scope='admin')}"}
        r = stock_client.get("/dashboard/", headers=h)
        hh = {k.lower(): v for k, v in r.headers.items()}
        assert hh.get("x-content-type-options") == "nosniff"
        assert hh.get("x-frame-options", "").upper() == "DENY"


class TestStrictCsp:
    def test_api_csp_is_strict(self, stock_client):
        csp = stock_client.get("/health").headers.get("content-security-policy", "")
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "default-src 'none'" in csp
        assert "unsafe-inline" not in csp

    def test_ui_mount_not_given_strict_csp(self, stock_client, server_module):
        # The inline-script UI must not get the strict CSP yet (it would break).
        from auth import create_access_token
        h = {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'}, scope='admin')}"}
        csp = stock_client.get("/playground/", headers=h).headers.get("content-security-policy")
        assert csp is None


class TestCorsDenyByDefault:
    def test_no_cors_allow_origin_by_default(self, stock_client):
        r = stock_client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


class TestErrorSanitization:
    def test_public_error_detail_passes_upstream_reasons(self):
        from utils import public_error_detail

        message = "Blocked by anti-bot protection: Cloudflare JS challenge"
        assert public_error_detail(message) == message
        assert len(public_error_detail("x" * 2000)) == 500

    def test_public_error_detail_genericizes_internal_shapes(self):
        from utils import public_error_detail

        internal = (
            "Unexpected error in _crawl_web at line 806 in _crawl_web "
            "(/app/crawl4ai/async_crawler_strategy.py):\nError: boom\n\n"
            "Code context:\n 805 | raise\n"
        )
        assert public_error_detail(internal) == "Crawl failed"
        assert public_error_detail(None) == "Crawl failed"
        assert public_error_detail("  ") == "Crawl failed"
        assert public_error_detail('File "/app/server.py", line 1') == "Crawl failed"

    def test_all_failed_crawl_returns_502_with_the_already_sanitized_reason(
        self, stock_client, server_module, monkeypatch
    ):
        """/crawl reads the aggregate handle_crawl_request computed rather than
        re-deriving it from projected per-result fields, and forwards the
        error_message that owner already sanitized."""
        from auth import create_access_token

        async def all_urls_failed(**_kwargs):
            return {
                "success": False,
                # A projected result: result_fields may omit "success", which
                # is why the verdict is read from the aggregate above.
                "results": [
                    {
                        "url": "https://example.com",
                        "error_message": "Crawl failed (correlation_id=deadbeefcafe)",
                    }
                ],
            }

        monkeypatch.setattr(server_module, "handle_crawl_request", all_urls_failed)
        h = {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'}, scope='data')}"}
        r = stock_client.post("/crawl", json={"urls": ["https://example.com"]}, headers=h)

        assert r.status_code == 502
        assert r.json()["detail"] == (
            "Crawl request failed: Crawl failed (correlation_id=deadbeefcafe)"
        )

    def test_public_error_detail_genericizes_container_paths_without_a_marker(self):
        """crawl4ai/async_dispatcher.py reports a bare str(e), which carries
        none of the get_error_context markers."""
        from utils import public_error_detail

        assert public_error_detail(
            "[Errno 2] No such file or directory: '/ms-playwright/chromium-1148/chrome'"
        ) == "Crawl failed"
        assert public_error_detail(
            "BrowserType.launch: Executable doesn't exist at /home/appuser/.cache/chrome"
        ) == "Crawl failed"
        # The crawled URL's own path segments must not trip it.
        upstream = "Blocked by anti-bot protection at https://example.com/app/login"
        assert public_error_detail(upstream) == upstream

    def test_public_error_detail_redacts_unlisted_container_path_prefixes(self):
        """Operators routinely override PLAYWRIGHT_BROWSERS_PATH / TMPDIR to
        top-level dirs absent from the default Dockerfile layout (e.g. /data,
        /mnt). The bare str(e) the dispatcher stores (async_dispatcher.py:357
        and :769) carries no get_error_context marker, so _CONTAINER_PATH is
        the only defense; it must match any absolute path, not a fixed
        allowlist of prefixes. Covers both exploit shapes: a missing browser
        binary (PLAYWRIGHT_BROWSERS_PATH) and a failed profile dir (TMPDIR)."""
        from utils import public_error_detail

        # FileNotFoundError shape under unlisted prefixes (the original hardcoded
        # denylist missed every one of these) and its "Failed to start browser"
        # re-raise shape from browser_manager.py.
        for path in (
            "/data/browsers/chrome-linux/chrome",
            "/mnt/cache/chrome-linux/chrome",
            "/scratch/tmp/browser-profile-abc123",
            "/cache/ms-playwright/chromium-1148/chrome",
        ):
            assert public_error_detail(
                f"[Errno 2] No such file or directory: '{path}'"
            ) == "Crawl failed", f"unlisted-prefix FileNotFoundError leaked: {path}"
            assert public_error_detail(
                f"Failed to start browser: [Errno 2] No such file or directory: '{path}'"
            ) == "Crawl failed", f"unlisted-prefix re-raise leaked: {path}"
        # TMPDIR override: the profile dir lives under an unlisted prefix and
        # Chromium fails on it — a different message shape than FileNotFoundError.
        assert public_error_detail(
            "Failed to start browser: could not create profile dir "
            "/data/tmp/browser-profile-abc123: Permission denied"
        ) == "Crawl failed"
        assert public_error_detail(
            "user_data_dir /scratch/tmp/crawl4ai-persistent-xyz is not empty: abort"
        ) == "Crawl failed"

    def test_public_error_detail_redacts_uppercase_container_path_prefixes(self):
        """Directory names may begin with an uppercase letter (e.g. /DATA/...,
        /CACHE/...). _CONTAINER_PATH must match those too, otherwise the bare
        str(e) the dispatcher stores (async_dispatcher.py:357 and :769) would
        leak the path to the client. URL path segments under an uppercase dir
        are still stripped first and must pass through intact."""
        from utils import public_error_detail

        for path in (
            "/DATA/profile-abc123",
            "/CACHE/ms-playwright/chromium-1148/chrome",
            "/Tmp/browser-profile-abc123",
            "/Home/appuser/.cache/ms-playwright/chrome",
        ):
            assert public_error_detail(
                f"[Errno 2] No such file or directory: '{path}'"
            ) == "Crawl failed", f"uppercase-prefix FileNotFoundError leaked: {path}"
        # URL path under an uppercase top-level dir is stripped first and must
        # not be mistaken for a container path.
        assert public_error_detail(
            "Blocked by anti-bot protection at https://example.com/Data/thing"
        ) == "Blocked by anti-bot protection at https://example.com/Data/thing"

    def test_public_error_detail_still_redacts_default_dockerfile_prefixes(self):
        """No regression for the default deployment, which keeps all assets
        under the prefixes the original allowlist covered. Marker-bearing
        payloads are avoided so this exercises the path regex, not the marker
        denylist."""
        from utils import public_error_detail

        for path in (
            "/ms-playwright/chromium-1148/chrome",
            "/home/appuser/.cache/ms-playwright/chrome",
            "/app/server.log",
            "/tmp/browser-profile-abc123",
            "/usr/local/bin/chrome",
            "/opt/crawl4ai/runtime/chrome",
            "/var/log/crawl4ai/server.log",
            "/etc/ssl/certs/ca-certificates.crt",
            "/root/.cache/ms-playwright/chrome",
            "/srv/www/index.html",
            "/proc/1/status",
        ):
            assert public_error_detail(
                f"[Errno 2] No such file or directory: '{path}'"
            ) == "Crawl failed", f"regression: previously-listed prefix now leaks: {path}"

    def test_public_error_detail_url_path_segments_do_not_trip_filter(self):
        """A crawled URL legitimately carries path segments — including under
        unlisted top-level dirs — and is stripped before the path check, so an
        upstream anti-bot / navigation message must pass through intact. This
        is the false-positive guard against over-broadening the regex."""
        from utils import public_error_detail

        for case in (
            # URL path uses a listed prefix; must still pass (existing case).
            "Blocked by anti-bot protection at https://example.com/app/login",
            # URL path uses an unlisted top-level dir; the URL is stripped first.
            "Blocked by anti-bot protection at https://example.com/data/thing",
            "Blocked at https://example.com/mnt/cache/login then retry",
            # Plain upstream reasons with no absolute path at all.
            "Blocked by anti-bot protection: Cloudflare JS challenge",
            "net::ERR_CONNECTION_RESET",
            "Page.goto: Timeout 30000ms exceeded.",
            "DNS_PROBE_FINISHED_NXDOMAIN",
        ):
            assert public_error_detail(case) == case[:500], (
                f"upstream reason incorrectly redacted: {case!r}"
            )

    def test_correlation_id_is_minted_only_when_something_was_withheld(self):
        from utils import public_crawl_error

        clean = "Blocked by anti-bot protection"
        assert public_crawl_error(clean) == clean
        # Merely capped at 500: the caller still sees the real reason.
        assert public_crawl_error("x" * 2000) == "x" * 500
        # Nothing to withhold.
        assert public_crawl_error("   ") == "Crawl failed"
        assert public_crawl_error(None) == "Crawl failed"
        assert public_crawl_error("Traceback (most recent call last)").startswith(
            "Crawl failed (correlation_id="
        )

    def test_withheld_navigation_failure_keeps_only_its_net_error_code(self):
        """A navigation failure outside the target-refusal set arrives wrapped in
        get_error_context; the client keeps the bare Chromium code (target vs
        transient) and the correlation id, never the path, call log, or source."""
        from utils import public_crawl_error

        internal = (
            "Unexpected error in _crawl_web at line 816 in _crawl_web "
            "(../usr/local/lib/python3.12/site-packages/crawl4ai/async_crawler_strategy.py):\n"
            "Error: Failed on navigating ACS-GOTO:\nPage.goto: net::ERR_CONNECTION_CLOSED "
            "at https://example.com/\nCall log:\n  - navigating to \"https://example.com/\"\n\n"
            "Code context:\n 816 →   raise RuntimeError(...)\n"
        )
        public = public_crawl_error(internal, "https://example.com/")

        assert re.fullmatch(
            r"Crawl failed: net::ERR_CONNECTION_CLOSED \(correlation_id=[0-9a-f]{12}\)", public
        )
        assert public.correlation_id

        # A code that only appears in the crawled URL echoed by the call log is not the error.
        timeout = (
            "Unexpected error in _crawl_web (/app/crawl4ai/async_crawler_strategy.py):\n"
            "Error: Failed on navigating ACS-GOTO:\nPage.goto: Timeout 30000ms exceeded.\n"
            "Call log:\n  - navigating to \"https://x.example/net::ERR_FAILED\"\n"
        )
        assert public_crawl_error(timeout).startswith("Crawl failed (correlation_id=")

    def test_execute_js_result_error_message_is_sanitized(
        self, stock_client, server_module, monkeypatch
    ):
        from auth import create_access_token

        class Result:
            success = True
            error_message = (
                "Unexpected error in _crawl_web (/app/crawl4ai/async_crawler_strategy.py)"
            )

            def model_dump(self):
                return {
                    "url": "https://example.com",
                    "success": True,
                    "error_message": self.error_message,
                }

        class Crawler:
            async def arun(self, **_kwargs):
                return [Result()]

        async def get_crawler(_config):
            return Crawler()

        monkeypatch.setattr(server_module, "EXECUTE_JS_ENABLED", True)
        monkeypatch.setattr(server_module, "get_crawler", get_crawler)
        monkeypatch.setattr(server_module, "validate_webhook_url", lambda _url: None)
        h = {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'}, scope='data')}"}
        r = stock_client.post(
            "/execute_js",
            json={"url": "https://example.com", "scripts": ["return 1"]},
            headers=h,
        )

        assert r.status_code == 200, r.text
        assert r.json()["error_message"].startswith("Crawl failed (correlation_id=")
        assert "async_crawler_strategy.py" not in r.text

    def test_5xx_is_generic_with_correlation_id(self, stock_client, server_module):
        from auth import create_access_token
        h = {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'}, scope='admin')}"}
        # /monitor/stats/reset has no monitor singleton without a lifespan -> the
        # handler raises -> our central handler returns a sanitized 500.
        r = stock_client.post("/monitor/stats/reset", headers=h)
        assert r.status_code >= 500
        body = r.json()
        assert body.get("error") == "Internal server error"
        assert "correlation_id" in body
        # No internal detail (exception text / paths) leaked.
        assert "Traceback" not in r.text and "/home/" not in r.text

    def test_4xx_detail_preserved(self, stock_client):
        # /token is public; with no api_token configured it raises a 4xx, which
        # the central handler passes through with its developer-facing detail.
        r = stock_client.post("/token", json={"email": "x@y.com"})
        assert 400 <= r.status_code < 500
        assert "detail" in r.json() and r.json()["detail"]


class TestWebhookHeaderSanitization:
    def test_crlf_in_value_rejected(self):
        from webhook import sanitize_webhook_headers
        with pytest.raises(ValueError):
            sanitize_webhook_headers({"X-Foo": "bar\r\nInjected: 1"})

    def test_hop_by_hop_denied(self):
        from webhook import sanitize_webhook_headers
        for bad in ("Host", "Content-Length", "Transfer-Encoding", "Authorization"):
            with pytest.raises(ValueError):
                sanitize_webhook_headers({bad: "x"})

    def test_bad_name_rejected(self):
        from webhook import sanitize_webhook_headers
        with pytest.raises(ValueError):
            sanitize_webhook_headers({"X Foo": "bar"})

    def test_good_headers_pass(self):
        from webhook import sanitize_webhook_headers
        assert sanitize_webhook_headers({"X-Trace-Id": "abc123"}) == {"X-Trace-Id": "abc123"}

    def test_schema_validator_rejects_early(self):
        from schemas import WebhookConfig
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            WebhookConfig(webhook_url="https://example.com/cb",
                          webhook_headers={"Host": "evil"})


class TestCRLFSafeLogging:
    def test_configured_level_applies_after_logging_has_started(self, monkeypatch):
        import io
        import logging
        from utils import setup_logging

        output = io.StringIO()
        handler = logging.StreamHandler(output)
        root = logging.getLogger()
        previous_level = root.level
        monkeypatch.setattr(root, "handlers", [handler])
        try:
            root.setLevel(logging.WARNING)
            setup_logging({"logging": {"level": "INFO", "format": "%(message)s"}})
            logging.getLogger("server").info("health probe start\r\nprobe_id=example")
            assert root.handlers == [handler]
            assert output.getvalue() == "health probe startprobe_id=example\n"
        finally:
            root.setLevel(previous_level)

    def test_crlf_stripped_from_log_message(self):
        import logging
        from utils import CRLFSafeFilter
        rec = logging.LogRecord("t", logging.INFO, __file__, 1,
                                "url=http://x/\r\nINJECTED admin login", None, None)
        CRLFSafeFilter().filter(rec)
        msg = rec.getMessage()
        assert "\r" not in msg and "\n" not in msg
        assert "INJECTED" in msg  # content kept, just de-fanged
