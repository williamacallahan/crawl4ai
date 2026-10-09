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

    def test_tunnel_swap_fires_for_clean_all_proxies_failed_message(self, monkeypatch):
        """Regression for the max_retries >= 1 path: the "All proxies failed: ..."
        shape is clean of every internal marker and container path, so it reaches
        public_error_detail's early-return branch. A recorded dead-target dial
        outcome must still drive the ERR_TUNNEL_CONNECTION_FAILED -> real code
        swap there; without a recording the message passes through verbatim."""
        import time
        import egress_proxy
        from utils import public_error_detail

        # Fresh dial-failure map so this test never sees or leaks state.
        failures = type(egress_proxy._dial_failures)()
        monkeypatch.setattr(egress_proxy, "_dial_failures", failures)

        def _all_proxies_failed(url):
            line = f"Page.goto: net::ERR_TUNNEL_CONNECTION_FAILED at {url}\n"
            return f"All proxies failed: Failed on navigating ACS-GOTO:\n{line}"

        # No recording: a transient proxy fault surfaces verbatim (pass-through).
        transient = _all_proxies_failed("https://transient.example:443/")
        assert public_error_detail(transient) == transient.strip()[:500]

        # Record a refused dial outcome for the same host:port -> swap fires.
        failures[("dead.example", 443)] = ("net::ERR_CONNECTION_REFUSED", time.monotonic())
        refused = _all_proxies_failed("https://dead.example:443/")
        assert public_error_detail(refused) == "Crawl failed: net::ERR_CONNECTION_REFUSED"

        # Default https port (443) is inferred when the URL omits the port.
        failures.clear()
        failures[("dead.example", 443)] = ("net::ERR_NAME_NOT_RESOLVED", time.monotonic())
        unresolvable = _all_proxies_failed("https://dead.example/")
        assert public_error_detail(unresolvable) == "Crawl failed: net::ERR_NAME_NOT_RESOLVED"

        # An expired recording (past the TTL) no-ops -> verbatim pass-through.
        failures.clear()
        failures[("stale.example", 443)] = (
            "net::ERR_ADDRESS_UNREACHABLE",
            time.monotonic() - egress_proxy._DIAL_FAILURE_TTL_S - 1,
        )
        stale = _all_proxies_failed("https://stale.example:443/")
        assert public_error_detail(stale) == stale.strip()[:500]

    def test_clean_non_tunnel_navigation_error_still_passes_through(self, monkeypatch):
        """A clean navigation error that is NOT ERR_TUNNEL_CONNECTION_FAILED
        (e.g. a cert error) must continue to pass through verbatim even though it
        takes the same early-return branch — the swap is tunnel-failure-only."""
        import egress_proxy
        from utils import public_error_detail

        failures = type(egress_proxy._dial_failures)()
        monkeypatch.setattr(egress_proxy, "_dial_failures", failures)

        cert = (
            "All proxies failed: Failed on navigating ACS-GOTO:\n"
            "Page.goto: net::ERR_CERT_AUTHORITY_INVALID at https://self-signed.example/\n"
        )
        assert public_error_detail(cert) == cert.strip()[:500]

    def test_public_crawl_error_mints_correlation_id_when_tunnel_swap_rewrites_clean_message(
        self, monkeypatch
    ):
        """When the tunnel swap rewrites a clean "All proxies failed" message,
        something is withheld (the verbatim message was replaced), so
        public_crawl_error must mint a correlation id — mirroring the contract
        that a correlation id is minted only when something was withheld."""
        import time
        import egress_proxy
        from utils import public_crawl_error

        failures = type(egress_proxy._dial_failures)()
        monkeypatch.setattr(egress_proxy, "_dial_failures", failures)
        failures[("dead.example", 443)] = ("net::ERR_CONNECTION_REFUSED", time.monotonic())

        message = (
            "All proxies failed: Failed on navigating ACS-GOTO:\n"
            "Page.goto: net::ERR_TUNNEL_CONNECTION_FAILED at https://dead.example:443/\n"
        )
        public = public_crawl_error(message, "https://dead.example:443/")
        assert re.fullmatch(
            r"Crawl failed: net::ERR_CONNECTION_REFUSED \(correlation_id=[0-9a-f]{12}\)",
            public,
        )
        assert public.correlation_id

        # No recording -> verbatim pass-through -> NO correlation id minted.
        failures.clear()
        transient = (
            "All proxies failed: Failed on navigating ACS-GOTO:\n"
            "Page.goto: net::ERR_TUNNEL_CONNECTION_FAILED at https://transient.example:443/\n"
        )
        assert public_crawl_error(transient) == transient.strip()

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
