"""Strict-by-default security response headers for every HTTP response."""

from starlette.datastructures import MutableHeaders

SECURITY_BASELINE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
# Strict CSP for the API / error surface (the injection-reflection paths).
STRICT_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'none'"
)
# The dashboard/playground still ship inline scripts/styles; until they are
# externalized (CSP-compat refactor) they must not receive the strict CSP or
# they break. They still get the baseline headers (nosniff, frame DENY).
UI_PREFIXES = ("/dashboard", "/playground", "/static")


class SecurityHeadersMiddleware:
    """Add missing security headers to each response as it starts.

    Pure ASGI on purpose: BaseHTTPMiddleware runs the endpoint in an anyio task
    group whose cancel scope re-delivers a cancellation on every loop tick, so
    a client disconnect would cancel the endpoint's cleanup too (an LLM permit
    release, a browser close) before its awaited I/O finished.
    """

    def __init__(self, app, *, hsts: bool):
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        strict_csp = not scope["path"].startswith(UI_PREFIXES)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_BASELINE_HEADERS.items():
                    headers.setdefault(name, value)
                if strict_csp:
                    headers.setdefault("Content-Security-Policy", STRICT_CSP)
                if self.hsts:
                    headers.setdefault(
                        "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
                    )
            await send(message)

        await self.app(scope, receive, send_with_headers)
