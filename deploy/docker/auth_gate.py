"""
AuthGateMiddleware - the single, fail-closed authentication boundary.

The previous design decided auth in a per-route FastAPI dependency that, when
`jwt_enabled` was false (the default), returned `lambda: None` - so every
`Depends(token_dep)` was decorative and the whole API was open. Static mounts,
the MCP transports and the Prometheus endpoint were never covered at all.

This middleware moves auth to the outermost ASGI layer so it covers EVERY
route, mount and sub-app (HTTP + WebSocket) uniformly, and it fails closed: a
request without a valid credential is rejected before it reaches any handler.

Accepted credentials:
  * the static operator API token (constant-time compared) -> admin scope, or
  * a valid HS256 JWT minted by this server -> the token's own scope claim.

Public paths (health, token issuance, and exact UI redirects) pass through.
Public prefixes (the UI static shells) also pass through - they serve no data.
On failure: HTTP 401 JSON, or WebSocket close 4401.
On success: the validated principal is attached at scope["state"]["principal"]
(readable downstream as request.state.principal) for scope/ownership checks.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable

import jwt
from auth import constant_time_eq, decode_token


class AuthGateMiddleware:
    def __init__(
        self,
        app,
        *,
        token_provider: Callable[[], str],
        public_paths: Iterable[str] = (),
        public_prefixes: Iterable[str] = (),
    ):
        self.app = app
        self._token_provider = token_provider
        self.public_paths = set(public_paths)
        self.public_prefixes = tuple(public_prefixes)

    # ─────────────────────────── ASGI entry ───────────────────────────
    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.public_paths:
            await self.app(scope, receive, send)
            return
        if any(
            path == prefix or path.startswith(prefix.rstrip("/") + "/")
            for prefix in self.public_prefixes
        ):
            await self.app(scope, receive, send)
            return

        # A real CORS preflight carries both headers.  Let CORSMiddleware
        # decide whether the origin/method is allowlisted; no endpoint data is
        # executed or returned by that middleware.  Bare OPTIONS requests stay
        # behind authentication.
        if scope["type"] == "http" and scope.get("method") == "OPTIONS":
            header_names = {name.lower() for name, _ in scope.get("headers", [])}
            if b"origin" in header_names and b"access-control-request-method" in header_names:
                await self.app(scope, receive, send)
                return

        principal = self._authenticate(scope)
        if principal is None:
            await self._reject(scope, receive, send)
            return

        # Expose the principal to downstream handlers/dependencies.
        state = scope.setdefault("state", {})
        state["principal"] = principal
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return

        has_authorization_header = any(
            name.lower() == b"authorization" for name, _ in scope.get("headers", [])
        )
        bearer_protocol = (
            None
            if has_authorization_header
            else self._websocket_bearer_protocol(scope)
        )
        if bearer_protocol is None:
            await self.app(scope, receive, send)
            return

        async def select_bearer_protocol(message):
            if message["type"] == "websocket.accept" and not message.get("subprotocol"):
                message = {**message, "subprotocol": bearer_protocol}
            await send(message)

        await self.app(scope, receive, select_bearer_protocol)

    # ──────────────────────────── helpers ─────────────────────────────
    def _authenticate(self, scope) -> dict | None:
        token = self._extract_token(scope)
        if not token:
            return None

        # 1) static operator token -> admin scope
        static_token = self._token_provider() or ""
        if static_token and constant_time_eq(token, static_token):
            return {"sub": "operator", "scope": "admin", "via": "api_token"}

        # 2) A valid HS256 JWT.  The effective JWT flag controls whether this
        # server issues new JWTs and whether a JWT-only deployment may expose
        # its socket.  Verification remains compatible with already-issued
        # credentials when issuance is disabled.
        try:
            claims = decode_token(token)
        except (jwt.InvalidTokenError, RuntimeError):
            return None
        claims.setdefault("scope", "data")
        return claims

    @staticmethod
    def _extract_token(scope) -> str | None:
        # Authorization: Bearer <token>
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                raw = value.decode("latin-1")
                if raw[:7].lower() == "bearer ":
                    return raw[7:].strip()
                return None
        # Browsers cannot set Authorization on WebSocket upgrades.  Carry a
        # base64url-encoded bearer token in a WebSocket subprotocol instead;
        # unlike a query parameter it does not enter URLs, access logs, browser
        # history, or referrer telemetry.  Non-browser clients should keep
        # using the Authorization header above.
        if scope["type"] == "websocket":
            protocol = AuthGateMiddleware._websocket_bearer_protocol(scope)
            if protocol is not None:
                encoded = protocol.removeprefix("crawl4ai.bearer.")
                try:
                    padding = "=" * (-len(encoded) % 4)
                    return base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    return None
        return None

    @staticmethod
    def _websocket_bearer_protocol(scope) -> str | None:
        if scope["type"] != "websocket":
            return None
        prefix = "crawl4ai.bearer."
        for protocol in scope.get("subprotocols", []):
            if protocol.startswith(prefix):
                return protocol
        return None

    async def _reject(self, scope, receive, send):
        if scope["type"] == "websocket":
            # Close before accept; 4401 = application "unauthorized".
            await send({"type": "websocket.close", "code": 4401})
            return
        body = json.dumps({"detail": "Authentication required"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
