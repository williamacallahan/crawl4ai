"""
egress_proxy.py - localhost pinning forward-proxy for the browser.

context.route() sees URLs, not IPs, so it cannot stop DNS rebinding: Chromium
resolves the target host itself at connect time, and an attacker can answer
"public" to our up-front validation and "169.254.169.254" to the browser.

This proxy is the real control. Chromium is pointed at it (proxy_config), so it
never resolves the target itself - it asks us to CONNECT host:port. We run the
single egress rule (egress_broker.resolve_and_pin: resolve once, reject any
non-global IP, pin one IP), dial the PINNED IP ourselves, and splice raw bytes.
TLS stays end-to-end (we tunnel ciphertext; Chromium verifies the cert/SNI
against the real host - no MITM).

Bound to 127.0.0.1 on an ephemeral port; started at server boot.

If HTTP_PROXY/HTTPS_PROXY (or CRAWL4AI_UPSTREAM_PROXY) is set, the local
pinning proxy still resolves and validates the target itself, then asks the
operator's upstream HTTP proxy to CONNECT to the PINNED IP.  The hostname is
never delegated to that proxy, so corporate proxy compatibility does not
reopen DNS rebinding.  NO_PROXY retains its usual bypass semantics.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import logging
import os
from urllib.parse import unquote, urlsplit

from egress_broker import EgressBlocked, resolve_and_pin

logger = logging.getLogger("crawl4ai.egress")

_CONNECT_OK = b"HTTP/1.1 200 Connection established\r\n\r\n"
_BLOCKED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 11\r\n\r\nURL blocked"
_BAD = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nBad Request"
_MAX_HEADER_BYTES = 64 * 1024


def upstream_proxy(scheme: str = "https"):
    """Return ``(host, port, auth_header | None)`` for an upstream proxy.

    Read the environment per call so runtime configuration and tests see
    changes.  Only plaintext HTTP proxies are supported: HTTPS/SOCKS values
    are ignored instead of being mis-dialed, and malformed higher-priority
    values fall through to the next conventional variable.
    """
    order = (
        ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
        if scheme == "http"
        else ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    )
    for name in ("CRAWL4AI_UPSTREAM_PROXY", *order):
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            parsed = urlsplit(raw if "://" in raw else "http://" + raw)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            logger.warning("ignoring %s: unparseable proxy URL", name)
            continue
        if not hostname:
            logger.warning("ignoring %s: unparseable proxy URL", name)
            continue
        if parsed.scheme != "http":
            logger.warning(
                "ignoring %s: unsupported proxy scheme %r", name, parsed.scheme
            )
            continue
        auth = None
        if parsed.username:
            credentials = (
                f"{unquote(parsed.username)}:{unquote(parsed.password or '')}"
            ).encode()
            auth = (
                b"Proxy-Authorization: Basic "
                + base64.b64encode(credentials)
                + b"\r\n"
            )
        return hostname, port or 80, auth
    return None


def _no_proxy_match(host: str, ip: str, port: int) -> bool:
    raw = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    for entry in entries:
        if entry == "*":
            return True
        try:
            if ipaddress.ip_address(ip) in ipaddress.ip_network(entry, strict=False):
                return True
            continue
        except ValueError:
            pass
        suffix = entry.lower().lstrip(".")
        head, separator, port_part = suffix.rpartition(":")
        if separator and port_part.isdigit():
            if int(port_part) != port:
                continue
            suffix = head
        lowered = host.lower()
        if lowered == suffix or lowered.endswith("." + suffix):
            return True
    return False


def _use_upstream(pin):
    upstream = upstream_proxy(pin.scheme)
    if upstream is None or _no_proxy_match(pin.host, pin.ip, pin.port):
        return None
    return upstream


def _bracket(ip: str) -> str:
    return f"[{ip}]" if ":" in ip else ip


async def _resolve_and_pin(url: str):
    """Resolve without blocking the proxy event loop.

    Use the module-level synchronous name so existing callers/tests can replace
    the single canonical primitive without having to patch a second owner.
    """
    return await asyncio.to_thread(resolve_and_pin, url)


class PinningProxy:
    """Async HTTP forward-proxy that connects only to pinned, global IPs."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self.bound_host: str | None = None
        self.bound_port: int | None = None

    @property
    def url(self) -> str | None:
        if self.bound_port is None:
            return None
        return f"http://{self.bound_host}:{self.bound_port}"

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle, self._host, self._port)
        sock = self._server.sockets[0]
        self.bound_host, self.bound_port = sock.getsockname()[:2]
        url = self.url
        if url is None:  # defensive: bound_port was assigned immediately above
            raise RuntimeError("pinning proxy did not expose a bound port")
        logger.info("egress pinning proxy listening on %s", url)
        upstream = upstream_proxy()
        if upstream is not None:
            logger.info(
                "egress pinning proxy chaining through upstream proxy %s:%s",
                upstream[0],
                upstream[1],
            )
        return url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except OSError as error:
                logger.debug("proxy server close error: %s", error)

    # ─────────────────────────── connection handling ───────────────────────────
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not request_line:
                return
            parts = request_line.split()
            if len(parts) < 3:
                await self._reply(writer, _BAD)
                return
            method = parts[0].decode("latin-1", "replace").upper()
            target = parts[1].decode("latin-1", "replace")

            if method == "CONNECT":
                await self._handle_connect(target, reader, writer)
            else:
                await self._handle_absolute(method, target, request_line, reader, writer)
        except asyncio.TimeoutError:
            await self._reply(writer, _BAD)
        except (OSError, ValueError, UnicodeError) as e:
            logger.debug("proxy connection error: %s", e)
            await self._safe_close(writer)

    async def _handle_connect(self, target, client_reader, client_writer):
        # target is "host:port"
        host, _, port_s = target.rpartition(":")
        if not host or not port_s.isdigit():
            await self._reply(client_writer, _BAD)
            return
        try:
            pin = await _resolve_and_pin(f"https://{host}:{port_s}")
        except EgressBlocked:
            await self._reply(client_writer, _BLOCKED)
            return

        # Drain the rest of the client's CONNECT headers.
        await self._drain_headers(client_reader)

        try:
            up_reader, up_writer = await self._dial(pin, int(port_s))
        except (OSError, ConnectionError, asyncio.TimeoutError):
            await self._reply(client_writer, _BLOCKED)
            return

        client_writer.write(_CONNECT_OK)
        await client_writer.drain()
        await self._splice(client_reader, client_writer, up_reader, up_writer)

    async def _handle_absolute(self, method, target, request_line, client_reader, client_writer):
        # Plain HTTP proxying: target is an absolute URI "http://host/path".
        sp = urlsplit(target)
        if sp.scheme != "http" or not sp.hostname:
            await self._reply(client_writer, _BAD)
            return
        port = sp.port or 80
        try:
            pin = await _resolve_and_pin(f"http://{sp.hostname}:{port}")
        except EgressBlocked:
            await self._reply(client_writer, _BLOCKED)
            return

        headers = await self._read_headers(client_reader)
        path = sp.path or "/"
        if sp.query:
            path += "?" + sp.query
        upstream = _use_upstream(pin)
        destination = (upstream[0], upstream[1]) if upstream else (pin.ip, port)
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(*destination), timeout=30
            )
        except (OSError, ConnectionError, asyncio.TimeoutError):
            await self._reply(client_writer, _BLOCKED)
            return
        # Direct dials use origin form.  An upstream proxy gets absolute form
        # containing only the pinned IP, never a hostname it could re-resolve.
        if upstream is None:
            out = f"{method} {path} HTTP/1.1\r\n".encode("latin-1")
        else:
            out = (
                f"{method} http://{_bracket(pin.ip)}:{port}{path} HTTP/1.1\r\n"
            ).encode("latin-1")
            if upstream[2]:
                out += upstream[2]
            # Only the first request on this connection was validated and
            # rewritten.  Force close so a keep-alive client cannot smuggle a
            # second unvalidated request through the same upstream socket.
            headers = b"".join(
                line + b"\r\n"
                for line in headers.split(b"\r\n")
                if line and not line.lower().startswith(b"connection:")
            ) + b"Connection: close\r\n"
        out += b"Host: " + sp.hostname.encode("latin-1")
        if sp.port:
            out += f":{sp.port}".encode("latin-1")
        out += b"\r\n" + headers + b"\r\n"
        up_writer.write(out)
        await up_writer.drain()
        await self._splice(client_reader, client_writer, up_reader, up_writer)

    # ─────────────────────────── helpers ───────────────────────────
    async def _dial(self, pin, port: int):
        """Open a direct connection or an upstream CONNECT tunnel to a pin."""
        upstream = _use_upstream(pin)
        if upstream is None:
            return await asyncio.wait_for(
                asyncio.open_connection(pin.ip, port), timeout=30
            )

        proxy_host, proxy_port, auth = upstream
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(proxy_host, proxy_port), timeout=30
        )
        try:
            destination = f"{_bracket(pin.ip)}:{port}"
            request = (
                f"CONNECT {destination} HTTP/1.1\r\nHost: {destination}\r\n"
            ).encode("latin-1")
            if auth:
                request += auth
            request += b"\r\n"
            writer.write(request)
            await writer.drain()
            status = await asyncio.wait_for(reader.readline(), timeout=30)
            parts = status.split()
            if len(parts) < 2 or parts[1] != b"200":
                logger.warning("upstream proxy refused CONNECT: %r", status[:64])
                raise ConnectionError("upstream proxy refused CONNECT")
            await self._drain_headers(reader)
        except Exception:
            await self._safe_close(writer)
            raise
        return reader, writer

    async def _drain_headers(self, reader):
        read = 0
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            read += len(line)
            if line in (b"\r\n", b"\n", b""):
                return
            if read > _MAX_HEADER_BYTES:
                return

    async def _read_headers(self, reader) -> bytes:
        buf = bytearray()
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            if line in (b"\r\n", b"\n", b""):
                break
            buf.extend(line)
            if len(buf) > _MAX_HEADER_BYTES:
                break
        # strip any proxy-only / connection headers
        kept = []
        for ln in bytes(buf).split(b"\r\n"):
            name = ln.split(b":", 1)[0].strip().lower()
            if name in (b"proxy-connection", b"proxy-authorization", b"host"):
                continue
            if ln:
                kept.append(ln)
        return (b"\r\n".join(kept) + b"\r\n") if kept else b""

    async def _splice(self, c_reader, c_writer, u_reader, u_writer):
        async def pipe(src, dst):
            try:
                while True:
                    data = await src.read(65536)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except (OSError, ConnectionError) as error:
                logger.debug("proxy splice closed: %s", error)
            finally:
                await self._safe_close(dst)

        await asyncio.gather(
            pipe(c_reader, u_writer),
            pipe(u_reader, c_writer),
        )

    async def _reply(self, writer, payload: bytes):
        try:
            writer.write(payload)
            await writer.drain()
        except (OSError, ConnectionError) as error:
            logger.debug("proxy reply failed: %s", error)
        await self._safe_close(writer)

    @staticmethod
    async def _safe_close(writer):
        try:
            if not writer.is_closing():
                writer.close()
        except (OSError, RuntimeError) as error:
            logger.debug("proxy writer close failed: %s", error)


async def start_pinning_proxy() -> PinningProxy:
    """Start and register one process-local pinning proxy.

    Both the API lifespan and durable worker process use this lifecycle owner;
    callers never instantiate/register a proxy independently.
    """
    from egress_broker import set_egress_proxy

    proxy = PinningProxy()
    set_egress_proxy(await proxy.start())
    return proxy


async def stop_pinning_proxy(proxy: PinningProxy | None) -> None:
    """Unregister and stop a proxy created by :func:`start_pinning_proxy`."""
    if proxy is None:
        return
    from egress_broker import get_egress_proxy, set_egress_proxy

    if get_egress_proxy() == proxy.url:
        set_egress_proxy(None)
    await proxy.stop()
