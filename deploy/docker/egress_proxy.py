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
import errno
import ipaddress
import logging
import os
import time
from collections import OrderedDict
from urllib.parse import unquote, urlsplit

from egress_broker import EgressBlocked, TargetUnresolvable, resolve_and_pin

logger = logging.getLogger("crawl4ai.egress")

_CONNECT_OK = b"HTTP/1.1 200 Connection established\r\n\r\n"
_BLOCKED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 11\r\n\r\nURL blocked"
_BAD = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nBad Request"
_MAX_HEADER_BYTES = 64 * 1024
_HEX_DIGITS = b"0123456789abcdefABCDEF"
# Hop-by-hop fields the proxy replaces with its own Connection: close.
_CONNECTION_FIELDS = (b"connection", b"proxy-connection", b"keep-alive")
# How long stop() keeps ending client connections before it reports failure.
_STOP_DEADLINE_S = 5.0
# How long one stop() pass waits for Server.wait_closed() before sweeping again.
_STOP_SWEEP_S = 0.1


def _body_framing(headers: bytes) -> tuple[int | None, bool]:
    """Return (Content-Length, chunked) parsed from ``headers`` (raw header
    lines joined by ``\\r\\n``).  Content-Length is ``None`` when absent or
    unparseable; chunked is True only when ``Transfer-Encoding: chunked`` is
    present."""
    content_length: int | None = None
    chunked = False
    for line in headers.split(b"\r\n"):
        if not line:
            continue
        name, _, value = line.partition(b":")
        lname = name.strip().lower()
        if lname == b"content-length":
            try:
                content_length = int(value.strip())
            except ValueError:
                content_length = None
        elif lname == b"transfer-encoding" and b"chunked" in value.lower():
            chunked = True
    return content_length, chunked

# Chromium turns every CONNECT answer other than 200/407 into
# net::ERR_TUNNEL_CONNECTION_FAILED (net/http/http_proxy_client_socket.cc,
# DoProcessResponseCode), so no reply can tell the browser that the target
# itself is dead.  The proxy instead records the code a direct connection
# would have produced, per host:port, and the crawl error path reads it back
# through tunnel_failure_code().  Policy blocks record nothing and keep the
# tunnel code.
_DIAL_FAILURE_TTL_S = 60.0
_DIAL_FAILURE_CAPACITY = 1024
_dial_failures: OrderedDict[tuple[str, int], tuple[str, float]] = OrderedDict()


def _dial_failure_code(error: BaseException) -> str | None:
    """The Chromium code for a direct dial that failed with ``error``."""
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "net::ERR_CONNECTION_TIMED_OUT"
    if isinstance(error, TargetUnresolvable):
        return "net::ERR_NAME_NOT_RESOLVED"
    if isinstance(error, ConnectionRefusedError):
        return "net::ERR_CONNECTION_REFUSED"
    if isinstance(error, OSError) and error.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH):
        return "net::ERR_ADDRESS_UNREACHABLE"
    return None


def _record_dial_outcome(host: str, port: int, error: BaseException | None) -> None:
    key = (host.strip("[]").lower(), port)  # urlsplit().hostname drops IPv6 brackets
    code = _dial_failure_code(error) if error is not None else None
    if code is None:
        _dial_failures.pop(key, None)
        return
    _dial_failures[key] = (code, time.monotonic())
    _dial_failures.move_to_end(key)
    while len(_dial_failures) > _DIAL_FAILURE_CAPACITY:
        _dial_failures.popitem(last=False)


def tunnel_failure_code(url: str) -> str | None:
    """The target-side code behind a recent tunnel failure to ``url``, if any."""
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    if not parsed.hostname:
        return None
    recorded = _dial_failures.get((parsed.hostname.lower(), port))
    if recorded is None or time.monotonic() - recorded[1] > _DIAL_FAILURE_TTL_S:
        return None
    return recorded[0]


def _request_body_length(headers: bytes) -> int | None:
    """The request body's Content-Length, or None for a chunked body.

    Raises ValueError for framing that two HTTP parsers could read differently,
    since the bytes after the body decide where the next request starts.
    """
    lengths: set[bytes] = set()
    codings: list[bytes] = []
    for line in headers.split(b"\r\n"):
        name, _, field = line.partition(b":")
        name = name.strip().lower()
        if name == b"content-length":
            lengths.add(field.strip())
        elif name == b"transfer-encoding":
            codings.append(field.strip().lower())
    if codings:
        if lengths or codings[-1].rsplit(b",", 1)[-1].strip() != b"chunked":
            raise ValueError("ambiguous request body framing")
        return None
    if not lengths:
        return 0
    length = lengths.pop()
    if lengths or not length.isdigit():
        raise ValueError("invalid Content-Length")
    return int(length)


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
        # IP/CIDR branch: try the full entry first, then a port-stripped
        # retry for unambiguous IPv4:port / CIDR:port / [IPv6]:port forms.
        match = _ip_match(ip, entry)
        if match is True:
            return True
        if match is False:
            # Entry parsed as a valid IP/CIDR but the pin IP is not in it;
            # preserve the original try/continue semantics by NOT falling
            # through to the hostname branch for IP-form entries.
            continue
        candidate = _strip_port(entry, port)
        if candidate is not None:
            retry = _ip_match(ip, candidate)
            if retry is True:
                return True
            if retry is False:
                continue
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


def _ip_match(ip: str, net: str) -> bool | None:
    """True if `ip` is in network `net`; False if `net` is a valid IP/CIDR but
    `ip` is not in it; None if `net` is not a valid IP/CIDR (e.g. a hostname
    suffix or an entry carrying a `:port`)."""
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(net, strict=False)
    except ValueError:
        return None


def _strip_port(entry: str, want_port: int) -> str | None:
    """Strip an unambiguous trailing port from a NO_PROXY entry, returning the
    port-stripped candidate (or None).

    `:` is ambiguous between an IPv6 group separator and a port delimiter, so
    only IPv4/CIDR (exactly one colon) and bracketed IPv6 (`[addr]:port`) forms
    are split. A port mismatch returns None so the caller falls through to the
    hostname-branch port check rather than matching against the wrong IP.
    """
    if entry.startswith("["):
        close = entry.find("]")
        if close == -1:
            return None
        addr = entry[1:close]
        port_part = entry[close + 1:].lstrip(":")
        if not port_part.isdigit():
            return None  # bare [IPv6] (no :port) -> leave to full-entry branch
        if int(port_part) != want_port:
            return None  # port mismatch
        return addr
    if entry.count(":") == 1:
        head, sep, port_part = entry.rpartition(":")
        if sep and port_part.isdigit():
            if int(port_part) != want_port:
                return None
            return head
    return None


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
        # asyncio.start_server drops its reference to a connection's handler
        # task once the client socket is lost, so the proxy owns each handler,
        # keyed to its client writer, until it finishes; stop() ends survivors.
        self._connections: dict[asyncio.Task, asyncio.StreamWriter] = {}
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
        if self._server is None:
            return
        self._server.close()
        # Server.wait_closed() waits for every client connection. Cancelling a
        # handler ends its tunnel, but closing a client transport that holds
        # unsent bytes for a peer that stopped reading never runs
        # connection_lost, so each client transport is also aborted. A
        # connection accepted just before close() registers its handler only on
        # a later loop pass, so sweeps repeat until the server reports closed.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _STOP_DEADLINE_S
        while True:
            handlers = dict(self._connections)
            for handler, client_writer in handlers.items():
                handler.cancel()
                client_writer.transport.abort()
            if handlers:
                await asyncio.wait(handlers, timeout=max(0.0, deadline - loop.time()))
            try:
                await asyncio.wait_for(
                    self._server.wait_closed(),
                    timeout=max(0.0, min(_STOP_SWEEP_S, deadline - loop.time())),
                )
                return
            except TimeoutError:
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"pinning proxy server did not close within "
                        f"{_STOP_DEADLINE_S}s of stop() "
                        f"({len(self._connections)} tunnel handler(s) still registered)"
                    ) from None

    # ─────────────────────────── connection handling ───────────────────────────
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        connection = asyncio.current_task()
        self._connections[connection] = writer
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
        finally:
            self._connections.pop(connection, None)
            await self._safe_close(writer)

    async def _handle_connect(self, target, client_reader, client_writer):
        # target is "host:port"
        host, _, port_s = target.rpartition(":")
        if not host or not port_s.isdigit():
            await self._reply(client_writer, _BAD)
            return
        port = int(port_s)
        try:
            pin = await _resolve_and_pin(f"https://{host}:{port}")
        except EgressBlocked as error:
            _record_dial_outcome(host, port, error)
            await self._reply(client_writer, _BLOCKED)
            return

        # Drain the rest of the client's CONNECT headers.
        await self._drain_headers(client_reader)

        try:
            up_reader, up_writer = await self._dial(pin, port)
        except (OSError, ConnectionError, asyncio.TimeoutError) as error:
            # A failure reaching an operator upstream proxy is not the target's.
            _record_dial_outcome(host, port, error if _use_upstream(pin) is None else None)
            await self._reply(client_writer, _BLOCKED)
            return
        _record_dial_outcome(host, port, None)

        # A cancel or error before the splice owns the upstream still closes it.
        try:
            client_writer.write(_CONNECT_OK)
            await client_writer.drain()
            await self._splice(
                client_writer,
                up_writer,
                self._pipe(client_reader, up_writer),
                self._pipe(up_reader, client_writer),
            )
        finally:
            await self._safe_close(up_writer)

    async def _handle_absolute(self, method, target, request_line, client_reader, client_writer):
        # Plain HTTP proxying: target is an absolute URI "http://host/path".
        sp = urlsplit(target)
        if sp.scheme != "http" or not sp.hostname:
            await self._reply(client_writer, _BAD)
            return
        port = sp.port or 80
        try:
            pin = await _resolve_and_pin(f"http://{_bracket(sp.hostname)}:{port}")
        except EgressBlocked as error:
            _record_dial_outcome(sp.hostname, port, error)
            await self._reply(client_writer, _BLOCKED)
            return

        headers = await self._read_headers(client_reader)
        try:
            _request_body_length(headers)
        except ValueError:
            await self._reply(client_writer, _BAD)
            return
        path = sp.path or "/"
        if sp.query:
            path += "?" + sp.query
        upstream = _use_upstream(pin)
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
        # Only the first request on this connection was validated and rewritten.
        # Strip every hop-by-hop header -- the client's ``Connection`` plus any
        # header it nominates by name (e.g. ``Connection: X-Foo, keep-alive``)
        # and the standard hop-by-hop set -- and force close, so a keep-alive
        # client cannot reuse this socket to smuggle a second, unvalidated
        # absolute-form request through to the first request's pinned IP.  The
        # generated ``Host`` is added below, after this filter, so it is never
        # stripped even if a malicious client writes ``Connection: Host``; and
        # ``Content-Length`` / ``Transfer-Encoding`` are preserved so the body
        # framing _relay_one forwards to the upstream stays correct.  This guard
        # previously lived only in the upstream-proxy branch; the direct branch
        # leaked the client's Connection header (HTTP/1.1 default keep-alive
        # preserves the socket) and the raw _splice then piped every later
        # request verbatim, unvalidated, to the first pinned destination.
        header_lines = [line for line in headers.split(b"\r\n") if line]
        nominated = set()
        expect_continue = False
        for line in header_lines:
            name, _, value = line.partition(b":")
            lname = name.strip().lower()
            if lname == b"connection":
                nominated.update(
                    token.strip().lower()
                    for token in value.split(b",")
                    if token.strip()
                )
            elif lname == b"expect" and b"100-continue" in value.strip().lower():
                expect_continue = True
        framing = {b"content-length", b"transfer-encoding"}
        drop = (nominated - framing) | {
            b"connection", b"keep-alive", b"te", b"trailer", b"upgrade",
        }
        if expect_continue:
            drop.add(b"expect")
        headers = b"".join(
            line + b"\r\n"
            for line in header_lines
            if line.split(b":", 1)[0].strip().lower() not in drop
        ) + b"Connection: close\r\n"
        out += b"Host: " + _bracket(sp.hostname).encode("latin-1")
        if sp.port:
            out += f":{sp.port}".encode("latin-1")
        out += b"\r\n" + headers + b"\r\n"

        destination = (upstream[0], upstream[1]) if upstream else (pin.ip, port)
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(*destination), timeout=30
            )
        except (OSError, ConnectionError, asyncio.TimeoutError) as error:
            _record_dial_outcome(sp.hostname, port, error if upstream is None else None)
            await self._reply(client_writer, _BLOCKED)
            return
        _record_dial_outcome(sp.hostname, port, None)
        # A cancel or error before the relay owns the upstream still closes it.
        try:
            up_writer.write(out)
            await up_writer.drain()
            # ``_relay_one`` reads the full request body before reading the
            # upstream response, so a client using ``Expect: 100-continue``
            # would deadlock waiting for an interim response we never forward.
            # We stripped ``Expect`` from the upstream request so the origin
            # does not wait for one either; send the interim response to the
            # client ourselves so it sends the body, which ``_relay_one`` then
            # forwards.
            if expect_continue:
                client_writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
                await client_writer.drain()
            # _relay_one forwards exactly this one request and its single
            # response, then closes both sockets -- it never pipes a subsequent
            # client request to the already-pinned upstream, so an origin that
            # ignores our forced ``Connection: close`` still cannot be used to
            # smuggle a second, unvalidated absolute-form request to the pinned
            # IP.  CONNECT keeps the raw bidirectional _splice because its
            # payload is opaque TLS.
            await self._relay_one(
                method, headers, client_reader, client_writer, up_reader, up_writer
            )
        finally:
            await self._safe_close(up_writer)

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
        except BaseException:
            # A cancel during the upstream handshake must close it too.
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

    @staticmethod
    async def _pipe(src, dst):
        while True:
            data = await src.read(65536)
            if not data:
                return
            dst.write(data)
            await dst.drain()

    async def _forward_request_body(self, c_reader, u_writer, length: int | None):
        """Forward the validated request's body, then end at the client's next byte.

        The upstream socket is dialed to this request's pinned IP, so any later
        request on the client connection never passed resolve_and_pin.
        """
        if length is None:
            while True:
                size_line = await c_reader.readline()
                size = size_line.split(b";", 1)[0].strip()
                if not size or size.translate(None, _HEX_DIGITS):
                    raise ValueError("invalid chunk size")
                u_writer.write(size_line)
                chunk_length = int(size, 16)
                if not chunk_length:
                    break
                u_writer.write(await c_reader.readexactly(chunk_length + 2))
                await u_writer.drain()
            while (line := await c_reader.readline()) not in (b"\r\n", b"\n", b""):
                u_writer.write(line)
            u_writer.write(b"\r\n")
        while length:
            data = await c_reader.read(min(length, 65536))
            if not data:
                return
            u_writer.write(data)
            await u_writer.drain()
            length -= len(data)
        await u_writer.drain()
        await c_reader.read(1)

    async def _relay_response(self, u_reader, c_writer):
        """Relay one response whose final head says Connection: close.

        A keep-alive response would let the browser send its next plain-HTTP
        request, for any origin, down this socket to the first request's pin.
        """
        while True:
            fields = [await u_reader.readline()]
            if not fields[0].startswith(b"HTTP/"):
                raise ValueError("upstream sent no HTTP status line")
            head_bytes = len(fields[0])
            while (line := await u_reader.readline()) not in (b"\r\n", b"\n", b""):
                fields.append(line)
                head_bytes += len(line)
                if head_bytes > _MAX_HEADER_BYTES:
                    raise ValueError("upstream response head too large")
            code = fields[0].split()[1:2]
            # 1xx heads precede the final response; 101 switches protocols.
            if code and code[0].startswith(b"1") and code[0] != b"101":
                c_writer.write(b"".join(fields) + b"\r\n")
                continue
            c_writer.write(
                b"".join(
                    line for line in fields
                    if line.split(b":", 1)[0].strip().lower() not in _CONNECTION_FIELDS
                )
                + b"Connection: close\r\n\r\n"
            )
            await c_writer.drain()
            break
        await self._pipe(u_reader, c_writer)

    async def _splice(self, c_writer, u_writer, *directions):
        async def run(direction):
            try:
                await direction
            except (OSError, ValueError, asyncio.IncompleteReadError) as error:
                logger.debug("proxy splice closed: %s", error)

        # The exchange ends when either direction ends. Closing a transport that
        # still holds unsent bytes never wakes a read waiting on that same
        # transport, so the surviving direction is cancelled, not left waiting.
        pipes = {asyncio.create_task(run(direction)) for direction in directions}
        try:
            await asyncio.wait(pipes, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for direction in pipes:
                direction.cancel()
            # The wait lets each transport read what is already in flight, so
            # close sends FIN rather than RST; a second cancel during the wait
            # must still close both writers.
            try:
                await asyncio.wait(pipes)
            finally:
                await self._safe_close(u_writer)
                await self._safe_close(c_writer)
        for direction in pipes:
            if not direction.cancelled():
                direction.result()

    async def _relay_one(self, method, req_headers, c_reader, c_writer,
                        u_reader, u_writer):
        """Relay exactly one HTTP/1.1 request and its response, then close both
        sockets.

        Used on the plain-HTTP path (direct dial and upstream-chained) instead
        of the raw bidirectional ``_splice``: only the first request on this
        connection was validated and re-pinned, so after forwarding it (and any
        request body) and its single framed response we close both ends.  A
        keep-alive client -- or an origin/upstream that ignores the forced
        ``Connection: close`` we appended -- cannot reuse this socket to
        smuggle a second, unvalidated absolute-form request to the pinned IP,
        because no client bytes are ever piped to the upstream after the first
        request body.  CONNECT keeps ``_splice`` because its payload is opaque
        TLS (or another tunnelled protocol) this proxy must not frame.
        """
        await self._forward_http_body(c_reader, u_writer, req_headers)
        try:
            await self._forward_http_response(method, u_reader, c_writer)
        except (OSError, ConnectionError, asyncio.TimeoutError) as error:
            logger.debug("proxy one-request response relay failed: %s", error)
        finally:
            await self._safe_close(c_writer)
            await self._safe_close(u_writer)

    async def _forward_http_body(self, c_reader, u_writer, req_headers):
        """Forward a request body framed by ``req_headers`` (Content-Length or
        chunked); without framing there is no body, since ``_read_headers``
        already consumed up to the terminating blank line."""
        content_length, chunked = _body_framing(req_headers)
        try:
            if chunked:
                await self._forward_chunked(c_reader, u_writer)
            elif content_length is not None and content_length > 0:
                remaining = content_length
                while remaining > 0:
                    chunk = await c_reader.read(min(65536, remaining))
                    if not chunk:
                        break
                    u_writer.write(chunk)
                    await u_writer.drain()
                    remaining -= len(chunk)
        except (OSError, ConnectionError, asyncio.TimeoutError) as error:
            logger.debug("proxy request body relay failed: %s", error)

    async def _forward_http_response(self, method, u_reader, c_writer):
        """Read one HTTP/1.1 response from ``u_reader``, forward it to
        ``c_writer``, and return as soon as the response body is complete so
        the caller can close both ends without waiting for the upstream to
        close (an origin that ignores ``Connection: close`` cannot hold the
        socket open for a smuggled second request).

        The response is accumulated into a buffer and written to the client
        in 64 KiB chunks (a small response is written as a single chunk), the
        same way the raw ``_splice`` delivered it -- so a client that issues
        one bounded ``read()`` for a small response observes the whole
        response in that read, then EOF when we close, instead of a fragment
        of it (status line or a single header line) per ``read()``.
        """
        # Forward any interim 1xx responses (100 Continue, 103 Early Hints) to
        # the client and keep reading until the final response.  The request
        # path strips ``Upgrade``, so 101 Switching Protocols does not occur
        # here; a 101 is treated as the final response (no body) below.
        while True:
            status_line = await asyncio.wait_for(u_reader.readline(), timeout=30)
            if not status_line:
                return
            status_code = 0
            parts = status_line.split(b" ", 2)
            if len(parts) >= 2:
                try:
                    status_code = int(parts[1])
                except ValueError:
                    status_code = 0
            content_length, chunked = None, False
            header_lines = []
            connection_value = b""
            while True:
                line = await asyncio.wait_for(u_reader.readline(), timeout=30)
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.partition(b":")
                lname = name.strip().lower()
                if lname == b"connection":
                    connection_value = value.strip().lower()
                    continue
                header_lines.append(line)
                if lname == b"content-length":
                    try:
                        content_length = int(value.strip())
                    except ValueError:
                        content_length = None
                elif lname == b"transfer-encoding" and b"chunked" in value.lower():
                    chunked = True
            # Rewrite hop-by-hop response headers: drop the origin's
            # ``Connection`` and every header it nominates by name, drop
            # ``Keep-Alive``/``TE``/``Trailer``/``Upgrade``, and append our own
            # ``Connection: close`` so the client treats this socket as
            # single-use and does not try to reuse it for a second (smuggled)
            # request.  ``Content-Length`` and ``Transfer-Encoding`` are kept so
            # the body framing the relay forwards to the client stays correct.
            framing = {b"content-length", b"transfer-encoding"}
            nominated = set()
            if connection_value:
                nominated.update(
                    token.strip()
                    for token in connection_value.split(b",")
                    if token.strip()
                )
            drop = (nominated - framing) | {
                b"keep-alive", b"te", b"trailer", b"upgrade",
            }
            buf = bytearray(status_line)
            for line in header_lines:
                if line.split(b":", 1)[0].strip().lower() in drop:
                    continue
                buf.extend(line)
            buf.extend(b"Connection: close\r\n\r\n")
            if 100 <= status_code < 200 and status_code != 101:
                c_writer.write(buf)
                await c_writer.drain()
                buf.clear()
                continue
            break

        async def flush(threshold: int = 65536):
            if len(buf) >= threshold:
                c_writer.write(buf)
                await c_writer.drain()
                buf.clear()

        no_body = (
            method.upper() == "HEAD"
            or status_code == 101
            or status_code in (204, 304)
        )
        if no_body:
            c_writer.write(buf)
            await c_writer.drain()
            return
        if chunked:
            while True:
                line = await asyncio.wait_for(u_reader.readline(), timeout=30)
                buf.extend(line)
                if line in (b"\r\n", b"\n", b""):
                    break
                size_field = line.strip().split(b";", 1)[0] or b"0"
                try:
                    size = int(size_field, 16)
                except ValueError:
                    break
                if size == 0:
                    while True:
                        trailer = await asyncio.wait_for(u_reader.readline(), timeout=30)
                        buf.extend(trailer)
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                remaining = size
                while remaining > 0:
                    chunk = await u_reader.read(min(65536, remaining))
                    if not chunk:
                        break
                    buf.extend(chunk)
                    remaining -= len(chunk)
                    await flush()
                crlf = await asyncio.wait_for(u_reader.readline(), timeout=30)
                buf.extend(crlf)
                await flush()
        elif content_length is not None:
            remaining = content_length
            while remaining > 0:
                chunk = await u_reader.read(min(65536, remaining))
                if not chunk:
                    break
                buf.extend(chunk)
                remaining -= len(chunk)
                await flush()
        else:
            while True:
                chunk = await u_reader.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                await flush()
        if buf:
            c_writer.write(buf)
            await c_writer.drain()

    async def _forward_chunked(self, src, dst):
        """Copy a chunked HTTP/1.1 body (including the terminating zero-size
        chunk and any trailers) from ``src`` to ``dst`` verbatim."""
        while True:
            line = await asyncio.wait_for(src.readline(), timeout=30)
            dst.write(line)
            await dst.drain()
            if line in (b"\r\n", b"\n", b""):
                break
            size_field = line.strip().split(b";", 1)[0] or b"0"
            try:
                size = int(size_field, 16)
            except ValueError:
                break
            if size == 0:
                while True:
                    trailer = await asyncio.wait_for(src.readline(), timeout=30)
                    dst.write(trailer)
                    await dst.drain()
                    if trailer in (b"\r\n", b"\n", b""):
                        break
                break
            remaining = size
            while remaining > 0:
                chunk = await src.read(min(65536, remaining))
                if not chunk:
                    break
                dst.write(chunk)
                await dst.drain()
                remaining -= len(chunk)
            crlf = await asyncio.wait_for(src.readline(), timeout=30)
            dst.write(crlf)
            await dst.drain()

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
