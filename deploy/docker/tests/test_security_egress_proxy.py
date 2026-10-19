"""
R3 browser egress-proxy tests (real loopback sockets, fully offline).

The pinning proxy is what actually stops DNS rebinding on the browser path:
Chromium is pointed at it, so it asks the proxy to CONNECT host:port; the proxy
resolves-and-pins (egress_broker.resolve_and_pin) and dials only the pinned,
global IP. We drive it with a raw asyncio client + a fake upstream, and stub
resolve_and_pin so a "public" host pins to the loopback upstream while an
"internal" host is refused. (The not-is_global rule itself is covered in
test_security_ssrf_egress.py.)
"""

import asyncio
import socket

import pytest

import egress_proxy
from egress_broker import EgressBlocked, PinnedTarget
from egress_proxy import PinningProxy

pytestmark = pytest.mark.posture

_PROXY_ENV = (
    "CRAWL4AI_UPSTREAM_PROXY", "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy",
)


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch):
    # Keep the suite deterministic on dev machines that sit behind a proxy.
    for name in _PROXY_ENV:
        monkeypatch.delenv(name, raising=False)


async def _fake_upstream():
    async def handle(reader, writer):
        await reader.read(65536)
        writer.write(b"UPSTREAM-OK")
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _fake_http_origin():
    """A loopback HTTP origin that replies 200 OK with a small body.

    Used by the plain-HTTP absolute-URI tests: the pinning proxy re-issues the
    request in origin form to the pinned IP and splices the origin's response
    back to the client, so the client observes a real HTTP status line.
    """
    async def handle(reader, writer):
        await reader.read(65536)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi")
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
class TestPinningProxy:
    async def test_connect_to_global_host_tunnels(self, monkeypatch):
        up, up_port = await _fake_upstream()

        # Pin "good.example" to the loopback upstream (stand-in for a global IP).
        def fake_pin(url):
            return PinnedTarget("https", "good.example", up_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT good.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"200" in status
            await r.readline()  # blank line after the 200
            w.write(b"hello")
            await w.drain()
            body = await asyncio.wait_for(r.read(100), timeout=5)
            assert b"UPSTREAM-OK" in body
            w.close()
        finally:
            await proxy.stop()
            up.close()

    async def test_connect_to_internal_host_blocked(self, monkeypatch):
        def fake_pin(url):
            raise EgressBlocked()
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"403" in status
            w.close()
        finally:
            await proxy.stop()

    async def test_proxy_dials_pinned_ip_not_requested_host(self, monkeypatch):
        # resolve_and_pin returns a pinned ip distinct from the CONNECT host;
        # assert the proxy dials the pinned ip.
        dialed = {}
        up, up_port = await _fake_upstream()

        def fake_pin(url):
            return PinnedTarget("https", "rebind.example", up_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        real_open = asyncio.open_connection

        async def spy_open(host, port, *a, **k):
            dialed["host"], dialed["port"] = host, port
            return await real_open(host, port, *a, **k)
        # patch only the name the proxy module uses
        monkeypatch.setattr(egress_proxy.asyncio, "open_connection", spy_open)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await real_open(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT rebind.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            await asyncio.wait_for(r.readline(), timeout=5)
            assert dialed.get("host") == "127.0.0.1"  # the pinned ip
            w.close()
        finally:
            await proxy.stop()
            up.close()

    async def test_stop_ends_open_tunnels(self, monkeypatch):
        # Server.wait_closed() waits for every open connection, so stop() used
        # to hang behind any idle tunnel; it must end them and return.
        async def silent_upstream(reader, writer):
            await reader.read()
        up = await asyncio.start_server(silent_upstream, "127.0.0.1", 0)
        up_port = up.sockets[0].getsockname()[1]
        monkeypatch.setattr(
            egress_proxy,
            "resolve_and_pin",
            lambda url: PinnedTarget("https", "idle.example", up_port, "127.0.0.1"),
        )

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT idle.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            assert b"200" in await asyncio.wait_for(r.readline(), timeout=5)
            await r.readline()
            await asyncio.wait_for(proxy.stop(), timeout=5)
            assert await asyncio.wait_for(r.read(), timeout=5) == b""
            w.close()
        finally:
            up.close()

    async def test_tunnel_ends_when_client_finishes_to_non_reading_upstream(
        self, monkeypatch
    ):
        # The upstream never reads, so bytes the client sends beyond the
        # kernel's small buffers stay queued on the proxy's upstream writer.
        # Closing that writer then never runs connection_lost, so a splice
        # that waits for both directions keeps reading the silent upstream
        # forever after the client's direction ends.
        # The kernel completes the handshake into the backlog; nothing accepts
        # or reads, and the accepted socket inherits the small receive buffer.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        up_port = listener.getsockname()[1]
        monkeypatch.setattr(
            egress_proxy,
            "resolve_and_pin",
            lambda url: PinnedTarget("https", "stuck.example", up_port, "127.0.0.1"),
        )
        real_open_connection = asyncio.open_connection

        async def small_send_buffer(*args, **kwargs):
            reader, writer = await real_open_connection(*args, **kwargs)
            writer.get_extra_info("socket").setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, 4096
            )
            return reader, writer

        monkeypatch.setattr(egress_proxy.asyncio, "open_connection", small_send_buffer)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await real_open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT stuck.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            assert b"200" in await asyncio.wait_for(r.readline(), timeout=5)
            await r.readline()
            # Above the kernel buffers, below the transport's 64 KiB
            # high-water mark, so the client direction reaches EOF.
            w.write(b"x" * (40 * 1024))
            w.write_eof()
            assert await asyncio.wait_for(r.read(), timeout=5) == b""
            w.close()
        finally:
            listener.close()
            await proxy.stop()

    async def test_stop_aborts_client_holding_unsent_bytes(self, monkeypatch):
        # A client that stops reading leaves the proxy's client writer holding
        # unsent bytes; closing it never runs connection_lost, so
        # Server.wait_closed() hung unless stop() aborts the transport.
        async def flooding_upstream(reader, writer):
            writer.write(b"x" * (8 * 1024 * 1024))
            await reader.read()
        up = await asyncio.start_server(flooding_upstream, "127.0.0.1", 0)
        up_port = up.sockets[0].getsockname()[1]
        monkeypatch.setattr(
            egress_proxy,
            "resolve_and_pin",
            lambda url: PinnedTarget("https", "flood.example", up_port, "127.0.0.1"),
        )

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT flood.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            assert b"200" in await asyncio.wait_for(r.readline(), timeout=5)
            await r.readline()
            # Proves the flood is flowing; the client reads nothing more.
            await asyncio.wait_for(r.readexactly(1), timeout=5)
            await asyncio.sleep(0.5)
            await asyncio.wait_for(proxy.stop(), timeout=5)
            w.close()
        finally:
            up.close()

    async def test_malformed_connect_400(self):
        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(b"CONNECT not-a-host-port HTTP/1.1\r\n\r\n")
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"400" in status
            w.close()
        finally:
            await proxy.stop()

    async def test_direct_plain_http_keep_alive_origin_cannot_take_next_request(
        self, monkeypatch
    ):
        """A keep-alive origin dialed directly never receives a second request
        on the same client connection: that request was never resolved or
        pinned, and the origin could answer it as another site (issue #483)."""
        seen = []

        async def keep_alive_origin(reader, writer):
            while request := await reader.read(65536):
                seen.append(request)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n"
                             b"Connection: keep-alive\r\n\r\nevil")
                await writer.drain()
            writer.close()
        origin = await asyncio.start_server(keep_alive_origin, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]
        monkeypatch.setattr(
            egress_proxy, "resolve_and_pin",
            lambda url: PinnedTarget("http", "attacker.example", origin_port, "127.0.0.1"),
        )

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            # The proxy dials the pinned IP at the requested URL's port.
            w.write(f"GET http://attacker.example:{origin_port}/ HTTP/1.1\r\n"
                    "Host: attacker.example\r\nConnection: keep-alive\r\n\r\n".encode())
            await w.drain()
            head = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=5)
            assert await asyncio.wait_for(r.readexactly(4), timeout=5) == b"evil"
            w.write(b"GET http://victim.example/ HTTP/1.1\r\nHost: victim.example\r\n\r\n")
            await w.drain()
            leftover = await asyncio.wait_for(r.read(200), timeout=5)
            w.close()
        finally:
            await proxy.stop()
            origin.close()
        assert len(seen) == 1 and b"victim.example" not in seen[0]
        assert leftover == b""
        assert head.startswith(b"HTTP/1.1 200")
        assert b"Connection: close\r\n" in head and b"keep-alive" not in head

    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("refused.example", "net::ERR_CONNECTION_REFUSED"),
            ("[2001:db8::1]", "net::ERR_CONNECTION_REFUSED"),
            ("unresolvable.example", "net::ERR_NAME_NOT_RESOLVED"),
            ("internal.example", "net::ERR_TUNNEL_CONNECTION_FAILED"),
        ],
    )
    async def test_dead_target_surfaces_its_own_code_not_the_tunnel_code(
        self, monkeypatch, host, expected
    ):
        """Chromium reports every refused CONNECT as ERR_TUNNEL_CONNECTION_FAILED;
        the public crawl error restores a dead target's code, while a policy block
        keeps the tunnel code."""
        from egress_broker import TargetUnresolvable
        from utils import public_error_detail

        monkeypatch.setattr(egress_proxy, "_dial_failures", type(egress_proxy._dial_failures)())
        proxy = PinningProxy()
        await proxy.start()
        # A port nothing listens on, taken after the proxy bound its own.
        closed = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        closed_port = closed.sockets[0].getsockname()[1]
        closed.close()
        await closed.wait_closed()

        def fake_pin(url):
            if host == "unresolvable.example":
                raise TargetUnresolvable()
            if host == "internal.example":
                raise EgressBlocked()
            return PinnedTarget("https", host, closed_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT {host}:{closed_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"403" in status
            w.close()
        finally:
            await proxy.stop()

        playwright_error = (
            "Unexpected error in _crawl_web (/app/crawl4ai/async_crawler_strategy.py):\n"
            "Error: Failed on navigating ACS-GOTO:\n"
            f"Page.goto: net::ERR_TUNNEL_CONNECTION_FAILED at https://{host}:{closed_port}/\n"
            f"Call log:\n  - navigating to \"https://{host}:{closed_port}/\"\n"
        )
        assert public_error_detail(playwright_error) == f"Crawl failed: {expected}"

    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("refused.example", "net::ERR_CONNECTION_REFUSED"),
            ("[2001:db8::1]", "net::ERR_CONNECTION_REFUSED"),
            ("unresolvable.example", "net::ERR_NAME_NOT_RESOLVED"),
            ("internal.example", None),
        ],
    )
    async def test_dead_target_surfaces_its_own_code_not_the_tunnel_code_all_proxies_failed_shape(
        self, monkeypatch, host, expected
    ):
        """Regression: the "All proxies failed: ..." message shape produced when
        CrawlerRunConfig.max_retries >= 1 (async_crawler_strategy.py:816 ->
        async_webcrawler.py:677) carries no internal-error marker and no
        container path, so it takes public_error_detail's early-return branch.
        The tunnel-code swap must still fire there, otherwise a dead target
        surfaces the misleading ERR_TUNNEL_CONNECTION_FAILED verbatim. A policy
        block records no dial outcome, so it keeps the verbatim pass-through."""
        from egress_broker import TargetUnresolvable
        from utils import public_error_detail

        monkeypatch.setattr(egress_proxy, "_dial_failures", type(egress_proxy._dial_failures)())
        proxy = PinningProxy()
        await proxy.start()
        closed = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        closed_port = closed.sockets[0].getsockname()[1]
        closed.close()
        await closed.wait_closed()

        def fake_pin(url):
            if host == "unresolvable.example":
                raise TargetUnresolvable()
            if host == "internal.example":
                raise EgressBlocked()
            return PinnedTarget("https", host, closed_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT {host}:{closed_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"403" in status
            w.close()
        finally:
            await proxy.stop()

        # The path-B shape: str(RuntimeError(...)) prefixed with "All proxies
        # failed: ", clean of every marker and container path.
        playwright_line = (
            f"Page.goto: net::ERR_TUNNEL_CONNECTION_FAILED at https://{host}:{closed_port}/\n"
            f"Call log:\n  - navigating to \"https://{host}:{closed_port}/\"\n"
        )
        runtime_err = str(RuntimeError(f"Failed on navigating ACS-GOTO:\n{playwright_line}"))
        message = f"All proxies failed: {runtime_err}"

        if expected is not None:
            assert public_error_detail(message) == f"Crawl failed: {expected}"
        else:
            # Policy block: nothing recorded -> swap no-ops -> verbatim pass-through
            # (public_error_detail strips then caps at 500).
            assert public_error_detail(message) == message.strip()[:500]


@pytest.mark.asyncio
class TestAbsoluteUriIpv6Literal:
    """Regression for the bracket-dropping resolve-URL reconstruction in
    ``_handle_absolute``.

    A plain-HTTP absolute-URI request to a bracketed IPv6 literal
    (``GET http://[2606:4700::1]/path``) must be tunneled for a global literal
    and rejected with 403 for a non-global literal. The bug rebuilt the URL
    handed to ``resolve_and_pin`` from ``urlsplit(target).hostname`` which
    strips the ``[]`` brackets, producing e.g. ``http://2606:4700::1:80`` that
    ``urlparse`` mis-parses (hostname becomes the first hex group, ``.port``
    raises ``ValueError``); the ``ValueError`` escaped the ``except
    EgressBlocked`` clause and the proxy closed the connection silently.
    """

    async def test_resolve_url_keeps_ipv6_brackets_and_tunnels(self, monkeypatch):
        """The URL handed to ``resolve_and_pin`` retains the IPv6 brackets so
        ``urlparse`` parses host+port correctly, the forwarded ``Host`` field is
        bracketed, and the request is tunneled."""
        sent = {}
        up_port = {"port": None}

        async def handle(reader, writer):
            request = await reader.read(65536)
            sent["request"] = request
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\nhi"
            )
            await writer.drain()
            writer.close()
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        up_port["port"] = server.sockets[0].getsockname()[1]

        seen = {}

        def fake_pin(url):
            seen["url"] = url
            return PinnedTarget("http", "2606:4700::1", up_port["port"], "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                f"GET http://[2606:4700::1]:{up_port['port']}/path HTTP/1.1\r\n"
                f"Host: [2606:4700::1]\r\n\r\n".encode()
            )
            await w.drain()
            data = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in data
            w.close()
        finally:
            await proxy.stop()
            server.close()
            await server.wait_closed()
        # Pre-fix the recorded URL was 'http://2606:4700::1:<port>' (no
        # brackets), which urlparse mis-parses. The fix preserves the brackets.
        assert seen["url"] == f"http://[2606:4700::1]:{up_port['port']}"
        # The forwarded Host field must bracket the IPv6 literal so an origin
        # that validates Host does not reject the request. The port is appended
        # by the proxy, so check the bracketed host prefix only.
        assert b"Host: [2606:4700::1]" in sent["request"]

    async def test_global_ipv6_tunnels_through_real_resolve(self, monkeypatch):
        """End-to-end through the REAL ``resolve_and_pin`` (only DNS mocked): a
        global IPv6 literal pins to itself, the proxy dials the pinned IP (not
        a re-resolved hostname), and the response is tunneled. Pre-fix the
        unbracketed rebuilt URL raised ``ValueError`` inside ``resolve_and_pin``
        and the proxy closed silently."""
        import egress_broker
        up, up_port = await _fake_http_origin()

        def fake_getaddrinfo(host, port, *a, **k):
            if host == "2606:4700::1":
                return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "",
                         (host, port, 0, 0))]
            raise OSError("no such host")
        monkeypatch.setattr(egress_broker.socket, "getaddrinfo", fake_getaddrinfo)

        real_open = asyncio.open_connection
        dialed = {}

        async def redirect_open(host, port, *a, **k):
            dialed["host"], dialed["port"] = host, port
            # The pinned IPv6 is not dialable offline; redirect to the origin.
            return await real_open("127.0.0.1", up_port)
        monkeypatch.setattr(egress_proxy.asyncio, "open_connection", redirect_open)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await real_open(proxy.bound_host, proxy.bound_port)
            w.write(
                f"GET http://[2606:4700::1]:{up_port}/path HTTP/1.1\r\n"
                f"Host: [2606:4700::1]\r\n\r\n".encode()
            )
            await w.drain()
            data = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in data
            w.close()
        finally:
            await proxy.stop()
            up.close()
        # The proxy dialed the pinned IPv6 literal, never re-resolving the host.
        assert dialed["host"] == "2606:4700::1"
        assert dialed["port"] == up_port

    async def test_non_global_ipv6_returns_403_not_silent_close(self, monkeypatch):
        """End-to-end through the REAL ``resolve_and_pin`` (only DNS mocked): a
        non-global IPv6 literal (ULA ``fc00::/7``) must produce the recognizable
        403 Forbidden, not a silent connection close. Pre-fix the unbracketed
        rebuilt URL raised ``ValueError`` before the ``is_global`` check, so the
        proxy closed silently."""
        import egress_broker

        def fake_getaddrinfo(host, port, *a, **k):
            if host == "fc00::1":
                return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "",
                         (host, port, 0, 0))]
            raise OSError("no such host")
        monkeypatch.setattr(egress_broker.socket, "getaddrinfo", fake_getaddrinfo)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(b"GET http://[fc00::1]:80/path HTTP/1.1\r\n"
                    b"Host: [fc00::1]\r\n\r\n")
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"403" in status
            w.close()
        finally:
            await proxy.stop()


async def _fake_corporate_proxy(seen):
    """Minimal HTTP proxy: records the CONNECT request line, replies 200, then
    answers any tunneled bytes with TUNNEL-OK."""
    async def handle(reader, writer):
        line = await reader.readline()
        seen.append(line)
        while True:  # drain CONNECT headers
            h = await reader.readline()
            if h in (b"\r\n", b"\n", b""):
                break
        writer.write(b"HTTP/1.1 200 Connection established\r\nVia: fake\r\n\r\n")
        await writer.drain()
        await reader.read(65536)
        writer.write(b"TUNNEL-OK")
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
class TestUpstreamChaining:
    async def test_chained_connect_pins_ip_and_blocks_before_upstream(self, monkeypatch):
        """The chained-CONNECT security contract: the upstream receives the
        PINNED IP (never a hostname to resolve), its response headers do not
        leak into the tunnel, and a blocked target produces an opaque 403
        with zero upstream traffic."""
        seen = []
        corp, corp_port = await _fake_corporate_proxy(seen)
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{corp_port}")

        def fake_pin(url):
            if "internal.example" in url:
                raise EgressBlocked()
            return PinnedTarget("https", "good.example", 443, "203.0.113.7")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(b"CONNECT good.example:443 HTTP/1.1\r\n\r\n")
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"200" in status
            await r.readline()  # blank line after the 200
            w.write(b"hello")
            await w.drain()
            body = await asyncio.wait_for(r.read(100), timeout=5)
            # Upstream's Via header must NOT leak into the tunnel.
            assert body == b"TUNNEL-OK"
            w.close()

            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(b"CONNECT internal.example:443 HTTP/1.1\r\n\r\n")
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"403" in status
            w.close()
        finally:
            await proxy.stop()
            corp.close()
        # The upstream saw ONLY the pinned IP of the allowed target.
        assert seen == [b"CONNECT 203.0.113.7:443 HTTP/1.1\r\n"]

    async def test_chained_plain_http_pinned_absolute_form_no_smuggling(self, monkeypatch):
        """Plain HTTP via upstream: the request is re-issued in absolute form
        against the PINNED IP (no name for the upstream to resolve), carries
        Connection: close, and a reused client connection cannot smuggle a
        second, unvalidated request upstream."""
        lines = []

        async def handle(reader, writer):
            req = b""
            while b"\r\n\r\n" not in req:
                req += await reader.read(4096)
            lines.append(req)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi")
            await writer.drain()
            writer.close()
        corp = await asyncio.start_server(handle, "127.0.0.1", 0)
        corp_port = corp.sockets[0].getsockname()[1]
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{corp_port}")

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", 80, "203.0.113.7")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(b"GET http://plain.example/ HTTP/1.1\r\n"
                    b"Host: plain.example\r\nConnection: keep-alive\r\n\r\n")
            await w.drain()
            first = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in first
            # Attempt to smuggle an unvalidated request on the same connection.
            w.write(b"GET http://rebind.evil/ HTTP/1.1\r\nHost: rebind.evil\r\n\r\n")
            await w.drain()
            leftover = await asyncio.wait_for(r.read(200), timeout=5)
            assert leftover == b""  # upstream closed; nothing came back
            w.close()
        finally:
            await proxy.stop()
            corp.close()
        sent = b"".join(lines)
        assert sent.startswith(b"GET http://203.0.113.7:80/ HTTP/1.1\r\n")
        assert b"Connection: close" in sent
        assert b"keep-alive" not in sent
        assert b"rebind.evil" not in sent  # the smuggled request never got upstream

    async def test_no_proxy_ip_port_form_dials_direct_not_through_upstream(self, monkeypatch):
        """End-to-end regression for be18eb4: an ``IP:port`` NO_PROXY entry must
        cause the proxy to dial the pinned IP directly and not chain a CONNECT
        through the corporate upstream (the bug silently routed the request
        through the corporate proxy)."""
        up, up_port = await _fake_upstream()  # stand-in for the pinned IP
        corp_seen = []

        async def corp_handle(reader, writer):
            corp_seen.append(await reader.readline())
            writer.close()

        corp = await asyncio.start_server(corp_handle, "127.0.0.1", 0)
        corp_port = corp.sockets[0].getsockname()[1]
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{corp_port}")
        monkeypatch.setenv("NO_PROXY", f"127.0.0.1:{up_port}")  # IP:port form

        def fake_pin(url):
            return PinnedTarget("https", "good.example", up_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        dialed = {}
        real_open = asyncio.open_connection

        async def spy_open(host, port, *a, **k):
            dialed["host"], dialed["port"] = host, port
            return await real_open(host, port, *a, **k)
        monkeypatch.setattr(egress_proxy.asyncio, "open_connection", spy_open)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await real_open(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT good.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"200" in status
            await r.readline()  # blank line after the 200
            w.write(b"hello")
            await w.drain()
            body = await asyncio.wait_for(r.read(100), timeout=5)
            assert b"UPSTREAM-OK" in body
            w.close()
        finally:
            await proxy.stop()
            up.close()
            corp.close()
        # Direct dial target was the pinned IP at the request port, NOT the
        # corporate proxy port.
        assert dialed["host"] == "127.0.0.1"
        assert dialed["port"] == up_port
        # Corporate proxy received zero CONNECT requests.
        assert corp_seen == []

    async def test_no_proxy_ip_port_mismatch_chains_through_upstream(self, monkeypatch):
        """The port scope on ``IP:port`` is honored end-to-end: a port mismatch
        must fall back to chaining through the corporate proxy (i.e. the bug
        fix must NOT over-broaden to ignore the port)."""
        up, up_port = await _fake_upstream()
        seen = []
        corp, corp_port = await _fake_corporate_proxy(seen)
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{corp_port}")
        monkeypatch.setenv("NO_PROXY", f"127.0.0.1:1")  # port != up_port

        def fake_pin(url):
            return PinnedTarget("https", "good.example", up_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(f"CONNECT good.example:{up_port} HTTP/1.1\r\n\r\n".encode())
            await w.drain()
            status = await asyncio.wait_for(r.readline(), timeout=5)
            assert b"200" in status
            await r.readline()
            w.write(b"hello")
            await w.drain()
            body = await asyncio.wait_for(r.read(100), timeout=5)
            assert body == b"TUNNEL-OK"
            w.close()
        finally:
            await proxy.stop()
            up.close()
            corp.close()
        # The corporate proxy DID receive a CONNECT to the pinned IP.
        assert seen == [f"CONNECT 127.0.0.1:{up_port} HTTP/1.1\r\n".encode()]


@pytest.mark.asyncio
class TestDirectDialNoSmuggling:
    """Regression for the egress proxy's direct-dial (no-upstream) keep-alive
    request-smuggling hole.

    ``_handle_absolute`` validates and rewrites only the FIRST request on a
    client connection, then hands both socket ends to a raw bidirectional
    ``_splice``.  The upstream-proxy branch already rejects a smuggled second
    request by stripping the client's ``Connection`` header and appending
    ``Connection: close`` to the rewritten request (covered by
    ``TestUpstreamChaining::test_chained_plain_http_pinned_absolute_form_no_smuggling``);
    the direct-dial branch lacked the same guard, so a keep-alive client's
    ``Connection`` header (or HTTP/1.1's persistence-by-default) survived into
    the direct-dial request, both sides stayed alive after the first response,
    and any later absolute-form request on the same socket was piped verbatim
    -- unvalidated, never re-pinned, never policy-checked -- to the FIRST
    request's pinned destination, letting an attacker-controlled plain-HTTP
    origin spoof a different plain-HTTP origin's content/cookies.  These
    tests mirror the upstream regression test for the direct branch.
    """

    async def test_direct_plain_http_pinned_absolute_form_no_smuggling(
        self, monkeypatch
    ):
        """Direct-dial plain HTTP: the request is re-issued in ORIGIN form
        (path only) to the PINNED IP, carries ``Connection: close``, the
        client's ``keep-alive`` is stripped, and a reused client connection
        cannot smuggle a second, unvalidated absolute-form request to the
        first origin.  Mirrors the upstream-proxy branch regression test."""
        lines = []

        async def handle(reader, writer):
            # Record exactly one complete request; the smuggled second request
            # must never arrive at this origin.
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            lines.append(req)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\nhi"
            )
            await writer.drain()
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        pin_calls = []

        def fake_pin(url):
            pin_calls.append(url)
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"GET http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n"
                 "Connection: keep-alive\r\n\r\n").encode()
            )
            await w.drain()
            first = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in first
            # Attempt to smuggle an unvalidated request on the same connection.
            w.write(b"GET http://rebind.evil/ HTTP/1.1\r\nHost: rebind.evil\r\n\r\n")
            await w.drain()
            leftover = await asyncio.wait_for(r.read(200), timeout=5)
            assert leftover == b""  # origin closed; nothing came back
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        sent = b"".join(lines)
        # Direct dial uses ORIGIN form (path only), not absolute form.
        assert sent.startswith(b"GET / HTTP/1.1\r\n")
        # The proxy must strip any client Connection header and force close.
        assert b"Connection: close" in sent
        assert b"keep-alive" not in sent.lower()
        # The smuggled request never reached the origin.
        assert b"rebind.evil" not in sent
        # The smuggled target was never re-resolved / re-pinned: only the
        # first request triggers resolve_and_pin (pre-fix this was also true
        # because the raw splice bypassed _handle_absolute, but we assert it
        # here to lock the invariant).
        assert len(pin_calls) == 1
        assert "rebind.evil" not in pin_calls[0]

    async def test_direct_path_force_close_without_explicit_client_connection_header(
        self, monkeypatch
    ):
        """HTTP/1.1 defaults to keep-alive, so a client that omits
        ``Connection`` entirely still reuses the socket for a second request.
        The direct branch must append ``Connection: close`` even when no
        client ``Connection`` header was sent, otherwise a relying-on-defaults
        keep-alive client can still smuggle a second request.  Mirrors the
        upstream branch's behavior."""
        lines = []

        async def handle(reader, writer):
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            lines.append(req)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\nhi"
            )
            await writer.drain()
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            # No Connection header at all - HTTP/1.1 keep-alive is the default.
            w.write(
                (f"GET http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n\r\n").encode()
            )
            await w.drain()
            first = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in first
            # Attempt to smuggle an unvalidated request on the same connection.
            w.write(b"GET http://rebind.evil/ HTTP/1.1\r\nHost: rebind.evil\r\n\r\n")
            await w.drain()
            leftover = await asyncio.wait_for(r.read(200), timeout=5)
            assert leftover == b""  # origin closed; nothing came back
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        sent = b"".join(lines)
        # The proxy added Connection: close even though the client sent none.
        assert b"Connection: close" in sent
        # The smuggled request never reached the origin.
        assert b"rebind.evil" not in sent

    async def test_direct_path_request_line_stays_origin_form(self, monkeypatch):
        """Sanity guard that the direct-path fix does NOT change the
        request-line form: a plain-HTTP absolute-URI request forwarded direct
        (no upstream) must still be rewritten to ORIGIN form (``GET /path``)
        against the pinned IP -- never forwarded verbatim in absolute form,
        which would expose the request-line host to the origin server and
        allow origin confusion.  The direct branch always had this; pin it
        down so a future change cannot regress it while adding the close."""
        sent = {}

        async def handle(reader, writer):
            request = await reader.read(65536)
            sent["request"] = request
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                b"Connection: close\r\n\r\nhi"
            )
            await writer.drain()
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"GET http://plain.example:{origin_port}/path?q=1 HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n\r\n").encode()
            )
            await w.drain()
            data = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in data
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        request = sent["request"]
        # Origin form: path + query preserved, no absolute-URI in request line.
        assert request.startswith(b"GET /path?q=1 HTTP/1.1\r\n")
        assert b"plain.example" not in request.split(b"\r\n")[0]
        # Host header carries the original hostname so the origin can route.
        assert b"Host: plain.example:" + str(origin_port).encode() in request
        # The proxy's own Connection: close is appended.
        assert b"Connection: close" in request

    async def test_direct_path_origin_ignoring_connection_close_no_smuggling(
        self, monkeypatch
    ):
        """The raw ``_splice`` relay relied on the origin honoring the forced
        ``Connection: close``: an origin that ignored it and kept the socket
        alive would let a keep-alive client pipe a second, unvalidated
        absolute-form request to the first request's pinned IP.  The
        one-request ``_relay_one`` must close both ends as soon as the
        *framed* first response is complete -- without waiting for the origin
        to close -- so smuggling is impossible even against a non-cooperating
        origin.  This origin deliberately keeps the connection open after the
        Content-Length response and never closes on its own."""
        received = []

        async def handle(reader, writer):
            # Read the first request only.
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            received.append(req)
            # Send a full Content-Length response and KEEP THE SOCKET OPEN,
            # ignoring the proxy's Connection: close.
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nhi"
            )
            await writer.drain()
            # Wait for any smuggled second request; record it if it arrives.
            try:
                second = b""
                while b"\r\n\r\n" not in second:
                    chunk = await asyncio.wait_for(reader.read(4096), timeout=2)
                    if not chunk:
                        break
                    second += chunk
                received.append(second)
            except asyncio.TimeoutError:
                pass
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"GET http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n"
                 "Connection: keep-alive\r\n\r\n").encode()
            )
            await w.drain()
            first = await asyncio.wait_for(r.read(200), timeout=5)
            assert b"200" in first
            assert first.endswith(b"hi")
            # Attempt to smuggle an unvalidated request on the same connection.
            w.write(b"GET http://rebind.evil/ HTTP/1.1\r\nHost: rebind.evil\r\n\r\n")
            await w.drain()
            # The proxy must have closed the client socket after the framed
            # response; the smuggled request yields EOF, not a second response.
            leftover = await asyncio.wait_for(r.read(200), timeout=5)
            assert leftover == b""
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        # Exactly one (non-empty) request reached the origin; the smuggled
        # second one never did -- at most an EOF from the proxy closing the
        # upstream socket after the framed response.
        assert [chunk for chunk in received if chunk] == [received[0]]
        assert b"rebind.evil" not in b"".join(received)

    async def test_direct_path_rewrites_response_hop_by_hop_headers(
        self, monkeypatch
    ):
        """The relay must not forward the origin's ``Connection`` and
        ``Keep-Alive`` headers verbatim: the proxy forces a single response
        and closes both sockets, so the client must see ``Connection: close``
        (not ``keep-alive``) or a keep-alive client will reuse the socket for
        a second, unvalidated request.  ``Content-Length`` and body framing
        must be preserved so the client still reads the response correctly."""
        async def handle(reader, writer):
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            # Origin sends keep-alive and nominates X-Custom via Connection.
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 2\r\n"
                b"Connection: keep-alive, X-Custom\r\n"
                b"Keep-Alive: timeout=30\r\n"
                b"X-Custom: should-be-stripped\r\n"
                b"\r\nhi"
            )
            await writer.drain()
            await asyncio.sleep(1)
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"GET http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n\r\n").encode()
            )
            await w.drain()
            data = await asyncio.wait_for(r.read(4096), timeout=5)
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        # The proxy rewrites the response so the client sees Connection: close.
        assert b"Connection: close" in data
        # Origin-nominated and standard hop-by-hop response headers are dropped.
        assert b"keep-alive" not in data.lower()
        assert b"x-custom" not in data.lower()
        # Body framing is preserved so the client can still read the body.
        assert b"Content-Length: 2" in data
        assert data.endswith(b"hi")

    async def test_direct_path_head_completes_without_body(self, monkeypatch):
        """A HEAD response has no body.  ``_handle`` decodes ``method`` to a
        ``str`` and uppercases it; the relay must compare with ``"HEAD"`` (not
        ``b"HEAD"``) or it falls through to the body-reading path.  With a
        keep-alive origin that sends ``Content-Length`` but no body, the body
        branch would block on ``u_reader.read(N)`` waiting for bytes the origin
        never sends; the HEAD path instead forwards the headers and closes both
        sockets without waiting for a body."""
        async def handle(reader, writer):
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            # HEAD response: headers only, no body, keep the socket open.
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 5\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            await writer.drain()
            # Hold the socket open well past the test's read timeout so the
            # buggy body-reading path (which waits for the 5 advertised body
            # bytes that never arrive) cannot accidentally complete via EOF.
            await asyncio.sleep(10)
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"HEAD http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n\r\n").encode()
            )
            await w.drain()
            # Must complete promptly without waiting for the 5-byte body the
            # origin never sends; the buggy body-reading path would block here
            # for the origin's 10s hold (or its EOF) and trip the 3s timeout.
            data = await asyncio.wait_for(r.read(4096), timeout=3)
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        assert b"HTTP/1.1 200 OK" in data
        assert b"Content-Length: 5" in data
        assert b"Connection: close" in data
        # No body bytes for a HEAD response.
        assert not data.endswith(b"hello")
        assert b"\r\n\r\n" in data

    async def test_direct_path_forwards_interim_1xx_then_final(self, monkeypatch):
        """An origin may emit an interim ``100 Continue`` (or ``103 Early
        Hints``) response before the final response on the same socket.  The
        relay must forward each interim block and keep reading until the final
        response, instead of treating the first ``1xx`` as the complete
        response and closing both sockets (which would drop the final
        response on the floor)."""
        async def handle(reader, writer):
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            # Interim 100 Continue, then the final 200 OK with a body.
            writer.write(
                b"HTTP/1.1 100 Continue\r\n\r\n"
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 5\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\nhello"
            )
            await writer.drain()
            await asyncio.sleep(1)
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"GET http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n\r\n").encode()
            )
            await w.drain()
            # Read until EOF: the proxy forwards the interim 100 Continue and
            # the final 200 OK as separate writes, then closes the client
            # socket in _relay_one's finally.  A single bounded read() may
            # return after just the interim block (timing-dependent), so drain
            # to EOF to observe both responses.
            data = b""
            while True:
                chunk = await asyncio.wait_for(r.read(8192), timeout=5)
                if not chunk:
                    break
                data += chunk
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        # Both the interim 100 Continue and the final 200 OK reach the client.
        assert b"HTTP/1.1 100 Continue" in data
        assert b"HTTP/1.1 200 OK" in data
        # The final body is forwarded after the interim response.
        assert data.endswith(b"hello")
        # The relay rewrites the final response's Connection to close.
        assert b"Connection: close" in data

    async def test_direct_path_expect_100_continue_does_not_deadlock(self, monkeypatch):
        """A client that sends ``Expect: 100-continue`` waits for an interim
        ``100 Continue`` before sending the body.  ``_relay_one`` reads the
        full request body before reading the upstream response, so without
        handling the interim handshake the client would block forever waiting
        for ``100 Continue`` while the proxy blocks in ``_forward_http_body``
        waiting for body bytes the client never sends.  The proxy strips
        ``Expect`` from the upstream request (the origin does not need to
        send its own ``100 Continue``) and sends the interim response to the
        client itself, after which the body flows and the final response is
        relayed normally."""
        received = {}

        async def handle(reader, writer):
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                req += chunk
            head, _, rest = req.partition(b"\r\n\r\n")
            received["head"] = head
            # The proxy must strip Expect: the origin should not see it.
            assert b"expect:" not in head.lower()
            # Read the body the client sends after the proxy's 100 Continue.
            # Content-Length is 5, so read until we have all of it.
            body = rest
            while len(body) < 5:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                body += chunk
            received["body"] = body
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 7\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\ngot it!"
            )
            await writer.drain()
            await asyncio.sleep(1)
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"POST http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n"
                 "Expect: 100-continue\r\n"
                 "Content-Length: 5\r\n\r\n").encode()
            )
            await w.drain()
            # The proxy sends 100 Continue so the client can send the body.
            # Read the interim response first; it must arrive before the body
            # is forwarded, so a bounded read can return just it.
            interim = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=5)
            assert interim == b"HTTP/1.1 100 Continue\r\n\r\n"
            # Now the client sends the body.
            w.write(b"hello")
            await w.drain()
            # Drain to EOF to collect the final response.
            data = b""
            while True:
                chunk = await asyncio.wait_for(r.read(8192), timeout=5)
                if not chunk:
                    break
                data += chunk
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        # The origin received the full body the client sent after 100 Continue.
        assert received.get("body") == b"hello"
        # The final 200 OK reached the client after the interim 100 Continue.
        assert b"HTTP/1.1 200 OK" in data
        assert data.endswith(b"got it!")
        assert b"Connection: close" in data

    async def test_direct_path_unsupported_expect_rejected_not_forwarded(self, monkeypatch):
        """A client that sends an ``Expect`` value other than ``100-continue``
        with a body must not hang the relay.  ``_relay_one`` reads the full
        request body before the upstream response, so forwarding the request as
        written would block the proxy on body bytes the client (correctly)
        never sends while it waits for a ``417``/``400`` it expects first.
        The proxy rejects the request locally with ``417 Expectation Failed``
        before opening the upstream connection, and never reads the body."""
        connected = {"value": False}

        async def handle(reader, writer):
            connected["value"] = True
            writer.close()

        origin = await asyncio.start_server(handle, "127.0.0.1", 0)
        origin_port = origin.sockets[0].getsockname()[1]

        def fake_pin(url):
            return PinnedTarget("http", "plain.example", origin_port, "127.0.0.1")
        monkeypatch.setattr(egress_proxy, "resolve_and_pin", fake_pin)

        proxy = PinningProxy()
        await proxy.start()
        try:
            r, w = await asyncio.open_connection(proxy.bound_host, proxy.bound_port)
            w.write(
                (f"POST http://plain.example:{origin_port}/ HTTP/1.1\r\n"
                 f"Host: plain.example:{origin_port}\r\n"
                 "Expect: x-token-continue\r\n"
                 "Content-Length: 5\r\n\r\n").encode()
            )
            await w.drain()
            # The proxy must reject before forwarding; the origin must not be
            # contacted, and the client must not have to send the body.
            data = await asyncio.wait_for(r.read(4096), timeout=3)
            w.close()
        finally:
            await proxy.stop()
            origin.close()
            await origin.wait_closed()
        assert not connected["value"], "origin was contacted for an unsupported Expect"
        assert b"417 Expectation Failed" in data
        assert b"Expectation Failed" in data


def test_upstream_proxy_env_parsing(monkeypatch):
    assert egress_proxy.upstream_proxy() is None
    monkeypatch.setenv("HTTP_PROXY", "http://192.168.180.254:56560")
    assert egress_proxy.upstream_proxy() == ("192.168.180.254", 56560, None)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:p%40ss@10.0.0.1:8080")
    host, port, auth = egress_proxy.upstream_proxy()
    assert (host, port) == ("10.0.0.1", 8080)
    import base64
    assert base64.b64decode(auth.split(b" ")[-1].strip()) == b"user:p@ss"
    # scheme-aware selection: http targets prefer HTTP_PROXY
    assert egress_proxy.upstream_proxy("http") == ("192.168.180.254", 56560, None)
    monkeypatch.setenv("CRAWL4AI_UPSTREAM_PROXY", "proxy.corp:3128")
    assert egress_proxy.upstream_proxy() == ("proxy.corp", 3128, None)
    # whitespace-only env var means unset, not a proxy named "   "
    monkeypatch.setenv("CRAWL4AI_UPSTREAM_PROXY", "   ")
    monkeypatch.delenv("HTTP_PROXY")
    monkeypatch.delenv("HTTPS_PROXY")
    assert egress_proxy.upstream_proxy() is None
    # a junk/unsupported candidate falls through to a valid fallback
    monkeypatch.setenv("HTTP_PROXY", "http://good:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://")
    assert egress_proxy.upstream_proxy() == ("good", 3128, None)
    monkeypatch.setenv("HTTPS_PROXY", "https://tls-proxy.corp")  # unsupported scheme
    assert egress_proxy.upstream_proxy() == ("good", 3128, None)
    monkeypatch.delenv("CRAWL4AI_UPSTREAM_PROXY")
    monkeypatch.delenv("HTTP_PROXY")
    assert egress_proxy.upstream_proxy() is None  # https:// alone -> refused, not mis-dialed
    # non-latin-1 credentials must not raise (encoded as UTF-8)
    monkeypatch.setenv("HTTPS_PROXY", "http://u:%E5%AF%86%E7%A0%81@10.0.0.1:8080")
    assert egress_proxy.upstream_proxy()[2] is not None
    # NO_PROXY routing: suffix and CIDR entries force a direct dial
    pin = PinnedTarget("https", "site.corp.example", 443, "203.0.113.7")
    assert egress_proxy._use_upstream(pin) is not None
    monkeypatch.setenv("NO_PROXY", ".corp.example")
    assert egress_proxy._use_upstream(pin) is None
    monkeypatch.setenv("NO_PROXY", "203.0.113.0/24")
    assert egress_proxy._use_upstream(pin) is None
    monkeypatch.setenv("NO_PROXY", "site.corp.example:443")  # host:port form
    assert egress_proxy._use_upstream(pin) is None


def test_no_proxy_ip_port_forms_bypass_with_port_scoping(monkeypatch):
    """Regression for be18eb4: IP/CIDR/[IPv6] NO_PROXY entries with a trailing
    ``:port`` were silently ignored because the port-stripping logic that the
    same commit added was wired only into the hostname-suffix branch, not the
    IP/CIDR branch directly above it. ``IP:port`` / ``CIDR:port`` /
    ``[IPv6]:port`` entries must bypass the upstream proxy (and must honor the
    port scope, like ``host:port`` already did).
    """
    monkeypatch.setenv("HTTP_PROXY", "http://192.168.180.254:56560")
    proxy = ("192.168.180.254", 56560, None)
    pin4 = PinnedTarget("https", "site.corp.example", 443, "203.0.113.7")
    pin6 = PinnedTarget("https", "site.corp.example", 443, "2606:4700::1")
    pin6_loop = PinnedTarget("https", "localhost", 443, "::1")
    pin4_other = PinnedTarget("https", "site.example", 443, "198.51.100.5")

    def expect(entry, pin, want_bypass):
        monkeypatch.setenv("NO_PROXY", entry)
        got = egress_proxy._use_upstream(pin)
        return (got is None) == want_bypass, got

    # Bugs being fixed: port-scoped IP/CIDR/[IPv6] now bypass.
    ok, got = expect("203.0.113.7:443", pin4, True)
    assert ok, f"203.0.113.7:443 expected bypass, got {got}"
    ok, got = expect("203.0.113.0/24:443", pin4, True)
    assert ok, f"203.0.113.0/24:443 expected bypass, got {got}"
    ok, got = expect("[2606:4700::1]:443", pin6, True)
    assert ok, f"[2606:4700::1]:443 expected bypass, got {got}"
    ok, got = expect("[2606:4700::/32]:443", pin6, True)
    assert ok, f"[2606:4700::/32]:443 expected bypass, got {got}"

    # Port scope is honored: mismatched port does NOT bypass (uses upstream).
    ok, got = expect("203.0.113.7:444", pin4, False)
    assert ok, f"203.0.113.7:444 expected upstream, got {got}"
    ok, got = expect("203.0.113.0/24:444", pin4, False)
    assert ok, f"203.0.113.0/24:444 expected upstream, got {got}"
    ok, got = expect("[2606:4700::1]:444", pin6, False)
    assert ok, f"[2606:4700::1]:444 expected upstream, got {got}"
    ok, got = expect("[2606:4700::/32]:444", pin6, False)
    assert ok, f"[2606:4700::/32]:444 expected upstream, got {got}"

    # Controls preserved: bare IP / CIDR / IPv6 still bypass all ports.
    assert expect("203.0.113.7", pin4, True)[0]
    assert expect("203.0.113.0/24", pin4, True)[0]
    assert expect("2606:4700::1", pin6, True)[0]
    assert expect("2606:4700::/32", pin6, True)[0]
    assert expect("::1", pin6_loop, True)[0]

    # Bare IPv6 is not split on a trailing group as a faux port (be18eb4 would
    # have regressed this if ``rpartition(':')`` had been moved above the IP
    # branch naively).
    assert expect("2606:4700::1", pin6, True)[0]

    # IDNA-style / suffix and host:port still behave (existing behavior).
    assert expect(".corp.example", pin4, True)[0]
    assert expect("site.corp.example:443", pin4, True)[0]
    assert expect("site.corp.example:444", pin4, False)[0]

    # Bare-bracket IPv6 without :port remains a no-match (existing behavior).
    assert expect("[2606:4700::1]", pin6, False)[0]

    # A bare-IP / CIDR entry that parses but does NOT contain the pin IP must
    # NOT fall through to a hostname-suffix match (preserves the original
    # try/continue semantics of the IP branch).
    assert expect("10.0.0.0/8", pin4_other, False)[0]
    assert expect("1.2.3.4", pin4_other, False)[0]

    # Wildcard still bypasses everything.
    assert expect("*", pin4, True)[0]


class TestEnforceEgressWiring:
    def test_enforce_egress_sets_proxy(self, monkeypatch):
        import egress_broker
        from crawl4ai import BrowserConfig
        monkeypatch.setattr(egress_broker, "_EGRESS_PROXY_URL", "http://127.0.0.1:9999")
        b = BrowserConfig()
        egress_broker.enforce_egress(b)
        assert b.proxy_config is not None
        assert b.proxy_config.server == "http://127.0.0.1:9999"
