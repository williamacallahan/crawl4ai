"""Process-local Web Unlocker fallback for server-validated crawl URLs."""

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnlockerSettings:
    """Owner of the fallback's defaults; config.yml `brightdata.unlocker` may override any field."""

    base_url: str = "https://api.brightdata.com"
    zone: str = "cli_unlocker"
    country: str = "us"
    connect_timeout_s: float = 5
    # Direct attempt plus this fallback must answer inside the back-end's 45 s /md read.
    total_timeout_s: float = 30
    max_concurrency: int = 4
    max_response_bytes: int = 8388608
    breaker_failures: int = 3
    breaker_open_s: float = 60

    def __post_init__(self):
        for name in ("max_concurrency", "max_response_bytes", "breaker_failures"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"brightdata.unlocker.{name} must be a positive integer")
        for name in ("connect_timeout_s", "total_timeout_s", "breaker_open_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"brightdata.unlocker.{name} must be positive seconds")
        for name in ("base_url", "zone", "country"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"brightdata.unlocker.{name} must be a nonempty string")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("brightdata.unlocker.base_url must be an HTTP(S) URL")


class UnlockerError(RuntimeError):
    """A concise fallback failure safe for the crawler's existing log line."""


class _ProviderFailure(UnlockerError):
    def __init__(self, reason, *, credential_refused=False):
        super().__init__(reason)
        self.credential_refused = credential_refused


class _TargetFailure(UnlockerError):
    pass


class _ResponseTooLarge(UnlockerError):
    pass


class BrightDataUnlocker:
    def __init__(self, api_key: str, settings: UnlockerSettings):
        self.settings = settings
        self._api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._failures = 0
        self._open_until = 0.0
        self._probe_in_flight = False
        self._generation = 0
        self._credential_refused = False

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _admit(self):
        # No awaits: admission and probe reservation are atomic on this loop.
        if self._open_until:
            if time.monotonic() < self._open_until or self._probe_in_flight:
                raise UnlockerError("Web Unlocker circuit is open")
            self._probe_in_flight = True
        return self._generation

    def _success(self, generation):
        if generation == self._generation:
            self._failures = 0
            self._open_until = 0.0
            self._credential_refused = False

    def _failure(self, generation, *, credential_refused=False):
        if generation != self._generation:
            return
        self._failures += 1
        self._credential_refused |= credential_refused
        if self._open_until or self._failures >= self.settings.breaker_failures:
            self._open_until = time.monotonic() + self.settings.breaker_open_s
            self._generation += 1
            self._probe_in_flight = False
            if self._credential_refused:
                logger.error("Web Unlocker credential refused; circuit opened")

    async def fetch(self, url: str) -> str:
        """Fetch once; callers have already passed the server URL/egress policy."""
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise UnlockerError("Web Unlocker requires an HTTP(S) target URL")
        # Skip an open circuit immediately, including while all permits are busy.
        if self._open_until and (
            time.monotonic() < self._open_until or self._probe_in_flight
        ):
            raise UnlockerError("Web Unlocker circuit is open")

        generation = None

        async def request():
            nonlocal generation
            async with self._semaphore:
                generation = self._admit()
                return await self._request(url)

        try:
            body = await asyncio.wait_for(request(), self.settings.total_timeout_s)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            self._failure(generation)
            raise UnlockerError("Web Unlocker request timed out") from None
        except httpx.TransportError:
            self._failure(generation)
            raise UnlockerError("Web Unlocker transport failed") from None
        except _ProviderFailure as exc:
            self._failure(generation, credential_refused=exc.credential_refused)
            raise UnlockerError(str(exc)) from None
        except (_TargetFailure, _ResponseTooLarge):
            # A relayed target status or a bounded body is a provider response,
            # not a provider outage. Both also release a half-open probe.
            self._success(generation)
            raise
        else:
            self._success(generation)
            return body
        finally:
            if generation == self._generation:
                self._probe_in_flight = False

    async def _request(self, url: str) -> str:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.settings.total_timeout_s,
                    connect=self.settings.connect_timeout_s,
                ),
                follow_redirects=False,
            )
        async with self._client.stream(
            "POST",
            self.settings.base_url.rstrip("/") + "/request",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "zone": self.settings.zone,
                "url": url,
                "format": "raw",
                "country": self.settings.country,
            },
        ) as response:
            # Bright Data reports a target's outcome (a wall it could not pass, a relayed target
            # status, a policy refusal, a page that would not load) in its error headers. Only a
            # bare 5xx or a bare 401/407 credential refusal says the provider itself is unhealthy.
            target_reason = response.headers.get("x-brd-error-code") or response.headers.get(
                "x-brd-error"
            )
            if target_reason:
                raise _TargetFailure(
                    f"Web Unlocker could not serve the target (HTTP {response.status_code}, {target_reason[:80]})"
                )
            credential_refused = response.status_code in (401, 407)
            if credential_refused or response.status_code >= 500:
                raise _ProviderFailure(
                    f"Web Unlocker provider refused request (HTTP {response.status_code})",
                    credential_refused=credential_refused,
                )
            if not 200 <= response.status_code < 300:
                raise _TargetFailure(
                    f"Web Unlocker target returned HTTP {response.status_code}"
                )
            body = bytearray()
            async for chunk in response.aiter_bytes(
                chunk_size=min(65536, self.settings.max_response_bytes + 1)
            ):
                if len(body) + len(chunk) > self.settings.max_response_bytes:
                    raise _ResponseTooLarge("Web Unlocker response exceeds max_response_bytes")
                body.extend(chunk)
            return body.decode(response.encoding or "utf-8", errors="replace")


_initialized = False
_unlocker: Optional[BrightDataUnlocker] = None


def get_unlocker_fallback(config):
    """Initialize once per process from the server's loaded config and env."""
    global _initialized, _unlocker
    if not _initialized:
        api_key = os.environ.get("BRIGHTDATA_API_KEY", "").strip()
        if api_key:
            settings = UnlockerSettings(**config.get("brightdata", {}).get("unlocker", {}))
            _unlocker = BrightDataUnlocker(api_key, settings)
        else:
            logger.info("Web Unlocker fallback disabled: BRIGHTDATA_API_KEY is not set")
        _initialized = True
    return _unlocker.fetch if _unlocker is not None else None


async def close_unlocker():
    if _unlocker is not None:
        await _unlocker.close()
