"""Multi-proxy anti-bot retry loop must attribute ``CrawlResult.error_message``
to the iteration that built the returned ``crawl_result``, not to a later
throwing proxy iteration.

``AsyncWebCrawler.arun`` runs an outer retry loop (``_max_attempts``) over an
inner proxy loop (``_proxy_list``).  In the inner loop, a successful-but-blocked
proxy response sets BOTH ``_blocked`` and ``_block_reason`` and builds the
``crawl_result`` that will be returned.  A later iteration that THROWS lands in
the ``except`` handler, which historically overwrote ``_block_reason`` with
``str(_crawl_err)`` WITHOUT replacing ``crawl_result`` or clearing ``_blocked``.
After the loops, the post-loop block stamped::

        crawl_result.error_message = f"Blocked by anti-bot protection: {_block_reason}"

This paired ``_blocked`` (from the iteration that built ``crawl_result``) with
``_block_reason`` (from a LATER, throwing iteration), producing a self-
contradictory ``error_message`` that misattributed a proxy/network failure as
the "reason" for an anti-bot block.  The dispatcher forwards
``CrawlResult.error_message`` into the per-URL monitor state that operators
read (``async_dispatcher.py``), so the misattribution was user-visible.

The fix captures the block reason from the iteration that produced
``crawl_result`` (``_block_reason_for_result``) and uses that captured value
when formatting ``error_message`` after the loop.

These tests lock in the regression and the fix's scope:
  * ``error_message`` reports the blocked iteration's reason, not a later
    throwing proxy's exception text (the reported bug);
  * the throwing proxy's failure is still emitted to the in-loop error stream
    (``ANTIBOT`` tag) so error monitoring is not misled;
  * ``_block_reason_for_result`` is re-captured on every iteration that
    rebuilds ``crawl_result`` (it is not frozen to the first blocked one);
  * the conflation is fixed across BOTH loop axes (inner proxy loop AND outer
    retry loop);
  * the ``else`` branch (``crawl_result is None`` because all proxies threw)
    is intentionally untouched and keeps using the last exception -- guarding
    the fix's scope.

All tests use a mocked ``crawler_strategy`` so they need no browser or
network.
"""

from unittest.mock import AsyncMock

import pytest

from crawl4ai.antibot_detector import is_blocked
from crawl4ai.async_configs import CrawlerRunConfig, ProxyConfig
from crawl4ai.async_crawler_strategy import TargetNavigationError
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import AsyncCrawlResponse, CrawlResult

URL = "https://example.com"
PROXY_A = ProxyConfig(server="http://proxy-a.example:8080")
PROXY_B = ProxyConfig(server="http://proxy-b.example:8080")

# A PerimeterX-style anti-bot challenge page -- 403 short HTML with the
# ``px.jsPerimeterX`` script marker.
PX_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<h1>Access denied</h1>"
    '<script src="/px.jsPerimeterX/x"></script>'
    '<div id="px-captcha"></div>'
    "</body></html>"
)
# A Cloudflare-style anti-bot challenge page -- also 403 short HTML, but with
# the ``__cf_chl_f_tk__`` form marker.  ``is_blocked`` classifies it with a
# DIFFERENT reason string so tests can tell the two blocked iterations apart.
CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "<h1>Just a moment...</h1>"
    "<p>Checking your browser before accessing the site.</p>"
    '<form id="challenge-form" action="/cdn-cgi/challenge-platform/">'
    '<input name="__cf_chl_f_tk__" value="token"/></form></body></html>'
)

PX_REASON = is_blocked(403, PX_HTML)[1]
CF_REASON = is_blocked(403, CF_HTML)[1]
# Sanity: the two blocked pages produce distinct reason strings so the
# multi-iteration tests can reliably tell them apart.
assert PX_REASON != CF_REASON, "test fixtures must classify with distinct reasons"


class _RecordingLogger:
    def __init__(self):
        self.verbose = True
        self.events = []

    def info(self, message, tag="INFO", **kwargs):
        self.events.append(("info", tag, message))

    def warning(self, message, tag="WARNING", **kwargs):
        self.events.append(("warning", tag, message))

    def error_status(self, url, error, tag="ERROR", **kwargs):
        self.events.append(("error", tag, error))

    def url_status(self, url, success, timing, tag="FETCH", **kwargs):
        self.events.append(("success" if success else "error", tag, url))


class _PerProxyStrategy:
    """``AsyncCrawlerStrategy`` stub whose response/exception depends on the
    currently-active proxy (set by ``arun``'s retry loop to one of the
    instances in ``_proxy_list``).

    ``behaviors`` maps a ``ProxyConfig`` instance (or ``None`` for the direct
    connection) to either an ``AsyncCrawlResponse`` (returned) or an
    ``Exception`` (raised).  Identity-keying matches the real loop, which
    reuses the exact ``ProxyConfig`` instances from the caller's list.
    """

    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.crawl_calls = 0

    async def crawl(self, url, config):
        self.crawl_calls += 1
        behavior = self.behaviors.get(config.proxy_config)
        if behavior is None:
            raise AssertionError(
                f"no behavior registered for proxy={config.proxy_config!r}"
            )
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    def update_user_agent(self, *args, **kwargs):
        pass


class _CallSequenceStrategy:
    """``AsyncCrawlerStrategy`` stub that serves a configured sequence of
    responses/exceptions in crawl-call order, regardless of which proxy is
    active.  Used to model multi-ATTEMPT scenarios where the SAME proxy is
    hit more than once (the outer retry loop re-tries the inner proxy loop)."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.crawl_calls = 0

    async def crawl(self, url, config):
        idx = min(self.crawl_calls, len(self.sequence) - 1)
        self.crawl_calls += 1
        behavior = self.sequence[idx]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    def update_user_agent(self, *args, **kwargs):
        pass


def _resp(html, status=403):
    return AsyncCrawlResponse(html=html, response_headers={}, status_code=status)


def _make_crawler(tmp_path, strategy):
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=strategy,
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True

    def _echo(url, html, **kwargs):
        return CrawlResult(url=url, html=html, success=True)

    crawler.aprocess_html = AsyncMock(side_effect=_echo)
    return crawler, logger


# ---------------------------------------------------------------------------
# The reported bug: a blocked proxy iteration followed by a throwing proxy
# iteration must not conflate their reasons.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_then_throw_reports_blocked_reason_not_exception(tmp_path):
    """Iter 1 (proxy A) returns a 403 anti-bot page; iter 2 (proxy B) raises a
    navigation ``RuntimeError`` (the real strategy's
    ``Failed on navigating ACS-GOTO:\\n...`` shape).  The returned
    ``crawl_result`` is iter 1's blocked page, so ``error_message`` must
    report iter 1's block reason and MUST NOT contain iter 2's proxy-exception
    text."""
    crawler, logger = _make_crawler(
        tmp_path,
        _PerProxyStrategy(
            {
                PROXY_A: _resp(PX_HTML, 403),
                PROXY_B: RuntimeError(
                    "Failed on navigating ACS-GOTO:\nnet::ERR_CONNECTION_REFUSED"
                ),
            }
        ),
    )
    config = CrawlerRunConfig(
        proxy_config=[PROXY_A, PROXY_B], cache_mode=CacheMode.BYPASS
    )
    result = await crawler.arun(URL, config=config)

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False
    assert result.html == PX_HTML
    # error_message reports iter 1's block reason...
    assert result.error_message == f"Blocked by anti-bot protection: {PX_REASON}"
    # ...and NOT iter 2's proxy-exception text.
    assert "ERR_CONNECTION_REFUSED" not in (result.error_message or "")
    assert "Failed on navigating" not in (result.error_message or "")


@pytest.mark.asyncio
async def test_blocked_then_throw_proxy_failure_still_logged_to_error_stream(
    tmp_path,
):
    """The throwing proxy's failure must still reach the in-loop error stream
    (``ANTIBOT`` tag) -- the fix only corrects the conflated ``error_message``
    on the returned ``CrawlResult``; it must NOT hide the proxy/network failure
    from error monitoring."""
    crawler, logger = _make_crawler(
        tmp_path,
        _PerProxyStrategy(
            {
                PROXY_A: _resp(PX_HTML, 403),
                PROXY_B: RuntimeError(
                    "Failed on navigating ACS-GOTO:\nnet::ERR_CONNECTION_REFUSED"
                ),
            }
        ),
    )
    config = CrawlerRunConfig(
        proxy_config=[PROXY_A, PROXY_B], cache_mode=CacheMode.BYPASS
    )
    await crawler.arun(URL, config=config)

    proxy_err_events = [
        e for e in logger.events if e[0] == "error" and "ERR_CONNECTION_REFUSED" in e[2]
    ]
    assert proxy_err_events, (
        "the in-loop except handler must still emit the throwing proxy's "
        "RuntimeError to the error stream (ANTIBOT tag) after the fix"
    )
    assert proxy_err_events[0][1] == "ANTIBOT"


# ---------------------------------------------------------------------------
# ``_block_reason_for_result`` must be re-captured on EVERY iteration that
# rebuilds ``crawl_result`` (it is not frozen to the first blocked one).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_blocked_uses_last_blocked_iteration_reason(tmp_path):
    """Iter 1 (proxy A) returns a PX 403 page; iter 2 (proxy B) returns a
    Cloudflare 403 page (distinct reason).  The returned ``crawl_result`` is
    iter 2's, so ``error_message`` must cite iter 2's (Cloudflare) reason --
    proving ``_block_reason_for_result`` is re-captured on EVERY iteration
    that rebuilds ``crawl_result``, not frozen to iter 1."""
    crawler, logger = _make_crawler(
        tmp_path,
        _PerProxyStrategy(
            {
                PROXY_A: _resp(PX_HTML, 403),
                PROXY_B: _resp(CF_HTML, 403),
            }
        ),
    )
    config = CrawlerRunConfig(
        proxy_config=[PROXY_A, PROXY_B], cache_mode=CacheMode.BYPASS
    )
    result = await crawler.arun(URL, config=config)

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False
    assert result.html == CF_HTML
    assert result.error_message == f"Blocked by anti-bot protection: {CF_REASON}"
    assert PX_REASON not in (result.error_message or "")


# ---------------------------------------------------------------------------
# Cross-axis conflation: the outer retry loop shares ``_block_reason`` with
# the inner proxy loop, so the same conflation can occur across RETRY
# attempts.  The fix covers this axis too.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_then_throw_across_retry_attempts(tmp_path):
    """``max_retries=1`` -> ``_max_attempts=2`` with a single-element proxy
    list ``[PROXY_A]``.  Attempt 1's crawl() returns the blocked page (builds
    ``crawl_result``, sets ``_block_reason_for_result``); attempt 2's crawl()
    to the SAME proxy raises a navigation ``RuntimeError`` (overwrites
    ``_block_reason`` but not the captured per-result reason).  Pre-fix,
    ``error_message`` conflated attempt 1's block with attempt 2's exception;
    post-fix it reports attempt 1's block reason."""
    crawler, logger = _make_crawler(
        tmp_path,
        _CallSequenceStrategy(
            [
                _resp(PX_HTML, 403),
                RuntimeError(
                    "Failed on navigating ACS-GOTO:\nnet::ERR_CONNECTION_REFUSED"
                ),
            ]
        ),
    )
    config = CrawlerRunConfig(
        proxy_config=[PROXY_A],
        max_retries=1,
        cache_mode=CacheMode.BYPASS,
    )
    result = await crawler.arun(URL, config=config)

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False
    assert result.html == PX_HTML
    assert result.error_message == f"Blocked by anti-bot protection: {PX_REASON}"
    assert "ERR_CONNECTION_REFUSED" not in (result.error_message or "")
    # The retry-attempt's proxy failure still reaches the in-loop error
    # stream -- the fix does not suppress cross-axis failures either.
    assert any(
        e[0] == "error" and "ERR_CONNECTION_REFUSED" in e[2] for e in logger.events
    )


# ---------------------------------------------------------------------------
# Scope guard: the ``else`` branch (``crawl_result is None`` because every
# proxy threw) was NOT buggy -- it builds ``failure_message`` from
# ``_block_reason``, which is consistently the LAST except.  The fix must not
# change that path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_throw_routes_to_else_branch_uses_last_exception(tmp_path):
    """Every proxy throws, so ``crawl_result`` stays ``None`` and the post-loop
    code takes the ``else`` branch that builds a minimal result.  That branch
    formats ``failure_message`` from ``_block_reason`` (the LAST except), not
    from ``_block_reason_for_result`` (which stays "" because no iteration
    built a ``crawl_result``).  Expected: ``"All proxies failed: <last
    exception>"`` -- NO anti-bot prefix.  This guards the fix's scope."""
    crawler, logger = _make_crawler(
        tmp_path,
        _PerProxyStrategy(
            {
                None: TargetNavigationError("Target navigation refused: DNS"),
                PROXY_B: RuntimeError(
                    "Failed on navigating ACS-GOTO:\nnet::ERR_CONNECTION_REFUSED"
                ),
            }
        ),
    )
    config = CrawlerRunConfig(
        proxy_config=[ProxyConfig.DIRECT, PROXY_B],
        cache_mode=CacheMode.BYPASS,
    )
    result = await crawler.arun(URL, config=config)

    assert crawler.crawler_strategy.crawl_calls == 2
    assert result.success is False
    assert result.html == ""
    # The else branch prefixes with "All proxies failed" and uses the LAST
    # except's exception text -- it must NOT carry the "Blocked by anti-bot
    # protection" prefix (no blocked response occurred).
    assert result.error_message == (
        "All proxies failed: Failed on navigating ACS-GOTO:\n"
        "net::ERR_CONNECTION_REFUSED"
    )
    assert "anti-bot protection" not in (result.error_message or "")
