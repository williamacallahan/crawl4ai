from unittest.mock import AsyncMock

import api
import crawl_job_queue as crawl_job_queue_module
import crawler_pool
import pytest
from auth import create_access_token
from test_crawl_job_queue import FakeRedis, queue_config


@pytest.mark.parametrize("error_message,expected_detail", [
    ("Wait condition failed: selector not found", "Crawl request failed: Wait condition failed: selector not found"),
    ("Permission denied: /app/private/cache.db", "Crawl request failed: Crawl failed"),
])
def test_all_failed_crawl_returns_sanitized_bad_gateway(
    stock_client, monkeypatch, error_message, expected_detail
):
    class FailedCrawler:
        async def arun(self, *args, **kwargs):
            return [{
                "url": "https://example.com",
                "success": False,
                "error_message": error_message,
            }]

    monkeypatch.setattr(api, "validate_url_destination", lambda url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=FailedCrawler()))
    monkeypatch.setattr(crawler_pool, "release_crawler", AsyncMock())

    token = create_access_token({"sub": "test@example.com"})
    response = stock_client.post(
        "/crawl",
        json={"urls": ["https://example.com"]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    if "/app/" in error_message:
        assert detail.startswith(expected_detail + " (correlation_id=")
        assert "/app/private/cache.db" not in response.text
    else:
        assert detail == expected_detail


def _library_failure(error: BaseException) -> str:
    """The error_message arun records for an exception escaping a crawl."""
    import sys

    from crawl4ai.utils import unexpected_error_message

    try:
        raise error
    except type(error):
        return unexpected_error_message("_crawl_web", sys.exc_info())


@pytest.mark.parametrize("error,public,level", [
    (TimeoutError("Page timeout after 60000ms at async_crawler_strategy.py:1270 "
                  "(html = await self._settled_content(page))"),
     "Crawl failed: Page timeout after 60000ms", "WARNING"),
    (RuntimeError("Target closed: browser drained by the resource watchdog"),
     "Crawl failed: Target closed: browser drained by the resource watchdog", "WARNING"),
    (RuntimeError("Target page crashed"), "Crawl failed", "ERROR"),
])
def test_withheld_failures_report_one_line_with_url_and_target_level(error, public, level, caplog):
    from utils import public_crawl_error

    raw = _library_failure(error)
    assert "\n" not in raw
    with caplog.at_level("WARNING", logger="utils"):
        detail = public_crawl_error(raw, "https://target.example/page")
    assert detail.startswith(public + " (correlation_id=")
    [record] = [r for r in caplog.records if r.name == "utils"]
    assert record.levelname == level
    assert "url=https://target.example/page" in record.getMessage()
    assert "\n" not in record.getMessage()


@pytest.mark.parametrize("raw,public", [
    ("All proxies failed: Page timeout after 1ms at x.py:1 (y)",
     "Crawl failed: Page timeout after 1ms"),
    ("All proxies failed: Target closed: browser drained by the resource watchdog at x.py:1 (y)",
     "Crawl failed: Target closed: browser drained by the resource watchdog"),
])
def test_retried_target_failures_keep_their_reason_and_target_level(raw, public, caplog):
    # max_retries >= 1 reports a crawl's failure as "All proxies failed: <reason>".
    from utils import public_crawl_error

    with caplog.at_level("WARNING", logger="utils"):
        detail = public_crawl_error(raw, "https://target.example/page")
    assert detail.startswith(public + " (correlation_id=")
    [record] = [r for r in caplog.records if r.name == "utils"]
    assert record.levelname == "WARNING"


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/crawl", {"urls": ["https://unavailable.example/"]}),
    ("POST", "/crawl/stream", {"urls": ["https://unavailable.example/"]}),
    ("POST", "/crawl", {"urls": ["https://unavailable.example/"], "crawler_config": {"stream": True}}),
    ("POST", "/md", {"url": "https://unavailable.example/", "f": "raw"}),
    ("POST", "/html", {"url": "https://unavailable.example/"}),
    ("POST", "/screenshot", {"url": "https://unavailable.example/"}),
    ("POST", "/pdf", {"url": "https://unavailable.example/"}),
    ("POST", "/execute_js", {"url": "https://unavailable.example/", "scripts": ["1"]}),
    ("GET", "/llm/https://unavailable.example/?q=explain", None),
    ("POST", "/llm/job", {"url": "https://unavailable.example/", "q": "extract"}),
    ("POST", "/llm/job", {"url": "https://good.example/", "q": "extract", "webhook_config": {"webhook_url": "https://unavailable.example/cb"}}),
])
def test_shared_routes_return_opaque_temporary_unavailable(
    stock_client, server_module, monkeypatch, offline_dns, method, path, body
):
    import socket
    import egress_broker
    import utils
    import monitor

    stats = monitor.MonitorStats(None)
    monkeypatch.setattr(monitor, "monitor_stats", stats)
    monkeypatch.setattr(egress_broker, "ALLOW_INTERNAL", False)
    monkeypatch.setattr(utils, "ALLOW_INTERNAL_URLS", False)
    monkeypatch.setattr(server_module, "EXECUTE_JS_ENABLED", True)

    def resolve(host, port, *args, **kwargs):
        if host == "unavailable.example":
            raise socket.gaierror(socket.EAI_AGAIN, "private resolver information")
        return offline_dns.getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    headers = {"Authorization": f"Bearer {create_access_token({'sub': 'temporary-route@example.com'})}"}
    response = stock_client.request(method, path, json=body, headers=headers)
    assert response.status_code == 503
    assert response.json() == {"detail": "Temporary URL resolution failure"}
    assert response.headers["Retry-After"] == "30"
    if path in ("/crawl", "/crawl/stream"):
        [completed] = stats.completed_requests
        assert completed["status_code"] == 503
        assert completed["error"] == "Temporary URL resolution failure"
        assert not stats.active_requests


@pytest.mark.parametrize("failure_site", ["first_seed", "later_seed", "webhook"])
def test_temporary_dns_refuses_before_enqueue_then_accepts(
    stock_client, monkeypatch, offline_dns, failure_site
):
    import socket
    import egress_broker
    import job
    import utils

    monkeypatch.setattr(egress_broker, "ALLOW_INTERNAL", False)
    monkeypatch.setattr(utils, "ALLOW_INTERNAL_URLS", False)
    monkeypatch.setattr(crawl_job_queue_module, "validate_url_destination", utils.validate_url_destination)
    redis = FakeRedis()
    monkeypatch.setattr(job, "_redis", redis)
    monkeypatch.setattr(job, "_config", queue_config())
    body = {"urls": ["https://first.example/", "https://later.example/"]}
    failed_host = {"first_seed": "first.example", "later_seed": "later.example", "webhook": "hook.example"}[failure_site]
    if failure_site == "webhook":
        body["webhook_config"] = {"webhook_url": "https://hook.example/cb"}
    failing = True

    def resolve(host, port, *args, **kwargs):
        if failing and host == failed_host:
            raise socket.gaierror(socket.EAI_AGAIN, "private resolver information")
        return offline_dns.getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    headers = {"Authorization": f"Bearer {create_access_token({'sub': 'dns-submit@example.com'})}"}
    refused = stock_client.post("/crawl/job", json=body, headers=headers)
    assert refused.status_code == 503
    assert refused.json() == {"detail": "Crawl job not queued: temporary URL resolution failure"}
    assert refused.headers["Retry-After"] == "30"
    assert redis.eval_scripts == []
    assert not redis.hashes and not redis.streams

    failing = False
    accepted = stock_client.post("/crawl/job", json=body, headers=headers)
    assert accepted.status_code == 202
    task_id = accepted.json()["task_id"]
    assert redis.hashes[f"task:{task_id}"]["status"] == "processing"
    assert len(redis.eval_scripts) == 1
    assert sum(map(len, redis.streams.values())) == 1


@pytest.mark.parametrize("failure_site", ["seed", "webhook"])
@pytest.mark.parametrize("failure", ["missing_name", "other_dns", "private", "mixed"])
def test_permanent_or_unsafe_dns_stays_rejected_before_enqueue(
    stock_client, monkeypatch, offline_dns, failure_site, failure
):
    import socket
    import egress_broker
    import job
    import utils

    monkeypatch.setattr(egress_broker, "ALLOW_INTERNAL", False)
    monkeypatch.setattr(utils, "ALLOW_INTERNAL_URLS", False)
    monkeypatch.setattr(crawl_job_queue_module, "validate_url_destination", utils.validate_url_destination)
    redis = FakeRedis()
    monkeypatch.setattr(job, "_redis", redis)
    monkeypatch.setattr(job, "_config", queue_config())
    body = {"urls": ["https://unsafe.example/"]}
    if failure_site == "webhook":
        body["urls"] = ["https://good.example/"]
        body["webhook_config"] = {"webhook_url": "https://unsafe.example/cb"}

    def resolve(host, port, *args, **kwargs):
        if host != "unsafe.example":
            return offline_dns.getaddrinfo(host, port, *args, **kwargs)
        if failure in ("missing_name", "other_dns"):
            number = socket.EAI_NONAME if failure == "missing_name" else socket.EAI_FAIL
            raise socket.gaierror(number, "private resolver information")
        ips = ["127.0.0.1"] if failure == "private" else ["93.184.216.34", "127.0.0.1"]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    headers = {"Authorization": f"Bearer {create_access_token({'sub': 'dns-reject@example.com'})}"}
    response = stock_client.post("/crawl/job", json=body, headers=headers)
    assert response.status_code == 400
    assert "not queued" not in response.text
    assert "unsafe.example" not in response.text and "127.0.0.1" not in response.text
    assert "private resolver" not in response.text
    assert redis.eval_scripts == []
    assert not redis.hashes and not redis.streams


@pytest.mark.parametrize("after_enqueue", [False, True])
@pytest.mark.parametrize("failure", ["redis", "unexpected"])
def test_enqueue_failures_never_claim_validation_prevented_enqueue(
    stock_client, monkeypatch, offline_dns, failure, after_enqueue
):
    import job
    from redis.exceptions import ConnectionError as RedisConnectionError

    redis = FakeRedis()
    original_eval = redis.eval

    async def fail_eval(*args, **kwargs):
        if after_enqueue:
            await original_eval(*args, **kwargs)
        if failure == "redis":
            raise RedisConnectionError("state store unavailable")
        raise RuntimeError("private post-enqueue information")

    monkeypatch.setattr(redis, "eval", fail_eval)
    monkeypatch.setattr(job, "_redis", redis)
    monkeypatch.setattr(job, "_config", queue_config())
    headers = {"Authorization": f"Bearer {create_access_token({'sub': 'enqueue-error@example.com'})}"}
    response = stock_client.post("/crawl/job", json={"urls": ["https://good.example/"]}, headers=headers)
    assert response.status_code == (500 if failure == "unexpected" else 503)
    assert "Crawl job not queued: temporary URL resolution failure" not in response.text
    assert "private post-enqueue" not in response.text
    assert sum(map(len, redis.streams.values())) == int(after_enqueue)
    if failure == "redis":
        assert response.json()["code"] == "redis_unavailable"
