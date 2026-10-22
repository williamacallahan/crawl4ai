"""Archive setup honors the crawl deadline and validates before side effects."""
import asyncio

import api
import pytest
from fastapi import HTTPException
from test_crawl_hook_lifecycle import FakeCrawler, crawl_config, install_fakes, run
from test_archive_budget import app_client, redis_url, _bearer  # noqa: F401


@pytest.mark.parametrize("streaming", [False, True])
def test_materialization_is_cancelled_at_crawl_deadline(monkeypatch, streaming):
    pooled = FakeCrawler()
    released, dedicated = install_fakes(monkeypatch, pooled)
    events = []

    async def slow_materialize(_redis, _config, urls):
        events.append("started")
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            events.append("cancelled")
            raise
        events.append("completed")
        return urls

    monkeypatch.setattr(api, "prefetch_archive", slow_materialize)
    handler = api.handle_stream_crawl_request if streaming else api.handle_crawl_request
    with pytest.raises(HTTPException) as error:
        run(handler(
            ["https://archive.org/details/deadline"], {}, {},
            crawl_config(wall_clock_s=0.01),
        ))
    assert error.value.status_code == 504
    assert error.value.detail == "Crawl exceeded the time limit"
    assert events == ["started", "cancelled"]
    assert not pooled.run_started.is_set()
    assert not dedicated.started
    assert released == []


@pytest.mark.parametrize("endpoint", ["/html", "/md"])
def test_archive_prefetch_deadline_is_504(endpoint, app_client, monkeypatch, offline_dns):
    import archive_budget
    import server

    monkeypatch.setenv("CRAWL4AI_NODE", "deadline-" + endpoint[1:])
    monkeypatch.setitem(server.config["limits"], "wall_clock_s", 0.1)

    async def slow_origin(url, maximum):
        await asyncio.Event().wait()

    monkeypatch.setattr(archive_budget, "_default_opener", slow_origin)
    response = app_client.post(endpoint, json={"url": "https://archive.org/details/deadline"},
                               headers=_bearer())
    assert response.status_code == 504
    assert response.json()["detail"] == "Crawl exceeded the time limit"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("spec", [
    {"action": "run_python", "params": {}},
    {"action": "wait_for_timeout", "params": {"timeout_ms": 60001}},
])
def test_invalid_hooks_reject_before_materialization(monkeypatch, streaming, spec):
    from hook_registry import build_declarative_hooks
    pooled = FakeCrawler()
    released, dedicated = install_fakes(monkeypatch, pooled)
    monkeypatch.setattr(api, "build_declarative_hooks", build_declarative_hooks)
    materialized = []

    async def materialize(_redis, _config, urls):
        materialized.extend(urls)
        return urls

    monkeypatch.setattr(api, "prefetch_archive", materialize)
    handler = api.handle_stream_crawl_request if streaming else api.handle_crawl_request
    with pytest.raises(HTTPException) as error:
        run(handler(
            ["https://archive.org/details/invalid-hook"], {}, {}, crawl_config(),
            hooks_config={"hooks": [spec]},
        ))
    assert error.value.status_code == 400
    assert error.value.detail.startswith("Rejected request:")
    assert materialized == []
    assert not dedicated.started
    assert not pooled.run_started.is_set()
    assert released == []
