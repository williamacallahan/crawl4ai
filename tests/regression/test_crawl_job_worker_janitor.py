"""Each crawl-job worker process must sweep its own browser pool.

`crawler_pool`'s HOT_POOL/COLD_POOL are module globals, and supervisord runs
`numprocs=2` of crawl_job_worker.py beside gunicorn. The worker used to import
only `close_all`, so nothing closed an idle browser for the life of the process:
its pool only grew, that memory counted against the same container limit
gunicorn's `get_crawler` checks, and none of it was visible to /monitor/*, which
reads the gunicorn process's pool.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "docker"))
)

import crawl_job_worker  # noqa: E402


@pytest.mark.asyncio
async def test_run_worker_sweeps_its_own_pool(monkeypatch):
    janitor_running = asyncio.Event()
    janitor_cancelled = asyncio.Event()

    async def fake_janitor():
        janitor_running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            janitor_cancelled.set()
            raise

    class _FakeRedis:
        async def aclose(self):
            pass

    class _FakeWorker:
        def __init__(self, *a, **kw):
            pass

        async def run(self):
            # The worker only returns once the pool janitor is up beside it.
            await asyncio.wait_for(janitor_running.wait(), timeout=5)

    monkeypatch.setattr(crawl_job_worker, "janitor", fake_janitor)
    monkeypatch.setattr(crawl_job_worker, "load_config", lambda: {})
    monkeypatch.setattr(crawl_job_worker, "setup_logging", lambda _c: None)
    monkeypatch.setattr(crawl_job_worker, "build_redis_url", lambda _c: "redis://localhost:6379/0")
    monkeypatch.setattr(crawl_job_worker.aioredis, "from_url", lambda *a, **kw: _FakeRedis())
    monkeypatch.setattr(crawl_job_worker, "CrawlJobQueue", lambda *a, **kw: object())
    monkeypatch.setattr(crawl_job_worker, "CrawlJobWorker", _FakeWorker)

    async def _noop_proxy():
        return None

    async def _noop_stop(_p):
        return None

    monkeypatch.setattr(crawl_job_worker, "start_pinning_proxy", _noop_proxy)
    monkeypatch.setattr(crawl_job_worker, "stop_pinning_proxy", _noop_stop)

    closed = asyncio.Event()

    async def fake_close_all():
        closed.set()

    monkeypatch.setattr(crawl_job_worker, "close_all", fake_close_all)

    await crawl_job_worker.run_worker()

    assert janitor_running.is_set(), "worker never started a pool janitor"
    assert janitor_cancelled.is_set(), "pool janitor outlived the worker"
    assert closed.is_set(), "worker did not close its pool on shutdown"
