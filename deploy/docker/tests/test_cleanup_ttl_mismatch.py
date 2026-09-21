"""
Regression coverage for the producer/consumer Redis task-TTL contract.

PR #1730 (commit 761664d) introduced an operator-facing ``redis.task_ttl_seconds``
knob (with a ``0 == disabled`` sentinel) and a ``REDIS_TASK_TTL`` env override,
and wired the *producer* (``hset_with_ttl`` on every task write) to honor it.
The *consumer* (the lazy delete-after-read gate inside ``handle_task_status``,
gated by ``should_cleanup_task``) was left on its hard-coded ``3600`` default
with no ``0 == disabled`` handling and no way to receive the operator config.

Effect: a completed/failed task whose configured ``task_ttl_seconds`` was
``86400`` (24h retention) -- or ``0`` (TTL explicitly disabled) -- was deleted
on the first status poll occurring more than one wall-clock hour after
creation, silently contradicting the operator's retention setting.

These tests exercise both halves of the contract -- the producer
``api.hset_with_ttl`` and the consumer ``api.handle_task_status`` -- against a
shared stateful async Redis stub, and assert producer/consumer agreement at
default, long, and disabled TTLs. They also cover the helper-level ``0``
sentinel directly, the ``keep`` opt-out, the ``config=None`` historical default,
the FAILED (not just COMPLETED) terminal state, and the ``handle_llm_request``
task-id wiring path.
"""

import os
import sys
from datetime import datetime, timedelta

# deploy/docker is not a package; tests import its modules by bare name, and
# conftest.py inserts the dir onto sys.path for the suite. Repeat the insert
# here so this file also runs standalone (mirrors test_crawl_job_contract.py).
DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import api  # noqa: E402
from utils import should_cleanup_task  # noqa: E402


class _FakeRedis:
    """Minimal async Redis stub recording hset/expire/delete on a task hash.

    Real redis-py returns bytes keys/values from ``hgetall`` and encodes
    str->bytes on ``hset``; ``decode_redis_hash`` expects that contract, so we
    preserve it. The stub also records every call so the producer
    (``hset``/``expire``) and consumer (``delete``) retention behaviors can be
    asserted against shared state.
    """

    def __init__(self, mapping=None):
        self._hash: dict[bytes, bytes] = {}
        if mapping:
            self._hash = {
                k.encode("utf-8"): str(v).encode("utf-8")
                for k, v in mapping.items()
            }
        self.hset_calls: list[tuple[str, dict]] = []
        self.expire_calls: list[tuple[str, int]] = []
        self.deleted_keys: list[str] = []

    async def hset(self, key, mapping=None):
        self.hset_calls.append((key, dict(mapping)))
        self._hash = {
            k.encode("utf-8"): str(v).encode("utf-8")
            for k, v in mapping.items()
        }
        return len(mapping)

    async def expire(self, key, ttl):
        self.expire_calls.append((key, int(ttl)))
        return True

    async def hgetall(self, key):
        return dict(self._hash)

    async def delete(self, key):
        self.deleted_keys.append(key)
        self._hash = {}
        return 1


def _config_with_ttl(ttl: int) -> dict:
    return {"redis": {"task_ttl_seconds": ttl}}


def _iso(age_hours: float) -> str:
    return (datetime.now() - timedelta(hours=age_hours)).isoformat()


def _completed_task(age_hours: float = 2.0) -> dict:
    return {
        "status": "completed",
        "created_at": _iso(age_hours),
        "url": "https://example.com",
        "result": "{}",
    }


def _failed_task(age_hours: float = 2.0) -> dict:
    return {
        "status": "failed",
        "created_at": _iso(age_hours),
        "url": "https://example.com",
        "error": "boom",
    }


# ────────────────────── end-to-end producer/consumer matrix ──────────────


class TestProducerConsumerTTLContract:
    """Both halves of the retention contract against shared Redis state.

    The producer (``hset_with_ttl``) records an ``expire(key, ttl)`` reflecting
    the operator config (and skips ``expire`` when ``ttl == 0``). The consumer
    (``handle_task_status``) must perform its lazy delete-after-read on the same
    schedule: delete only when the task age exceeds the configured TTL, and
    never when TTL is disabled.
    """

    def test_long_retention_ttl_asymmetry_end_to_end(self):
        """config=86400 (24h), task age=2h: producer keeps 24h, consumer keeps."""
        config = _config_with_ttl(86400)
        redis = _FakeRedis()
        key = "task:llm_abc"

        # Producer side: write the task with the configured 24h TTL.
        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(2.0), config))

        assert redis.expire_calls == [(key, 86400)], (
            "producer should record expire(key, 86400) for 24h retention"
        )

        # Consumer side: status-read a 2h-old completed task.
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.deleted_keys == [], (
            "consumer must NOT delete a task younger than the configured 24h TTL; "
            "before the fix the hard-coded 3600s gate silently deleted it"
        )

    def test_ttl_disabled_asymmetry_end_to_end(self):
        """config=0 (TTL disabled), task age=2h: producer skips expire, consumer keeps."""
        config = _config_with_ttl(0)
        redis = _FakeRedis()
        key = "task:llm_abc"

        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(2.0), config))

        assert redis.expire_calls == [], (
            "producer must skip expire() entirely when ttl == 0 (disabled sentinel)"
        )

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.deleted_keys == [], (
            "consumer must NOT delete when TTL is explicitly disabled (0); "
            "before the fix should_cleanup_task(t, 0) returned True"
        )

    def test_default_ttl_producer_and_consumer_agree_fresh_task(self):
        """config=3600 (default), fresh task: both agree to keep."""
        config = _config_with_ttl(3600)
        redis = _FakeRedis()
        key = "task:llm_abc"

        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(0.0), config))

        assert redis.expire_calls == [(key, 3600)]

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.deleted_keys == [], (
            "a fresh task well inside the 3600s default TTL must not be cleaned up"
        )

    def test_default_ttl_producer_and_consumer_agree_aged_task(self):
        """config=3600 (default), aged task (2h): both agree to drop."""
        config = _config_with_ttl(3600)
        redis = _FakeRedis()
        key = "task:llm_abc"

        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(2.0), config))

        assert redis.expire_calls == [(key, 3600)]

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.deleted_keys == [key], (
            "an aged task past the 3600s default TTL must be cleaned up"
        )


# ───────────────────────── helper-level sentinel ─────────────────────────


class TestShouldCleanupTaskHelper:
    def test_disabled_sentinel_returns_false(self):
        """ttl_seconds=0 must short-circuit to False (0 == disabled)."""
        assert should_cleanup_task(_iso(2.0), ttl_seconds=0) is False

    def test_default_ttl_aged_task_cleans_up(self):
        assert should_cleanup_task(_iso(2.0)) is True

    def test_default_ttl_fresh_task_kept(self):
        assert should_cleanup_task(_iso(0.0)) is False


# ──────────────────── consumer wiring + opt-out paths ────────────────────


class TestHandleTaskStatusWiring:
    def test_config_none_preserves_historical_3600_default(self):
        """Callers that omit config keep the 3600s historical behavior."""
        redis = _FakeRedis(_completed_task(2.0))
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job",
            )
        )
        assert resp.status_code == 200
        assert redis.deleted_keys == ["task:llm_abc"], (
            "config=None -> get_redis_task_ttl({}) returns 3600 -> 2h-aged task deleted"
        )

    def test_keep_true_short_circuits_even_when_aged(self):
        """keep=True must bypass the cleanup branch regardless of TTL/age."""
        redis = _FakeRedis(_completed_task(2.0))
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job",
                keep=True, config=_config_with_ttl(3600),
            )
        )
        assert resp.status_code == 200
        assert redis.deleted_keys == [], "keep=True must never delete"

    def test_failed_task_respects_configured_long_ttl(self):
        """FAILED (not just COMPLETED) terminal state honors the configured TTL."""
        redis = _FakeRedis(_failed_task(2.0))
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job",
                config=_config_with_ttl(86400),
            )
        )
        assert resp.status_code == 200
        assert redis.deleted_keys == [], (
            "FAILED task under 24h retention must not be deleted at 2h age"
        )


# ────────────────────── handle_llm_request wiring ────────────────────────


class TestHandleLlmRequestTaskIdPath:
    """The third status-read caller threads config into handle_task_status.

    handle_llm_request already accepts ``config`` and, when ``input_path`` is a
    task id, delegates to handle_task_status. This verifies the wiring fix so
    an operator's long-retention config reaches the cleanup gate through this
    path too (previously a wiring defect: config was in scope but not passed).
    """

    def test_task_id_path_honors_long_retention(self, monkeypatch):
        from fastapi import BackgroundTasks
        from types import SimpleNamespace

        # is_task_id() must recognize our synthetic id so the task-id branch runs.
        monkeypatch.setattr(api, "is_task_id", lambda s: True)

        long_config = _config_with_ttl(86400)
        redis = _FakeRedis(_completed_task(2.0))
        # get_base_url() reads request.url.scheme/.netloc.
        request = SimpleNamespace(
            url=SimpleNamespace(scheme="http", netloc="t")
        )

        async def call():
            return await api.handle_llm_request(
                redis,
                BackgroundTasks(),
                request,
                "llm_abc",
                config=long_config,
            )

        resp = asyncio_run(call())
        assert resp.status_code == 200
        assert redis.deleted_keys == [], (
            "handle_llm_request -> handle_task_status must forward config; "
            "at 24h retention a 2h-old task must survive"
        )

    def test_task_id_path_cleans_up_at_default_ttl(self, monkeypatch):
        from fastapi import BackgroundTasks
        from types import SimpleNamespace

        monkeypatch.setattr(api, "is_task_id", lambda s: True)

        redis = _FakeRedis(_completed_task(2.0))
        request = SimpleNamespace(
            url=SimpleNamespace(scheme="http", netloc="t")
        )

        async def call():
            return await api.handle_llm_request(
                redis,
                BackgroundTasks(),
                request,
                "llm_abc",
                config=_config_with_ttl(3600),
            )

        resp = asyncio_run(call())
        assert resp.status_code == 200
        assert redis.deleted_keys == ["task:llm_abc"]


# ───────────────────────────── helpers ───────────────────────────────────


def asyncio_run(coro):
    """Run a one-shot coroutine under a fresh loop (sync test wrappers)."""
    import asyncio as _asyncio

    return _asyncio.run(coro)


# Make the async-end-to-end tests also runnable under pytest-asyncio if a
# future contributor converts the file; the sync+asyncio.run style matches
# test_crawl_job_contract.py and avoids strict-mode asyncio marker churn.
