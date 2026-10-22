"""
Regression coverage for the producer/consumer Redis task-TTL contract.

PR #1730 (commit 761664d) introduced an operator-facing ``redis.task_ttl_seconds``
knob (with a ``0 == disabled`` sentinel) and a ``REDIS_TASK_TTL`` env override,
and wired the *producer* (``hset_with_ttl`` on every task write) to honor it.
The *consumer* (the lazy delete-after-read gate inside ``handle_task_status``,
gated by ``should_cleanup_task``) was left on its hard-coded ``3600`` default
with no ``0 == disabled`` handling and no way to receive the operator config.

A positive ``task_ttl_seconds`` is set by terminal writes. Status reads leave
that expiry unchanged, including after a resume probe or a lost response.
``0`` disables producer expiry; reads never arm an expiry themselves.

These tests exercise producer expiry and read-only retention against a shared
stateful async Redis stub, including the disabled sentinel, the compatible
``keep`` keyword, omitted config, FAILED state, and task-id delegation.
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
    """Minimal async Redis stub holding one task hash and its TTL.

    Real redis-py returns bytes keys/values from ``hgetall`` and encodes
    str->bytes on ``hset``; ``decode_redis_hash`` expects that contract, so we
    preserve it. The stub stores the key TTL and deletion so retention is
    asserted from Redis state.
    """

    def __init__(self, mapping=None):
        self._hash: dict[bytes, bytes] = {}
        if mapping:
            self._hash = {
                k.encode("utf-8"): str(v).encode("utf-8")
                for k, v in mapping.items()
            }
        self.hset_calls: list[tuple[str, dict]] = []
        self._ttl: dict[str, int] = {}
        self.deleted_keys: list[str] = []

    async def hset(self, key, mapping=None):
        self.hset_calls.append((key, dict(mapping)))
        self._hash = {
            k.encode("utf-8"): str(v).encode("utf-8")
            for k, v in mapping.items()
        }
        return len(mapping)

    async def expire(self, key, ttl, lt=False):
        # LT: a key with no TTL counts as infinite, as in Redis.
        current = self._ttl.get(key)
        if lt and current is not None and current <= int(ttl):
            return False
        self._ttl[key] = int(ttl)
        return True

    def ttl_seconds(self, key) -> int | None:
        return self._ttl.get(key)

    async def hgetall(self, key):
        return dict(self._hash)

    async def delete(self, key):
        self.deleted_keys.append(key)
        self._hash = {}
        self._ttl.pop(key, None)
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

    The producer (``hset_with_ttl``) records an ``expire(key, ttl)`` on a
    terminal write (and skips ``expire`` when ``ttl == 0``). That TTL is how
    long a finished hash lives. The consumer (``handle_task_status``) leaves
    that expiry unchanged, regardless of task age or whether TTL is disabled.
    """

    def test_long_retention_ttl_is_preserved_after_read(self):
        """config=86400: producer arms a 24h TTL; a fetch preserves it."""
        config = _config_with_ttl(86400)
        redis = _FakeRedis()
        key = "task:llm_abc"

        # Producer side: a terminal write arms the configured terminal TTL.
        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(2.0), config))

        assert redis.ttl_seconds(key) == 86400, (
            "a terminal write arms the configured terminal TTL"
        )

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.ttl_seconds(key) == 86400, (
            "a terminal status read preserves the producer TTL"
        )
        assert redis.deleted_keys == []

    def test_ttl_disabled_asymmetry_end_to_end(self):
        """config=0 (TTL disabled), task age=2h: producer skips expire, consumer keeps."""
        config = _config_with_ttl(0)
        redis = _FakeRedis()
        key = "task:llm_abc"

        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(2.0), config))

        assert redis.ttl_seconds(key) is None, (
            "ttl == 0 leaves the terminal hash with no expiry"
        )

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.ttl_seconds(key) is None, (
            "TTL disabled (0): the read arms no expiry"
        )
        assert redis.deleted_keys == []

    def test_default_ttl_producer_and_consumer_agree_fresh_task(self):
        """config=3600: producer arms the terminal TTL; the fetch preserves it."""
        config = _config_with_ttl(3600)
        redis = _FakeRedis()
        key = "task:llm_abc"

        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(0.0), config))

        assert redis.ttl_seconds(key) == 3600

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.ttl_seconds(key) == 3600
        assert redis.deleted_keys == []

    def test_default_ttl_producer_and_consumer_agree_aged_task(self):
        """An old creation time does not shorten retention set at completion."""
        config = _config_with_ttl(3600)
        redis = _FakeRedis()
        key = "task:llm_abc"

        asyncio_run(api.hset_with_ttl(redis, key, _completed_task(2.0), config))

        assert redis.ttl_seconds(key) == 3600

        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job", config=config,
            )
        )

        assert resp.status_code == 200
        assert redis.ttl_seconds(key) == 3600
        assert redis.deleted_keys == []


# ───────────────────────── helper-level sentinel ─────────────────────────


class TestShouldCleanupTaskHelper:
    def test_disabled_sentinel_returns_false(self):
        """ttl_seconds=0 must short-circuit to False (0 == disabled)."""
        assert should_cleanup_task(_iso(2.0), ttl_seconds=0) is False

    def test_default_ttl_aged_task_cleans_up(self):
        assert should_cleanup_task(_iso(2.0)) is True

    def test_default_ttl_fresh_task_kept(self):
        assert should_cleanup_task(_iso(0.0)) is False


# ──────────────────── consumer wiring + compatibility paths ────────────────────


class TestHandleTaskStatusWiring:
    def test_config_none_leaves_unexpired_hash_unchanged(self):
        """Omitting config does not arm an expiry on an existing hash."""
        redis = _FakeRedis(_completed_task(2.0))
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job",
            )
        )
        assert resp.status_code == 200
        assert redis.ttl_seconds("task:llm_abc") is None
        assert redis.deleted_keys == []

    def test_keep_true_remains_compatible_even_when_aged(self):
        """The compatible keep keyword leaves expiry unchanged regardless of age."""
        redis = _FakeRedis(_completed_task(2.0))
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job",
                keep=True, config=_config_with_ttl(3600),
            )
        )
        assert resp.status_code == 200
        assert redis.ttl_seconds("task:llm_abc") is None, "keep=True leaves the TTL alone"
        assert redis.deleted_keys == []

    def test_failed_terminal_read_does_not_arm_expiry(self):
        """A FAILED hash without expiry remains unchanged after a fetch."""
        redis = _FakeRedis(_failed_task(2.0))
        resp = asyncio_run(
            api.handle_task_status(
                redis, "llm_abc", "http://t/", collection="llm/job",
                config=_config_with_ttl(86400),
            )
        )
        assert resp.status_code == 200
        assert redis.ttl_seconds("task:llm_abc") is None
        assert redis.deleted_keys == []


# ────────────────────── handle_llm_request wiring ────────────────────────


class TestHandleLlmRequestTaskIdPath:
    """Task-id delegation preserves retention with disabled or positive TTL.

    These calls exercise the handle_llm_request status path, which must not
    arm an expiry on a pre-existing hash.
    """

    def test_task_id_path_preserves_disabled_ttl(self, monkeypatch):
        from fastapi import BackgroundTasks
        from types import SimpleNamespace

        # is_task_id() must recognize our synthetic id so the task-id branch runs.
        monkeypatch.setattr(api, "is_task_id", lambda s: True)

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
                config=_config_with_ttl(0),
            )

        resp = asyncio_run(call())
        assert resp.status_code == 200
        assert redis.ttl_seconds("task:llm_abc") is None, (
            "task-id delegation must leave expiry unchanged with TTL 0"
        )

    def test_task_id_path_preserves_existing_expiry_at_default_ttl(self, monkeypatch):
        from fastapi import BackgroundTasks
        from types import SimpleNamespace

        monkeypatch.setattr(api, "is_task_id", lambda s: True)

        redis = _FakeRedis(_completed_task(2.0))
        asyncio_run(redis.expire("task:llm_abc", 3600))
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
        assert redis.ttl_seconds("task:llm_abc") == 3600


# ───────────────────────────── helpers ───────────────────────────────────


def asyncio_run(coro):
    """Run a one-shot coroutine under a fresh loop (sync test wrappers)."""
    import asyncio as _asyncio

    return _asyncio.run(coro)


# Make the async-end-to-end tests also runnable under pytest-asyncio if a
# future contributor converts the file; the sync+asyncio.run style matches
# test_crawl_job_contract.py and avoids strict-mode asyncio marker churn.
