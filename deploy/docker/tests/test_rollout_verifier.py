from __future__ import annotations

import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import verify_rollout as rollout_verifier
from verify_rollout import (
    MAX_RESPONSE_BYTES,
    REQUIRED_STOP_GRACE_NS,
    _curl_json,
    _has_task_error,
    verify_rollout,
)


def _application(**overrides):
    application = {
        "replicas": 1,
        "appName": "crawl4ai",
        "labelsSwarm": rollout_verifier._observability_labels("target"),
        "env": (
            f"LLM_PROVIDER={rollout_verifier.LLM_PROVIDER}\n"
            f"LLM_BASE_URL={rollout_verifier.LLM_BASE_URL}\n"
            "LLM_API_KEY=nonempty"
        ),
        "mounts": [
            {
                "type": "volume",
                "volumeName": rollout_verifier.REDIS_VOLUME,
                "mountPath": rollout_verifier.REDIS_MOUNT_PATH,
            }
        ],
        "healthCheckSwarm": {"Test": rollout_verifier.REDIS_HEALTHCHECK},
        "placementSwarm": {
            "Constraints": [],
            "MaxReplicas": rollout_verifier.CRAWL_MAX_REPLICAS_PER_NODE,
        },
        "updateConfigSwarm": {"Order": "start-first", "Parallelism": 1, "MaxFailureRatio": 0},
        "rollbackConfigSwarm": {"Order": "start-first", "Parallelism": 1, "MaxFailureRatio": 0},
    }
    application.update(overrides)
    return application


def _task_config(container_id: str, labels: dict[str, str] | None = None) -> dict:
    return {
        "Status": {"ContainerStatus": {"ContainerID": container_id}},
        "Spec": {
            "ContainerSpec": {
                "Labels": rollout_verifier._observability_labels("target")
                if labels is None
                else labels
            }
        },
    }


def _running_task(task_id: str, node: str, seconds: int = 1) -> dict:
    return {
        "containerId": task_id,
        "state": "running",
        "currentState": f"Running {seconds}s",
        "node": node,
        "error": "",
    }


def _health(instance: str, revision: str = "target") -> dict:
    return {
        "instance": instance,
        "revision": revision,
        "status": "ok",
        "components": {"api": "ready", "redis": "ready"},
    }


def _fake_read(handlers):
    def read_json(url, _api_key):
        operation = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
        if (
            operation == "docker.getServiceContainersByAppName"
            and "appName=crawl4ai-redis" in url
        ):
            return [_running_task("redis-task", "haiku-18")]
        if operation not in handlers:
            raise AssertionError(f"unexpected operation: {operation}")
        handler = handlers[operation]
        result = handler(url) if callable(handler) else handler
        if operation == "application.one" and isinstance(result, dict):
            supplied = result
            result = _application(**result)
            if "applicationId=redis" in url:
                result["appName"] = "crawl4ai-redis"
                if "placementSwarm" not in supplied:
                    result["placementSwarm"] = {
                        "Constraints": [rollout_verifier.REDIS_NODE_CONSTRAINT],
                        "MaxReplicas": 1,
                    }
            result.setdefault("stopGracePeriodSwarm", REQUIRED_STOP_GRACE_NS)
        return result

    return read_json


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("", False),
        ("Error:", False),
        ("Error: task failed", True),
        ("task failed", True),
    ],
)
def test_task_error_ignores_only_empty_dokploy_error_boilerplate(error, expected):
    assert _has_task_error({"error": error}) is expected


def test_curl_json_rejects_streamed_responses_over_64_kib():
    class OversizedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * (MAX_RESPONSE_BYTES + 1))

        def log_message(self, format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OversizedHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        with pytest.raises(ValueError, match="exceeds 64 KiB"):
            _curl_json(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"labelsSwarm": {}}, "observability labels"),
        (
            {
                "env": (
                    "LLM_PROVIDER=openai/gpt-4o-mini\n"
                    "LLM_BASE_URL=https://api.llm-gateway.iocloudhost.net/v1\n"
                    "LLM_API_KEY=nonempty"
                )
            },
            "LLM_PROVIDER",
        ),
        (
            {
                "env": (
                    "LLM_PROVIDER=openai/qwen3.8-27b\n"
                    "LLM_BASE_URL=https://wrong.example/v1\n"
                    "LLM_API_KEY=nonempty"
                )
            },
            "LLM_BASE_URL",
        ),
        (
            {
                "env": (
                    "LLM_PROVIDER=openai/qwen3.8-27b\n"
                    "LLM_BASE_URL=https://api.llm-gateway.iocloudhost.net/v1\n"
                    "LLM_API_KEY="
                )
            },
            "LLM_API_KEY must be nonempty",
        ),
        ({"placementSwarm": {"MaxReplicas": 1}}, "temporary overlap task"),
        (
            {"updateConfigSwarm": {"Order": "stop-first", "Parallelism": 1, "MaxFailureRatio": 0}},
            "updateConfigSwarm must use start-first",
        ),
        (
            {"rollbackConfigSwarm": {"Order": "stop-first", "Parallelism": 1, "MaxFailureRatio": 0}},
            "rollbackConfigSwarm must use start-first",
        ),
    ],
)
def test_verifier_rejects_invalid_release_observability_configuration(overrides, message):
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": overrides,
        }
    )

    with pytest.raises(ValueError, match=message):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mounts": []}, "crawl4ai-redis-data"),
        ({"healthCheckSwarm": {"Test": ["CMD-SHELL", "redis-cli ping"]}}, "healthcheck"),
        ({"placementSwarm": {"Constraints": [], "MaxReplicas": 1}}, "haiku-18"),
        (
            {
                "placementSwarm": {
                    "Constraints": ["node.hostname==haiku-18"],
                    "MaxReplicas": 2,
                }
            },
            "MaxReplicas",
        ),
    ],
)
def test_verifier_rejects_invalid_external_redis_configuration(overrides, message):
    def application_for_id(url):
        return overrides if "applicationId=redis" in url else {}

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": application_for_id,
        }
    )

    with pytest.raises(ValueError, match=message):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
        )


def test_main_reads_external_redis_application_id(monkeypatch):
    captured = {}
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy.example")
    monkeypatch.setenv("DOKPLOY_API_KEY", "secret")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setenv("REDIS_APPLICATION_ID", "redis")
    monkeypatch.setenv("GITHUB_SHA", "target")
    monkeypatch.setattr(
        rollout_verifier, "verify_rollout", lambda **kwargs: captured.update(kwargs)
    )

    rollout_verifier.main()

    assert captured["redis_application_id"] == "redis"


def test_verifier_requires_two_identical_complete_task_and_instance_snapshots_with_runtime_labels():
    container_id = "a" * 64
    task_snapshots = iter(
        [
            [_running_task("task-a", "a")],
            [_running_task("task-a", "a", 6)],
        ]
    )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
            "docker.getConfig": _task_config(
                container_id,
                {
                    **rollout_verifier._observability_labels("target"),
                    "com.docker.swarm.task.id": "task-a",
                },
            ),
        }
    )

    verify_rollout(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        redis_application_id="redis",
        revision="target",
        health_url="https://crawl.example/health",
        read_json=read_json,
        probe_health=lambda _url, count: [_health(container_id[:12])] * count,
        sleep=lambda _seconds: None,
    )


@pytest.mark.parametrize(
    "second_tasks,second_health,second_labels",
    [
        (
            [_running_task("task-b", "b")],
            _health("b" * 12),
            None,
        ),
        (
            [_running_task("task-a", "a", 6)],
            _health("a" * 12, "rolled-back"),
            None,
        ),
        (
            [_running_task("task-a", "a", 6)],
            _health("a" * 12),
            rollout_verifier._observability_labels("wrong"),
        ),
    ],
)
def test_verifier_rejects_membership_churn_revision_rollback_or_runtime_label_drift(
    second_tasks, second_health, second_labels
):
    clock = iter([0, 0, 0, 1, 1, 700])
    task_snapshots = iter(
        [
            [_running_task("task-a", "a")],
            second_tasks,
        ]
    )
    health_snapshots = iter([_health("a" * 12), second_health])
    config_labels = iter([None, second_labels])

    def task_config(url):
        task_id = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)[
            "containerId"
        ][0]
        container_id = ("a" if task_id == "task-a" else "b") * 64
        return _task_config(container_id, next(config_labels))

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
            "docker.getConfig": task_config,
        }
    )

    with pytest.raises(TimeoutError, match="did not stabilize"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            probe_health=lambda _url, count: [next(health_snapshots)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )


def test_verifier_rejects_unbounded_replica_counts():
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 17, "appName": "crawl4ai"},
        }
    )

    with pytest.raises(ValueError, match="between 1 and 16"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_verifier_rejects_stop_grace_shorter_than_the_process_drain():
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {
                "replicas": 1,
                "appName": "crawl4ai",
                "stopGracePeriodSwarm": REQUIRED_STOP_GRACE_NS - 1,
            },
        }
    )

    with pytest.raises(ValueError, match="must outlive"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_verifier_rejects_shutdown_desired_tasks_that_are_still_running():
    clock = iter([0, 0, 0, 1, 700])
    get_config_calls = 0
    tasks = [
        _running_task("task-current", "a", 5),
        {
            "containerId": "task-stale",
            "state": "shutdown",
            "currentState": "Running 1s",
            "node": "b",
            "error": "",
        },
    ]

    def task_config(_url):
        nonlocal get_config_calls
        get_config_calls += 1
        return {"Status": {"ContainerStatus": {"ContainerID": "a" * 64}}}

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": tasks,
            "docker.getConfig": task_config,
        }
    )

    with pytest.raises(TimeoutError, match="did not stabilize"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            probe_health=lambda _url, count: [_health("a" * 12)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )
    assert get_config_calls == 0


def test_verifier_rejects_desired_running_tasks_that_are_still_pending():
    clock = iter([0, 0, 0, 1, 700])
    tasks = [
        _running_task("task-a", "a", 5),
        _running_task("task-b", "b", 5),
        {
            "containerId": "task-pending",
            "state": "running",
            "currentState": "Pending 1s",
            "node": "",
            "error": "",
        },
    ]

    def unexpected_inspection(_url):
        raise AssertionError(
            "impossible task sets must short-circuit before inspection"
        )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 2, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": tasks,
            "docker.getConfig": unexpected_inspection,
        }
    )

    with pytest.raises(TimeoutError, match="did not stabilize"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            probe_health=lambda _url, count: [_health("a" * 12), _health("b" * 12)]
            * (count // 2),
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )
