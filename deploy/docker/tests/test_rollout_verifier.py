from __future__ import annotations

import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from verify_rollout import (
    MAX_RESPONSE_BYTES,
    REQUIRED_STOP_GRACE_NS,
    _curl_json,
    _has_task_error,
    verify_rollout,
)


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
        if operation not in handlers:
            raise AssertionError(f"unexpected operation: {operation}")
        handler = handlers[operation]
        result = handler(url) if callable(handler) else handler
        if operation == "application.one" and isinstance(result, dict):
            result = dict(result)
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


def test_verifier_requires_two_identical_complete_task_and_instance_snapshots():
    container_id = "a" * 64
    task_snapshots = iter(
        [
            [
                {
                    "containerId": "task-a",
                    "state": "running",
                    "currentState": "Running 1s",
                    "node": "a",
                    "error": "",
                }
            ],
            [
                {
                    "containerId": "task-a",
                    "state": "running",
                    "currentState": "Running 6s",
                    "node": "a",
                    "error": "",
                }
            ],
        ]
    )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
            "docker.getConfig": {
                "Status": {"ContainerStatus": {"ContainerID": container_id}}
            },
        }
    )

    verify_rollout(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        revision="target",
        health_url="https://crawl.example/health",
        read_json=read_json,
        probe_health=lambda _url, count: [_health(container_id[:12])] * count,
        sleep=lambda _seconds: None,
    )


@pytest.mark.parametrize(
    "second_tasks,second_health",
    [
        (
            [
                {
                    "containerId": "task-b",
                    "state": "running",
                    "currentState": "Running 1s",
                    "node": "b",
                    "error": "",
                }
            ],
            _health("b" * 12),
        ),
        (
            [
                {
                    "containerId": "task-a",
                    "state": "running",
                    "currentState": "Running 6s",
                    "node": "a",
                    "error": "",
                }
            ],
            _health("a" * 12, "rolled-back"),
        ),
    ],
)
def test_verifier_rejects_membership_churn_and_revision_rollback(
    second_tasks, second_health
):
    clock = iter([0, 0, 0, 1, 1, 700])
    task_snapshots = iter(
        [
            [
                {
                    "containerId": "task-a",
                    "state": "running",
                    "currentState": "Running 1s",
                    "node": "a",
                    "error": "",
                }
            ],
            second_tasks,
        ]
    )
    health_snapshots = iter([_health("a" * 12), second_health])

    def task_config(url):
        task_id = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)[
            "containerId"
        ][0]
        container_id = ("a" if task_id == "task-a" else "b") * 64
        return {"Status": {"ContainerStatus": {"ContainerID": container_id}}}

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
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_verifier_rejects_shutdown_desired_tasks_that_are_still_running():
    clock = iter([0, 0, 0, 1, 700])
    get_config_calls = 0
    tasks = [
        {
            "containerId": "task-current",
            "state": "running",
            "currentState": "Running 5s",
            "node": "a",
            "error": "",
        },
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
        {
            "containerId": "task-a",
            "state": "running",
            "currentState": "Running 5s",
            "node": "a",
            "error": "",
        },
        {
            "containerId": "task-b",
            "state": "running",
            "currentState": "Running 5s",
            "node": "b",
            "error": "",
        },
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
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            probe_health=lambda _url, count: [_health("a" * 12), _health("b" * 12)]
            * (count // 2),
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )
