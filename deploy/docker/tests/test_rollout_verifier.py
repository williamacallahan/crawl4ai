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

ORIGINAL_INSPECT_TASK_RUNTIME = rollout_verifier._inspect_task_runtime


@pytest.fixture(autouse=True)
def _task_runtime(monkeypatch):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 1)
    def inspect(task_id, _network_id):
        instance = ("b" if "b" in task_id else "a") * 12
        address = "10.0.1.12" if instance.startswith("b") else "10.0.1.11"
        return (
            instance,
            frozenset({address}),
            rollout_verifier._observability_labels("target"),
        )

    monkeypatch.setattr(rollout_verifier, "_inspect_task_runtime", inspect)


def _readiness_root_labels():
    prefix = "traefik.http.services.crawl4ai-1.loadbalancer"
    readiness = rollout_verifier.CRAWL_READINESS_CHECK
    return {
        "traefik.enable": "true",
        "traefik.swarm.network": "dokploy-network",
        "traefik.swarm.lbswarm": "false",
        "traefik.http.routers.crawl4ai-1-websecure.rule": "Host(`crawl.example`)",
        f"{prefix}.server.port": "11235",
        f"{prefix}.healthcheck.path": readiness["Path"],
        f"{prefix}.healthcheck.interval": f'{readiness["Interval"]}ns',
        f"{prefix}.healthcheck.unhealthyinterval": f'{readiness["UnhealthyInterval"]}ns',
        f"{prefix}.healthcheck.timeout": f'{readiness["Timeout"]}ns',
        f"{prefix}.healthcheck.status": str(readiness["Status"]),
        f"{prefix}.healthcheck.initialstatus": "down",
    }


def _runtime_state(*, traefik=None, **overrides):
    service = {
        "replicas": 1,
        "taskLabels": rollout_verifier._observability_labels("target"),
        "rootLabels": _readiness_root_labels(),
        "image": "registry.example/crawl4ai@sha256:" + "d" * 64,
        "healthCheck": {
            "Test": rollout_verifier.CRAWL_HEALTHCHECK_TEST,
            **rollout_verifier.CRAWL_HEALTHCHECK_TIMING,
        },
        "placement": {
            "Constraints": [],
            "MaxReplicas": rollout_verifier.CRAWL_MAX_REPLICAS_PER_NODE,
        },
        "updateConfig": {
            "Order": "start-first",
            "Parallelism": 1,
            "Delay": 150_000_000_000,
            "FailureAction": "rollback",
            "Monitor": 150_000_000_000,
            "MaxFailureRatio": 0,
        },
        "rollbackConfig": {
            "Order": "start-first",
            "Parallelism": 1,
            "Delay": 150_000_000_000,
            "FailureAction": "pause",
            "Monitor": 150_000_000_000,
            "MaxFailureRatio": 0,
        },
        "stopGracePeriod": REQUIRED_STOP_GRACE_NS,
        "volumeMounts": [],
    }
    service.update(overrides)
    return {
        "application": {"appName": "crawl4ai"},
        "service": service,
        "traefik": traefik
        or {
            "routers": [
                {
                    "routerId": "crawl4ai-1-websecure@swarm",
                    "status": "enabled",
                    "service": "crawl4ai-1@swarm",
                }
            ],
            "services": [
                {
                    "serviceId": "crawl4ai-1@swarm",
                    "status": "enabled",
                    "serverStatus": {"http://10.0.1.11:11235/": "UP"},
                }
            ],
        },
    }


def _redis_runtime_state(**overrides):
    service = {
        "replicas": 1,
        "taskLabels": {},
        "rootLabels": {},
        "image": "redis:7.4",
        "healthCheck": {"Interval": 1_000_000_000},
        "placement": {
            "Constraints": [rollout_verifier.REDIS_NODE_CONSTRAINT],
            "MaxReplicas": 1,
        },
        "volumeMounts": [
            {
                "Type": "volume",
                "Source": rollout_verifier.REDIS_VOLUME,
                "Target": rollout_verifier.REDIS_MOUNT_PATH,
            }
        ],
    }
    service.update(overrides)
    state = _runtime_state(traefik={"routers": [], "services": []}, **service)
    state["application"]["appName"] = "crawl4ai-redis"
    return state


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
        if operation == "application.runtimeServiceState" and isinstance(result, dict):
            result = (
                _redis_runtime_state(**result)
                if "applicationId=redis" in url
                else _runtime_state(**result)
            )
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


def test_task_runtime_reads_only_the_dokploy_overlay(monkeypatch):
    output = (
        '"' + "a" * 64 + '"\t'
        + '{"otel.service.version":"target"}\t'
        + '[{"Network":{"ID":"other"},"Addresses":["10.9.0.2/24"]},'
        + '{"Network":{"ID":"network"},"Addresses":["10.0.1.11/24"]}]\n'
    )
    monkeypatch.setattr(
        rollout_verifier.subprocess,
        "run",
        lambda *_args, **_kwargs: rollout_verifier.subprocess.CompletedProcess(
            [], 0, output, ""
        ),
    )

    instance, addresses, labels = ORIGINAL_INSPECT_TASK_RUNTIME(
        "task", "network"
    )

    assert instance == "a" * 12
    assert addresses == frozenset({"10.0.1.11"})
    assert labels == {"otel.service.version": "target"}


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
        ({"replicas": 2}, "exactly three"),
        ({"taskLabels": {}}, "observability labels"),
        ({"image": "registry.example/crawl4ai:wrong"}, "image does not match"),
        ({"placement": {"MaxReplicas": 2}}, "one replica per node"),
        ({"healthCheck": None}, "admission healthcheck"),
        ({"rootLabels": {}}, "native fail-closed"),
        (
            {"traefik": {"routers": [], "services": []}},
            "runtime routing is unavailable",
        ),
        (
            {"updateConfig": {"Order": "stop-first", "Parallelism": 1, "MaxFailureRatio": 0}},
            "updateConfig must use start-first",
        ),
        (
            {"rollbackConfig": {"Order": "stop-first", "Parallelism": 1, "MaxFailureRatio": 0}},
            "rollbackConfig must use start-first",
        ),
        (
            {
                "updateConfig": {
                    **_runtime_state()["service"]["updateConfig"],
                    "FailureAction": "continue",
                }
            },
            "fail closed with rollback",
        ),
        (
            {
                "rollbackConfig": {
                    **_runtime_state()["service"]["rollbackConfig"],
                    "FailureAction": "rollback",
                }
            },
            "fail closed with pause",
        ),
        (
            {
                "updateConfig": {
                    **_runtime_state()["service"]["updateConfig"],
                    "Monitor": rollout_verifier.CRAWL_UPDATE_MONITOR_NS - 1,
                }
            },
            "monitor must cover candidate startup",
        ),
        (
            {
                "rollbackConfig": {
                    **_runtime_state()["service"]["rollbackConfig"],
                    "Delay": rollout_verifier.CRAWL_UPDATE_DELAY_NS - 1,
                }
            },
            "delay must preserve peers",
        ),
    ],
)
def test_verifier_rejects_invalid_release_observability_configuration(overrides, message):
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.runtimeServiceState": overrides,
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
            network_id="network",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"volumeMounts": []}, "crawl4ai-redis-data"),
        ({"healthCheck": None}, "healthcheck"),
        ({"placement": {"Constraints": [], "MaxReplicas": 1}}, "haiku-18"),
        (
            {
                "placement": {
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
            "application.runtimeServiceState": application_for_id,
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
            network_id="network",
        )


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
            "application.runtimeServiceState": {},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
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
        network_id="network",
        probe_health=lambda _url, count: [_health(container_id[:12])] * count,
        sleep=lambda _seconds: None,
    )


def test_verifier_rejects_traefik_admission_outside_the_current_task_set():
    clock = iter([0, 0, 0, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
    traefik = {
        "routers": [
            {
                "routerId": "crawl4ai-1-websecure@swarm",
                "status": "enabled",
                "service": "crawl4ai-1@swarm",
            }
        ],
        "services": [
            {
                "serviceId": "crawl4ai-1@swarm",
                "status": "enabled",
                "serverStatus": {"http://10.0.1.99:11235/": "UP"},
            }
        ],
    }

    def runtime_for_id(url):
        return {} if "applicationId=redis" in url else {"traefik": traefik}

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.runtimeServiceState": runtime_for_id,
            "docker.getServiceContainersByAppName": [_running_task("task-a", "a")],
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
            network_id="network",
            probe_health=lambda _url, count: [_health("a" * 12)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )


@pytest.mark.parametrize(
    "second_tasks,second_health,second_labels,error_type,error_message",
    [
        (
            [_running_task("task-b", "b")],
            _health("b" * 12),
            None,
            TimeoutError,
            "did not stabilize",
        ),
        (
            [_running_task("task-a", "a", 6)],
            _health("a" * 12, "rolled-back"),
            None,
            TimeoutError,
            "did not stabilize",
        ),
        (
            [_running_task("task-a", "a", 6)],
            _health("a" * 12),
            rollout_verifier._observability_labels("wrong"),
            ValueError,
            "observability labels",
        ),
    ],
)
def test_verifier_rejects_membership_churn_revision_rollback_or_runtime_label_drift(
    second_tasks, second_health, second_labels, error_type, error_message
):
    clock = iter([0, 0, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
    task_snapshots = iter(
        [
            [_running_task("task-a", "a")],
            second_tasks,
        ]
    )
    health_snapshots = iter([_health("a" * 12), second_health])
    config_labels = iter([None, second_labels])

    def inspect_task(task_id, _network_id):
        container_id = ("a" if task_id == "task-a" else "b") * 64
        labels = next(config_labels)
        return (
            container_id[:12],
            frozenset({"10.0.1.11" if task_id == "task-a" else "10.0.1.12"}),
            rollout_verifier._observability_labels("target")
            if labels is None
            else labels,
        )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.runtimeServiceState": {},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
        }
    )

    with pytest.raises(error_type, match=error_message):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            health_url="https://crawl.example/health",
            read_json=read_json,
            network_id="network",
            probe_health=lambda _url, count: [next(health_snapshots)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
            inspect_task_runtime=inspect_task,
        )


def test_verifier_rejects_stop_grace_shorter_than_the_process_drain():
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.runtimeServiceState": {
                "stopGracePeriod": REQUIRED_STOP_GRACE_NS - 1,
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
            network_id="network",
            sleep=lambda _seconds: None,
        )


def test_verifier_rejects_shutdown_desired_tasks_that_are_still_running():
    clock = iter([0, 0, 0, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
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

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.runtimeServiceState": {},
            "docker.getServiceContainersByAppName": tasks,
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
            network_id="network",
            probe_health=lambda _url, count: [_health("a" * 12)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )


def test_verifier_rejects_desired_running_tasks_that_are_still_pending():
    clock = iter([0, 0, 0, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
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

    def unexpected_inspection(_task, _network):
        raise AssertionError(
            "impossible task sets must short-circuit before inspection"
        )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.runtimeServiceState": {},
            "docker.getServiceContainersByAppName": tasks,
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
            network_id="network",
            probe_health=lambda _url, count: [_health("a" * 12), _health("b" * 12)]
            * (count // 2),
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
            inspect_task_runtime=unexpected_inspection,
        )
