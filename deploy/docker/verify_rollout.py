"""Fail closed until Dokploy and every public replica serve one exact revision."""

from __future__ import annotations

import concurrent.futures
import configparser
import json
import os
import subprocess
import time
import urllib.parse
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 65_536
MAX_REPLICAS = 16
LLM_PROVIDER = "openai/qwen3.8-27b"
LLM_BASE_URL = "https://api.llm-gateway.iocloudhost.net/v1"
REDIS_VOLUME = "crawl4ai-redis-data"
REDIS_MOUNT_PATH = "/data"
REDIS_HEALTHCHECK = ["CMD", "redis-cli", "ping"]
REDIS_NODE_CONSTRAINT = "node.hostname==haiku-18"


def _observability_labels(revision: str) -> dict[str, str]:
    return {
        "otel.logs.enabled": "true",
        "otel.service.name": "crawl4ai",
        "otel.deployment.environment.name": "production",
        "otel.service.version": revision,
    }


def _environment_values(environment: str) -> dict[str, str]:
    values = {}
    for line in environment.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _verify_release_configuration(application: Any, revision: str) -> None:
    if not isinstance(application, dict):
        raise ValueError("application configuration response is invalid")
    if application.get("labelsSwarm") != _observability_labels(revision):
        raise ValueError("observability labels do not match the release")
    environment = application.get("env")
    if not isinstance(environment, str):
        raise ValueError("LLM environment is not configured")
    values = _environment_values(environment)
    if values.get("LLM_PROVIDER") != LLM_PROVIDER:
        raise ValueError("LLM_PROVIDER does not match the release")
    if values.get("LLM_BASE_URL") != LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL does not match the release")
    if not values.get("LLM_API_KEY", "").strip():
        raise ValueError("LLM_API_KEY must be nonempty")


def _verify_redis_configuration(application: Any) -> None:
    if not isinstance(application, dict):
        raise ValueError("external Redis configuration response is invalid")
    mounts = application.get("mounts")
    if not isinstance(mounts, list) or not any(
        isinstance(mount, dict)
        and mount.get("type") == "volume"
        and mount.get("volumeName") == REDIS_VOLUME
        and mount.get("mountPath") == REDIS_MOUNT_PATH
        for mount in mounts
    ):
        raise ValueError("external Redis must mount crawl4ai-redis-data at /data")
    healthcheck = application.get("healthCheckSwarm")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != REDIS_HEALTHCHECK:
        raise ValueError("external Redis must use redis-cli ping healthcheck")
    placement = application.get("placementSwarm")
    if not isinstance(placement, dict):
        raise ValueError("external Redis placement is not configured")
    constraints = placement.get("Constraints")
    if not isinstance(constraints, list) or REDIS_NODE_CONSTRAINT not in constraints:
        raise ValueError("external Redis must be placed on haiku-18")
    if placement.get("MaxReplicas") != 1:
        raise ValueError("external Redis MaxReplicas must be 1")


def _required_stop_grace_ns() -> int:
    parser = configparser.RawConfigParser()
    parser.read(Path(__file__).with_name("supervisord.conf"))
    return (parser.getint("program:gunicorn", "stopwaitsecs") + 1) * 1_000_000_000


REQUIRED_STOP_GRACE_NS = _required_stop_grace_ns()


def _curl_json(url: str, api_key: str | None = None) -> Any:
    """Fetch bounded JSON with a true transport deadline and no secret in argv."""
    config = ""
    if api_key:
        escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
        config = f'header = "x-api-key: {escaped}"\n'
    curl = subprocess.Popen(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--connect-timeout",
            "5",
            "--max-time",
            "15",
            "--config",
            "-",
            url,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert curl.stdin is not None and curl.stdout is not None
    curl.stdin.write(config.encode())
    curl.stdin.close()
    try:
        response = subprocess.run(
            ["head", "-c", str(MAX_RESPONSE_BYTES + 1)],
            stdin=curl.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=True,
        )
    finally:
        curl.stdout.close()
        if curl.poll() is None:
            curl.kill()
        curl.wait()
    if len(response.stdout) > MAX_RESPONSE_BYTES:
        raise ValueError("HTTP response exceeds 64 KiB")
    if curl.returncode != 0:
        raise RuntimeError("HTTP request failed")
    return json.loads(response.stdout)


def _dokploy_url(base_url: str, operation: str, **params: str) -> str:
    return f"{base_url.rstrip('/')}/api/{operation}?{urllib.parse.urlencode(params)}"


def _is_exact_health(health: Any, revision: str) -> bool:
    return (
        isinstance(health, dict)
        and bool(health.get("instance"))
        and health.get("revision") == revision
        and health.get("status") == "ok"
        and health.get("components", {}).get("api") == "ready"
        and health.get("components", {}).get("redis") == "ready"
    )


def _has_task_error(task: dict[str, Any]) -> bool:
    error = str(task.get("error", "")).strip()
    return bool(error.removeprefix("Error:").strip())


def verify_rollout(
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    redis_application_id: str,
    revision: str,
    health_url: str,
    read_json: Callable[[str, str | None], Any] = _curl_json,
    probe_health: Callable[[str, int], list[Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Require a stable task set and matching public instances at one revision."""
    deployment_deadline = monotonic() + 600
    while monotonic() < deployment_deadline:
        deployments = read_json(
            _dokploy_url(dokploy_url, "deployment.all", applicationId=application_id),
            api_key,
        )
        status = deployments[0]["status"] if deployments else "none"
        if status == "done":
            break
        if status == "error":
            raise RuntimeError("Dokploy reported a failed deployment")
        sleep(15)
    else:
        raise TimeoutError("Dokploy deployment did not finish within 10 minutes")

    expected_labels = _observability_labels(revision)

    if probe_health is None:

        def probe_health(url: str, count: int) -> list[Any]:
            def probe(_: int) -> Any:
                try:
                    return read_json(f"{url}?rollout={uuid.uuid4()}", None)
                except Exception:
                    return None

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(16, count)
            ) as executor:
                return list(executor.map(probe, range(count)))

    proof_deadline = monotonic() + 600
    candidate: tuple[frozenset[str], frozenset[str], frozenset[str]] | None = None
    stable_rounds = 0
    while monotonic() < proof_deadline:
        application = read_json(
            _dokploy_url(dokploy_url, "application.one", applicationId=application_id),
            api_key,
        )
        _verify_release_configuration(application, revision)
        redis_application = read_json(
            _dokploy_url(
                dokploy_url, "application.one", applicationId=redis_application_id
            ),
            api_key,
        )
        _verify_redis_configuration(redis_application)
        redis_tasks = read_json(
            _dokploy_url(
                dokploy_url,
                "docker.getServiceContainersByAppName",
                appName=redis_application["appName"],
            ),
            api_key,
        )
        redis_running = frozenset(
            task["containerId"]
            for task in redis_tasks
            if task.get("state") == "running"
            and str(task.get("currentState", "")).startswith("Running ")
            and task.get("node")
            and not _has_task_error(task)
        )
        if len(redis_running) != 1:
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue
        replicas = int(application["replicas"])
        if not 1 <= replicas <= MAX_REPLICAS:
            raise ValueError(
                f"configured replicas must be between 1 and {MAX_REPLICAS}"
            )
        stop_grace_ns = int(application.get("stopGracePeriodSwarm") or 0)
        if stop_grace_ns < REQUIRED_STOP_GRACE_NS:
            raise ValueError(
                "configured Swarm stop grace must outlive the supervised process drain"
            )
        app_name = application["appName"]
        tasks = read_json(
            _dokploy_url(
                dokploy_url, "docker.getServiceContainersByAppName", appName=app_name
            ),
            api_key,
        )
        actual_running_tasks = frozenset(
            task["containerId"]
            for task in tasks
            if str(task.get("currentState", "")).startswith("Running ")
            and task.get("node")
            and not _has_task_error(task)
        )
        desired_running_tasks = frozenset(
            task["containerId"] for task in tasks if task.get("state") == "running"
        )
        if (
            len(actual_running_tasks) != replicas
            or actual_running_tasks != desired_running_tasks
        ):
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue

        def inspect_task(task_id: str) -> str:
            task = read_json(
                _dokploy_url(dokploy_url, "docker.getConfig", containerId=task_id),
                api_key,
            )
            labels = task["Config"]["Labels"]
            if not isinstance(labels, dict) or any(
                labels.get(key) != value for key, value in expected_labels.items()
            ):
                raise ValueError("container observability labels do not match release")
            return task["Status"]["ContainerStatus"]["ContainerID"][:12]

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=replicas
            ) as executor:
                authoritative_instances = frozenset(
                    executor.map(inspect_task, actual_running_tasks)
                )
        except Exception:
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue

        responses = probe_health(health_url, max(4, replicas * 4))
        matches = [_is_exact_health(health, revision) for health in responses]
        public_instances = frozenset(
            health["instance"]
            for health, matches_revision in zip(responses, matches)
            if matches_revision
        )
        complete = public_instances == authoritative_instances and all(matches)
        snapshot = (actual_running_tasks, public_instances, redis_running)
        if complete and snapshot == candidate:
            stable_rounds += 1
        elif complete:
            candidate = snapshot
            stable_rounds = 1
        else:
            candidate = None
            stable_rounds = 0
        if stable_rounds >= 2:
            print(f"verified stable {replicas}-replica rollout at {revision}")
            return
        sleep(5)
    raise TimeoutError(
        f"exact rollout did not stabilize at {revision}; last snapshot: {candidate}"
    )


def main() -> None:
    verify_rollout(
        dokploy_url=os.environ["DOKPLOY_URL"],
        api_key=os.environ["DOKPLOY_API_KEY"],
        application_id=os.environ["APPLICATION_ID"],
        redis_application_id=os.environ["REDIS_APPLICATION_ID"],
        revision=os.environ["GITHUB_SHA"],
        health_url=os.environ.get("HEALTH_URL", "https://crawl4ai.haiku.host/health"),
    )


if __name__ == "__main__":
    main()
