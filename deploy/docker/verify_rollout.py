"""Fail closed until Dokploy and every public replica serve one exact revision."""

from __future__ import annotations

import concurrent.futures
import configparser
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from readiness_routing import (
    MAX_RESPONSE_BYTES,
    ingress_backends,
    ingress_task_stats_url,
    verify_monitor_evidence,
)

MAX_REPLICAS = 16
CRAWL_REPLICAS = 3
LLM_PROVIDER = "openai/qwen3.8-27b"
LLM_BASE_URL = "https://api.llm-gateway.iocloudhost.net/v1"
REDIS_VOLUME = "crawl4ai-redis-data"
REDIS_MOUNT_PATH = "/data"
REDIS_HEALTHCHECK = ["CMD", "redis-cli", "ping"]
REDIS_NODE_CONSTRAINT = "node.hostname==haiku-18"
CRAWL_HEALTHCHECK = ["CMD", "curl", "-f", "http://localhost:11235/health"]
CRAWL_HEALTHCHECK_MIN_RETRIES = 3
CRAWL_HEALTHCHECK_MIN_INTERVAL_NS = 5_000_000_000
CRAWL_MAX_REPLICAS_PER_NODE = 1
ROLLOUT_PROOF_TIMEOUT_SECONDS = 900


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
    if application.get("replicas") != CRAWL_REPLICAS:
        raise ValueError("Crawl4AI must keep exactly three replicas")
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
    placement = application.get("placementSwarm")
    if not isinstance(placement, dict) or placement.get("MaxReplicas") != CRAWL_MAX_REPLICAS_PER_NODE:
        raise ValueError("Crawl4AI placement must keep one replica per node")
    healthcheck = application.get("healthCheckSwarm")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != CRAWL_HEALTHCHECK:
        raise ValueError("Crawl4AI must use the routing admission healthcheck")
    # A 1s/1-retry probe kills tasks on a single transient /health 503;
    # require tolerance for short blips.
    if (
        healthcheck.get("Retries", 0) < CRAWL_HEALTHCHECK_MIN_RETRIES
        or healthcheck.get("Interval", 0) < CRAWL_HEALTHCHECK_MIN_INTERVAL_NS
    ):
        raise ValueError(
            "Crawl4AI healthcheck must tolerate transient blips (Retries>=3, Interval>=5s)"
        )
    for field in ("updateConfigSwarm", "rollbackConfigSwarm"):
        config = application.get(field)
        if not isinstance(config, dict) or config.get("Order") != "start-first":
            raise ValueError(f"{field} must use start-first order")
        if config.get("Parallelism") != 1 or config.get("MaxFailureRatio") != 0:
            raise ValueError(f"{field} must replace one task at a time and fail closed")


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
    command = application.get("command")
    if not isinstance(command, str) or "--appendfsync everysec" not in command:
        # fsync-per-write blocks the single Redis thread on disk stalls, which
        # 503s every replica's /health at once.
        raise ValueError("external Redis must persist with --appendfsync everysec")
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
    entrypoint = Path(__file__).with_name("entrypoint.sh").read_text()
    match = re.search(r"CRAWL4AI_DRAIN_DELAY_SECONDS:-([0-9]+)", entrypoint)
    if match is None:
        raise ValueError("entrypoint drain delay is not configured")
    return (
        parser.getint("program:gunicorn", "stopwaitsecs")
        + int(match.group(1))
        + 1
    ) * 1_000_000_000


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


def _inspect_task_runtime(
    task_id: str, network_id: str
) -> tuple[str, frozenset[str], dict[str, str]]:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            task_id,
            "--format",
            "{{json .Status.ContainerStatus.ContainerID}}\t"
            "{{json .Spec.ContainerSpec.Labels}}\t{{json .NetworksAttachments}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container, labels, attachments = result.stdout.rstrip("\n").split("\t", 2)
    addresses = frozenset(
        address.split("/", 1)[0]
        for attachment in json.loads(attachments) or []
        if attachment.get("Network", {}).get("ID") == network_id
        for address in attachment.get("Addresses", [])
    )
    return json.loads(container)[:12], addresses, json.loads(labels) or {}


def _dokploy_network_id() -> str:
    network = subprocess.run(
        ["docker", "network", "inspect", "dokploy-network", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not network:
        raise ValueError("dokploy-network could not be resolved")
    return network


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
    monitor_evidence_path: Path | None = None,
    ingress_stats_url: str | None = None,
    inspect_task_runtime: Callable[
        [str, str], tuple[str, frozenset[str], dict[str, str]]
    ]
    | None = None,
) -> None:
    """Require a stable task set and matching public instances at one revision."""
    inspect_task_runtime = inspect_task_runtime or _inspect_task_runtime
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
    network = _dokploy_network_id()

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

    proof_deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
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

        def inspect_task(task_id: str) -> tuple[str, frozenset[str]]:
            instance, addresses, labels = inspect_task_runtime(task_id, network)
            if not isinstance(labels, dict) or any(
                labels.get(key) != value for key, value in expected_labels.items()
            ):
                raise ValueError("container observability labels do not match release")
            if not addresses:
                raise ValueError("Crawl4AI task has no overlay address")
            return instance, addresses

        with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as executor:
            inspected_tasks = list(executor.map(inspect_task, actual_running_tasks))
        authoritative_instances = frozenset(instance for instance, _ in inspected_tasks)
        authoritative_addresses = frozenset(
            address for _, addresses in inspected_tasks for address in addresses
        )

        if ingress_stats_url is not None:
            admitted_backends = frozenset(ingress_backends(ingress_stats_url))
            if len(admitted_backends) != len(authoritative_addresses):
                candidate = None
                stable_rounds = 0
                sleep(5)
                continue
        else:
            admitted_backends = authoritative_addresses

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
            verify_monitor_evidence(
                monitor_evidence_path,
                authoritative_instances,
                admitted_backends,
            )
            print(
                json.dumps(
                    {
                        "revision": revision,
                        "tasks": sorted(actual_running_tasks),
                        "instances": sorted(authoritative_instances),
                        "taskAddresses": sorted(authoritative_addresses),
                        "admittedBackends": sorted(admitted_backends),
                    },
                    separators=(",", ":"),
                )
            )
            return
        sleep(5)
    raise TimeoutError(
        f"exact rollout did not stabilize at {revision}; last snapshot: {candidate}"
    )


def main(arguments: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments:
        raise ValueError(f"Unsupported verifier arguments: {arguments}")
    verify_rollout(
        dokploy_url=os.environ["DOKPLOY_URL"],
        api_key=os.environ["DOKPLOY_API_KEY"],
        application_id=os.environ["APPLICATION_ID"],
        redis_application_id=os.environ["REDIS_APPLICATION_ID"],
        revision=os.environ["GITHUB_SHA"],
        health_url=os.environ.get("HEALTH_URL", "https://crawl4ai.haiku.host/health"),
        monitor_evidence_path=(
            Path(os.environ["ROLLOUT_MONITOR_PATH"])
            if os.environ.get("ROLLOUT_MONITOR_PATH")
            else None
        ),
        ingress_stats_url=ingress_task_stats_url(
            os.environ["INGRESS_APPLICATION_APP_NAME"]
        ),
    )


if __name__ == "__main__":
    main()
