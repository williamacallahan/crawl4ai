"""Deploy one immutable Crawl4AI image through stock Dokploy and Swarm."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import yaml

REPLICAS = 3
ELIGIBLE_NODES = frozenset({"haiku-4", "haiku-5", "haiku-9", "haiku-18"})
LLM_PROVIDER = "openai/qwen3.8-27b"
LLM_BASE_URL = "https://api.llm-gateway.iocloudhost.net/v1"
HEALTH_URLS = (
    "https://crawl4ai.haiku.host/health",
    "https://crawl4ai.popos-sf0.com/health",
)
HEALTHCHECK = {
    "Test": ["CMD", "curl", "-f", "http://localhost:11235/health"],
    "Interval": 10_000_000_000,
    "Timeout": 5_000_000_000,
    "StartPeriod": 120_000_000_000,
    "Retries": 5,
}
NODE_CONSTRAINT = "node.labels.crawl4ai-eligible==true"
# A down eligible node is a capacity event, not a rollout failure. Placement
# carries no MaxReplicas cap: the hard cap turned any single down node into a
# rollout deadlock — start-first had no legal overlap slot, for the update and
# for its automatic rollback alike (2026-08-30, haiku-4). The scheduler's
# default same-service spread keeps replicas apart when capacity allows, and
# _verify_tasks fail-closes on the end state if they ever land together.
PLACEMENT = {"Constraints": [NODE_CONSTRAINT]}
# ponytail: legacy capped placement accepted while records/services converge;
# delete _LEGACY_PLACEMENT after the first green deploy on this shape.
_LEGACY_PLACEMENT = {"Constraints": [NODE_CONSTRAINT], "MaxReplicas": 1}
RESOURCES = {
    "Reservations": {"NanoCPUs": 500_000_000, "MemoryBytes": 1_073_741_824},
    "Limits": {"NanoCPUs": 2_000_000_000, "MemoryBytes": 4_294_967_296},
}
DELAY_NS = 400_000_000_000
STOP_GRACE_NS = 390_000_000_000
TIMEOUT_SECONDS = 4_000
# Crawl4AI's durable job queue and task state live in this one external Redis.
# Nothing in the deploy path proved its wiring, so a lost volume, an unpinned
# or duplicated replica, or a reintroduced appendfsync=always would first
# surface as a production incident. The appName is immutable in Dokploy.
REDIS_SERVICE = "crawl4ai-redis-8xt08u"
REDIS_NODE = "haiku-18"
# everysec, never always: appendfsync=always was half of the 2026-08-28 fleet
# kill, stalling every crawl behind Redis. Pinned so a console edit cannot
# reintroduce it silently.
REDIS_COMMAND = [
    "redis-server",
    "--dir", "/data",
    "--appendonly", "yes",
    "--appendfsync", "everysec",
    "--loglevel", "notice",
]
REDIS_MOUNTS = [{"Type": "volume", "Source": "crawl4ai-redis-data", "Target": "/data"}]
REDIS_HEALTHCHECK_TEST = ["CMD", "redis-cli", "ping"]
# Floors, not equalities. The 2026-08-28 kill came from a healthcheck that was
# too tight, so raising Retries or StartPeriod is a safety improvement and must
# not read as drift. Only the probe itself is bidirectional.
REDIS_HEALTHCHECK_FLOORS = {
    "Interval": 5_000_000_000,
    "Timeout": 1_000_000_000,
    "StartPeriod": 10_000_000_000,
    "Retries": 3,
}


def _labels(revision: str) -> dict[str, str]:
    return {
        "otel.logs.enabled": "true",
        "otel.service.name": "crawl4ai",
        "otel.deployment.environment.name": "production",
        "otel.service.version": revision,
    }


def _environment_values(environment: Any) -> dict[str, str]:
    if isinstance(environment, str):
        lines = environment.splitlines()
    elif isinstance(environment, list) and all(
        isinstance(line, str) for line in environment
    ):
        lines = environment
    else:
        raise ValueError("Crawl4AI environment is not configured")
    values = {}
    protected = {"LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY"}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            if key in protected and key in values:
                raise ValueError(f"duplicate {key}")
            values[key] = value
    return values


def _verify_llm_environment(environment: Any) -> None:
    values = _environment_values(environment)
    if values.get("LLM_PROVIDER") != LLM_PROVIDER:
        raise ValueError("LLM_PROVIDER drifted")
    if values.get("LLM_BASE_URL") != LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL drifted")
    if not values.get("LLM_API_KEY", "").strip():
        raise ValueError("LLM_API_KEY is empty")


def _rollout_policy(failure_action: str) -> dict[str, Any]:
    return {
        "Parallelism": 1,
        "Delay": DELAY_NS,
        "FailureAction": failure_action,
        "Monitor": DELAY_NS,
        "MaxFailureRatio": 0,
        "Order": "start-first",
    }


def _request_json(url: str, api_key: str | None = None) -> Any:
    config = ""
    if api_key:
        escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
        config = f'header = "x-api-key: {escaped}"\n'
    response = subprocess.run(
        [
            "curl", "--silent", "--show-error", "--fail",
            "--connect-timeout", "5", "--max-time", "15",
            "--max-filesize", "65536", "--config", "-", url,
        ],
        input=config,
        text=True,
        capture_output=True,
        timeout=20,
    )
    if response.returncode:
        raise RuntimeError("HTTP request failed")
    return json.loads(response.stdout) if response.stdout else None


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> Any:
    escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
    response = subprocess.run(
        [
            "curl", "--silent", "--show-error", "--fail",
            "--connect-timeout", "5", "--max-time", "15",
            "--max-filesize", "65536",
            "--request", "POST", "--header", "content-type: application/json",
            "--data", json.dumps(payload, separators=(",", ":")),
            "--config", "-", url,
        ],
        input=f'header = "x-api-key: {escaped}"\n',
        text=True,
        capture_output=True,
        timeout=20,
    )
    if response.returncode:
        raise RuntimeError("HTTP request failed; state is ambiguous")
    return json.loads(response.stdout) if response.stdout else None


def _url(base: str, operation: str, **params: str) -> str:
    return f"{base.rstrip('/')}/api/{operation}?{urllib.parse.urlencode(params)}"


def _service_spec(app_name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "service", "inspect", app_name, "--format", "{{json .Spec}}"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _update_state(app_name: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "service",
            "inspect",
            app_name,
            "--format",
            "{{if .UpdateStatus}}{{.UpdateStatus.State}}{{end}}",
        ],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _task_runtime(task_id: str, network_id: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "docker", "inspect", task_id, "--format",
            "{{if .Status.ContainerStatus}}{{json .Status.ContainerStatus.ContainerID}}"
            '{{else}}""{{end}}\t{{json .Spec.ContainerSpec.Labels}}\t'
            "{{json .Spec.ContainerSpec.Image}}\t{{json .NetworksAttachments}}",
        ],
        check=True, capture_output=True, text=True,
    )
    container, labels, image, attachments = result.stdout.rstrip("\n").split("\t", 3)
    addresses = {
        address.split("/", 1)[0]
        for attachment in json.loads(attachments) or []
        if attachment.get("Network", {}).get("ID") == network_id
        for address in attachment.get("Addresses", [])
    }
    return {
        "container": json.loads(container)[:12],
        "labels": json.loads(labels) or {},
        "image": json.loads(image),
        "addresses": addresses,
    }


def _task_state(task_id: str) -> tuple[str, str]:
    result = subprocess.run(
        [
            "docker", "inspect", task_id, "--format",
            "{{json .DesiredState}}\t{{json .Status.State}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [json.loads(value) for value in result.stdout.rstrip("\n").split("\t")]
    return values[0], values[1]


def _eligible_nodes() -> frozenset[str]:
    """Ready/Active nodes carrying the eligibility label.

    A labeled node that is down or drained is excluded from the result but is
    not an error: node loss is a capacity event and must not read as rollout
    failure. Label membership drifting from the contract is an error
    regardless of node state.
    """
    listed = subprocess.run(
        ["docker", "node", "ls", "--format", "{{.ID}}"],
        check=True, capture_output=True, text=True,
    )
    node_ids = [line for line in listed.stdout.splitlines() if line]
    inspected = subprocess.run(
        [
            "docker", "node", "inspect", "--format",
            "{{json .Description.Hostname}}\t{{json .Spec.Labels}}\t"
            "{{json .Status.State}}\t{{json .Spec.Availability}}",
            *node_ids,
        ],
        check=True, capture_output=True, text=True,
    )
    labeled = set()
    ready = set()
    for line in inspected.stdout.splitlines():
        hostname, labels, state, availability = line.split("\t", 3)
        if (json.loads(labels) or {}).get("crawl4ai-eligible") == "true":
            labeled.add(json.loads(hostname))
            if (json.loads(state), json.loads(availability)) == ("ready", "active"):
                ready.add(json.loads(hostname))
    if frozenset(labeled) != ELIGIBLE_NODES:
        raise RuntimeError("Crawl4AI eligible-node inventory drifted")
    return frozenset(ready)


def _drain_budget_ns(environment: Any) -> int:
    """What PID 1 needs to shut down cleanly, read from the sources that own it.

    Swarm SIGKILLs a task the moment StopGracePeriod expires, so a stop grace
    shorter than supervisord's stop wait plus the entrypoint's pre-termination
    delay truncates the drain. Nothing tied STOP_GRACE_NS to either number.

    The delay is read from the deployed environment when it sets one: the
    literal in entrypoint.sh is only the fallback, and reading it alone would
    miss a value raised in the Dokploy console.
    """
    here = Path(__file__).resolve().parent
    # The slowest program in the group bounds supervisord's own shutdown.
    stop_waits = re.findall(
        r"^stopwaitsecs=(\d+)\s*$", (here / "supervisord.conf").read_text(), re.MULTILINE
    )
    fallback = re.search(
        r"CRAWL4AI_DRAIN_DELAY_SECONDS:-(\d+)", (here / "entrypoint.sh").read_text()
    )
    if not stop_waits or not fallback:
        raise ValueError("container drain budget is no longer readable from its own sources")
    deployed = _environment_values(environment).get("CRAWL4AI_DRAIN_DELAY_SECONDS", "").strip()
    drain = int(deployed) if deployed.isdigit() else int(fallback.group(1))
    return (max(int(value) for value in stop_waits) + drain) * 1_000_000_000


def _verify_redis() -> None:
    """Prove the external Redis is still durable, pinned and single before
    rolling anything that depends on it."""
    spec = _service_spec(REDIS_SERVICE)
    task = spec.get("TaskTemplate") or {}
    container = task.get("ContainerSpec") or {}
    if container.get("Command") != REDIS_COMMAND:
        raise ValueError("Crawl4AI Redis command drifted")
    if container.get("Mounts") != REDIS_MOUNTS:
        raise ValueError("Crawl4AI Redis durable volume drifted")
    healthcheck = container.get("Healthcheck") or {}
    if healthcheck.get("Test") != REDIS_HEALTHCHECK_TEST:
        raise ValueError("Crawl4AI Redis healthcheck probe drifted")
    for field, floor in REDIS_HEALTHCHECK_FLOORS.items():
        if int(healthcheck.get(field) or 0) < floor:
            raise ValueError(f"Crawl4AI Redis healthcheck {field} fell below its floor")
    if task.get("Placement") != {
        "Constraints": [f"node.hostname=={REDIS_NODE}"],
        "MaxReplicas": 1,
    }:
        raise ValueError("Crawl4AI Redis placement drifted")
    if spec.get("Mode") != {"Replicated": {"Replicas": 1}}:
        raise ValueError("Crawl4AI Redis must stay single-replica")
    running = [
        row for row in _service_tasks(REDIS_SERVICE)
        if str(row.get("DesiredState", "")).lower() == "running"
    ]
    if (
        len(running) != 1
        or not str(running[0].get("CurrentState", "")).startswith("Running ")
        or running[0].get("Node") != REDIS_NODE
    ):
        raise RuntimeError("Crawl4AI Redis task did not converge on its pinned node")


def _policy(application: dict[str, Any]) -> None:
    if application.get("sourceType") != "docker":
        raise ValueError("Crawl4AI must remain a stock Dokploy Docker-image app")
    _verify_llm_environment(application.get("env"))
    if application.get("replicas") != REPLICAS:
        raise ValueError("Crawl4AI must keep three replicas")
    if application.get("healthCheckSwarm") != HEALTHCHECK:
        raise ValueError("Crawl4AI admission healthcheck drifted")
    if application.get("placementSwarm") not in (PLACEMENT, _LEGACY_PLACEMENT):
        raise ValueError("Crawl4AI placement drifted")
    if application.get("endpointSpecSwarm") != {"Mode": "vip", "Ports": []}:
        raise ValueError("Crawl4AI must use the native Swarm VIP")
    for field, action in (("updateConfigSwarm", "rollback"), ("rollbackConfigSwarm", "pause")):
        if application.get(field) != _rollout_policy(action):
            raise ValueError(f"{field} drifted")
    # Strictly greater: a grace equal to the budget leaves no room for the
    # SIGTERM itself to be delivered and observed.
    if STOP_GRACE_NS <= _drain_budget_ns(application.get("env")):
        raise ValueError("Crawl4AI stop grace no longer covers the container drain budget")
    if int(application.get("stopGracePeriodSwarm") or 0) != STOP_GRACE_NS:
        raise ValueError("Crawl4AI stop grace drifted")
    for field, expected in (
        ("cpuReservation", "500000000"), ("cpuLimit", "2000000000"),
        ("memoryReservation", "1073741824"), ("memoryLimit", "4294967296"),
    ):
        if application.get(field) != expected:
            raise ValueError(f"Crawl4AI {field} drifted")


def _running_spec(
    spec: dict[str, Any],
    image: str,
    labels: dict[str, str],
    placements: tuple[dict[str, Any], ...] = (PLACEMENT, _LEGACY_PLACEMENT),
) -> None:
    task = spec.get("TaskTemplate") or {}
    container = task.get("ContainerSpec") or {}
    if container.get("Image") != image or container.get("Labels") != labels:
        raise ValueError("running artifact or labels differ from Dokploy")
    _verify_llm_environment(container.get("Env"))
    if container.get("Healthcheck") != HEALTHCHECK:
        raise ValueError("running healthcheck drifted")
    if task.get("Placement") not in placements:
        raise ValueError("running placement drifted")
    if task.get("Resources") != RESOURCES:
        raise ValueError("running resources drifted")
    if container.get("StopGracePeriod", 0) != STOP_GRACE_NS:
        raise ValueError("running stop grace drifted")
    if spec.get("Mode") != {"Replicated": {"Replicas": REPLICAS}}:
        raise ValueError("running replica count drifted")
    for field, action in (("UpdateConfig", "rollback"), ("RollbackConfig", "pause")):
        if spec.get(field) != _rollout_policy(action):
            raise ValueError(f"running {field} drifted")
    endpoint = spec.get("EndpointSpec") or {}
    if endpoint.get("Mode") != "vip" or endpoint.get("Ports") not in (None, []):
        raise ValueError("running endpoint mode drifted")


def verify_route(base: str, api_key: str, application_id: str, app_name: str) -> None:
    route = _request_json(
        _url(base, "application.readTraefikConfig", applicationId=application_id),
        api_key,
    )
    config = yaml.safe_load(route)
    http = config.get("http", {}) if isinstance(config, dict) else {}
    if http.get("serversTransports"):
        raise ValueError("stock Crawl4AI route contains a custom transport")
    services = http.get("services") or {}
    routers = http.get("routers") or {}
    expected_url = f"http://{app_name}:11235"
    for service in services.values():
        load_balancer = service.get("loadBalancer", {})
        if load_balancer.get("servers") != [{"url": expected_url}]:
            raise ValueError("Dokploy route does not target the native Swarm VIP")
        if load_balancer.get("serversTransport"):
            raise ValueError("Dokploy route uses a custom transport")
    hosts = tuple(urllib.parse.urlparse(url).hostname for url in HEALTH_URLS)
    for router in routers.values():
        rule = str(router.get("rule"))
        if (
            not any(f"`{host}`" in rule for host in hosts)
            or router.get("service") not in services
        ):
            raise ValueError("Dokploy router is not bound to a native VIP service")
    if not services or not routers or any(
        not any(f"`{host}`" in str(router.get("rule")) for router in routers.values())
        for host in hosts
    ):
        raise ValueError("Dokploy does not own both Crawl4AI domains")


def _application(base: str, api_key: str, application_id: str) -> dict[str, Any]:
    application = _request_json(
        _url(base, "application.one", applicationId=application_id), api_key
    )
    if not isinstance(application, dict):
        raise ValueError("invalid application response")
    return application


def _deployments(base: str, api_key: str, application_id: str) -> list[dict[str, Any]]:
    deployments = _request_json(_url(base, "deployment.all", applicationId=application_id), api_key)
    if not isinstance(deployments, list):
        raise ValueError("invalid deployment response")
    return deployments


def _wait_deployment(
    base: str,
    api_key: str,
    application_id: str,
    prior_ids: set[str],
    title: str,
    description: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        new = [
            row
            for row in _deployments(base, api_key, application_id)
            if row.get("deploymentId") not in prior_ids
        ]
        matching = [
            row
            for row in new
            if row.get("title") == title and row.get("description") == description
        ]
        if len(new) > 1 or len(matching) > 1 or (new and not matching):
            raise RuntimeError("deployment submission is ambiguous; no recovery write is allowed")
        if matching:
            owned = matching[0]
            status = owned.get("status")
            if status == "done":
                return owned
            if status in {"error", "cancelled"}:
                raise RuntimeError(f"stock Dokploy deployment ended in {status}")
        time.sleep(5)
    raise TimeoutError("stock Dokploy deployment did not finish")


def _exact_health(health: Any, revision: str | None = None) -> bool:
    return (
        isinstance(health, dict)
        and bool(health.get("instance"))
        and (revision is None or health.get("revision") == revision)
        and health.get("status") == "ok"
        and health.get("components", {}).get("api") == "ready"
        and health.get("components", {}).get("redis") == "ready"
    )


def _service_tasks(app_name: str) -> list[dict[str, Any]]:
    listed = subprocess.run(
        ["docker", "service", "ps", "--no-trunc", "--format", "{{json .}}", app_name],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in listed.stdout.splitlines() if line]


def _verify_tasks(
    app_name: str, image: str, revision: str, ready: frozenset[str], converged: bool = True
) -> dict[str, Any]:
    rows = _service_tasks(app_name)
    current = [
        row for row in rows
        if str(row.get("DesiredState", "")).lower() == "running"
    ]
    if len(current) != REPLICAS or any(
        not str(row.get("CurrentState", "")).startswith("Running ")
        for row in current
    ):
        raise RuntimeError("Crawl4AI tasks did not converge")
    nodes = {row.get("Node") for row in current}
    if not nodes <= ELIGIBLE_NODES:
        raise RuntimeError("Crawl4AI tasks are not on eligible nodes")
    # Distinct placement is this deploy's converged obligation. The baseline is
    # not held to it: a node death heals replicas onto the surviving nodes, and
    # refusing to deploy from that state would leave co-location permanent.
    if converged and len(nodes) != REPLICAS:
        raise RuntimeError("Crawl4AI tasks are not on distinct eligible nodes")
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(str(row.get("Name")), []).append(row)
    for candidate in current:
        history = by_slot[str(candidate.get("Name"))]
        predecessor = next(
            (row for row in history[1:] if str(row.get("DesiredState", "")).lower() == "shutdown"),
            None,
        )
        if predecessor is None:
            raise RuntimeError("Crawl4AI task has no predecessor withdrawal evidence")
        desired, state = _task_state(str(candidate["ID"])[:12])
        if desired != "running" or state != "running":
            raise RuntimeError("Swarm task history contradicts the start-first rollout")
        stranded = predecessor.get("Node") in ELIGIBLE_NODES - ready
        if not converged and stranded:
            # A predecessor stranded on a down eligible node can never confirm
            # its own shutdown; its desired state already records the
            # withdrawal. Only that exact case is excused: an unassigned or
            # rejected attempt (no node) or a task on a foreign node is not
            # withdrawal evidence and stays strict. The deploy's own
            # withdrawals happen on ready nodes and stay fully confirmed.
            continue
        predecessor_desired, predecessor_state = _task_state(str(predecessor["ID"])[:12])
        if (
            predecessor_desired != "shutdown"
            or predecessor_state not in {"shutdown", "complete"}
        ):
            raise RuntimeError("Swarm task history contradicts the start-first rollout")
    network = subprocess.run(
        ["docker", "network", "inspect", "dokploy-network", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    instances: set[str] = set()
    for row in current:
        runtime = _task_runtime(str(row["ID"])[:12], network)
        if (
            runtime["image"] != image
            or runtime["labels"] != _labels(revision)
            or not runtime["addresses"]
        ):
            raise RuntimeError("Crawl4AI task identity drifted")
        health = _request_json(f"http://{next(iter(runtime['addresses']))}:11235/health")
        if not _exact_health(health, revision) or health["instance"] != runtime["container"]:
            raise RuntimeError("Crawl4AI task failed direct overlay readiness")
        instances.add(runtime["container"])
    if len(instances) != REPLICAS:
        raise RuntimeError("Crawl4AI task instances are not unique")
    return {
        "tasks": sorted(str(row["ID"])[:12] for row in current),
        "nodes": sorted(nodes),
        "instances": sorted(instances),
    }


def _record_converged(app: dict[str, Any], candidate: str, revision: str) -> bool:
    return (
        app.get("dockerImage") == candidate
        and app.get("labelsSwarm") == _labels(revision)
        and app.get("placementSwarm") == PLACEMENT
    )


def deploy() -> None:
    base = os.environ["DOKPLOY_URL"]
    api_key = os.environ["DOKPLOY_API_KEY"]
    application_id = os.environ["APPLICATION_ID"]
    revision = os.environ["GITHUB_SHA"]
    candidate = f"{os.environ['IMAGE']}@{os.environ['IMAGE_DIGEST']}"
    application = _application(base, api_key, application_id)
    _policy(application)
    baseline = str(application.get("dockerImage", ""))
    baseline_labels = application.get("labelsSwarm")
    if "@sha256:" not in baseline or not isinstance(baseline_labels, dict):
        raise ValueError("stock Dokploy baseline is not immutable")
    baseline_revision = baseline_labels.get("otel.service.version")
    if not isinstance(baseline_revision, str) or not baseline_revision:
        raise ValueError("stock Dokploy baseline has no revision")
    app_name = str(application["appName"])
    if _update_state(app_name) != "completed":
        raise RuntimeError("Crawl4AI already has a nonterminal Swarm update")
    # The live service may keep the legacy cap only while the Dokploy record
    # still carries it; once the record has converged, a capped live spec is
    # reintroduced drift, not transition residue.
    baseline_placements = (
        (PLACEMENT, _LEGACY_PLACEMENT)
        if application.get("placementSwarm") == _LEGACY_PLACEMENT
        else (PLACEMENT,)
    )
    _running_spec(
        _service_spec(app_name), baseline, baseline_labels, placements=baseline_placements
    )
    _verify_redis()
    verify_route(base, api_key, application_id, app_name)
    # Captured adjacent to its use: this snapshot decides which unconfirmable
    # predecessors the baseline may excuse.
    ready = _eligible_nodes()
    if len(ready) < REPLICAS:
        raise RuntimeError("not enough Ready eligible nodes to place every replica")
    _verify_tasks(app_name, baseline, baseline_revision, ready, False)
    for url in HEALTH_URLS:
        if not _exact_health(_request_json(f"{url}?baseline={uuid.uuid4()}"), baseline_revision):
            raise RuntimeError("public Crawl4AI baseline is not ready")
    prior_deployments = _deployments(base, api_key, application_id)
    if any(
        row.get("status") not in {"done", "error", "cancelled"}
        for row in prior_deployments
    ):
        raise RuntimeError("Crawl4AI already has a nonterminal Dokploy deployment")
    prior_ids = {str(row.get("deploymentId")) for row in prior_deployments}
    title = f"crawl4ai-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-{revision}"
    description = f"candidate={candidate};baseline={baseline}"
    current = _application(base, api_key, application_id)
    _policy(current)
    if current.get("dockerImage") != baseline or current.get("labelsSwarm") != baseline_labels:
        raise RuntimeError("baseline metadata changed before submission")
    verify_route(base, api_key, application_id, app_name)
    _post_json(
        f"{base.rstrip('/')}/api/application.update",
        api_key,
        {
            "applicationId": application_id,
            "dockerImage": candidate,
            "labelsSwarm": _labels(revision),
            "placementSwarm": PLACEMENT,
        },
    )
    updated = _application(base, api_key, application_id)
    _policy(updated)
    if not _record_converged(updated, candidate, revision):
        raise RuntimeError("candidate metadata did not converge; no deploy was submitted")
    verify_route(base, api_key, application_id, app_name)
    _post_json(
        f"{base.rstrip('/')}/api/application.deploy",
        api_key,
        {"applicationId": application_id, "title": title, "description": description},
    )
    deployment = _wait_deployment(base, api_key, application_id, prior_ids, title, description)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while True:
        state = _update_state(app_name)
        if state == "completed":
            break
        if state in {"paused", "rollback_paused", "rollback_completed"}:
            raise RuntimeError(f"Swarm update ended in {state}; manual reconciliation required")
        if time.monotonic() >= deadline:
            raise TimeoutError("Swarm update did not reach completed")
        time.sleep(5)
    final = _application(base, api_key, application_id)
    _policy(final)
    if not _record_converged(final, candidate, revision):
        raise RuntimeError("foreign application metadata replaced the candidate")
    _running_spec(
        _service_spec(app_name), candidate, _labels(revision), placements=(PLACEMENT,)
    )
    ready = _eligible_nodes()
    if len(ready) < REPLICAS:
        raise RuntimeError("not enough Ready eligible nodes to place every replica")
    task_proof = _verify_tasks(app_name, candidate, revision, ready)
    verify_route(base, api_key, application_id, app_name)
    for url in HEALTH_URLS:
        if not all(
            _exact_health(_request_json(f"{url}?verify={uuid.uuid4()}"), revision)
            for _ in range(4)
        ):
            raise RuntimeError("public Crawl4AI health did not converge")
    print(
        json.dumps(
            {"deploymentId": deployment["deploymentId"], "revision": revision, **task_proof},
            separators=(",", ":"),
        )
    )


def monitor() -> None:
    evidence = Path(os.environ["ROLLOUT_MONITOR_PATH"])
    armed = Path(os.environ["ROLLOUT_MONITOR_ARMED_PATH"])
    stop = Path(os.environ["ROLLOUT_MONITOR_STOP_PATH"])
    evidence.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    first_round = True
    with evidence.open("a") as output:
        while not stop.exists():
            for url in HEALTH_URLS:
                try:
                    health = _request_json(f"{url}?rollout={uuid.uuid4()}")
                    sample = {
                        "ok": _exact_health(health),
                        "url": url,
                        "timestamp": time.time(),
                        "revision": health.get("revision"),
                        "instance": health.get("instance"),
                    }
                except Exception as error:
                    sample = {
                        "ok": False,
                        "url": url,
                        "timestamp": time.time(),
                        "error": type(error).__name__,
                    }
                failures += not sample["ok"]
                output.write(json.dumps(sample, separators=(",", ":")) + "\n")
                output.flush()
            if first_round:
                if failures:
                    raise RuntimeError("public rollout baseline is not ready")
                armed.touch()
                first_round = False
            time.sleep(0.5)
    if failures:
        raise RuntimeError(f"public rollout monitor recorded {failures} failures")


def evidence() -> None:
    rows = [
        json.loads(line)
        for line in Path(os.environ["ROLLOUT_MONITOR_PATH"])
        .read_text()
        .splitlines()
        if line
    ]
    if not rows or any(not row.get("ok") for row in rows):
        raise RuntimeError("public rollout evidence contains a failure")
    revision = os.environ["GITHUB_SHA"]
    if any(
        not any(
            row.get("revision") == revision
            for row in rows
            if row.get("url") == url
        )
        for url in HEALTH_URLS
    ):
        raise RuntimeError("each public domain must observe the candidate revision")
    print(json.dumps({"publicSuccesses": len(rows), "publicFailures": 0}, separators=(",", ":")))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    {"deploy": deploy, "monitor": monitor, "evidence": evidence}[command]()


if __name__ == "__main__":
    main()
