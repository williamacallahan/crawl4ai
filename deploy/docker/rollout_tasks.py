"""Read and verify live Swarm service, task, node, and public-route state."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import yaml
import rollout_http
import rollout_policy


# A withdrawn `docker service ps` row whose Status.State never reached RUNNING
# never started a backend and withdrew nothing. Swarm can retain such a frozen
# row after reaping the real terminal predecessor: the orchestrator sets the
# row's DesiredState to shutdown, but Status.State freezes at a never-started
# dispatch point (new/pending/assigned/accepted/ready/preparing/starting) when
# the node dies mid-dispatch or the update is superseded. Terminal states
# (shutdown/complete/failed/rejected) and orphaned (which may have started a
# backend and is covered by the stranded-node rule) are not excused here.
NEVER_STARTED_STATES = frozenset({
    "new", "pending", "assigned", "accepted", "ready", "preparing", "starting",
})


def _service_spec(app_name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "service", "inspect", app_name, "--format", "{{json .Spec}}"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _update_state(app_name: str) -> tuple[str, str]:
    result = subprocess.run(
        [
            "docker",
            "service",
            "inspect",
            app_name,
            "--format",
            "{{json .UpdateStatus}}",
        ],
        check=True, capture_output=True, text=True,
    )
    status = json.loads(result.stdout) or {}
    return status.get("State", ""), status.get("StartedAt", "")


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


def _eligible_nodes() -> tuple[frozenset[str], frozenset[str]]:
    """Read labelled membership and Ready/Active capacity from the Swarm owner."""
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
    return frozenset(labeled), frozenset(ready)


def _verify_redis() -> None:
    """Prove the external Redis is still durable, pinned and single before
    rolling anything that depends on it."""
    spec = _service_spec(rollout_policy.REDIS_SERVICE)
    task = spec.get("TaskTemplate") or {}
    container = task.get("ContainerSpec") or {}
    if container.get("Command") != rollout_policy.REDIS_COMMAND:
        raise ValueError("Crawl4AI Redis command drifted")
    if container.get("Mounts") != rollout_policy.REDIS_MOUNTS:
        raise ValueError("Crawl4AI Redis durable volume drifted")
    healthcheck = container.get("Healthcheck") or {}
    if healthcheck.get("Test") != rollout_policy.REDIS_HEALTHCHECK_TEST:
        raise ValueError("Crawl4AI Redis healthcheck probe drifted")
    for field, floor in rollout_policy.REDIS_HEALTHCHECK_FLOORS.items():
        if int(healthcheck.get(field) or 0) < floor:
            raise ValueError(f"Crawl4AI Redis healthcheck {field} fell below its floor")
    if task.get("Placement") != {
        "Constraints": [f"node.hostname=={rollout_policy.REDIS_NODE}"],
        "MaxReplicas": 1,
    }:
        raise ValueError("Crawl4AI Redis placement drifted")
    if spec.get("Mode") != {"Replicated": {"Replicas": 1}}:
        raise ValueError("Crawl4AI Redis must stay single-replica")
    running = [
        row for row in _service_tasks(rollout_policy.REDIS_SERVICE)
        if str(row.get("DesiredState", "")).lower() == "running"
    ]
    if (
        len(running) != 1
        or not str(running[0].get("CurrentState", "")).startswith("Running ")
        or running[0].get("Node") != rollout_policy.REDIS_NODE
    ):
        raise RuntimeError("Crawl4AI Redis task did not converge on its pinned node")


def verify_route(base: str, api_key: str, application_id: str, app_name: str) -> None:
    route = rollout_http._request_json(
        rollout_http._url(base, "application.readTraefikConfig", applicationId=application_id),
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
    hosts = tuple(urllib.parse.urlparse(url).hostname for url in rollout_policy.HEALTH_URLS)
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


def _verify_ingress_container() -> str:
    ingress = subprocess.run(
        [
            "docker",
            "ps",
            "--no-trunc",
            "--filter",
            "name=^/dokploy-traefik$",
            "--filter",
            "status=running",
            "--format",
            "{{.ID}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container = ingress.stdout.splitlines()
    if len(container) != 1:
        raise RuntimeError("rollout verifier is not running on the public ingress host")
    return container[0]


def _verify_ingress_host() -> int:
    container = _verify_ingress_container()
    inspected = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.State.Pid}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        pid = int(inspected.stdout.strip())
    except ValueError as error:
        raise RuntimeError("public ingress network namespace PID is invalid") from error
    if pid <= 0:
        raise RuntimeError("public ingress network namespace PID is invalid")
    return pid


def _verify_tasks(
    app_name: str, image: str, revision: str,
    eligibility: tuple[frozenset[str], frozenset[str]], converged: bool = True,
    placement: dict[str, Any] = rollout_policy.PLACEMENT,
) -> dict[str, Any]:
    eligible, ready = eligibility
    ingress_container = _verify_ingress_container()
    rows = _service_tasks(app_name)
    current = [
        row for row in rows
        if str(row.get("DesiredState", "")).lower() == "running"
    ]
    if len(current) != rollout_policy.REPLICAS or any(
        not str(row.get("CurrentState", "")).startswith("Running ")
        for row in current
    ):
        raise RuntimeError("Crawl4AI tasks did not converge")
    nodes = {row.get("Node") for row in current}
    if not nodes <= eligible:
        raise RuntimeError("Crawl4AI tasks are not on eligible nodes")
    if not nodes <= ready:
        raise RuntimeError("Crawl4AI tasks are not on Ready eligible nodes")
    # MaxReplicas is a hard scheduler rule; hostname spread is only a preference.
    if placement.get("MaxReplicas") == 1 and len(nodes) != rollout_policy.REPLICAS:
        raise RuntimeError("Crawl4AI tasks are not on distinct eligible nodes")
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(str(row.get("Name")), []).append(row)
    for candidate in current:
        desired, state = _task_state(str(candidate["ID"])[:12])
        if desired != "running" or state != "running":
            raise RuntimeError("Swarm task history contradicts the start-first rollout")
        # A withdrawn task whose Status.State never reached RUNNING never started
        # a backend and withdrew nothing; Swarm can retain such a frozen row
        # even after reaping the real terminal predecessor. Drop every never-
        # started row from the slot's history so it cannot become its real
        # predecessor and trip the strict check below; terminal and orphaned
        # rows are not excused and stay strict here and via the stranded rule.
        history = [
            row for row in by_slot[str(candidate.get("Name"))]
            if not (
                str(row.get("DesiredState", "")).lower() == "shutdown"
                and _task_state(str(row["ID"])[:12])[1] in NEVER_STARTED_STATES
            )
        ]
        # Swarm reaps terminal predecessors with task-history-limit=1, but
        # retains a predecessor that is still running. Verify retained history
        # when present; exact VIP membership below also excludes old backends.
        if len(history) == 1:
            continue
        withdrawn = [
            row for row in history[1:]
            if str(row.get("DesiredState", "")).lower() == "shutdown"
        ]
        if not converged:
            # A task an eligible node refused never started, so it withdrew
            # nothing and is not this slot's predecessor; a slot holding only
            # those was placed by scheduler recovery and has no earlier rollout
            # to prove. An unassigned or foreign-node attempt is still not
            # withdrawal evidence, and the deploy's own census stays strict.
            withdrawn = [
                row for row in withdrawn
                if not (
                    str(row.get("CurrentState", "")).startswith("Rejected")
                    and row.get("Node") in eligible
                )
            ]
            if not withdrawn:
                continue
        if not withdrawn:
            raise RuntimeError("Crawl4AI task has no predecessor withdrawal evidence")
        predecessor = withdrawn[0]
        stranded = predecessor.get("Node") in eligible - ready
        if not converged and stranded:
            # A predecessor stranded on a down eligible node can never confirm
            # its own shutdown; its desired state already records the
            # withdrawal. Only that exact case is excused: an unassigned or
            # rejected attempt (no node) or a task on a foreign node is not
            # withdrawal evidence and stays strict. The deploy's own
            # withdrawals happen on ready nodes and stay fully confirmed.
            continue
        predecessor_desired, predecessor_state = _task_state(str(predecessor["ID"])[:12])
        allowed_predecessor_states = {"shutdown", "complete"}
        if not converged and predecessor.get("Node") in eligible:
            # A recovered baseline may follow a terminal task failure from an
            # earlier outage. This deploy's own withdrawal proof stays strict.
            allowed_predecessor_states.add("failed")
        if (
            predecessor_desired != "shutdown"
            or predecessor_state not in allowed_predecessor_states
        ):
            raise RuntimeError("Swarm task history contradicts the start-first rollout")
    network = subprocess.run(
        ["docker", "network", "inspect", "--verbose", "dokploy-network"],
        check=True,
        capture_output=True,
        text=True,
    )
    network_details = json.loads(network.stdout)[0]
    vip_tasks = (network_details.get("Services") or {}).get(app_name, {}).get("Tasks") or []
    vip_by_task = {
        str(task.get("Name", "")).rsplit(".", 1)[-1]: task for task in vip_tasks
    }
    current_ids = {str(row["ID"]) for row in current}
    if set(vip_by_task) != current_ids:
        raise RuntimeError("native Swarm VIP membership differs from running tasks")
    instances: set[str] = set()
    for row in current:
        task_id = str(row["ID"])
        runtime = _task_runtime(task_id[:12], str(network_details["Id"]))
        if (
            runtime["image"] != image
            or not (rollout_policy._labels(revision).items() <= runtime["labels"].items())
            or runtime["addresses"] != {vip_by_task[task_id].get("EndpointIP")}
        ):
            raise RuntimeError("Crawl4AI task identity drifted")
        health_url = f"http://{next(iter(runtime['addresses']))}:11235/health"
        health = rollout_http._request_json(health_url, ingress_container=ingress_container)
        if not _exact_health(health, revision) or health["instance"] != runtime["container"]:
            raise RuntimeError("Crawl4AI task failed direct overlay readiness")
        instances.add(runtime["container"])
    if len(instances) != rollout_policy.REPLICAS:
        raise RuntimeError("Crawl4AI task instances are not unique")
    return {
        "tasks": sorted(str(row["ID"])[:12] for row in current),
        "nodes": sorted(nodes),
        "instances": sorted(instances),
    }


def _verify_public(revision: str, instances: list[str]) -> dict[str, list[str]]:
    expected = set(instances)
    observed = {url: set() for url in rollout_policy.HEALTH_URLS}
    # Client connection churn alone does not control Traefik's upstream reuse;
    # the gate passes only after each route returns every authoritative identity.
    for url, seen in observed.items():
        for _ in range(rollout_policy.PUBLIC_PROOF_ATTEMPTS):
            health = rollout_http._request_json(f"{url}?verify={uuid.uuid4()}")
            if not _exact_health(health, revision):
                raise RuntimeError(
                    "public Crawl4AI returned unhealthy or wrong-revision health"
                )
            instance = str(health["instance"])
            if instance not in expected:
                raise RuntimeError("public Crawl4AI returned an unknown task instance")
            seen.add(instance)
            if seen == expected:
                break
        if seen != expected:
            raise RuntimeError("each public domain did not observe every authoritative task")
    return {url: sorted(seen) for url, seen in observed.items()}


def _task_proof_path() -> Path:
    return Path(f"{os.environ['ROLLOUT_MONITOR_PATH']}.tasks.json")
