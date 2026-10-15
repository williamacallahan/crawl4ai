"""Finite, per-node Engine event journals for one Swarm rollout."""

from __future__ import annotations

import inspect
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import TypedDict


class Task(TypedDict):
    task: str
    container: str
    node: str


class Event(TypedDict):
    action: str
    task: str
    container: str
    exit: str


class Journal(TypedDict):
    app: str
    nodes: dict[str, str]
    baseline: list[Task]
    candidate: list[Task]
    events: dict[str, list[Event]]


def _collect(app: str, arm: str, finish: str, seconds: int) -> None:
    # This function runs in the immutable application image on each exact node.
    import http.client
    import json
    import socket
    import time

    def request(method, path, payload=None, end_marker=None):
        connection = http.client.HTTPConnection("localhost", timeout=15)
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(15)
        connection.sock.connect("/var/run/docker.sock")
        connection.request(method, path, None if payload is None else json.dumps(payload),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        try:
            if end_marker is None:
                body = response.read(4_194_305)
            else:
                lines = []
                for _ in range(256):
                    line = response.readline(65537)
                    if not line or len(line) > 65536:
                        raise RuntimeError("Engine journal lost its checkpoint anchor: stream ended")
                    lines.append(line)
                    record = json.loads(line)
                    if (record["Type"] == "container"
                            and record["Actor"]["ID"] == own_id
                            and record["Actor"]["Attributes"].get("execID") == end_marker):
                        break
                else:
                    raise RuntimeError("Engine journal checkpoint left the native cache")
                body = b"".join(lines)
        finally:
            connection.close()
        if response.status not in {200, 201, 204} or len(body) > 4_194_304:
            raise RuntimeError(f"Engine journal request failed: {method} {response.status}")
        return body

    own = json.loads(request("GET", "/containers/" + socket.gethostname() + "/json"))
    own_id = own["Id"]
    deadline = time.monotonic() + seconds
    events = []
    armed = False
    previous = None
    count = 0
    while time.monotonic() < deadline:
        # Exec creation logs synchronously. Never start it: its configuration
        # belongs to this already-owned collector and creates no process.
        marker = json.loads(request("POST", "/containers/" + own_id + "/exec",
            {"Cmd": ["python", "-c", "pass"], "AttachStdout": False,
             "AttachStderr": False, "AttachStdin": False}))["Id"]
        # Read the already-cached anchor immediately. Waiting for a wall-clock
        # cutoff let unrelated engine traffic evict it before the snapshot.
        cached = [json.loads(line) for line in request(
            "GET", "/events?since=1", end_marker=marker
        ).splitlines()]
        # Engine appends/evicts under one mutex. Both anchors in the same
        # retrospective snapshot prove the entire intervening insertion window.
        anchors = [index for index, record in enumerate(cached)
                   if record["Type"] == "container" and record["Action"].startswith("exec_create: ")
                   and record["Actor"]["ID"] == own_id
                   and record["Actor"]["Attributes"].get("execID") == marker]
        prior = [index for index, record in enumerate(cached)
                 if record["Type"] == "container" and record["Action"].startswith("exec_create: ")
                 and record["Actor"]["ID"] == own_id
                 and record["Actor"]["Attributes"].get("execID") == previous]
        if len(anchors) != 1 or (previous is not None and (len(prior) != 1 or prior[0] >= anchors[0])):
            raise RuntimeError("Engine event cache lost a withdrawal journal anchor")
        window = cached[prior[0] + 1:anchors[0]] if previous is not None else []
        if previous is None:
            print(json.dumps({"ready": True}), flush=True)
        previous = marker
        finished = False
        for event in window:
            if event["Type"] != "container":
                continue
            attributes = event["Actor"]["Attributes"]
            service = attributes.get("com.docker.swarm.service.name")
            action = event["Action"]
            if service not in {app, arm, finish}:
                continue
            if action not in {"create", "die", "oom"}:
                continue
            count += 1
            if count > 128:
                raise RuntimeError("Engine journal exceeded its 128-event bound")
            if service == arm and action == "create":
                if armed:
                    raise RuntimeError("arming marker was replaced")
                armed = True
                print(json.dumps({"armed": True}), flush=True)
            elif service == finish and action == "create":
                if not armed:
                    raise RuntimeError("final marker arrived before arming")
                finished = True
            elif service == app:
                if not armed:
                    raise RuntimeError("application changed before journal arming")
                if action in {"create", "die", "oom"}:
                    events.append({"action": action, "task": attributes["com.docker.swarm.task.id"],
                                   "container": event["Actor"]["ID"], "exit": attributes.get("exitCode", "")})
        if finished:
            print(json.dumps({"complete": True, "events": events}), flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError("Engine withdrawal journal expired")


def _docker(*arguments: str) -> str:
    return subprocess.run(["docker", *arguments], check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def ready_nodes() -> dict[str, str]:
    identifiers = _docker("node", "ls", "-q").splitlines()
    if not identifiers:
        raise RuntimeError("Swarm has no nodes")
    nodes = json.loads(_docker("node", "inspect", *identifiers))
    return {node["ID"]: node["Description"]["Hostname"] for node in nodes
            if node["Spec"].get("Labels", {}).get("crawl4ai-eligible") == "true"
            and node["Status"]["State"] == "ready"
            and node["Spec"]["Availability"] == "active"}


def census(app: str) -> list[Task]:
    identifiers = _docker("service", "ps", "--no-trunc", "--filter", "desired-state=running", "-q", app).splitlines()
    tasks = json.loads(_docker("inspect", *identifiers)) if identifiers else []
    return sorted([
        {"task": task["ID"], "node": task["NodeID"],
         "container": task["Status"]["ContainerStatus"]["ContainerID"]}
        for task in tasks if task["DesiredState"] == "running"
        and task["Status"]["State"] == "running"
    ], key=lambda task: task["task"])


def validate(journal: Journal, app: str, baseline: list[Task], candidate: list[Task]) -> None:
    if journal["app"] != app or journal["baseline"] != baseline or journal["candidate"] != candidate:
        raise RuntimeError("withdrawal journal identity differs from the rollout census")
    nodes = journal["nodes"]
    if not nodes or set(journal["events"]) != set(nodes):
        raise RuntimeError("withdrawal journal does not cover every frozen node")
    if any(task["node"] not in nodes for task in baseline + candidate):
        raise RuntimeError("rollout task escaped frozen journal coverage")
    baseline_by_container = {task["container"]: task for task in baseline}
    candidate_by_container = {task["container"]: task for task in candidate}
    if len(baseline_by_container) != len(baseline) or len(candidate_by_container) != len(candidate):
        raise RuntimeError("rollout census contains duplicate container identities")
    created: dict[str, Task] = {}
    died: set[str] = set()
    for node, events in journal["events"].items():
        if len(events) > 128:
            raise RuntimeError("withdrawal journal exceeded its event bound")
        for event in events:
            container = event["container"]
            identity: Task = {"container": container, "task": event["task"], "node": node}
            if event["action"] == "create":
                if container in created or container in baseline_by_container:
                    raise RuntimeError("withdrawal journal repeats a container create")
                created[container] = identity
            elif event["action"] == "die":
                if event["exit"] != "0" or container in died:
                    raise RuntimeError("rollout container did not exit cleanly")
                if (created.get(container) or baseline_by_container.get(container)) != identity:
                    raise RuntimeError("withdrawal journal has an unknown container exit")
                died.add(container)
            else:
                raise RuntimeError("rollout container had an OOM or unknown event")
    if not set(baseline_by_container) <= died:
        raise RuntimeError("baseline container withdrawal was not confirmed")
    if any(created.get(container) != identity for container, identity in candidate_by_container.items()):
        raise RuntimeError("candidate container has no observed create event")
    if set(candidate_by_container) & died or set(created) - died != set(candidate_by_container):
        raise RuntimeError("rollout attempt is neither live candidate nor cleanly withdrawn")


class Withdrawals:
    def __init__(self, app: str, image: str, path: Path, seconds: int):
        self.app, self.image, self.path, self.seconds = app, image, path, seconds
        self.prefix = f"withdrawal-{uuid.uuid4().hex[:20]}"
        self.nodes = ready_nodes()
        self.services: list[str] = []
        self.collectors: dict[str, tuple[str, str, str]] = {}
        self.baseline: list[Task] = []
        self.deadline = time.monotonic() + seconds

    def _save(self) -> None:
        self.path.write_text(json.dumps({"owner": self.prefix, "services": self.services}, separators=(",", ":")))

    def _create(self, name: str, node: str, program: str, collector: bool = False) -> str:
        # Save the exact attempted name before a possibly ambiguous create.
        self.services.append(name)
        self._save()
        arguments = ["service", "create", "--detach", "--with-registry-auth", "--name", name,
                     "--constraint", f"node.id=={node}", "--restart-condition", "none",
                     "--read-only", "--user", "0", "--no-healthcheck",
                     "--entrypoint", "python", "--stop-grace-period", "5s",
                     "--limit-cpu", "0.1", "--limit-memory", "64m", "--limit-pids", "32",
                     "--label", f"crawl4ai.withdrawal.owner={self.prefix}"]
        if collector:
            arguments += ["--mount", "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock"]
        return _docker(*arguments, self.image, "-u", "-c", program)

    def _task(self, service: str, pending: bool = False) -> dict:
        identifiers = _docker("service", "ps", "--no-trunc", "-q", service).splitlines()
        if not identifiers and pending:
            return {}
        if len(identifiers) != 1:
            raise RuntimeError("withdrawal collector is missing or was replaced")
        return json.loads(_docker("inspect", identifiers[0]))[0]

    def _marker(self, service: str, node: str) -> None:
        identifier = None
        while time.monotonic() < self.deadline:
            task = self._task(service, pending=identifier is None)
            if task:
                if identifier is not None and task["ID"] != identifier:
                    raise RuntimeError("withdrawal marker was replaced")
                identifier = task["ID"]
                status = task["Status"]
                if task.get("NodeID") and task["NodeID"] != node:
                    raise RuntimeError("withdrawal marker escaped its exact node")
                if status["State"] == "complete":
                    if status.get("Err") or status["ContainerStatus"].get("ExitCode") != 0:
                        raise RuntimeError("withdrawal marker did not exit cleanly")
                    return
                if status["State"] in {"failed", "rejected", "shutdown"}:
                    raise RuntimeError("withdrawal marker failed")
            time.sleep(1)
        raise TimeoutError("withdrawal marker did not complete")

    def _logs(self, service: str) -> list[dict]:
        return [json.loads(line) for line in _docker("service", "logs", "--raw", service).splitlines() if line]

    def start(self) -> list[Task]:
        if not self.nodes:
            raise RuntimeError("no Ready eligible nodes for withdrawal journals")
        source = inspect.getsource(_collect)
        for node in self.nodes:
            name = f"{self.prefix}-{node}"
            program = source + f"\n_collect({self.app!r}, {name + '-arm'!r}, {name + '-end'!r}, {self.seconds})"
            service = self._create(name, node, program, True)
            self.collectors[node] = (service, "", "")
        for node, (service, _, _) in list(self.collectors.items()):
            while True:
                task = self._task(service, pending=True)
                if not task:
                    if time.monotonic() >= self.deadline:
                        raise RuntimeError("withdrawal collector was not scheduled")
                    time.sleep(1)
                    continue
                state = task["Status"]["State"]
                if state == "running":
                    self.collectors[node] = (service, task["ID"], task["Status"]["ContainerStatus"]["ContainerID"])
                    break
                if state in {"failed", "rejected", "complete", "shutdown"} or time.monotonic() >= self.deadline:
                    raise RuntimeError("withdrawal collector did not start")
                time.sleep(1)
            while not any(row.get("ready") for row in self._logs(service)):
                self._guard(node)
                time.sleep(1)
            # The collector has already observed its initial native cache anchor.
            marker = f"{self.prefix}-{node}-arm"
            marker_service = self._create(marker, node, "pass")
            while not any(row.get("armed") for row in self._logs(service)):
                self._guard(node)
                time.sleep(1)
            self._marker(marker_service, node)
        self.baseline = census(self.app)
        if ready_nodes() != self.nodes:
            raise RuntimeError("eligible nodes changed while arming withdrawal journals")
        return self.baseline

    def _guard(self, node: str, complete: bool = False) -> None:
        service, identifier, container = self.collectors[node]
        task = self._task(service)
        status = task["Status"]
        if (task["ID"] != identifier or task["NodeID"] != node
                or status.get("ContainerStatus", {}).get("ContainerID") != container
                or status.get("Err") or time.monotonic() >= self.deadline
                or status["State"] not in ({"running", "complete"} if complete else {"running"})):
            raise RuntimeError("withdrawal collector failed, expired, or was replaced")
        if status["State"] == "complete" and status["ContainerStatus"].get("ExitCode") != 0:
            raise RuntimeError("withdrawal collector exited unsuccessfully")

    def finish(self, candidate: list[Task]) -> Journal:
        if ready_nodes() != self.nodes or census(self.app) != candidate:
            raise RuntimeError("rollout census or node coverage changed before final barrier")
        for node in self.nodes:
            self._guard(node)
            marker = self._create(f"{self.prefix}-{node}-end", node, "pass")
            self._marker(marker, node)
        events = {}
        for node, (service, _, _) in self.collectors.items():
            while True:
                self._guard(node, complete=True)
                records = [row for row in self._logs(service) if row.get("complete")]
                if records:
                    if len(records) != 1:
                        raise RuntimeError("withdrawal collector produced duplicate final records")
                    events[node] = records[0]["events"]
                    if self._task(service)["Status"]["State"] == "complete":
                        break
                time.sleep(1)
        journal: Journal = {"app": self.app, "nodes": self.nodes, "baseline": self.baseline,
                            "candidate": candidate, "events": events}
        self.path.with_suffix(".evidence.json").write_text(json.dumps(journal, separators=(",", ":")))
        validate(journal, self.app, self.baseline, candidate)
        if census(self.app) != candidate or ready_nodes() != self.nodes:
            raise RuntimeError("rollout census changed across withdrawal barrier")
        return journal

def cleanup(path: Path) -> None:
    if not path.exists():
        return
    record = json.loads(path.read_text())
    for service in list(record.get("services", [])):
        # Removal is restricted to names saved by this invocation, not a search.
        found = subprocess.run(["docker", "service", "inspect", service], capture_output=True,
                               text=True, timeout=30)
        if found.returncode:
            if "no such service" not in found.stderr.lower():
                raise RuntimeError("could not inspect withdrawal service for cleanup")
        else:
            specification = json.loads(found.stdout)[0]["Spec"]
            owner = specification.get("Labels", {}).get("crawl4ai.withdrawal.owner")
            if (owner != record.get("owner") or specification["Name"] != service
                    or not service.startswith(owner + "-")):
                raise RuntimeError("withdrawal cleanup service ownership changed")
            _docker("service", "rm", service)
        record["services"].remove(service)
        path.write_text(json.dumps(record, separators=(",", ":")))
    path.unlink()
