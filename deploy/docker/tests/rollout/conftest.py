"""Shared constants, fixtures, and record builders for the rollout test files."""

import copy
import json
import subprocess

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout


BASELINE = "registry.example/crawl4ai@sha256:baseline"


CANDIDATE = "registry.example/crawl4ai@sha256:candidate"


REVISION = "a" * 40


ELIGIBLE_NODES = frozenset({"haiku-5", "haiku-6", "haiku-9", "haiku-18"})
# Extra nodes let a one-per-node cap hold every replica plus one start-first batch.
ELIGIBLE_NODES |= {
    f"node-{n}" for n in range(rollout_policy.REPLICAS + rollout_policy.PARALLELISM - len(ELIGIBLE_NODES))
}


SLOTS = range(1, rollout_policy.REPLICAS + 1)


INSTANCES = [f"instance{slot}" for slot in SLOTS]


def task_proof(nodes=None, tag=""):
    """A _verify_tasks result naming REPLICAS tasks and instances; tag sets them apart."""
    if nodes is None:
        nodes = sorted(ELIGIBLE_NODES)[:rollout_policy.REPLICAS]
    return {
        "tasks": [f"{tag}{slot}" for slot in SLOTS],
        "nodes": sorted(nodes),
        "instances": [f"{tag}instance{slot}" for slot in SLOTS],
    }


@pytest.fixture(autouse=True)
def deploy_event_boundary(monkeypatch, tmp_path, request):
    if not request.node.name.startswith("test_deploy"):
        return

    class Collector:
        nodes = {}

        def __init__(self, *_args):
            pass

        def start(self):
            return []

        def finish(self, _candidate):
            return None

        def refresh(self):
            pass

        def check(self):
            pass

    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout.withdrawals, "Withdrawals", Collector)
    monkeypatch.setattr(rollout.withdrawals, "census", lambda _app: [])
    monkeypatch.setattr(rollout.withdrawals, "ready_nodes", lambda: {})


def application(image=BASELINE, revision="baseline", placement=rollout_policy.PLACEMENT):
    return {
        "applicationId": "app",
        "appName": "crawl4ai",
        "sourceType": "docker",
        "dockerImage": image,
        "labelsSwarm": rollout_policy._labels(revision),
        "env": (
            f"LLM_PROVIDER={rollout_policy.LLM_PROVIDER}\n"
            f"LLM_BASE_URL={rollout_policy.LLM_BASE_URL}\n"
            "LLM_API_KEY=secret"
        ),
        "replicas": rollout_policy.REPLICAS,
        "healthCheckSwarm": copy.deepcopy(rollout_policy.HEALTHCHECK),
        "placementSwarm": copy.deepcopy(placement),
        "endpointSpecSwarm": {"Mode": "vip", "Ports": []},
        "updateConfigSwarm": {
            "Parallelism": rollout_policy.PARALLELISM,
            "Delay": rollout_policy.DELAY_NS,
            "FailureAction": "rollback",
            "Monitor": rollout_policy.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "rollbackConfigSwarm": {
            "Parallelism": rollout_policy.PARALLELISM,
            "Delay": rollout_policy.DELAY_NS,
            "FailureAction": "pause",
            "Monitor": rollout_policy.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "stopGracePeriodSwarm": rollout_policy.STOP_GRACE_NS,
        "cpuReservation": "500000000",
        "cpuLimit": "2000000000",
        "memoryReservation": "1073741824",
        "memoryLimit": "4294967296",
    }


def service_spec(image=BASELINE, revision="baseline", placement=rollout_policy.PLACEMENT):
    return {
        "TaskTemplate": {
            "ContainerSpec": {
                "Image": image,
                "Labels": rollout_policy._labels(revision),
                "Env": [
                    f"LLM_PROVIDER={rollout_policy.LLM_PROVIDER}",
                    f"LLM_BASE_URL={rollout_policy.LLM_BASE_URL}",
                    "LLM_API_KEY=secret",
                ],
                "Healthcheck": copy.deepcopy(rollout_policy.HEALTHCHECK),
                "StopGracePeriod": rollout_policy.STOP_GRACE_NS,
            },
            "Placement": copy.deepcopy(placement),
            "Resources": copy.deepcopy(rollout_policy.RESOURCES),
        },
        "Mode": {"Replicated": {"Replicas": rollout_policy.REPLICAS}},
        "UpdateConfig": {
            "Parallelism": rollout_policy.PARALLELISM,
            "Delay": rollout_policy.DELAY_NS,
            "FailureAction": "rollback",
            "Monitor": rollout_policy.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "RollbackConfig": {
            "Parallelism": rollout_policy.PARALLELISM,
            "Delay": rollout_policy.DELAY_NS,
            "FailureAction": "pause",
            "Monitor": rollout_policy.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "EndpointSpec": {"Mode": "vip"},
    }


def health(instance="container123", revision=REVISION):
    return {
        "instance": instance,
        "revision": revision,
        "status": "ok",
        "components": {"api": "ready", "redis": "ready"},
    }


def _candidate_runtimes():
    # On a no-op rerun the candidate already is the baseline, so one set of
    # runtimes satisfies both the baseline proof and the candidate proof.
    return {
        f"task{index}": {
            "container": f"container{index}",
            "labels": rollout_policy._labels(REVISION),
            "image": CANDIDATE,
            "addresses": {f"10.0.1.{index}"},
        }
        for index in SLOTS
    }


def _healed_baseline_rows():
    # A node died holding slot 1; Swarm healed the replica onto the Redis node,
    # and the ghost task on the dead node can never confirm its shutdown.
    # Every other slot runs on its own Ready node beside its withdrawn
    # predecessor on the next one, leaving one Ready node spare.
    rows = [
        {"ID": "task1", "Name": "crawl4ai.1", "Node": rollout_policy.REDIS_NODE,
         "DesiredState": "Running", "CurrentState": "Running 1m"},
        {"ID": "ghost1", "Name": "crawl4ai.1", "Node": "haiku-6",
         "DesiredState": "Shutdown", "CurrentState": "Running 12 hours ago"},
    ]
    nodes = sorted(ELIGIBLE_NODES - {rows[1]["Node"], rollout_policy.REDIS_NODE})
    for slot in SLOTS[1:]:
        rows += [
            {"ID": f"task{slot}", "Name": f"crawl4ai.{slot}", "Node": nodes[slot - 2],
             "DesiredState": "Running", "CurrentState": "Running 1h"},
            {"ID": f"old{slot}", "Name": f"crawl4ai.{slot}", "Node": nodes[slot - 1],
             "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
        ]
    runtimes = {
        f"task{index}": {
            "container": f"container{index}",
            "labels": rollout_policy._labels("baseline"),
            "image": BASELINE,
            "addresses": {f"10.0.1.{index}"},
        }
        for index in SLOTS
    }
    return rows, runtimes


# The Ready set after slot 1's node died; the healed baseline runs on it.
HEALED_READY = ELIGIBLE_NODES - {_healed_baseline_rows()[0][1]["Node"]}


def _healed_colocation_rows():
    # The healed slot 1 doubled up beside slot 2 instead of moving to a
    # spare; its ghost is stranded on the dead node.
    rows, runtimes = _healed_baseline_rows()
    rows[0]["Node"] = rows[2]["Node"]
    return rows, runtimes


def _running_nodes(rows):
    return sorted({row["Node"] for row in rows if row["DesiredState"] == "Running"})


def _wire_verify_tasks(monkeypatch, rows, runtimes, revision="baseline"):
    current = [
        row for row in rows if str(row.get("DesiredState", "")).lower() == "running"
    ]
    network = [{
        "Id": "network",
        "Services": {
            "crawl4ai": {
                "Tasks": [
                    {
                        "Name": f"{row['Name']}.{row['ID']}",
                        "EndpointIP": next(iter(runtimes[row["ID"]]["addresses"])),
                    }
                    for row in current
                ]
            }
        },
    }]

    def run(command, **_kwargs):
        if command[1:3] == ["service", "ps"]:
            return subprocess.CompletedProcess(
                command, 0, "\n".join(json.dumps(row) for row in rows), ""
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(network), "")

    monkeypatch.setattr(rollout.subprocess, "run", run)
    monkeypatch.setattr(rollout_tasks, "_verify_ingress_container", lambda: "a" * 64)
    monkeypatch.setattr(
        rollout_tasks,
        "_task_state",
        lambda task: (
            ("running", "running") if task.startswith("task")
            else ("shutdown", "running") if task.startswith("ghost")
            else ("shutdown", "rejected") if task.startswith("reject")
            else ("shutdown", "shutdown")
        ),
    )
    monkeypatch.setattr(rollout_tasks, "_task_runtime", lambda task, _network: runtimes[task])
    monkeypatch.setattr(
        rollout_http,
        "_request_json",
        lambda url, *_args, ingress_container: health(
            next(r["container"] for r in runtimes.values() if next(iter(r["addresses"])) in url),
            revision=revision,
        ),
    )


def _deploy_env(monkeypatch):
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setenv("IMAGE", "registry.example/crawl4ai")
    monkeypatch.setenv("IMAGE_DIGEST", "sha256:candidate")
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    # Seeding the rollback source is an infrastructure precondition orthogonal
    # to what these tests assert; it has its own tests below.
    monkeypatch.setattr(rollout_dokploy, "_ensure_rollback_source", lambda _image: None)
