import json
import subprocess

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import verify_rollout as rollout
from conftest import BASELINE, CANDIDATE, ELIGIBLE_NODES, REVISION, _healed_baseline_rows, _wire_verify_tasks, health


@pytest.mark.parametrize("retain_predecessors", [True, False])
def test_verify_tasks_proves_each_overlay_backend(monkeypatch, retain_predecessors):
    rows = [
        {
            "ID": f"task{index}",
            "Name": f"crawl4ai.{index}",
            "Node": node,
            "DesiredState": "Running",
            "CurrentState": "Running 1m",
        }
        for index, node in enumerate(("haiku-6", "haiku-5", "haiku-9"), 1)
    ]
    predecessors = [
        {
            "ID": f"old{index}",
            "Name": f"crawl4ai.{index}",
            "Node": node,
            "DesiredState": "Shutdown",
            "CurrentState": "Shutdown 1m",
        }
        for index, node in enumerate(("haiku-5", "haiku-9", "haiku-18"), 1)
    ]
    if retain_predecessors:
        rows += predecessors
    runtimes = {
        f"task{index}": {
            "container": f"container{index}",
            "labels": rollout_policy._labels(REVISION),
            "image": CANDIDATE,
            "addresses": {f"10.0.1.{index}"},
        }
        for index in range(1, 4)
    }
    # Dokploy's own stamp beside the record's labels must not read as drift.
    runtimes["task1"]["labels"]["dokploy.deployment.id"] = "row"
    network = [{
        "Id": "network",
        "Services": {
            "crawl4ai": {
                "Tasks": [
                    {
                        "Name": f"{row['Name']}.{row['ID']}",
                        "EndpointIP": next(iter(runtimes[row["ID"]]["addresses"])),
                    }
                    for row in rows[:3]
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
    ingress_calls = []
    monkeypatch.setattr(
        rollout_tasks, "_verify_ingress_container", lambda: ingress_calls.append(None) or "a" * 64
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_task_state",
        lambda task: (
            ("running", "running")
            if task.startswith("task")
            else ("shutdown", "shutdown")
        ),
    )
    monkeypatch.setattr(
        rollout_tasks, "_task_runtime", lambda task, _network: runtimes[task]
    )
    direct_probe = []

    def direct_health(url, *, ingress_container):
        direct_probe.append((url, ingress_container))
        return health(
            next(runtime["container"] for runtime in runtimes.values() if next(iter(runtime["addresses"])) in url)
        )

    monkeypatch.setattr(rollout_http, "_request_json", direct_health)
    proof = rollout_tasks._verify_tasks("crawl4ai", CANDIDATE, REVISION, (ELIGIBLE_NODES, ELIGIBLE_NODES))
    assert proof["nodes"] == ["haiku-5", "haiku-6", "haiku-9"]
    assert len(proof["instances"]) == 3
    assert ingress_calls == [None]
    assert len(direct_probe) == rollout_policy.REPLICAS
    assert {url for url, _container in direct_probe} == {
        f"http://10.0.1.{index}:11235/health" for index in range(1, 4)
    }
    assert {container for _url, container in direct_probe} == {"a" * 64}


@pytest.mark.parametrize("drift", ["missing", "extra"])
def test_verify_tasks_rejects_vip_membership_drift(monkeypatch, drift):
    rows, runtimes = _healed_baseline_rows()
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    original_run = subprocess.run

    def run(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[1:4] == ["network", "inspect", "--verbose"]:
            network = json.loads(result.stdout)
            tasks = network[0]["Services"]["crawl4ai"]["Tasks"]
            if drift == "missing":
                tasks.pop()
            else:
                tasks.append({"Name": "crawl4ai.4.foreign", "EndpointIP": "10.0.1.99"})
            return subprocess.CompletedProcess(command, 0, json.dumps(network), "")
        return result

    monkeypatch.setattr(rollout.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="VIP membership"):
        rollout_tasks._verify_tasks(
            "crawl4ai",
            BASELINE,
            "baseline",
            (ELIGIBLE_NODES, frozenset({"haiku-5", "haiku-9", "haiku-18"})),
            False,
        )


@pytest.mark.parametrize("drift", ["missing", "changed"])
def test_verify_tasks_rejects_record_label_drift_beside_control_plane_labels(
    monkeypatch, drift
):
    rows, runtimes = _healed_baseline_rows()
    labels = runtimes["task2"]["labels"]
    labels["dokploy.deployment.id"] = "row"
    if drift == "missing":
        del labels["otel.logs.enabled"]
    else:
        labels["otel.service.version"] = "other"
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    with pytest.raises(RuntimeError, match="identity drifted"):
        rollout_tasks._verify_tasks(
            "crawl4ai",
            BASELINE,
            "baseline",
            (ELIGIBLE_NODES, frozenset({"haiku-5", "haiku-9", "haiku-18"})),
            False,
        )


@pytest.mark.parametrize("retain_terminal_predecessor", [False, True])
def test_converged_rollout_ignores_a_never_started_assigned_predecessor(
    monkeypatch, retain_terminal_predecessor,
):
    rows, runtimes = _healed_baseline_rows()
    rows[1].update(ID="assigned1", CurrentState="Assigned 12 hours ago")
    if retain_terminal_predecessor:
        rows.insert(2, dict(rows[1], ID="old1", CurrentState="Shutdown 1m"))
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    task_state = rollout_tasks._task_state
    monkeypatch.setattr(rollout_tasks, "_task_state", lambda task: (
        ("shutdown", "assigned") if task == "assigned1" else task_state(task)
    ))

    proof = rollout_tasks._verify_tasks(
        "crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ELIGIBLE_NODES),
    )

    assert proof["tasks"] == ["task1", "task2", "task3"]
    assert proof["instances"] == ["container1", "container2", "container3"]
    assert len(proof["nodes"]) == 3 and set(proof["nodes"]) <= ELIGIBLE_NODES


def test_baseline_tolerates_a_ghost_predecessor_on_a_down_node(monkeypatch):
    rows, runtimes = _healed_baseline_rows()
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    proof = rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), False)
    assert sorted(proof["nodes"]) == ["haiku-18", "haiku-5", "haiku-9"]
    # The same state must still fail a converged proof: the ghost's shutdown
    # was never confirmed, so this deploy's own withdrawals stay strict.
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready))


@pytest.mark.parametrize("converged", [False, True])
def test_capped_rollout_rejects_colocation(monkeypatch, converged):
    rows, runtimes = _healed_baseline_rows()
    for row in rows:
        if row["ID"] == "task1":
            row["Node"] = "haiku-5"  # healed replica doubled up
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    proof = rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), False)
    assert sorted(proof["nodes"]) == ["haiku-5", "haiku-9"]
    with pytest.raises(RuntimeError, match="not on distinct eligible nodes"):
        rollout_tasks._verify_tasks(
            "crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), converged,
            placement=rollout_policy._CAPPED_PLACEMENT,
        )


def test_verify_tasks_does_not_retry_an_ingress_exec_failure(monkeypatch):
    rows, runtimes = _healed_baseline_rows()
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    ingress_calls = []
    monkeypatch.setattr(
        rollout_tasks, "_verify_ingress_container", lambda: ingress_calls.append(None) or "a" * 64
    )
    probes = []

    def fail(url, *, ingress_container):
        probes.append((url, ingress_container))
        raise rollout_http.CurlError("HTTP request failed: docker exec exit 1: refused", 1)

    monkeypatch.setattr(rollout_http, "_request_json", fail)

    with pytest.raises(rollout_http.CurlError, match="docker exec exit 1"):
        rollout_tasks._verify_tasks(
            "crawl4ai",
            BASELINE,
            "baseline",
            (ELIGIBLE_NODES, frozenset({"haiku-5", "haiku-9", "haiku-18"})),
            False,
        )

    assert ingress_calls == [None]
    assert probes == [("http://10.0.1.1:11235/health", "a" * 64)]


@pytest.mark.parametrize("placement", [rollout_policy.PLACEMENT, rollout_policy._SPREAD_PLACEMENT])
def test_uncapped_rollout_accepts_colocation_with_clean_withdrawals(monkeypatch, placement):
    rows, runtimes = _healed_baseline_rows()
    rows[0]["Node"] = "haiku-5"
    rows[1].update(ID="old1", CurrentState="Shutdown 1m")
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    proof = rollout_tasks._verify_tasks(
        "crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ELIGIBLE_NODES),
        placement=placement,
    )
    assert proof["nodes"] == ["haiku-5", "haiku-9"]
    assert len(proof["instances"]) == 3


@pytest.mark.parametrize("converged", [False, True])
def test_verify_tasks_rejects_a_current_task_on_an_unavailable_node(monkeypatch, converged):
    rows, runtimes = _healed_baseline_rows()
    ready = frozenset(ELIGIBLE_NODES - {"haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    with pytest.raises(RuntimeError, match="not on Ready eligible nodes"):
        rollout_tasks._verify_tasks(
            "crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), converged,
        )


def test_baseline_looks_past_a_rejection_on_an_eligible_node(monkeypatch):
    # A node that refuses a task leaves an attempt that never started, so it
    # withdrew nothing. The real predecessor behind it still has to confirm.
    rows, runtimes = _healed_baseline_rows()
    rows.insert(3, {"ID": "reject2", "Name": "crawl4ai.2", "Node": "haiku-18",
                    "DesiredState": "Shutdown", "CurrentState": "Rejected 1h"})
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), False)

    # The deploy's own census never looks past its own withdrawals.
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), True)


def test_baseline_accepts_a_slot_placed_only_by_scheduler_recovery(monkeypatch):
    # Every attempt in the slot was refused before one finally ran, so there is
    # no earlier rollout whose withdrawal could be proved.
    rows, runtimes = _healed_baseline_rows()
    rows[3] = {"ID": "reject2", "Name": "crawl4ai.2", "Node": "haiku-18",
               "DesiredState": "Shutdown", "CurrentState": "Rejected 1h"}
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), False)

    # The deploy's own census still demands the withdrawal it just performed.
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), True)


def test_baseline_keeps_unassigned_attempts_strict(monkeypatch):
    # A rejected scheduling attempt has no node; it is not a stranded ghost
    # and must not excuse the withdrawal proof.
    rows, runtimes = _healed_baseline_rows()
    rows.insert(1, {"ID": "reject1", "Name": "crawl4ai.1", "Node": "",
                    "DesiredState": "Shutdown", "CurrentState": "Rejected 1h"})
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ready), False)


def test_baseline_accepts_terminal_failed_predecessor_but_rollout_does_not(monkeypatch):
    rows, runtimes = _healed_baseline_rows()
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    monkeypatch.setattr(rollout_tasks, "_task_state", lambda task: (
        ("running", "running") if task.startswith("task")
        else ("shutdown", "failed") if task.startswith("ghost")
        else ("shutdown", "shutdown")
    ))
    proof = rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ELIGIBLE_NODES), False)
    assert len(proof["instances"]) == rollout_policy.REPLICAS
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (ELIGIBLE_NODES, ELIGIBLE_NODES))
