import copy
import json

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout
from conftest import CANDIDATE, ELIGIBLE_NODES, REVISION, _candidate_runtimes, _deploy_env, _wire_verify_tasks, application, health, service_spec


def _wire_noop_deploy_with_real_verify_tasks(monkeypatch, tmp_path, rows, runtimes, ready):
    # Unlike the no-op rerun tests above, this wires the real _verify_tasks
    # through deploy() so the lenient vs. strict converged branch is exercised
    # end-to-end, not stubbed away.
    candidate = CANDIDATE
    state = application(image=candidate, revision=REVISION)
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_dokploy, "_wait_deployment", lambda *_args: pytest.fail("no deployment may be submitted")
    )
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    _wire_verify_tasks(monkeypatch, rows, runtimes, revision=REVISION)

    def request_json(url, *_args, ingress_container=None, **_kwargs):
        # Per-task overlay probes carry the task's overlay IP; public probes
        # fall through to candidate health (the baseline is the candidate here).
        for runtime in runtimes.values():
            if next(iter(runtime["addresses"])) in url:
                return health(runtime["container"], revision=REVISION)
        return health(revision=REVISION)

    monkeypatch.setattr(rollout_http, "_request_json", request_json)


def test_deploy_noop_rerun_still_requires_replicas_ready_nodes(monkeypatch, tmp_path):
    # Exempting the no-op rerun from the start-first spare gate does not exempt
    # it from capacity: with fewer than REPLICAS Ready eligible nodes it cannot
    # place every replica, so the post-rollout < REPLICAS bar must still fire.
    candidate = "registry.example/crawl4ai@sha256:candidate"
    state = application(image=candidate, revision=REVISION)
    ready = frozenset({"haiku-5", "haiku-18"})
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_dokploy, "_wait_deployment", lambda *_args: pytest.fail("no deployment may be submitted")
    )
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: {
            "tasks": ["1", "2", "3"],
            "nodes": sorted(ready),
            "instances": ["a", "b", "c"],
        },
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    with pytest.raises(RuntimeError, match="not enough Ready eligible nodes to place every replica"):
        rollout.deploy()


def test_deploy_noop_rerun_proves_healed_colocation_in_place(monkeypatch, tmp_path, capsys):
    # A healed-but-not-respread fleet doubles a replica onto a surviving node
    # and strands a ghost on the down spare. The lenient baseline proof
    # tolerates that; the strict converged proof rejects it as a placement
    # fault. A no-op rerun performed no start-first placement, so its final
    # proof must be lenient and re-prove the live SHA in place.
    rows = [
        {"ID": "task1", "Name": "crawl4ai.1", "Node": "haiku-5",
         "DesiredState": "Running", "CurrentState": "Running 1m"},
        {"ID": "ghost1", "Name": "crawl4ai.1", "Node": "haiku-6",
         "DesiredState": "Shutdown", "CurrentState": "Running 12 hours ago"},
        {"ID": "task2", "Name": "crawl4ai.2", "Node": "haiku-5",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "old2", "Name": "crawl4ai.2", "Node": "haiku-9",
         "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
        {"ID": "task3", "Name": "crawl4ai.3", "Node": "haiku-9",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "old3", "Name": "crawl4ai.3", "Node": "haiku-18",
         "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
    ]
    runtimes = _candidate_runtimes()
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_noop_deploy_with_real_verify_tasks(monkeypatch, tmp_path, rows, runtimes, ready)

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == CANDIDATE
    assert receipt["revision"] == REVISION
    # The lenient proof accepted the healed co-location: two replicas sit on
    # haiku-5. A strict converged proof would have raised before reaching here.
    assert receipt["nodes"] == ["haiku-5", "haiku-9"]


def test_deploy_noop_rerun_proves_a_failed_predecessor_in_place(monkeypatch, tmp_path, capsys):
    # A predecessor that crashed during an earlier drain leaves a
    # DesiredState=shutdown / Status.State=failed task on a ready eligible
    # node. The lenient baseline proof tolerates that "failed" exemption; the
    # strict converged proof rejects it. A no-op rerun made no withdrawal of
    # its own, so its final proof must stay lenient and re-prove in place.
    rows = [
        {"ID": "task1", "Name": "crawl4ai.1", "Node": "haiku-5",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "failed1", "Name": "crawl4ai.1", "Node": "haiku-9",
         "DesiredState": "Shutdown", "CurrentState": "Failed 30m ago"},
        {"ID": "task2", "Name": "crawl4ai.2", "Node": "haiku-9",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "old2", "Name": "crawl4ai.2", "Node": "haiku-5",
         "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
        {"ID": "task3", "Name": "crawl4ai.3", "Node": "haiku-18",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "old3", "Name": "crawl4ai.3", "Node": "haiku-9",
         "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
    ]
    runtimes = _candidate_runtimes()
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_noop_deploy_with_real_verify_tasks(monkeypatch, tmp_path, rows, runtimes, ready)
    monkeypatch.setattr(
        rollout_tasks,
        "_task_state",
        lambda task: (
            ("running", "running") if task.startswith("task")
            else ("shutdown", "failed") if task.startswith("failed")
            else ("shutdown", "shutdown")
        ),
    )

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == CANDIDATE
    assert receipt["revision"] == REVISION
    assert receipt["nodes"] == ["haiku-18", "haiku-5", "haiku-9"]


def test_deploy_noop_rerun_passes_lenient_converged_to_the_candidate_proof(
    monkeypatch, tmp_path
):
    # The no-op rerun owes no converged obligation, so every _verify_tasks call
    # it makes (the baseline proof and both candidate proofs) must run lenient
    # (converged=False). A regression that defaults the candidate proof back
    # to converged=True surfaces here as a True recording.
    calls = []

    def spy(_app, image, _revision, _ready, converged=True, placement=rollout_policy.PLACEMENT):
        calls.append((image, converged))
        return {
            "tasks": ["1", "2", "3"],
            "nodes": sorted(ELIGIBLE_NODES),
            "instances": ["a", "b", "c"],
        }

    candidate = CANDIDATE
    state = application(image=candidate, revision=REVISION)
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_dokploy, "_wait_deployment", lambda *_args: pytest.fail("no deployment may be submitted")
    )
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", spy)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    rollout.deploy()

    assert calls, "the no-op rerun must still take its candidate proof"
    assert all(converged is False for _image, converged in calls), calls
