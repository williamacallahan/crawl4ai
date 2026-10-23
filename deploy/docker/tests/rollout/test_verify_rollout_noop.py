import copy
import json

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout
from conftest import (
    CANDIDATE, ELIGIBLE_NODES, HEALED_READY, REVISION, _candidate_runtimes, _deploy_env, _healed_baseline_rows,
    _healed_colocation_rows, _running_nodes, _wire_verify_tasks, application, health, service_spec, task_proof,
)


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
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", "2026-10-21T10:00:00Z"))
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


@pytest.mark.parametrize("placement", rollout_policy.PLACEMENTS)
@pytest.mark.parametrize("ready_count", range(len(ELIGIBLE_NODES) + 1))
def test_deploy_noop_rerun_still_requires_replicas_ready_nodes(
    monkeypatch, tmp_path, capsys, placement, ready_count
):
    # Exempting the no-op rerun from the start-first spare gate does not exempt
    # it from capacity: when the Ready eligible nodes cannot hold REPLICAS
    # tasks under the placement's per-node cap, the post-rollout bar must
    # still fire. Capacity short only of the spare batch must prove in place.
    candidate = "registry.example/crawl4ai@sha256:candidate"
    state = application(image=candidate, revision=REVISION, placement=placement)
    ready = frozenset(sorted(ELIGIBLE_NODES)[:ready_count])
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
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", "2026-10-21T10:00:00Z"))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION, placement))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: task_proof(ready))
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    if rollout_policy.placement_capacity(ready_count, placement) < rollout_policy.REPLICAS:
        with pytest.raises(RuntimeError, match="not enough Ready eligible nodes to place every replica"):
            rollout.deploy()
    else:
        rollout.deploy()
        receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert receipt["deploymentId"] is None
        assert receipt["nodes"] == sorted(ready)


def test_deploy_noop_rerun_proves_healed_colocation_in_place(monkeypatch, tmp_path, capsys):
    # A healed-but-not-respread fleet doubles a replica onto a surviving node
    # and strands a ghost on the down spare. The lenient baseline proof
    # tolerates that; the strict converged proof rejects it as a placement
    # fault. A no-op rerun performed no start-first placement, so its final
    # proof must be lenient and re-prove the live SHA in place.
    rows, _baseline_runtimes = _healed_colocation_rows()
    runtimes = _candidate_runtimes()
    _wire_noop_deploy_with_real_verify_tasks(monkeypatch, tmp_path, rows, runtimes, HEALED_READY)

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == CANDIDATE
    assert receipt["revision"] == REVISION
    # The lenient proof accepted the healed co-location: two replicas sit on
    # haiku-5. A strict converged proof would have raised before reaching here.
    assert receipt["nodes"] == _running_nodes(rows)
    assert len(receipt["nodes"]) == rollout_policy.REPLICAS - 1


def test_deploy_noop_rerun_proves_a_failed_predecessor_in_place(monkeypatch, tmp_path, capsys):
    # A predecessor that crashed during an earlier drain leaves a
    # DesiredState=shutdown / Status.State=failed task on a ready eligible
    # node. The lenient baseline proof tolerates that "failed" exemption; the
    # strict converged proof rejects it. A no-op rerun made no withdrawal of
    # its own, so its final proof must stay lenient and re-prove in place.
    rows, _baseline_runtimes = _healed_baseline_rows()
    rows[1].update(ID="failed1", Node=rows[3]["Node"], CurrentState="Failed 30m ago")
    runtimes = _candidate_runtimes()
    _wire_noop_deploy_with_real_verify_tasks(monkeypatch, tmp_path, rows, runtimes, HEALED_READY)
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
    assert receipt["nodes"] == _running_nodes(rows)
    assert len(receipt["nodes"]) == rollout_policy.REPLICAS


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
        return task_proof()

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
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", "2026-10-21T10:00:00Z"))
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
