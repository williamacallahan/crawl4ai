import copy
import json

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout
from conftest import BASELINE, CANDIDATE, ELIGIBLE_NODES, INSTANCES, REVISION, _deploy_env, application, health, service_spec, task_proof


PRIOR_UPDATE_STARTED = "2026-10-21T10:00:00.123456789Z"
CURRENT_UPDATE_STARTED = "2026-10-21T10:00:01.123456789Z"


@pytest.mark.parametrize("placement", rollout_policy.PLACEMENTS)
def test_deploy_uses_only_stock_update_and_deploy(monkeypatch, tmp_path, capsys, placement):
    state = application(placement=placement)
    posts = []
    events = []
    deployments = [{"deploymentId": "old", "status": "done"}]

    def post(url, _key, payload):
        posts.append((url, payload))
        events.append(url.rsplit("/", 1)[-1])
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
            state["placementSwarm"] = payload["placementSwarm"]

    _deploy_env(monkeypatch)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: copy.deepcopy(deployments),
    )
    monkeypatch.setattr(rollout_http, "_post_json", post)
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", CURRENT_UPDATE_STARTED if state["dockerImage"] == CANDIDATE else PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"], placement))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: events.append("verify_redis"))
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_dokploy,
        "_wait_deployment",
        lambda *_args: {"deploymentId": "new", "status": "done"},
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: task_proof(),
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(
        rollout_http,
        "_request_json",
        lambda url, *_args: health(revision="baseline" if "baseline=" in url else REVISION),
    )
    rollout.deploy()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["publicInstances"] == {
        url: INSTANCES for url in rollout_policy.HEALTH_URLS
    }
    # The Redis proof must precede the first write, not merely happen.
    assert events == ["verify_redis", "application.update", "application.deploy"]
    assert [url.rsplit("/", 1)[-1] for url, _ in posts] == [
        "application.update",
        "application.deploy",
    ]
    assert "idempotencyKey" not in posts[1][1]
    assert posts[0][1]["placementSwarm"] == placement
    assert "expectedDockerImage" not in posts[0][1]
    proof = json.loads(rollout_tasks._task_proof_path().read_text())
    assert proof["instances"] == INSTANCES
    assert set(proof["publicInstances"]) == set(rollout_policy.HEALTH_URLS)
    assert proof["placement"] == placement

    state = application(placement=placement)
    posts.clear()
    events.clear()
    deployments[:] = [{"deploymentId": "running", "status": "running"}]
    with pytest.raises(RuntimeError, match="nonterminal Dokploy deployment"):
        rollout.deploy()
    assert not posts

    state = application(placement=placement)
    posts.clear()
    events.clear()
    deployments[:] = [{"deploymentId": "old", "status": "done"}]
    # One slot is rescheduled onto another node between the census reads.
    rescheduled = task_proof(sorted(ELIGIBLE_NODES)[1:rollout_policy.REPLICAS + 1])
    rescheduled["tasks"][-1] = "replacement"
    rescheduled["instances"][-1] = "replacement-instance"
    proofs = iter([
        task_proof(tag="old"),
        task_proof(tag="old"),
        task_proof(),
        rescheduled,
    ])
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: next(proofs))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", CURRENT_UPDATE_STARTED if state["dockerImage"] == CANDIDATE else PRIOR_UPDATE_STARTED))
    with pytest.raises(RuntimeError, match="task census changed"):
        rollout.deploy()


def test_deploy_proves_an_already_deployed_candidate_without_resubmitting(
    monkeypatch, tmp_path, capsys
):
    # Dokploy makes no spec change for an image the record already carries, so
    # it opens no deployment row: a rerun that resubmits waits out the whole
    # admission timeout on a row that never appears.
    candidate = "registry.example/crawl4ai@sha256:candidate"
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
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: task_proof(),
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == candidate
    assert receipt["revision"] == REVISION
    assert receipt["publicInstances"] == {
        url: INSTANCES for url in rollout_policy.HEALTH_URLS
    }


@pytest.mark.parametrize("down_node", ["haiku-9", "haiku-6"])
def test_deploy_proves_an_already_deployed_candidate_with_a_non_redis_spare_down(
    monkeypatch, tmp_path, capsys, down_node
):
    # A rerun of a revision that is already live performs no start-first
    # rollout: it submits no update, starts no replacement, and retires no
    # predecessor, so it needs no spare overlap slot. With a one-per-node cap,
    # exactly REPLICAS Ready eligible nodes, and the down node a non-Redis
    # spare (so the Redis node still passes), the no-op rerun must prove it
    # in place rather than abort under the start-first spare gate. Regression
    # for the no-op rerun path added in f43ef89, whose spare gate ran above
    # the branch.
    candidate = "registry.example/crawl4ai@sha256:candidate"
    placement = rollout_policy._CAPPED_PLACEMENT
    state = application(image=candidate, revision=REVISION, placement=placement)
    ready = frozenset(sorted(ELIGIBLE_NODES - {down_node})[:rollout_policy.REPLICAS])
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
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION, placement))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: task_proof(ready),
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    rollout.deploy()

    assert len(ready) == rollout_policy.REPLICAS
    assert down_node not in ready and rollout_policy.REDIS_NODE in ready
    assert rollout_policy.placement_capacity(len(ready), placement) < (
        rollout_policy.REPLICAS + rollout_policy.PARALLELISM
    )
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == candidate
    assert receipt["revision"] == REVISION
    assert receipt["publicInstances"] == {
        url: INSTANCES for url in rollout_policy.HEALTH_URLS
    }


def test_deploy_real_rollout_passes_strict_converged_to_the_candidate_proof(
    monkeypatch, tmp_path
):
    # A real start-first rollout performs the withdrawals the converged proof
    # exists to check, so its candidate proof must stay strict
    # (converged=True) while only the baseline proof stays lenient.
    calls = []

    def spy(_app, image, _revision, _ready, converged=True, placement=rollout_policy.PLACEMENT):
        calls.append((image, converged))
        return task_proof()

    state = application()  # baseline running; candidate not yet deployed
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )

    def post(url, _key, payload):
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
            state["placementSwarm"] = payload["placementSwarm"]

    monkeypatch.setattr(rollout_http, "_post_json", post)
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", CURRENT_UPDATE_STARTED if state["dockerImage"] == CANDIDATE else PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(
        rollout_tasks, "_service_spec",
        lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"]),
    )
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_dokploy, "_wait_deployment", lambda *_args: {"deploymentId": "new", "status": "done"})
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", spy)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(
        rollout_http, "_request_json",
        lambda url, *_args: health(revision="baseline" if "baseline=" in url else REVISION),
    )

    rollout.deploy()

    # The baseline proof (image == BASELINE) stays lenient; every candidate
    # proof (image == CANDIDATE) the rollout takes for itself stays strict.
    assert calls[0] == (BASELINE, False), calls
    assert all(converged is (image == CANDIDATE) for image, converged in calls), calls


@pytest.mark.parametrize("fail_on", [1, 2])
def test_deploy_does_not_compensate_for_ambiguous_write(monkeypatch, fail_on):
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setenv("IMAGE", "registry.example/crawl4ai")
    monkeypatch.setenv("IMAGE_DIGEST", "sha256:candidate")
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    state = application()
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_args: [])
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec())
    monkeypatch.setattr(rollout_dokploy, "_ensure_rollback_source", lambda _image: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: {})
    monkeypatch.setattr(
        rollout_http,
        "_request_json",
        lambda _url, *_args: health(revision="baseline"),
    )
    calls = []

    def fail(url, _key, payload):
        calls.append(url)
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
            state["placementSwarm"] = payload["placementSwarm"]
        if len(calls) == fail_on:
            raise RuntimeError("HTTP request failed; state is ambiguous")

    monkeypatch.setattr(rollout_http, "_post_json", fail)
    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout.deploy()
    assert len(calls) == fail_on


def _deploy_until_capacity_gate(monkeypatch, capsys, placement, ready_count):
    """Run deploy() with ready_count Ready nodes; return the gate failure or None if it passed."""
    _deploy_env(monkeypatch)
    ready = frozenset(sorted(ELIGIBLE_NODES)[:ready_count])
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application(placement=placement))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(placement=placement))
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(
        rollout_tasks, "_verify_tasks",
        lambda *_args: (_ for _ in ()).throw(SystemExit("capacity gate passed")),
    )
    monkeypatch.setattr(rollout.sys, "argv", ["verify_rollout.py", "deploy"])
    try:
        rollout.main()
    except SystemExit as passed:
        assert str(passed) == "capacity gate passed"
        return None
    except RuntimeError as failure:
        # The message names nodes, so the public line carries only its class.
        assert capsys.readouterr().err == (
            "rollout failure: command=deploy check=_deploy error=RuntimeError\n"
        )
        return str(failure)
    pytest.fail("deploy() must stop at the capacity gate or the baseline proof")


@pytest.mark.parametrize("placement", rollout_policy.PLACEMENTS)
@pytest.mark.parametrize("ready_count", range(len(ELIGIBLE_NODES) + 1))
def test_deploy_capacity_follows_placement(monkeypatch, capsys, placement, ready_count):
    failure = _deploy_until_capacity_gate(monkeypatch, capsys, placement, ready_count)
    capacity = rollout_policy.placement_capacity(ready_count, placement)
    if capacity < rollout_policy.REPLICAS:
        assert failure.startswith("not enough Ready eligible nodes to place every replica")
    elif capacity < rollout_policy.REPLICAS + rollout_policy.PARALLELISM:
        assert failure.startswith("start-first needs a spare eligible node")
    else:
        assert failure is None


def _spread_capped(max_replicas):
    return next(
        placement for placement in rollout_policy.PLACEMENTS
        if placement.get("MaxReplicas") == max_replicas and placement.get("Preferences")
    )


# Six replicas in start-first batches of two need eight placement slots.
@pytest.mark.parametrize(("max_replicas", "ready_count", "expected"), [
    (3, 3, None),
    (2, 3, "start-first needs a spare eligible node"),
    (2, 4, None),
    (3, 1, "not enough Ready eligible nodes to place every replica"),
    (1, 5, "not enough Ready eligible nodes to place every replica"),
    (1, 7, "start-first needs a spare eligible node"),
    (1, 8, None),
])
def test_deploy_capped_capacity_counts_slots_per_node(
    monkeypatch, capsys, max_replicas, ready_count, expected
):
    failure = _deploy_until_capacity_gate(
        monkeypatch, capsys, _spread_capped(max_replicas), ready_count
    )
    if expected is None:
        assert failure is None
    else:
        assert failure.startswith(expected)


@pytest.mark.parametrize("placement", [rollout_policy.PLACEMENT, rollout_policy._SPREAD_PLACEMENT])
def test_deploy_uncapped_placement_still_needs_three_ready_nodes(monkeypatch, capsys, placement):
    # Without a per-node cap Swarm could stack every replica on one node.
    assert _deploy_until_capacity_gate(monkeypatch, capsys, placement, 2).startswith(
        "not enough Ready eligible nodes to place every replica"
    )
    assert _deploy_until_capacity_gate(monkeypatch, capsys, placement, 3) is None


def test_main_keeps_the_original_failure_when_its_summary_fails(monkeypatch, capsys):
    _deploy_env(monkeypatch)

    def application_failure(*_args):
        raise ValueError("application lookup failed")

    monkeypatch.setattr(rollout_dokploy, "_application", application_failure)
    monkeypatch.setattr(
        rollout, "_public_failure", lambda *_args: (_ for _ in ()).throw(OSError("unreadable"))
    )
    monkeypatch.setattr(rollout.sys, "argv", ["verify_rollout.py", "deploy"])
    with pytest.raises(ValueError, match="application lookup failed"):
        rollout.main()
    assert capsys.readouterr().err == "rollout failure: command=deploy error=ValueError\n"


def test_deploy_rejects_a_reintroduced_cap_once_the_record_converged(monkeypatch):
    # Record already migrated to the capless shape; a capped LIVE spec is
    # drift, not transition residue.
    _deploy_env(monkeypatch)
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application())
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: spec)
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    with pytest.raises(ValueError, match="running placement drifted"):
        rollout.deploy()


def test_deploy_accepts_capped_live_spec_when_record_matches(monkeypatch):
    _deploy_env(monkeypatch)
    state = application()
    state["placementSwarm"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: spec)
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_tasks, "_verify_redis", lambda: (_ for _ in ()).throw(SystemExit("baseline spec accepted"))
    )
    with pytest.raises(SystemExit, match="baseline spec accepted"):
        rollout.deploy()
