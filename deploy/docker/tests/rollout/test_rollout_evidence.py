import copy
import json
import subprocess

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import rollout_evidence
import verify_rollout as rollout
from conftest import (
    CANDIDATE, ELIGIBLE_NODES, HEALED_READY, INSTANCES, REVISION, _candidate_runtimes, _healed_colocation_rows,
    _running_nodes, _wire_verify_tasks, application, health, service_spec, task_proof,
)


def _wire_withdrawal_journal(monkeypatch):
    path = rollout_tasks._task_proof_path()
    proof = json.loads(path.read_text())
    candidate = [
        {"task": identifier, "container": container, "node": proof["nodes"][index % len(proof["nodes"]) ]}
        for index, (identifier, container) in enumerate(zip(proof["tasks"], proof["instances"]))
    ]
    baseline = [{"task": "old-task", "container": "old-container", "node": proof["nodes"][0]}]
    nodes = {node: node for node in proof["nodes"]}
    events = {node: [] for node in nodes}
    events[baseline[0]["node"]].append({"action": "die", "task": "old-task", "container": "old-container", "exit": "0"})
    for identity in candidate:
        events[identity["node"]].append({"action": "create", "task": identity["task"], "container": identity["container"], "exit": ""})
    proof["baselineTasks"] = baseline
    proof["withdrawals"] = {"app": "crawl4ai", "nodes": nodes, "baseline": baseline,
                            "candidate": candidate, "events": events}
    path.write_text(json.dumps(proof))
    monkeypatch.setattr(rollout.withdrawals, "census", lambda _app: candidate)
    monkeypatch.setattr(rollout.withdrawals, "ready_nodes", lambda: nodes)


def _wire_real_verify_tasks_for_evidence(monkeypatch, tmp_path, rows, runtimes, revision, baseline_revision, placement=rollout_policy.PLACEMENT):
    # The leniency fix is in evidence() too: it derives converged from
    # baselineRevision == revision (no-op rerun) vs != (real rollout). Wire the
    # real _verify_tasks so the converged branch is exercised, not stubbed away.
    candidate = CANDIDATE
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"ok": True, "url": url, "revision": revision, "instance": "container1"})
            for url in rollout_policy.HEALTH_URLS
        )
        + "\n"
    )
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", revision)
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    rollout_tasks._task_proof_path().write_text(json.dumps({
        "revision": revision,
        "image": candidate,
        "baselineRevision": baseline_revision,
        "placement": copy.deepcopy(placement),
        "tasks": sorted(runtimes),
        "nodes": _running_nodes(rows),
        "instances": _containers(runtimes),
        "publicInstances": {url: _containers(runtimes) for url in rollout_policy.HEALTH_URLS},
    }))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application(candidate, revision, placement))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, HEALED_READY))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, revision, placement))
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: sorted(instances) for url in rollout_policy.HEALTH_URLS},
    )
    _wire_verify_tasks(monkeypatch, rows, runtimes, revision=revision)
    _wire_withdrawal_journal(monkeypatch)
    return path


def _containers(runtimes):
    return sorted(runtime["container"] for runtime in runtimes.values())


def _wire_final_evidence(monkeypatch, tasks):
    _wire_withdrawal_journal(monkeypatch)
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(CANDIDATE, REVISION))
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {
            url: sorted(instances) for url in rollout_policy.HEALTH_URLS
        },
    )
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: tasks)


def test_evidence_requires_candidate_on_both_domains(monkeypatch, tmp_path, capsys):
    path = tmp_path / "evidence.jsonl"
    rows = [
        {"ok": True, "url": url, "revision": REVISION, "instance": INSTANCES[0]}
        for url in rollout_policy.HEALTH_URLS
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    rollout_tasks._task_proof_path().write_text(json.dumps({
        "revision": REVISION,
        "image": CANDIDATE,
        "baselineRevision": "baseline",
        "placement": copy.deepcopy(rollout_policy.PLACEMENT),
        **task_proof(),
        "publicInstances": {
            url: INSTANCES[::-1] for url in rollout_policy.HEALTH_URLS
        },
    }))
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setattr(
        rollout_dokploy, "_application", lambda *_args: application(CANDIDATE, REVISION)
    )
    _wire_final_evidence(
        monkeypatch,
        task_proof(),
    )
    rollout_evidence.evidence()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["publicInstances"] == {
        url: sorted(INSTANCES) for url in rollout_policy.HEALTH_URLS
    }
    rows[0]["revision"] = "baseline"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="each public domain"):
        rollout_evidence.evidence()
    rows[0]["revision"] = "foreign"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="foreign revision"):
        rollout_evidence.evidence()
    rows[0]["revision"] = REVISION
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(
        rollout_tasks,
        "verify_route",
        lambda *_args: (_ for _ in ()).throw(ValueError("route changed")),
    )
    with pytest.raises(ValueError, match="route changed"):
        rollout_evidence.evidence()


@pytest.mark.parametrize("failure", ["missing", "unknown-proof", "unknown-monitor"])
def test_evidence_rejects_public_instance_set_mismatch(monkeypatch, tmp_path, failure):
    path = tmp_path / "evidence.jsonl"
    rows = [
        {"ok": True, "url": url, "revision": REVISION, "instance": INSTANCES[0]}
        for url in rollout_policy.HEALTH_URLS
    ]
    coverage = {url: INSTANCES for url in rollout_policy.HEALTH_URLS}
    if failure == "missing":
        coverage[rollout_policy.HEALTH_URLS[0]] = INSTANCES[:-1]
    elif failure == "unknown-proof":
        coverage[rollout_policy.HEALTH_URLS[0]] = [*INSTANCES[:-1], "unknown"]
    else:
        rows[0]["instance"] = "unknown"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    rollout_tasks._task_proof_path().write_text(json.dumps({
        "revision": REVISION,
        "image": CANDIDATE,
        "baselineRevision": "baseline",
        "placement": copy.deepcopy(rollout_policy.PLACEMENT),
        **task_proof(),
        "publicInstances": coverage,
    }))
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setattr(
        rollout_dokploy, "_application", lambda *_args: application(CANDIDATE, REVISION)
    )
    _wire_final_evidence(
        monkeypatch,
        task_proof(),
    )
    message = (
        "unknown candidate instance"
        if failure == "unknown-monitor"
        else "every authoritative task"
    )
    with pytest.raises(RuntimeError, match=message):
        rollout_evidence.evidence()


@pytest.mark.parametrize("drift", ["task", "placement"])
def test_evidence_rejects_changes_after_deploy(monkeypatch, tmp_path, drift):
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {"ok": True, "url": url, "revision": REVISION, "instance": INSTANCES[0]}
            )
            for url in rollout_policy.HEALTH_URLS
        )
        + "\n"
    )
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    rollout_tasks._task_proof_path().write_text(json.dumps({
        "revision": REVISION,
        "image": CANDIDATE,
        "baselineRevision": "baseline",
        "placement": copy.deepcopy(rollout_policy.PLACEMENT),
        **task_proof(),
        "publicInstances": {url: INSTANCES for url in rollout_policy.HEALTH_URLS},
    }))
    monkeypatch.setattr(
        rollout_dokploy, "_application", lambda *_args: application(
            CANDIDATE, REVISION,
            rollout_policy._SPREAD_PLACEMENT if drift == "placement" else rollout_policy.PLACEMENT,
        )
    )
    # One slot was rescheduled onto another node after the deploy's proof.
    rescheduled = task_proof(sorted(ELIGIBLE_NODES)[1:rollout_policy.REPLICAS + 1])
    rescheduled["tasks"][-1] = "replacement"
    rescheduled["instances"][-1] = "replacement-instance"
    _wire_final_evidence(monkeypatch, rescheduled)
    message = "candidate metadata changed" if drift == "placement" else "task census changed"
    with pytest.raises(RuntimeError, match=message):
        rollout_evidence.evidence()


def test_evidence_noop_rerun_proves_healed_colocation_in_place(monkeypatch, tmp_path, capsys):
    # evidence() re-proves a no-op rerun (baselineRevision == revision) leniently.
    # An older ghost predecessor is not attributed to this no-op run.
    rows, _baseline_runtimes = _healed_colocation_rows()
    runtimes = _candidate_runtimes()
    _wire_real_verify_tasks_for_evidence(
        monkeypatch, tmp_path, rows, runtimes, REVISION, REVISION
    )

    rollout_evidence.evidence()

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["publicFailures"] == 0
    assert receipt["publicInstances"] == {
        url: _containers(runtimes) for url in rollout_policy.HEALTH_URLS
    }


def test_evidence_real_rollout_rejects_unconfirmed_predecessor(monkeypatch, tmp_path):
    # Uncapped placement permits co-location, but a real rollout must still
    # confirm predecessor shutdown before accepting its evidence.
    rows, _baseline_runtimes = _healed_colocation_rows()
    runtimes = _candidate_runtimes()
    _wire_real_verify_tasks_for_evidence(
        monkeypatch, tmp_path, rows, runtimes, REVISION, "baseline"
    )

    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout_evidence.evidence()


@pytest.mark.parametrize("placement", rollout_policy.PLACEMENTS)
def test_final_evidence_uses_recorded_placement(monkeypatch, tmp_path, placement):
    rows, _baseline_runtimes = _healed_colocation_rows()
    rows[1].update(ID="old1", CurrentState="Shutdown 1m")
    _wire_real_verify_tasks_for_evidence(
        monkeypatch, tmp_path, rows, _candidate_runtimes(), REVISION, "baseline", placement
    )
    if placement.get("MaxReplicas") == 1:
        with pytest.raises(RuntimeError, match="not on distinct eligible nodes"):
            rollout_evidence.evidence()
    else:
        rollout_evidence.evidence()


def test_monitor_arms_only_after_both_domains_are_ready(monkeypatch, tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    armed_path = tmp_path / "armed"
    stop_path = tmp_path / "stop"
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(evidence_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_ARMED_PATH", str(armed_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_STOP_PATH", str(stop_path))
    calls = []

    def ready(url, *_args, **_kwargs):
        assert not armed_path.exists()
        calls.append(url)
        if len(calls) == len(rollout_policy.HEALTH_URLS):
            stop_path.touch()
        return health()

    monkeypatch.setattr(rollout_http, "_request_json", ready)
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    rollout_evidence.monitor()
    assert armed_path.exists()
    assert len(evidence_path.read_text().splitlines()) == len(rollout_policy.HEALTH_URLS)


@pytest.mark.parametrize("revision", ["baseline", REVISION])
def test_monitor_requires_candidate_echo_but_accepts_legacy_tasks(monkeypatch, tmp_path, revision):
    evidence_path = tmp_path / "evidence.jsonl"
    stop_path = tmp_path / "stop"
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(evidence_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_ARMED_PATH", str(tmp_path / "armed"))
    monkeypatch.setenv("ROLLOUT_MONITOR_STOP_PATH", str(stop_path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)

    def legacy_response(*_args, probe, **_kwargs):
        stop_path.touch()
        probe["correlation_verified"] = False
        return {**health(), "revision": revision}

    monkeypatch.setattr(rollout_http, "_request_json", legacy_response)
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    if revision == REVISION:
        with pytest.raises(RuntimeError, match="baseline is not ready"):
            rollout_evidence.monitor()
    else:
        rollout_evidence.monitor()
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert all(row["ok"] == (revision != REVISION) for row in rows)


def test_monitor_failure_sample_is_attributable(monkeypatch, tmp_path):
    """A failed probe must record what failed — the 2026-08-31 run went red
    on a single sample that said only {"error": "RuntimeError"}."""
    evidence_path = tmp_path / "evidence.jsonl"
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(evidence_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_ARMED_PATH", str(tmp_path / "armed"))
    monkeypatch.setenv("ROLLOUT_MONITOR_STOP_PATH", str(tmp_path / "stop"))
    calls = []

    def probe(url, *_args, **_kwargs):
        calls.append(url)
        if len(calls) <= len(rollout_policy.HEALTH_URLS):
            if len(calls) == len(rollout_policy.HEALTH_URLS):
                pass  # first round healthy: arms the monitor
            return health()
        (tmp_path / "stop").touch()
        raise rollout_http.CurlError(
            "HTTP request failed: curl exit 56: https://example.test/health?rollout=secret",
            56,
        )

    monkeypatch.setattr(rollout_http, "_request_json", probe)
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="recorded"):
        rollout_evidence.monitor()
    samples = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    failed = [s for s in samples if not s["ok"]]
    assert failed
    assert failed[0]["curl_exit"] == 56
    assert failed[0]["error"].startswith("CurlError: HTTP request failed: curl exit 56")
    assert "?rollout=" not in failed[0]["error"]
    assert failed[0]["probe_id"]
    assert failed[0]["started_at"] <= failed[0]["timestamp"]


def test_monitor_timeout_sample_keeps_correlation_without_command_details(monkeypatch, tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    stop_path = tmp_path / "stop"
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(evidence_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_ARMED_PATH", str(tmp_path / "armed"))
    monkeypatch.setenv("ROLLOUT_MONITOR_STOP_PATH", str(stop_path))
    calls = []

    def probe(url, *_args, **_kwargs):
        calls.append(url)
        if len(calls) <= len(rollout_policy.HEALTH_URLS):
            return health()
        stop_path.touch()
        raise subprocess.TimeoutExpired(["curl", url], 20)

    monkeypatch.setattr(rollout_http, "_request_json", probe)
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="recorded"):
        rollout_evidence.monitor()

    samples = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    failed = next(sample for sample in samples if not sample["ok"])
    assert failed["curl_exit"] is None
    assert failed["error"] == "TimeoutExpired: curl wrapper exceeded 20 seconds"
    assert "?rollout=" not in failed["error"]
    assert failed["probe_id"]
    assert failed["elapsed_ms"] >= 0
