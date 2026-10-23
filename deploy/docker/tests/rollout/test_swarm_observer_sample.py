import subprocess
from pathlib import Path

import pytest

import swarm_observer as observer
import rollout_policy
import rollout_http
import rollout_tasks
from conftest import REVISION, health


def test_sample_reuses_rollout_task_and_health_owners(monkeypatch):
    rows = [
        {
            "ID": f"task-{index}",
            "Node": f"node-{index}",
            "DesiredState": "Running",
            "CurrentState": "Running 1m",
        }
        for index in range(3)
    ]
    runtimes = {
        f"task-{index}": {
            "container": f"container-{index}",
            "addresses": {f"10.0.1.{index + 10}"},
            "labels": {"otel.service.version": REVISION},
        }
        for index in range(3)
    }
    monkeypatch.setattr(rollout_tasks, "_verify_ingress_host", lambda: 4321)
    namespace_probe = []
    monkeypatch.setattr(
        observer,
        "_verify_ingress_namespace",
        lambda pid: namespace_probe.append(pid),
    )
    monkeypatch.setattr(rollout_tasks, "_service_tasks", lambda _service: rows)
    monkeypatch.setattr(
        observer,
        "_network_details",
        lambda: {
            "Id": "network",
            "Services": {
                "crawl4ai": {
                    "Tasks": [
                        {
                            "Name": task_id,
                            "Info": {"Host IP": f"100.0.0.{index + 10}"},
                        }
                        for index, task_id in enumerate(runtimes)
                    ]
                }
            },
        },
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_task_runtime",
        lambda task_id, _network: runtimes[task_id],
    )
    direct_probe = []

    def direct_health(url, *, network_namespace_pid):
        direct_probe.append((url, network_namespace_pid))
        return health(
            next(
                runtime["container"]
                for runtime in runtimes.values()
                if next(iter(runtime["addresses"])) in url
            ),
            revision=REVISION,
        )

    monkeypatch.setattr(rollout_http, "_request_json", direct_health)
    monkeypatch.setattr(observer, "_public_coverage", lambda _direct, _deadline: 2)
    monkeypatch.setattr(
        observer,
        "_fdb_comparison",
        lambda *_args: observer.NetworkDbFdbComparison.CONSISTENT,
    )

    snapshot, comparison = observer.sample("crawl4ai")

    assert snapshot == observer.CoverageSnapshot(3, 3, 2)
    assert comparison is observer.NetworkDbFdbComparison.CONSISTENT
    assert len(direct_probe) == 3
    assert {namespace_pid for _url, namespace_pid in direct_probe} == {4321}
    assert namespace_probe == [4321]


def test_sample_rejects_a_task_census_change(monkeypatch):
    before = [
        {
            "ID": "task-before",
            "Node": "node-1",
            "DesiredState": "Running",
            "CurrentState": "Pending 1s",
        }
    ]
    after = [{**before[0], "ID": "task-after"}]
    calls = iter((before, after))
    monkeypatch.setattr(rollout_tasks, "_verify_ingress_host", lambda: None)
    monkeypatch.setattr(observer, "_verify_ingress_namespace", lambda _pid: None)
    monkeypatch.setattr(
        rollout_tasks, "_service_tasks", lambda _service: next(calls)
    )
    monkeypatch.setattr(observer, "_network_details", lambda: {"Id": "network"})

    with pytest.raises(RuntimeError, match="task census changed"):
        observer.sample("crawl4ai")


def test_verify_ingress_namespace_runs_a_bounded_nsenter_preflight(monkeypatch):
    invocation = []

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(observer.subprocess, "run", run)

    assert observer._verify_ingress_namespace(4321) is None
    assert invocation == [
        (
            ["nsenter", "--net=/proc/4321/ns/net", "/bin/true"],
            {"capture_output": True, "text": True, "timeout": 10},
        )
    ]


def test_verify_ingress_namespace_raises_loudly_with_the_denial_reason(monkeypatch):
    monkeypatch.setattr(
        observer.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "",
            "nsenter: cannot open '/proc/4321/ns/net': Permission denied",
        ),
    )
    with pytest.raises(RuntimeError, match="cannot enter ingress network namespace") as error:
        observer._verify_ingress_namespace(4321)
    assert "Permission denied" in str(error.value)
    assert "4321" in str(error.value)


def test_sample_fails_loudly_when_it_cannot_enter_the_ingress_namespace(monkeypatch):
    # Before the preflight, a systemic nsenter denial raised CurlError per task
    # and was swallowed by the loop. Fail before probing individual tasks.
    monkeypatch.setattr(rollout_tasks, "_verify_ingress_host", lambda: 4321)

    def run(command, **_kwargs):
        assert command == ["nsenter", "--net=/proc/4321/ns/net", "/bin/true"], command
        return subprocess.CompletedProcess(
            command, 1, "",
            "nsenter: cannot open '/proc/4321/ns/net': Permission denied",
        )

    monkeypatch.setattr(observer.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="cannot enter ingress network namespace") as error:
        observer.sample("crawl4ai")
    assert "Permission denied" in str(error.value)


def test_sampling_failure_does_not_report_bogus_zero_coverage_for_a_healthy_fleet(
    monkeypatch, tmp_path, capsys
):
    # A healthy full fleet whose sampler cannot enter the ingress namespace
    # must read as "sampling failed" (0/0/0), not the misleading
    # authoritative=REPLICAS/direct_healthy=0/public_covered=0 signature.
    rows = [
        {
            "ID": f"task-{index}",
            "Node": f"node-{index}",
            "DesiredState": "Running",
            "CurrentState": "Running 1m",
        }
        for index in range(rollout_policy.REPLICAS)
    ]
    runtimes = {
        f"task-{index}": {
            "container": f"container-{index}",
            "addresses": {f"10.0.1.{index + 10}"},
            "labels": {"otel.service.version": REVISION},
        }
        for index in range(rollout_policy.REPLICAS)
    }
    monkeypatch.setenv("CRAWL4AI_SWARM_SERVICE", "crawl4ai")
    monkeypatch.setenv("CRAWL4AI_OBSERVER_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("CRAWL4AI_OBSERVER_METRICS_PATH", str(tmp_path / "metrics"))
    monkeypatch.setattr(rollout_tasks, "_verify_ingress_host", lambda: 4321)
    monkeypatch.setattr(rollout_tasks, "_service_tasks", lambda _service: rows)
    monkeypatch.setattr(
        observer,
        "_network_details",
        lambda: {"Id": "network", "Services": {"crawl4ai": {"Tasks": []}}},
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_task_runtime",
        lambda task_id, _network: runtimes[task_id],
    )
    monkeypatch.setattr(
        observer.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "",
            "nsenter: cannot open '/proc/4321/ns/net': Permission denied",
        ),
    )
    monkeypatch.setattr(observer, "_public_coverage", lambda _direct, _deadline: 0)
    monkeypatch.setattr(
        observer,
        "_fdb_comparison",
        lambda *_args: observer.NetworkDbFdbComparison.INCONCLUSIVE,
    )

    monkeypatch.setattr(observer.time, "time", lambda: 1789000000.125)
    observer.write_sample()

    out = capsys.readouterr().out
    assert "crawl4ai coverage sampling failed: RuntimeError" in out
    assert f"authoritative_tasks={rollout_policy.REPLICAS}" not in out
    assert (tmp_path / "metrics").read_text() == (
        "crawl4ai_authoritative_task_count 0\n"
        "crawl4ai_direct_healthy_task_count 0\n"
        "crawl4ai_public_covered_task_count 0\n"
        "crawl4ai_coverage_complete 0\n"
        "crawl4ai_coverage_sample_timestamp_seconds 1789000000.125\n"
    )


def test_sample_still_swallows_a_per_task_health_failure(monkeypatch):
    # The preflight only fails loudly for a systemic namespace denial; one
    # flaky task must still be swallowed so it cannot abort a whole sample.
    rows = [
        {
            "ID": f"task-{index}",
            "Node": f"node-{index}",
            "DesiredState": "Running",
            "CurrentState": "Running 1m",
        }
        for index in range(3)
    ]
    runtimes = {
        f"task-{index}": {
            "container": f"container-{index}",
            "addresses": {f"10.0.1.{index + 10}"},
            "labels": {"otel.service.version": REVISION},
        }
        for index in range(3)
    }
    monkeypatch.setattr(rollout_tasks, "_verify_ingress_host", lambda: 4321)
    monkeypatch.setattr(observer, "_verify_ingress_namespace", lambda _pid: None)
    monkeypatch.setattr(rollout_tasks, "_service_tasks", lambda _service: rows)
    monkeypatch.setattr(
        observer,
        "_network_details",
        lambda: {"Id": "network", "Services": {"crawl4ai": {"Tasks": []}}},
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_task_runtime",
        lambda task_id, _network: runtimes[task_id],
    )

    def direct_health(url, *, network_namespace_pid):
        if "10.0.1.10" in url:
            raise rollout_http.CurlError(
                "HTTP request failed: curl exit 7: Connection refused", 7
            )
        runtime = next(
            r for r in runtimes.values() if next(iter(r["addresses"])) in url
        )
        return health(runtime["container"], revision=REVISION)

    monkeypatch.setattr(rollout_http, "_request_json", direct_health)
    monkeypatch.setattr(observer, "_public_coverage", lambda _direct, _deadline: 2)
    monkeypatch.setattr(
        observer,
        "_fdb_comparison",
        lambda *_args: observer.NetworkDbFdbComparison.INCONCLUSIVE,
    )

    snapshot, comparison = observer.sample("crawl4ai")

    assert snapshot == observer.CoverageSnapshot(3, 2, 2)
    assert comparison is observer.NetworkDbFdbComparison.INCONCLUSIVE


def test_sample_service_unit_grants_cap_sys_ptrace_for_nsenter():
    # The direct probe opens /proc/<ingress-pid>/ns/net, a ptrace-read check.
    # The sampler runs as root while dokploy-traefik runs non-root, so the
    # uid-match fast path does not apply and CAP_SYS_PTRACE must stay in the
    # unit's CapabilityBoundingSet. Pin it so a future edit cannot drop it and
    # silently pin every coverage gauge at zero.
    service = Path(__file__).resolve().parents[2] / "crawl4ai-swarm-observer-sample.service"
    line = next(
        line for line in service.read_text().splitlines()
        if line.lstrip().startswith("CapabilityBoundingSet=")
    )
    capabilities = line.split("=", 1)[1].split()
    assert "CAP_SYS_ADMIN" in capabilities
    assert "CAP_SYS_PTRACE" in capabilities


def test_public_coverage_accepts_each_task_revision_and_intersects_routes(monkeypatch):
    direct = {"one": "revision-one", "two": "revision-two"}
    route_call = {}

    def public_health(url):
        route = url.split("/health", 1)[0]
        route_call[route] = route_call.get(route, 0) + 1
        instance = "one" if route_call[route] == 1 else "two"
        return health(instance, revision=direct[instance])

    monkeypatch.setattr(rollout_http, "_request_json", public_health)
    monkeypatch.setattr(rollout_policy, "PUBLIC_PROOF_ATTEMPTS", 2)

    assert observer._public_coverage(direct, observer.time.monotonic() + 10) == 2


def test_public_coverage_stops_at_the_sample_deadline(monkeypatch):
    monkeypatch.setattr(
        rollout_http,
        "_request_json",
        lambda _url: pytest.fail("deadline must stop the public request"),
    )

    with pytest.raises(TimeoutError, match="deadline"):
        observer._public_coverage({"one": REVISION}, observer.time.monotonic() - 1)


def test_sampling_failure_replaces_a_stale_healthy_sample(monkeypatch, tmp_path):
    state_path = tmp_path / "state"
    metrics_path = tmp_path / "metrics"
    monkeypatch.setenv("CRAWL4AI_SWARM_SERVICE", "crawl4ai")
    monkeypatch.setenv("CRAWL4AI_OBSERVER_STATE_PATH", str(state_path))
    monkeypatch.setenv("CRAWL4AI_OBSERVER_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        observer,
        "sample",
        lambda _service: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    monkeypatch.setattr(observer.time, "time", lambda: 1789000000.125)
    observer.write_sample()

    assert metrics_path.read_text() == (
        "crawl4ai_authoritative_task_count 0\n"
        "crawl4ai_direct_healthy_task_count 0\n"
        "crawl4ai_public_covered_task_count 0\n"
        "crawl4ai_coverage_complete 0\n"
        "crawl4ai_coverage_sample_timestamp_seconds 1789000000.125\n"
    )
    assert state_path.read_bytes() == b"open\n"
