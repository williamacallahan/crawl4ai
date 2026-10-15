import copy
import io
import json
import socket
import time

import pytest

import rollout_withdrawals as withdrawals


def task(identifier, node="node1"):
    return {"task": identifier, "container": f"container-{identifier}", "node": node}


def event(identity, action, exit_code=""):
    return {"task": identity["task"], "container": identity["container"],
            "action": action, "exit": exit_code}


def journal():
    baseline, intermediate, candidate = task("old"), task("middle"), task("new")
    return {"app": "crawl4ai", "nodes": {"node1": "worker"}, "baseline": [baseline],
            "candidate": [candidate], "events": {"node1": [
                event(intermediate, "create"), event(baseline, "die", "0"),
                event(intermediate, "die", "0"), event(candidate, "create")]}}


def test_reaped_intermediate_attempt_is_proved_without_task_history():
    proof = journal()
    withdrawals.validate(proof, "crawl4ai", proof["baseline"], proof["candidate"])


@pytest.mark.parametrize("mutation", ["failed", "oom", "missing-create", "missing-die",
                                     "baseline-live", "candidate-died", "foreign-node",
                                     "wrong-task", "overflow", "missing-node"])
def test_journal_rejects_incomplete_or_failed_attempts(mutation):
    proof = journal()
    events = proof["events"]["node1"]
    if mutation == "failed":
        events[2]["exit"] = "137"
    elif mutation == "oom":
        events.insert(2, event(task("middle"), "oom"))
    elif mutation == "missing-create":
        events.pop()
    elif mutation == "missing-die":
        events.pop(2)
    elif mutation == "baseline-live":
        events.pop(1)
    elif mutation == "candidate-died":
        events.append(event(task("new"), "die", "0"))
    elif mutation == "foreign-node":
        proof["candidate"][0]["node"] = "outside"
    elif mutation == "wrong-task":
        events[2]["task"] = "different"
    elif mutation == "overflow":
        events.extend([copy.deepcopy(events[0])] * 128)
    else:
        proof["nodes"]["node2"] = "other-worker"
    with pytest.raises(RuntimeError):
        withdrawals.validate(proof, "crawl4ai", proof["baseline"], proof["candidate"])


def test_journal_is_bound_to_both_native_censuses():
    proof = journal()
    for baseline, candidate in [([task("other")], proof["candidate"]),
                                (proof["baseline"], [task("other")])]:
        with pytest.raises(RuntimeError, match="identity differs"):
            withdrawals.validate(proof, "crawl4ai", baseline, candidate)


def engine_event(service, action="create"):
    return {"Type": "container", "Action": action, "Actor": {"ID": f"container-{service}", "Attributes": {
        "com.docker.swarm.service.name": service, "com.docker.swarm.task.id": service,
        "name": f"{service}.1.task", "exitCode": "0"}}}


def wire_engine(monkeypatch, records, evict=False, wrong_owner=False):
    import http.client
    heartbeat = []
    requests = []

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.sock = None

        def request(self, method, path, body=None, headers=None):
            requests.append((method, path, body))
            self.path = path

        def getresponse(self):
            if self.path == "/containers/collector-id/exec":
                identifier = f"heartbeat{len(heartbeat)}"
                marker = engine_event("collector", "exec_create: python -c pass")
                marker["Actor"]["ID"] = "foreign-collector" if wrong_owner else "collector-id"
                marker["Actor"]["Attributes"]["execID"] = identifier
                heartbeat.append(marker)
                body = json.dumps({"Id": identifier}).encode()
            elif self.path.startswith("/containers/") and self.path.endswith("/json"):
                body = b'{"Id":"collector-id"}'
            elif self.path.startswith("/containers/"):
                raise AssertionError(f"unexpected container mutation: {self.path}")
            elif self.path == "/info":
                body = b'{"SystemTime":"2026-10-15T00:00:00Z"}'
            elif self.path.startswith("/events?"):
                if len(heartbeat) == 1:
                    snapshot = heartbeat
                elif len(heartbeat) == 2:
                    snapshot = ([heartbeat[0]] if not evict else []) + records + [heartbeat[1]]
                else:
                    raise RuntimeError("journal exhausted test events")
                body = b"".join(json.dumps(record).encode() + b"\n" for record in snapshot)
            else:
                raise AssertionError(self.path)
            response = io.BytesIO(body)
            response.status = 200
            return response

        def close(self):
            pass

    class Socket:
        def settimeout(self, timeout):
            assert timeout > 0

        def connect(self, path):
            assert path == "/var/run/docker.sock"

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(socket, "socket", lambda *_args: Socket())
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return requests


def test_collector_requires_observed_markers_and_keeps_all_attempts(monkeypatch, capsys):
    requests = wire_engine(monkeypatch, [engine_event("crawl4ai-arm"), engine_event("crawl4ai"),
                              engine_event("crawl4ai", "die"), engine_event("crawl4ai-end")])
    withdrawals._collect("crawl4ai", "crawl4ai-arm", "crawl4ai-end", 30)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0] == {"ready": True}
    assert records[1] == {"armed": True}
    assert records[2]["complete"] is True
    assert [record["action"] for record in records[2]["events"]] == ["create", "die"]
    mutations = [(method, path, json.loads(body)) for method, path, body in requests if method != "GET"]
    assert len(mutations) == 2
    assert all(method == "POST" and path == "/containers/collector-id/exec"
               and payload == {"Cmd": ["python", "-c", "pass"], "AttachStdout": False,
                               "AttachStderr": False, "AttachStdin": False}
               for method, path, payload in mutations)


@pytest.mark.parametrize("records", [[], [engine_event("crawl4ai-end")],
    [engine_event("crawl4ai-arm")] + [engine_event("crawl4ai", "create")] * 128
    + [engine_event("crawl4ai-end")]])
def test_collector_rejects_eof_missing_arming_and_overflow(monkeypatch, records):
    wire_engine(monkeypatch, records)
    with pytest.raises(RuntimeError):
        withdrawals._collect("crawl4ai", "crawl4ai-arm", "crawl4ai-end", 30)


def test_collector_rejects_cache_eviction_even_with_a_final_barrier(monkeypatch):
    wire_engine(monkeypatch, [engine_event("crawl4ai-arm"), engine_event("crawl4ai-end")], evict=True)
    with pytest.raises(RuntimeError, match="lost.*anchor"):
        withdrawals._collect("crawl4ai", "crawl4ai-arm", "crawl4ai-end", 30)


def test_collector_rejects_an_exec_anchor_owned_by_another_container(monkeypatch):
    wire_engine(monkeypatch, [], wrong_owner=True)
    with pytest.raises(RuntimeError, match="lost.*anchor"):
        withdrawals._collect("crawl4ai", "crawl4ai-arm", "crawl4ai-end", 30)


def test_cleanup_removes_only_recorded_services(monkeypatch, tmp_path):
    path = tmp_path / "lifecycle.json"
    path.write_text(json.dumps({"owner": "exact", "services": ["exact-first", "exact-second"]}))
    removed = []
    def inspect(arguments, **_kwargs):
        class Inspection:
            returncode = 0
            stdout = json.dumps([{"Spec": {"Name": arguments[-1], "Labels": {
                "crawl4ai.withdrawal.owner": "exact"}}}])
        return Inspection()
    monkeypatch.setattr(withdrawals.subprocess, "run", inspect)
    monkeypatch.setattr(withdrawals, "_docker", lambda *arguments: removed.append(arguments))
    withdrawals.cleanup(path)
    assert removed == [("service", "rm", "exact-first"), ("service", "rm", "exact-second")]
    assert not path.exists()


def test_failed_collector_reports_its_final_state_and_output(monkeypatch, tmp_path):
    monkeypatch.setattr(withdrawals, "ready_nodes", lambda: {"node1": "worker"})
    collector = withdrawals.Withdrawals("crawl4ai", "image", tmp_path / "lifecycle.json", 30)
    collector.collectors["node1"] = ("collector", "task1", "container1")
    failed = {"ID": "task1", "NodeID": "node1", "Status": {
        "State": "failed", "Err": "task: non-zero exit (1)", "ContainerStatus": {
            "ContainerID": "container1", "ExitCode": 1}}}
    monkeypatch.setattr(collector, "_task", lambda _service: failed)

    class Logs:
        stdout = '{"ready": true}\n'
        stderr = "TimeoutError: timed out\n"
    monkeypatch.setattr(withdrawals.subprocess, "run", lambda *_args, **_kwargs: Logs())
    with pytest.raises(RuntimeError) as failure:
        collector._guard("node1")
    detail = json.loads(str(failure.value).split(": ", 1)[1])
    assert detail["state"] == "failed" and detail["exit"] == 1 and not detail["replaced"]
    assert detail["output"].endswith("TimeoutError: timed out\n")
