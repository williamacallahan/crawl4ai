import json
import subprocess

import pytest

import rollout_http
import verify_rollout as rollout
from conftest import health


def test_post_keeps_api_key_out_of_argv(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(rollout.subprocess, "run", run)
    rollout_http._post_json("https://dokploy/api/application.update", "secret", {"x": 1})
    assert "secret" not in " ".join(observed["command"])
    assert 'header = "x-api-key: secret"' in observed["input"]


def test_request_json_failure_carries_curl_diagnosis(monkeypatch):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 56, "", "curl: (56) Recv failure: Connection reset by peer"
        ),
    )
    with pytest.raises(rollout_http.CurlError) as error:
        rollout_http._request_json("https://example.test/health")
    assert error.value.curl_exit == 56
    assert "curl exit 56" in str(error.value)
    assert "Connection reset" in str(error.value)


def test_request_json_records_probe_phase_timings(monkeypatch):
    invocation = []
    probe_id = "01234567-89ab-4cde-8fab-0123456789ab"

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(health("one"))
            + f"{rollout_http._PROBE_TIMING_MARKER}200|0.001|0.002|0.003|0.004|0.005|{probe_id}",
            "",
        )

    monkeypatch.setattr(rollout.subprocess, "run", run)
    probe = {"probe_id": probe_id}

    assert rollout_http._request_json(
        "https://example.test/health?rollout=probe", probe=probe
    ) == health("one")
    assert "--write-out" in invocation[0][0]
    assert probe == {
        "probe_id": probe_id,
        "echoed_probe_id": probe_id,
        "correlation_verified": True,
        "curl_exit": 0,
        "response_code": 200,
        "dns_ms": 1.0,
        "connect_ms": 2.0,
        "tls_ms": 3.0,
        "start_transfer_ms": 4.0,
        "curl_total_ms": 5.0,
    }


@pytest.mark.parametrize(
    ("echo", "error"),
    [("", None), ("11234567-89ab-4cde-8fab-0123456789ab", "probe echo mismatch"),
     ("untrusted-header", "invalid probe echo"),
     ("invalid|echo", "invalid probe echo")],
)
def test_probe_echo_distinguishes_legacy_tasks_from_bad_correlation(monkeypatch, echo, error):
    probe_id = "01234567-89ab-4cde-8fab-0123456789ab"
    monkeypatch.setattr(
        rollout.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0,
            json.dumps(health()) + f"{rollout_http._PROBE_TIMING_MARKER}200|0|0|0|0|0|{echo}",
            "",
        ),
    )
    probe = {"probe_id": probe_id}
    assert rollout_http._request_json("https://example.test/health", probe=probe) == health()
    assert probe["correlation_verified"] is False
    assert probe.get("correlation_error") == error
    assert "untrusted-header" not in json.dumps(probe)


def test_request_json_uses_namespace_argv_and_existing_bounds(monkeypatch):
    invocation = []
    url = "http://10.0.1.38:11235/health"

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(health("one")), "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout_http._request_json(url, network_namespace_pid=4321) == health("one")
    assert invocation == [
        (
            [
                "nsenter",
                "--net=/proc/4321/ns/net",
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--connect-timeout",
                "5",
                "--max-time",
                "15",
                "--max-filesize",
                "65536",
                "--config",
                "-",
                url,
            ],
            {"input": "", "text": True, "capture_output": True, "timeout": 20},
        )
    ]


def test_request_json_uses_ingress_container_exec_argv_and_bounds(monkeypatch):
    invocation = []
    container = "a" * 64
    url = "http://10.0.1.38:11235/health"

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(health("one")), "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout_http._request_json(url, ingress_container=container) == health("one")
    assert invocation == [
        (
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                'set -o pipefail; wget -q -T 15 -O - "$1" | head -c 65537',
                "sh",
                url,
            ],
            {"input": "", "text": True, "capture_output": True, "timeout": 20},
        )
    ]


@pytest.mark.parametrize(
    ("ingress_container", "api_key", "network_namespace_pid"),
    [
        ("short", None, None),
        ("g" * 64, None, None),
        ("a" * 64, "key", None),
        ("a" * 64, None, 4321),
    ],
)
def test_request_json_rejects_invalid_or_ambiguous_ingress_transport(
    monkeypatch, ingress_container, api_key, network_namespace_pid
):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid ingress transport must not execute a command"),
    )

    with pytest.raises(ValueError, match="ingress container"):
        rollout_http._request_json(
            "http://10.0.1.38:11235/health",
            api_key,
            network_namespace_pid=network_namespace_pid,
            ingress_container=ingress_container,
        )


def test_request_json_reports_ingress_exec_failure(monkeypatch):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "wget: connection refused"
        ),
    )

    with pytest.raises(rollout_http.CurlError) as error:
        rollout_http._request_json("http://10.0.1.38:11235/health", ingress_container="a" * 64)

    assert error.value.curl_exit == 1
    assert "docker exec exit 1" in str(error.value)
    assert "connection refused" in str(error.value)


def test_request_json_rejects_an_oversized_ingress_response(monkeypatch):
    oversized = json.dumps({"value": "€" * 21_846}, ensure_ascii=False)
    assert len(oversized) < 65536 < len(oversized.encode())
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, oversized, ""),
    )

    with pytest.raises(ValueError, match="exceeds 65536 bytes"):
        rollout_http._request_json("http://10.0.1.38:11235/health", ingress_container="a" * 64)


@pytest.mark.parametrize("network_namespace_pid", [True, 1.5, 0])
def test_request_json_rejects_an_invalid_network_namespace_pid(
    monkeypatch, network_namespace_pid
):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid PID must not execute a command"),
    )

    with pytest.raises(ValueError, match="PID must be a positive integer"):
        rollout_http._request_json(
            "http://10.0.1.38:11235/health",
            network_namespace_pid=network_namespace_pid,
        )
