"""Bounded curl transport for Dokploy, Swarm-adjacent, and public health requests."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import uuid
from typing import Any


class CurlError(RuntimeError):
    """A bounded HTTP transport exited non-zero; curl_exit carries its code.

    For curl, 22 means the service answered badly (--fail on a 4xx/5xx);
    7/28/35/52/56 point at transport between the runner and the host, not the
    service.
    """

    def __init__(self, message: str, curl_exit: int):
        super().__init__(message)
        self.curl_exit = curl_exit


_PROBE_TIMING_MARKER = "\n__crawl4ai_probe_timing__"
_PROBE_TIMING_FIELDS = (
    "response_code",
    "dns_ms",
    "connect_ms",
    "tls_ms",
    "start_transfer_ms",
    "curl_total_ms",
)


def _record_probe_timing(probe: dict[str, Any], stdout: str) -> str:
    """Strip curl timing output and retain only numeric phase measurements."""
    probe.update(echoed_probe_id=None, correlation_verified=False)
    body, marker, raw_timing = stdout.rpartition(_PROBE_TIMING_MARKER)
    if not marker:
        probe["timing_error"] = "curl timing output missing"
        return stdout
    values = raw_timing.strip().split("|", len(_PROBE_TIMING_FIELDS))
    if len(values) != len(_PROBE_TIMING_FIELDS) + 1:
        probe["timing_error"] = "curl timing output malformed"
        return body
    try:
        probe.update(
            zip(
                _PROBE_TIMING_FIELDS,
                (
                    int(values[0]),
                    *(round(float(value) * 1_000, 3) for value in values[1:-1]),
                ),
            )
        )
    except ValueError:
        probe["timing_error"] = "curl timing output malformed"
    if values[-1]:
        try:
            if len(values[-1]) != 36:
                raise ValueError("invalid probe ID length")
            probe["echoed_probe_id"] = str(uuid.UUID(values[-1]))
        except ValueError:
            probe["correlation_error"] = "invalid probe echo"
        else:
            if probe["echoed_probe_id"] != probe["probe_id"]:
                probe["correlation_error"] = "probe echo mismatch"
    # Older tasks do not echo the header during a mixed-version rollout.
    probe["correlation_verified"] = probe["echoed_probe_id"] == probe["probe_id"]
    return body


def _request_json(
    url: str,
    api_key: str | None = None,
    *,
    network_namespace_pid: int | None = None,
    ingress_container: str | None = None,
    probe: dict[str, Any] | None = None,
) -> Any:
    config = ""
    if ingress_container is not None:
        if (
            api_key is not None
            or network_namespace_pid is not None
            or not isinstance(ingress_container, str)
            or re.fullmatch(r"[0-9a-f]{64}", ingress_container) is None
        ):
            raise ValueError(
                "ingress container must be a full hexadecimal Docker ID without "
                "API key or namespace PID"
            )
        command = [
            "docker",
            "exec",
            ingress_container,
            "sh",
            "-c",
            'set -o pipefail; wget -q -T 15 -O - "$1" | head -c 65537',
            "sh",
            url,
        ]
    else:
        if api_key:
            escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
            config = f'header = "x-api-key: {escaped}"\n'
        command = [
            "curl", "--silent", "--show-error", "--fail",
            "--connect-timeout", "5", "--max-time", "15",
            "--max-filesize", "65536", "--config", "-", url,
        ]
        if probe is not None:
            command.extend(
                [
                    "--header", f"X-Crawl4AI-Health-Probe: {probe['probe_id']}",
                    "--write-out",
                    (
                        f"{_PROBE_TIMING_MARKER}%{{response_code}}|%{{time_namelookup}}|"
                        "%{time_connect}|%{time_appconnect}|%{time_starttransfer}|"
                        "%{time_total}|%header{x-crawl4ai-health-probe}"
                    ),
                ]
            )
        if network_namespace_pid is not None:
            if (
                isinstance(network_namespace_pid, bool)
                or not isinstance(network_namespace_pid, int)
                or network_namespace_pid <= 0
            ):
                raise ValueError("network namespace PID must be a positive integer")
            command = [
                "nsenter",
                f"--net=/proc/{network_namespace_pid}/ns/net",
                *command,
            ]
    response = subprocess.run(
        command,
        input=config,
        text=True,
        capture_output=True,
        timeout=20,
    )
    stdout = response.stdout
    if probe is not None:
        probe["curl_exit"] = response.returncode
        stdout = _record_probe_timing(probe, stdout)
    if response.returncode:
        # Carry curl's own diagnosis: a bare "HTTP request failed" made the
        # 2026-08-31 single-sample monitor failure unattributable (connection
        # reset vs TLS vs 5xx), leaving the gate's verdict unactionable.
        transport = "docker exec" if ingress_container is not None else "curl"
        raise CurlError(
            f"HTTP request failed: {transport} exit {response.returncode}: "
            f"{(response.stderr or '').strip()[:200]}",
            response.returncode,
        )
    if ingress_container is not None and len(response.stdout.encode()) > 65536:
        raise ValueError("ingress health response exceeds 65536 bytes")
    return json.loads(stdout) if stdout else None


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> Any:
    escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
    response = subprocess.run(
        [
            "curl", "--silent", "--show-error", "--fail",
            "--connect-timeout", "5", "--max-time", "15",
            "--max-filesize", "65536",
            "--request", "POST", "--header", "content-type: application/json",
            "--data", json.dumps(payload, separators=(",", ":")),
            "--config", "-", url,
        ],
        input=f'header = "x-api-key: {escaped}"\n',
        text=True,
        capture_output=True,
        timeout=20,
    )
    if response.returncode:
        raise CurlError(
            "HTTP request failed; state is ambiguous: "
            f"curl exit {response.returncode}: {(response.stderr or '').strip()[:200]}",
            response.returncode,
        )
    return json.loads(response.stdout) if response.stdout else None


def _url(base: str, operation: str, **params: str) -> str:
    return f"{base.rstrip('/')}/api/{operation}?{urllib.parse.urlencode(params)}"
