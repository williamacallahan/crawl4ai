"""Privileged Crawl4AI coverage sampler with identity-free durable output."""

from __future__ import annotations

import ipaddress
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import verify_rollout as rollout


DEFAULT_STATE_PATH = "/var/lib/crawl4ai-swarm-observer/episode.state"
DEFAULT_METRICS_PATH = "/var/lib/crawl4ai-swarm-observer/metrics.prom"
SAMPLE_DEADLINE_SECONDS = 45.0


def _atomic_replace(
    path: str | Path, content: bytes, mode: int, *, durable: bool
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), mode)
            output.write(content)
            if durable:
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_counts(
    authoritative_task_count: int,
    direct_healthy_count: int,
    public_covered_count: int,
) -> None:
    for name, count in (
        ("authoritative_task_count", authoritative_task_count),
        ("direct_healthy_count", direct_healthy_count),
        ("public_covered_count", public_covered_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer")
        if count < 0:
            raise ValueError(f"{name} must not be negative")
    if direct_healthy_count > authoritative_task_count:
        raise ValueError("direct_healthy_count exceeds authoritative_task_count")
    if public_covered_count > direct_healthy_count:
        raise ValueError("public_covered_count exceeds direct_healthy_count")


@dataclass(frozen=True, slots=True)
class CoverageSnapshot:
    """One low-cardinality coverage sample."""

    authoritative_task_count: int
    direct_healthy_count: int
    public_covered_count: int

    def __post_init__(self) -> None:
        _validate_counts(
            self.authoritative_task_count,
            self.direct_healthy_count,
            self.public_covered_count,
        )

    @property
    def complete(self) -> int:
        return int(
            self.authoritative_task_count == rollout.REPLICAS
            and self.authoritative_task_count
            == self.direct_healthy_count
            == self.public_covered_count
        )

    def metrics_text(self) -> str:
        """Return the four fixed, unlabeled Prometheus samples."""
        return (
            f"crawl4ai_authoritative_task_count {self.authoritative_task_count}\n"
            f"crawl4ai_direct_healthy_task_count {self.direct_healthy_count}\n"
            f"crawl4ai_public_covered_task_count {self.public_covered_count}\n"
            f"crawl4ai_coverage_complete {self.complete}\n"
        )


class NetworkDbFdbComparison(str, Enum):
    """Bounded NetworkDB and ingress-FDB comparison outcomes."""

    CONSISTENT = "consistent"
    DESTINATION_MISMATCH = "destination_mismatch"
    MISSING = "missing"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class CoverageDiagnostic:
    """The one redacted diagnostic emitted for an incomplete episode."""

    authoritative_task_count: int
    direct_healthy_count: int
    public_covered_count: int
    networkdb_fdb_comparison: NetworkDbFdbComparison

    def __post_init__(self) -> None:
        _validate_counts(
            self.authoritative_task_count,
            self.direct_healthy_count,
            self.public_covered_count,
        )
        if not isinstance(self.networkdb_fdb_comparison, NetworkDbFdbComparison):
            raise TypeError("networkdb_fdb_comparison must be a NetworkDbFdbComparison")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CoverageSnapshot,
        networkdb_fdb_comparison: NetworkDbFdbComparison,
    ) -> CoverageDiagnostic:
        if not isinstance(snapshot, CoverageSnapshot):
            raise TypeError("snapshot must be a CoverageSnapshot")
        return cls(
            snapshot.authoritative_task_count,
            snapshot.direct_healthy_count,
            snapshot.public_covered_count,
            networkdb_fdb_comparison,
        )

    def text(self) -> str:
        return (
            "crawl4ai coverage incomplete: "
            f"authoritative_tasks={self.authoritative_task_count} "
            f"direct_healthy={self.direct_healthy_count} "
            f"public_covered={self.public_covered_count} "
            f"networkdb_fdb={self.networkdb_fdb_comparison.value}"
        )


class CoverageEpisodeLatch:
    """Persist whether the current incomplete-coverage episode was reported."""

    _OPEN = b"open\n"
    _PENDING = b"pending\n"
    _CLOSED = b"closed\n"

    def __init__(self, state_path: str | Path) -> None:
        self._state_path = Path(state_path)
        if not self._state_path.name:
            raise ValueError("state_path must name a file")
        self._state = self._load()

    def observe(
        self,
        snapshot: CoverageSnapshot,
        networkdb_fdb_comparison: NetworkDbFdbComparison,
    ) -> CoverageDiagnostic | None:
        """Return one diagnostic per incomplete episode and persist the latch."""
        if not isinstance(snapshot, CoverageSnapshot):
            raise TypeError("snapshot must be a CoverageSnapshot")
        if not isinstance(networkdb_fdb_comparison, NetworkDbFdbComparison):
            raise TypeError("networkdb_fdb_comparison must be a NetworkDbFdbComparison")
        if snapshot.complete:
            if self._state != self._CLOSED:
                self._persist(self._CLOSED)
                self._state = self._CLOSED
            return None
        if self._state == self._OPEN:
            return None
        if self._state == self._CLOSED:
            self._persist(self._PENDING)
            self._state = self._PENDING
        return CoverageDiagnostic.from_snapshot(snapshot, networkdb_fdb_comparison)

    def acknowledge(self) -> None:
        if self._state != self._PENDING:
            raise RuntimeError("coverage diagnostic is not pending")
        self._persist(self._OPEN)
        self._state = self._OPEN

    def _load(self) -> bytes:
        try:
            state = self._state_path.read_bytes()
        except FileNotFoundError:
            return self._CLOSED
        if state in {self._OPEN, self._PENDING, self._CLOSED}:
            return state
        raise ValueError("coverage episode state is invalid")

    def _persist(self, state: bytes) -> None:
        _atomic_replace(self._state_path, state, 0o600, durable=True)


def _network_details() -> dict[str, Any]:
    inspected = subprocess.run(
        ["docker", "network", "inspect", "--verbose", "dokploy-network"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    details = json.loads(inspected.stdout)
    if len(details) != 1 or not isinstance(details[0], dict):
        raise ValueError("invalid overlay network response")
    return details[0]


def _overlay_mac(address: str) -> str:
    octet = ipaddress.IPv4Address(address).packed
    return "02:42:" + ":".join(f"{part:02x}" for part in octet)


def _fdb_comparison(
    network_id: str,
    task: list[dict[str, Any]],
    runtime_by_task: dict[str, dict[str, Any]],
    data_path_by_task: dict[str, str],
) -> NetworkDbFdbComparison:
    expected = {}
    for row in task:
        task_id = str(row["ID"])
        address = runtime_by_task[task_id]["addresses"]
        if len(address) != 1 or task_id not in data_path_by_task:
            return NetworkDbFdbComparison.INCONCLUSIVE
        expected[_overlay_mac(next(iter(address)))] = data_path_by_task[task_id]
    namespace = Path("/var/run/docker/netns") / f"1-{network_id[:12]}"
    result = subprocess.run(
        [
            "nsenter",
            f"--net={namespace}",
            "bridge",
            "fdb",
            "show",
            "dev",
            "vxlan0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    observed = {}
    seen_mac = set()
    for line in result.stdout.splitlines():
        part = line.split()
        if not part or part[0].lower() not in expected:
            continue
        seen_mac.add(part[0].lower())
        if len(part) >= 4 and "dst" in part:
            observed.setdefault(part[0].lower(), set()).add(part[part.index("dst") + 1])
    if any(mac in seen_mac and mac not in observed for mac in expected):
        return NetworkDbFdbComparison.INCONCLUSIVE
    if any(mac not in observed for mac in expected):
        return NetworkDbFdbComparison.MISSING
    if any(observed[mac] != {expected[mac]} for mac in expected):
        return NetworkDbFdbComparison.DESTINATION_MISMATCH
    return NetworkDbFdbComparison.CONSISTENT


def _public_coverage(direct_instance: dict[str, str], deadline: float) -> int:
    if not direct_instance:
        return 0
    observed = []
    for url in rollout.HEALTH_URLS:
        seen = set()
        for _ in range(rollout.PUBLIC_PROOF_ATTEMPTS):
            if time.monotonic() >= deadline:
                raise TimeoutError("coverage sampling deadline expired")
            try:
                health = rollout._request_json(f"{url}?observer={uuid.uuid4()}")
            except Exception:
                break
            instance = (
                str(health.get("instance", "")) if isinstance(health, dict) else ""
            )
            if instance in direct_instance and rollout._exact_health(
                health, direct_instance[instance]
            ):
                seen.add(instance)
            if len(seen) == len(direct_instance):
                break
        observed.append(seen)
    return len(set.intersection(*observed))


def sample(service_name: str) -> tuple[CoverageSnapshot, NetworkDbFdbComparison]:
    """Reuse the rollout verifier's task and health owners for one stable sample."""
    deadline = time.monotonic() + SAMPLE_DEADLINE_SECONDS
    rollout._verify_ingress_host()
    before = rollout._service_tasks(service_name)
    current = [
        row for row in before if str(row.get("DesiredState", "")).lower() == "running"
    ]
    authoritative_task_count = len(current)
    network = _network_details()
    data_path_by_task = {
        str(task.get("Name", "")).rsplit(".", 1)[-1]: str(
            (task.get("Info") or {}).get("Host IP", "")
        )
        for task in (network.get("Services") or {})
        .get(service_name, {})
        .get("Tasks", [])
        if (task.get("Info") or {}).get("Host IP")
    }
    runtime_by_task = {}
    direct_instance = {}
    for row in current:
        if time.monotonic() >= deadline:
            raise TimeoutError("coverage sampling deadline expired")
        if not str(row.get("CurrentState", "")).startswith("Running "):
            continue
        task_id = str(row["ID"])
        try:
            runtime = rollout._task_runtime(task_id[:12], str(network["Id"]))
            runtime_by_task[task_id] = runtime
            address = runtime["addresses"]
            revision = str(runtime["labels"].get("otel.service.version", ""))
            if len(address) != 1:
                continue
            health = rollout._request_json(f"http://{next(iter(address))}:11235/health")
            if (
                rollout._exact_health(health, revision)
                and health["instance"] == runtime["container"]
            ):
                direct_instance[runtime["container"]] = revision
        except Exception:
            continue
    snapshot = CoverageSnapshot(
        authoritative_task_count,
        len(direct_instance),
        _public_coverage(direct_instance, deadline),
    )
    if snapshot.complete or len(runtime_by_task) != authoritative_task_count:
        comparison = NetworkDbFdbComparison.INCONCLUSIVE
    else:
        try:
            comparison = _fdb_comparison(
                str(network["Id"]), current, runtime_by_task, data_path_by_task
            )
        except Exception:
            comparison = NetworkDbFdbComparison.INCONCLUSIVE
    after_ids = {
        str(row["ID"])
        for row in rollout._service_tasks(service_name)
        if str(row.get("DesiredState", "")).lower() == "running"
    }
    if after_ids != {str(row["ID"]) for row in current}:
        raise RuntimeError("Crawl4AI task census changed during coverage sampling")
    return snapshot, comparison


def _write_metrics(path: str | Path, metrics: str) -> None:
    _atomic_replace(path, metrics.encode(), 0o644, durable=False)


def write_sample() -> None:
    latch = CoverageEpisodeLatch(
        os.environ.get("CRAWL4AI_OBSERVER_STATE_PATH", DEFAULT_STATE_PATH)
    )
    try:
        snapshot, comparison = sample(os.environ["CRAWL4AI_SWARM_SERVICE"])
    except Exception as error:
        snapshot = CoverageSnapshot(0, 0, 0)
        comparison = NetworkDbFdbComparison.INCONCLUSIVE
        print(f"crawl4ai coverage sampling failed: {type(error).__name__}", flush=True)
    diagnostic = latch.observe(snapshot, comparison)
    if diagnostic is not None:
        print(diagnostic.text(), flush=True)
        latch.acknowledge()
    _write_metrics(
        os.environ.get("CRAWL4AI_OBSERVER_METRICS_PATH", DEFAULT_METRICS_PATH),
        snapshot.metrics_text(),
    )


if __name__ == "__main__":
    write_sample()
