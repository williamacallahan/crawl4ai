"""The fixed Crawl4AI rollout policy and the checks that hold Dokploy and Swarm to it."""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Any

import yaml


REPLICAS = 3
LLM_PROVIDER = yaml.safe_load(Path(__file__).with_name("config.yml").read_text())["llm"]["provider"]
LLM_BASE_URL = "https://api.llm-gateway.iocloudhost.net/v1"
HEALTH_URLS = (
    "https://crawl4ai.haiku.host/health",
    "https://crawl4ai.iocloudhost.net/health",
)
HEALTHCHECK = {
    "Test": ["CMD", "curl", "-f", "http://localhost:11235/health"],
    "Interval": 10_000_000_000,
    "Timeout": 5_000_000_000,
    "StartPeriod": 120_000_000_000,
    "Retries": 5,
}
NODE_CONSTRAINT = "node.labels.crawl4ai-eligible==true"


# Placement belongs to Dokploy. Accept the native hostname spread and optional
# one-per-node cap, but preserve the exact baseline policy during image updates.
PLACEMENT = {"Constraints": [NODE_CONSTRAINT]}
_CAPPED_PLACEMENT = {**PLACEMENT, "MaxReplicas": 1}
_SPREAD_PLACEMENT = {
    **PLACEMENT,
    "Preferences": [{"Spread": {"SpreadDescriptor": "node.hostname"}}],
}
PLACEMENTS = (
    PLACEMENT, _CAPPED_PLACEMENT, _SPREAD_PLACEMENT,
    {**_SPREAD_PLACEMENT, "MaxReplicas": 1},
    {**_SPREAD_PLACEMENT, "MaxReplicas": 2},
)
RESOURCES = {
    "Reservations": {"NanoCPUs": 500_000_000, "MemoryBytes": 1_073_741_824},
    "Limits": {"NanoCPUs": 2_000_000_000, "MemoryBytes": 4_294_967_296},
}
DELAY_NS = 400_000_000_000
STOP_GRACE_NS = 390_000_000_000
TIMEOUT_SECONDS = 4_000
PUBLIC_PROOF_ATTEMPTS = REPLICAS * 8


# Crawl4AI's durable job queue and task state live in this one external Redis.
# Nothing in the deploy path proved its wiring, so a lost volume, an unpinned
# or duplicated replica, or a reintroduced appendfsync=always would first
# surface as a production incident. The appName is immutable in Dokploy.
REDIS_SERVICE = "crawl4ai-redis-8xt08u"
REDIS_NODE = "haiku-18"


# everysec, never always: appendfsync=always was half of the 2026-08-28 fleet
# kill, stalling every crawl behind Redis. Pinned so a console edit cannot
# reintroduce it silently.
# Bound below the 1GiB limit; volatile-lru excludes the no-TTL queue and in-flight
# task hashes. Terminal results and the TTL-bearing LLM permit remain evictable;
# periodic permit renewal does not guarantee protection from eviction.
REDIS_COMMAND = [
    "redis-server",
    "--dir", "/data",
    "--appendonly", "yes",
    "--appendfsync", "everysec",
    "--loglevel", "notice",
    "--maxmemory", "700mb",
    "--maxmemory-policy", "volatile-lru",
]
REDIS_MOUNTS = [{"Type": "volume", "Source": "crawl4ai-redis-data", "Target": "/data"}]
REDIS_HEALTHCHECK_TEST = ["CMD", "redis-cli", "ping"]


# Floors, not equalities. The 2026-08-28 kill came from a healthcheck that was
# too tight, so raising Retries or StartPeriod is a safety improvement and must
# not read as drift. Only the probe itself is bidirectional.
REDIS_HEALTHCHECK_FLOORS = {
    "Interval": 5_000_000_000,
    "Timeout": 1_000_000_000,
    "StartPeriod": 10_000_000_000,
    "Retries": 3,
}


def _labels(revision: str) -> dict[str, str]:
    return {
        "otel.logs.enabled": "true",
        "otel.service.name": "crawl4ai",
        "otel.deployment.environment.name": "production",
        "otel.service.version": revision,
    }


def _environment_values(environment: Any) -> dict[str, str]:
    if isinstance(environment, str):
        lines = environment.splitlines()
    elif isinstance(environment, list) and all(
        isinstance(line, str) for line in environment
    ):
        lines = environment
    else:
        raise ValueError("Crawl4AI environment is not configured")
    values = {}
    protected = {"LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY"}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            if key in protected and key in values:
                raise ValueError(f"duplicate {key}")
            values[key] = value
    return values


def _verify_llm_environment(environment: Any) -> None:
    values = _environment_values(environment)
    if values.get("LLM_PROVIDER") != LLM_PROVIDER:
        raise ValueError("LLM_PROVIDER drifted")
    if values.get("LLM_BASE_URL") != LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL drifted")
    if not values.get("LLM_API_KEY", "").strip():
        raise ValueError("LLM_API_KEY is empty")


def _rollout_policy(failure_action: str) -> dict[str, Any]:
    return {
        "Parallelism": 1,
        "Delay": DELAY_NS,
        "FailureAction": failure_action,
        "Monitor": DELAY_NS,
        "MaxFailureRatio": 0,
        "Order": "start-first",
    }


_SLEEP_UNITS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400}


# GNU sleep uses strtod: leading ASCII whitespace, decimal/exponent or hex
# numbers, optional lower-case unit, but no trailing whitespace. Infinite
# intervals parse but cannot fit a finite stop grace and are rejected below.
_SLEEP_INTERVAL = re.compile(
    r"[ \t\n\r\v\f]*([+-]?(?:"
    r"0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?"
    r"|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?))([smhd])?"
)


def _parse_sleep_seconds(value: str) -> float:
    """Parse a finite GNU sleep interval, preserving fractional seconds.

    entrypoint.sh runs ``sleep "${CRAWL4AI_DRAIN_DELAY_SECONDS:-2}"``, so the
    deployed value reaches ``sleep`` as a single quoted argument. GNU sleep
    skips leading whitespace (a strtod artifact), then accepts one (optionally
    fractional) number with an optional single ``s``/``m``/``h``/``d`` suffix;
    trailing whitespace it rejects. Anything ``sleep`` would reject is rejected
    here too so the deploy fails loudly rather than silently falling back to the
    entrypoint literal. Falling back when the operator has set a value would
    undermeasure the real shutdown and let a rolling update SIGKILL a task
    mid-drain — the exact data-loss class the STOP_GRACE_NS gate exists to
    prevent.
    """
    # A bare leading '-' is parsed by sleep as an option, even for -0.
    match = None if value.startswith("-") else _SLEEP_INTERVAL.fullmatch(value)
    if match is None:
        raise ValueError(
            "CRAWL4AI_DRAIN_DELAY_SECONDS must be a number bash sleep accepts"
            f" (a number with an optional s/m/h/d suffix); got {value!r}"
        )
    number = match.group(1)
    try:
        seconds = (float.fromhex(number) if "0x" in number.lower() else float(number))
    except OverflowError as error:
        raise ValueError("CRAWL4AI_DRAIN_DELAY_SECONDS exceeds a finite drain budget") from error
    seconds *= _SLEEP_UNITS[match.group(2) or "s"]
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("CRAWL4AI_DRAIN_DELAY_SECONDS must have a finite nonnegative drain budget")
    return seconds


def _drain_budget_ns(environment: Any) -> int:
    """What PID 1 needs to shut down cleanly, read from the sources that own it.

    Swarm SIGKILLs a task the moment StopGracePeriod expires, so a stop grace
    shorter than supervisord's stop wait plus the entrypoint's pre-termination
    delay truncates the drain. Nothing tied STOP_GRACE_NS to either number.

    The delay is read from the deployed environment when it sets one: the
    literal in entrypoint.sh is only the fallback, and reading it alone would
    miss a value raised in the Dokploy console. The deployed value is parsed
    as the single GNU sleep interval the container's ``sleep`` honors — a
    number with an optional ``s``/``m``/``h``/``d`` suffix, fractional or not —
    so a value the container actually sleeps (``1m`` or ``31.5``) is what is
    measured, and a value that is set but unparseable fails the deploy loudly
    instead of being masked by the fallback literal.
    """
    here = Path(__file__).resolve().parent
    # The slowest program in the group bounds supervisord's own shutdown.
    stop_waits = re.findall(
        r"^stopwaitsecs=(\d+)\s*$", (here / "supervisord.conf").read_text(), re.MULTILINE
    )
    fallback = re.search(
        r"CRAWL4AI_DRAIN_DELAY_SECONDS:-(\d+)", (here / "entrypoint.sh").read_text()
    )
    if not stop_waits or not fallback:
        raise ValueError("container drain budget is no longer readable from its own sources")
    raw = _environment_values(environment).get("CRAWL4AI_DRAIN_DELAY_SECONDS", "")
    # Bash's ${VAR:-2} falls back only for unset or empty, not whitespace.
    if raw == "":
        drain = int(fallback.group(1))
    else:
        drain = _parse_sleep_seconds(raw)
    return max(int(value) for value in stop_waits) * 1_000_000_000 + math.ceil(drain * 1_000_000_000)


def _policy(application: dict[str, Any]) -> None:
    if application.get("sourceType") != "docker":
        raise ValueError("Crawl4AI must remain a stock Dokploy Docker-image app")
    _verify_llm_environment(application.get("env"))
    if application.get("replicas") != REPLICAS:
        raise ValueError("Crawl4AI must keep three replicas")
    if application.get("healthCheckSwarm") != HEALTHCHECK:
        raise ValueError("Crawl4AI admission healthcheck drifted")
    if application.get("placementSwarm") not in PLACEMENTS:
        raise ValueError("Crawl4AI placement drifted")
    if application.get("endpointSpecSwarm") != {"Mode": "vip", "Ports": []}:
        raise ValueError("Crawl4AI must use the native Swarm VIP")
    for field, action in (("updateConfigSwarm", "rollback"), ("rollbackConfigSwarm", "pause")):
        if application.get(field) != _rollout_policy(action):
            raise ValueError(f"{field} drifted")
    # Strictly greater: a grace equal to the budget leaves no room for the
    # SIGTERM itself to be delivered and observed.
    if STOP_GRACE_NS <= _drain_budget_ns(application.get("env")):
        raise ValueError("Crawl4AI stop grace no longer covers the container drain budget")
    if int(application.get("stopGracePeriodSwarm") or 0) != STOP_GRACE_NS:
        raise ValueError("Crawl4AI stop grace drifted")
    for field, expected in (
        ("cpuReservation", "500000000"), ("cpuLimit", "2000000000"),
        ("memoryReservation", "1073741824"), ("memoryLimit", "4294967296"),
    ):
        if application.get(field) != expected:
            raise ValueError(f"Crawl4AI {field} drifted")


def _running_spec(
    spec: dict[str, Any],
    image: str,
    labels: dict[str, str],
    placements: tuple[dict[str, Any], ...] = PLACEMENTS,
) -> None:
    task = spec.get("TaskTemplate") or {}
    container = task.get("ContainerSpec") or {}
    # The record owns its labels; Dokploy stamps its own (dokploy.deployment.id
    # since the 2026-09-01 control-plane build) beside them, so require the
    # record's labels rather than exact equality.
    if container.get("Image") != image or not (
        labels.items() <= (container.get("Labels") or {}).items()
    ):
        raise ValueError("running artifact or labels differ from Dokploy")
    _verify_llm_environment(container.get("Env"))
    if container.get("Healthcheck") != HEALTHCHECK:
        raise ValueError("running healthcheck drifted")
    if task.get("Placement") not in placements:
        raise ValueError("running placement drifted")
    if task.get("Resources") != RESOURCES:
        raise ValueError("running resources drifted")
    if container.get("StopGracePeriod", 0) != STOP_GRACE_NS:
        raise ValueError("running stop grace drifted")
    if spec.get("Mode") != {"Replicated": {"Replicas": REPLICAS}}:
        raise ValueError("running replica count drifted")
    for field, action in (("UpdateConfig", "rollback"), ("RollbackConfig", "pause")):
        if spec.get(field) != _rollout_policy(action):
            raise ValueError(f"running {field} drifted")
    endpoint = spec.get("EndpointSpec") or {}
    if endpoint.get("Mode") != "vip" or endpoint.get("Ports") not in (None, []):
        raise ValueError("running endpoint mode drifted")
