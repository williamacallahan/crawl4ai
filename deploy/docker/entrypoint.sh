#!/usr/bin/env bash
# entrypoint.sh - resolve the socket-level egress/auth posture before supervisord.
#
# This is the authoritative socket-level guard (gunicorn binds the socket, not
# Python). It must agree with the in-process _resolve_auth() check in server.py.
set -euo pipefail

# --- Redis password: prefer a mounted secret, else an existing env var. ------
if [[ -z "${REDIS_PASSWORD:-}" && -f /run/secrets/redis_password ]]; then
    REDIS_PASSWORD="$(cat /run/secrets/redis_password)"
fi
if [[ -z "${REDIS_PASSWORD:-}" && -n "${REDIS_PASSWORD_FROM_HOST:-}" ]]; then
    REDIS_PASSWORD="${REDIS_PASSWORD_FROM_HOST}"
fi
unset REDIS_PASSWORD_FROM_HOST
case "${REDIS_HOST:-localhost}" in
    ""|localhost|127.0.0.1|::1) EMBEDDED_REDIS=true ;;
    *) EMBEDDED_REDIS=false ;;
esac
if [[ -z "${REDIS_PASSWORD:-}" && "${EMBEDDED_REDIS}" == "true" ]]; then
    echo "entrypoint: embedded Redis requires an existing operator-managed REDIS_PASSWORD." >&2
    exit 1
fi
# An external Redis may deliberately be protected by network isolation only.
# Do not fabricate a password that only the client knows: that would make an
# unprotected sidecar reject every authenticated request.  Export an empty
# value so supervisord interpolation and redis_config share the same posture.
export REDIS_PASSWORD="${REDIS_PASSWORD:-}"

# --- API token: prefer a mounted secret, else an existing env var. -----------
if [[ -z "${CRAWL4AI_API_TOKEN:-}" && -f /run/secrets/api_token ]]; then
    export CRAWL4AI_API_TOKEN="$(cat /run/secrets/api_token)"
fi
if [[ -z "${CRAWL4AI_API_TOKEN:-}" && -n "${CRAWL4AI_API_TOKEN_FROM_HOST:-}" ]]; then
    export CRAWL4AI_API_TOKEN="${CRAWL4AI_API_TOKEN_FROM_HOST}"
fi
unset CRAWL4AI_API_TOKEN_FROM_HOST

# --- Bind resolution: loopback unless a credential is present. ---------------
PORT="${CRAWL4AI_PORT:-11235}"
read -r CONFIG_API_TOKEN_SET CONFIG_JWT_ENABLED < <(
    python3 -c 'import yaml; c=yaml.safe_load(open("config.yml")) or {}; s=c.get("security", {}); j=s.get("jwt_enabled", False); assert isinstance(j, bool), "security.jwt_enabled must be a YAML boolean"; print(str(bool(s.get("api_token", ""))).lower(), str(j).lower())'
)
JWT_ENABLED="${CRAWL4AI_JWT_ENABLED_FROM_HOST:-${CRAWL4AI_JWT_ENABLED:-}}"
unset CRAWL4AI_JWT_ENABLED_FROM_HOST
if [[ -z "${JWT_ENABLED}" ]]; then
    JWT_ENABLED="${CONFIG_JWT_ENABLED}"
fi
case "${JWT_ENABLED,,}" in
    true|1|yes|on) JWT_ENABLED=true ;;
    false|0|no|off) JWT_ENABLED=false ;;
    *) echo "entrypoint: invalid CRAWL4AI_JWT_ENABLED value" >&2; exit 1 ;;
esac
export CRAWL4AI_JWT_ENABLED="${JWT_ENABLED}"

if [[ -n "${CRAWL4AI_API_TOKEN:-}" || "${CONFIG_API_TOKEN_SET}" == "true" || "${JWT_ENABLED}" == "true" ]]; then
    # A credential is configured -> the operator may expose all interfaces.
    GUNICORN_BIND="${GUNICORN_BIND:-0.0.0.0:${PORT}}"
else
    # No credential -> refuse to expose; serve loopback only.
    GUNICORN_BIND="127.0.0.1:${PORT}"
    echo "entrypoint: no CRAWL4AI_API_TOKEN set; binding loopback only (${GUNICORN_BIND})." >&2
fi
export GUNICORN_BIND

supervisord -c supervisord.conf --pidfile /tmp/supervisord.pid &
SUPERVISORD_PID=$!

begin_drain() {
    trap '' TERM INT
    # Swarm withdraws the task before SIGTERM. Keep established VIP connections
    # ready through the gossip delay instead of poisoning them with a 503.
    sleep "${CRAWL4AI_DRAIN_DELAY_SECONDS:-2}"
    kill -TERM "${SUPERVISORD_PID}" 2>/dev/null || true
    wait "${SUPERVISORD_PID}" || exit $?
    exit 0
}

trap begin_drain TERM INT
wait "${SUPERVISORD_PID}"
