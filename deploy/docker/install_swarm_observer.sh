#!/bin/sh
set -eu

action=${1:-}
revision=${2:-}
root=/opt/crawl4ai-swarm-observer
server_unit=crawl4ai-swarm-observer.service
sample_unit=crawl4ai-swarm-observer-sample.service
timer_unit=crawl4ai-swarm-observer.timer

case "$revision" in
    *[!0-9a-f]*|'')
        echo "revision must be a lowercase hexadecimal Git SHA" >&2
        exit 64
        ;;
esac
[ "${#revision}" -eq 40 ] || {
    echo "revision must contain 40 characters" >&2
    exit 64
}

point_current() {
    target=$1
    link="$root/.current-$target-$$"
    ln -s "releases/$target" "$link"
    mv -Tf "$link" "$root/current"
}

activate() {
    target=$1
    release="$root/releases/$target"
    point_current "$target"
    install -m 0600 "$release/swarm-observer.env" /etc/crawl4ai/swarm-observer.env
    install -m 0444 "$release/$server_unit" "/etc/systemd/system/$server_unit"
    install -m 0444 "$release/$sample_unit" "/etc/systemd/system/$sample_unit"
    install -m 0444 "$release/$timer_unit" "/etc/systemd/system/$timer_unit"
    systemctl daemon-reload
    systemctl enable "$server_unit" "$timer_unit"
    systemctl restart "$server_unit"
    systemctl start "$timer_unit"
    systemctl is-active --quiet "$server_unit" "$timer_unit"
    # shellcheck disable=SC1091
    . /etc/crawl4ai/swarm-observer.env
    curl --fail --silent --show-error --max-time 10 \
        "http://$CRAWL4AI_OBSERVER_BIND:$CRAWL4AI_OBSERVER_PORT/metrics"
}

restore_on_error() {
    status=$?
    trap - EXIT
    set +e
    if [ -n "$prior" ] && [ -d "$root/$prior" ]; then
        activate "${prior#releases/}"
    else
        systemctl stop "$sample_unit"
        systemctl disable --now "$server_unit" "$timer_unit"
        unlink "$root/current"
    fi
    exit "$status"
}

case "$action" in
    install)
        service_name=${CRAWL4AI_SWARM_SERVICE:?set CRAWL4AI_SWARM_SERVICE}
        bind=${CRAWL4AI_OBSERVER_BIND:?set CRAWL4AI_OBSERVER_BIND}
        port=${CRAWL4AI_OBSERVER_PORT:-9476}
        case "$service_name$bind$port" in
            *[!A-Za-z0-9_.:-]*)
                echo "observer settings contain unsupported characters" >&2
                exit 64
                ;;
        esac
        tailscale ip -4 | grep -Fx -- "$bind" >/dev/null || {
            echo "CRAWL4AI_OBSERVER_BIND must be this host's tailnet address" >&2
            exit 64
        }
        /usr/bin/python3 -c 'import yaml' || {
            echo "system Python must provide PyYAML for the sampler" >&2
            exit 69
        }
        source_dir=$(dirname -- "$0")
        source_dir=$(CDPATH='' cd -- "$source_dir" && pwd)
        release="$root/releases/$revision"
        install -d -m 0755 "$release" /etc/crawl4ai
        install -m 0555 "$source_dir/swarm_observer.py" "$release/swarm_observer.py"
        install -m 0555 "$source_dir/swarm_observer_server.py" "$release/swarm_observer_server.py"
        install -m 0444 "$source_dir/verify_rollout.py" "$release/verify_rollout.py"
        install -m 0444 "$source_dir/$server_unit" "$release/$server_unit"
        install -m 0444 "$source_dir/$sample_unit" "$release/$sample_unit"
        install -m 0444 "$source_dir/$timer_unit" "$release/$timer_unit"
        environment=$(mktemp "$release/.swarm-observer.env.XXXXXX")
        chmod 0600 "$environment"
        {
            printf 'CRAWL4AI_SWARM_SERVICE=%s\n' "$service_name"
            printf 'CRAWL4AI_OBSERVER_BIND=%s\n' "$bind"
            printf 'CRAWL4AI_OBSERVER_PORT=%s\n' "$port"
            printf 'CRAWL4AI_OBSERVER_STATE_PATH=/var/lib/crawl4ai-swarm-observer/episode.state\n'
            printf 'CRAWL4AI_OBSERVER_METRICS_PATH=/var/lib/crawl4ai-swarm-observer/metrics.prom\n'
        } >"$environment"
        mv -f "$environment" "$release/swarm-observer.env"
        ;;
    rollback)
        [ -d "$root/releases/$revision" ] || {
            echo "requested observer release is not installed" >&2
            exit 66
        }
        ;;
    *)
        echo "usage: $0 {install|rollback} <40-character-revision>" >&2
        exit 64
        ;;
esac
prior=$(readlink "$root/current" 2>/dev/null || true)
trap restore_on_error EXIT
activate "$revision"
trap - EXIT
