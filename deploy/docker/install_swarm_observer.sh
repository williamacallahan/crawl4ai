#!/bin/sh
set -eu

action=${1:-}
revision=${2:-}
root=/opt/crawl4ai-swarm-observer
unit=crawl4ai-swarm-observer.service

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
    link="$root/.current-$revision"
    ln -s "releases/$target" "$link"
    mv -Tf "$link" "$root/current"
}

case "$action" in
    install)
        service_name=${CRAWL4AI_SWARM_SERVICE:?set CRAWL4AI_SWARM_SERVICE}
        bind=${CRAWL4AI_OBSERVER_BIND:?set CRAWL4AI_OBSERVER_BIND}
        port=${CRAWL4AI_OBSERVER_PORT:-9476}
        interval=${CRAWL4AI_OBSERVER_INTERVAL_SECONDS:-15}
        case "$service_name$bind$port$interval" in
            *[!A-Za-z0-9_.:-]*)
                echo "observer settings contain unsupported characters" >&2
                exit 64
                ;;
        esac
        source_dir=$(dirname -- "$0")
        source_dir=$(CDPATH='' cd -- "$source_dir" && pwd)
        release="$root/releases/$revision"
        install -d -m 0755 "$release" /etc/crawl4ai
        install -m 0555 "$source_dir/swarm_observer.py" "$release/swarm_observer.py"
        install -m 0444 "$source_dir/verify_rollout.py" "$release/verify_rollout.py"
        install -m 0444 "$source_dir/crawl4ai-swarm-observer.service" \
            "/etc/systemd/system/$unit"
        environment=$(mktemp /etc/crawl4ai/.swarm-observer.env.XXXXXX)
        chmod 0600 "$environment"
        {
            printf 'CRAWL4AI_SWARM_SERVICE=%s\n' "$service_name"
            printf 'CRAWL4AI_OBSERVER_BIND=%s\n' "$bind"
            printf 'CRAWL4AI_OBSERVER_PORT=%s\n' "$port"
            printf 'CRAWL4AI_OBSERVER_INTERVAL_SECONDS=%s\n' "$interval"
        } >"$environment"
        mv -f "$environment" /etc/crawl4ai/swarm-observer.env
        point_current "$revision"
        systemctl daemon-reload
        systemctl enable "$unit"
        systemctl restart "$unit"
        systemctl is-active --quiet "$unit"
        curl --fail --silent --show-error --max-time 10 \
            "http://$bind:$port/metrics"
        ;;
    rollback)
        [ -d "$root/releases/$revision" ] || {
            echo "requested observer release is not installed" >&2
            exit 66
        }
        point_current "$revision"
        systemctl restart "$unit"
        systemctl is-active --quiet "$unit"
        ;;
    *)
        echo "usage: $0 {install|rollback} <40-character-revision>" >&2
        exit 64
        ;;
esac
