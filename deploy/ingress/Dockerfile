# syntax=docker/dockerfile:1

FROM haproxy:3.2.23-alpine3.24@sha256:93de1368b406157be4cded231bb34336e8477e8db24e90f5fb830bec99142331

COPY --chown=haproxy:haproxy deploy/ingress/haproxy.cfg /usr/local/etc/haproxy/haproxy.cfg

USER haproxy
EXPOSE 11235 8404
