# Crawl4AI Fork Rules

## Scope and ownership

- This user-owned fork owns explicitly requested Crawl4AI application-source
  and repository-owned deployment-verification changes. It does not own Docker
  daemon settings, Dokploy resource configuration, generated Traefik routing,
  fleet topology, or central observability configuration.
- Use Crawl4AI and its dependencies through their documented configuration and
  APIs. An incident or deployment task is not authorization to patch application
  or upstream source when the native configuration surface owns the behavior.
- Source development requires an explicit user request naming this repository
  and source behavior. Keep that work separate from live operational recovery.
- The `upstream` repository is read-only. Never open or update an upstream pull
  request. A push to this user-owned fork requires separate explicit approval.
- Verify live Dokploy, Swarm, ingress, image, and telemetry state through their
  authoritative owners; never reconstruct those inventories in this repository.

For explicitly authorized source changes, preserve the existing repository
architecture and use its repository-owned validation lanes.
