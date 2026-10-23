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
  request.
- Verify live Dokploy, Swarm, ingress, image, and telemetry state through their
  authoritative owners; never reconstruct those inventories in this repository.

For explicitly authorized source changes, preserve the existing repository
architecture and use its repository-owned validation lanes.

## Landing and delivery

- The working branch is `main`, which is also production: every push to it
  runs `CI & Deploy`. Land with `git-land <repo-root> <task-slug> main`; the
  fork carries upstream's stale `staging` branch, which `git-land`'s default
  branch choice would otherwise pick.
- Every workflow job runs on this fork's self-hosted runners
  (`tests/regression/test_workflows_self_hosted.py` enforces it); never add a
  GitHub-hosted runner or a `pull_request` trigger.
- A stored Redis format change ships its readers one deploy before its
  writers: start-first rolling updates run old and new replicas against the
  same Redis for about 28 minutes.
