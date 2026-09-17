# Phase 1 Task Plan — Containerized Foundation

## Principle

Do not build the entire foundation in one Codex turn. Each task must end with a human review point.

## Task 1.1 — Inspect Repository and Establish Root Package Metadata

Goal:
- Inspect existing files.
- Establish or normalize root `package.json` only.

Expected code change:
- `package.json`

Validation:
- `cat package.json`
- `pnpm --version`

Stop after this task.

## Task 1.2 — Add pnpm Workspace Declaration

Goal:
- Create `pnpm-workspace.yaml`.

Expected content:

```yaml
packages:
  - "apps/*"
  - "services/*"
  - "packages/*"
```

Validation:
- `pnpm install`

Stop.

## Task 1.3 — Add Repository Ignore / Environment Templates

Create:
- `.gitignore`
- `.env.example`

Do not add secrets.

Stop.

## Task 1.4 — Create Docker Compose Skeleton

Create or modify only:
- `docker-compose.yml`

At this step, bring up PostgreSQL only.

Requirements:
- PostgreSQL image with pgvector support
- named persistent volume
- healthcheck
- environment variables from `.env`

Validation:
- `docker compose config`
- `docker compose up -d postgres`
- `docker compose ps`

Stop.

## Task 1.5 — Verify PostgreSQL + pgvector

Goal:
- Verify database connectivity.
- Verify `vector` extension.

Do not modify schema yet.

Stop.

## Task 1.6 — Add Redis

Add only the Redis service to Compose.

Validation:
- `docker compose up -d redis`
- Redis health / ping

Stop.

## Task 1.7 — Add MinIO

Add only MinIO to Compose.

Validation:
- container healthy/running
- console endpoint reachable

Stop.

## Task 1.8 — Add App Containers Skeleton

Add minimal service containers for:
- `web`
- `api`
- `ai`
- `ai-worker`

No business logic yet.

Only establish build/start contracts.

Stop.

## Task 1.9 — Add Nginx Reverse Proxy

Add Nginx only after app containers exist.

Validate routing separately.

Stop.

## Task 1.10 — Add Playwright Foundation

Add Playwright to the web project and create a single smoke test.

Do not add full user-flow tests yet.

Stop.

## Task 1.11 — Add CI Foundation

Run:
- install
- lint if available
- typecheck if available
- test if available
- build if available
- Docker config validation

Stop.

## Task 1.12 — Foundation Acceptance

Acceptance criteria:

- repository structure is stable
- Compose starts required local infrastructure
- app containers build
- Playwright smoke test works
- CI is green
- no RabbitMQ
- no AI model implementation
- no product feature implementation
