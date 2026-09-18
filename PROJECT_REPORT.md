# AI Dating RWD Platform — Project Report for Codex

## 0. Project Intent

Build an AI-powered responsive dating web application inspired by the interaction model of modern dating apps, while keeping the product architecture modular and testable.

Core flow:

```text
Register / Login
    -> Selfie Upload
    -> Face + Liveness Verification
    -> Basic Profile
    -> Dating Preferences
    -> Interests / Hobbies / Foods
    -> AI Feature Generation
    -> Discovery Recommendations
    -> Like / Pass
    -> Match
    -> Realtime Chat
    -> Conversation Feature Extraction
    -> User Feature Update
    -> Re-clustering / Re-ranking
    -> Better Recommendations
```

## 1. Finalized Technology Decisions

### Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS
- shadcn/ui
- TanStack Query
- Zustand
- React Hook Form
- Zod
- Playwright for E2E

### Backend
- Node.js
- NestJS
- TypeScript
- Prisma
- Socket.IO
- BullMQ

### Data
- PostgreSQL
- pgvector
- Redis
- MinIO for local object storage; S3-compatible storage in production

### AI
- Python
- FastAPI
- PyTorch
- Transformers
- scikit-learn
- InsightFace / ArcFace or equivalent face recognition stack
- Liveness / anti-spoofing model
- Text / image embedding models
- LLM for structured conversation feature extraction

### Infrastructure
- Docker
- Docker Compose
- Nginx
- GitHub Actions

### Explicit Decision: RabbitMQ

Do **not** add RabbitMQ in the initial architecture. BullMQ + Redis is sufficient for the current asynchronous workload. Revisit RabbitMQ only through an ADR if scale, routing, delivery semantics, or worker isolation create a concrete requirement.

## 2. Architecture Decision

Node.js is the product backend. Python is the AI service boundary.

```text
Browser
  -> Next.js
  -> NestJS API
      -> PostgreSQL / pgvector
      -> Redis / BullMQ
      -> MinIO / S3
      -> Python AI service / workers
```

### Node.js responsibilities
- Authentication
- Users
- Profiles
- Preferences
- Interests / hobbies / foods
- Photo metadata
- Verification orchestration
- Discovery orchestration
- Like / pass / match
- Conversations / messages
- Notifications
- Recommendation orchestration
- API contracts
- WebSocket / Socket.IO
- Queue creation and job orchestration

### Python AI responsibilities
- Face detection
- Liveness / anti-spoofing
- Face verification / embeddings
- Image embeddings
- Text embeddings
- Conversation analysis
- User feature generation
- Clustering
- Ranking model inference
- AI evaluations

## 3. Core Architecture Rule

Never place a large AI model directly inside the NestJS process.

Preferred pattern:

```text
NestJS
  -> BullMQ / Redis
  -> Python AI Worker
  -> PostgreSQL / object storage
```

For low-latency verification, a synchronous internal HTTP call to the Python AI service may be used.

## 4. Recommended Monorepo

```text
ai-dating-platform/
├── AGENTS.md
├── README.md
├── ARCHITECTURE.md
├── SECURITY.md
├── PRIVACY.md
├── CONTRIBUTING.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── package.json
├── pnpm-workspace.yaml
│
├── apps/
│   ├── web/
│   └── api/
│
├── services/
│   └── ai/
│
├── packages/
│   ├── contracts/
│   └── types/
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── database/
│   ├── ai/
│   └── tasks/
│
├── infrastructure/
│   ├── docker/
│   ├── nginx/
│   ├── postgres/
│   └── monitoring/
│
├── scripts/
└── .github/
    └── workflows/
```

## 5. Docker Philosophy

Codex should own local Docker operations after the repository instructions are established.

Target local environment:

```text
nginx
web
api
redis
heartlink-pg        # PostgreSQL 16 + pgvector (was: postgres)
heartlink-minio     # object storage (was: minio)
```

The `ai` / `ai-worker` services were removed: face verification was never wired to a real
model, so they only idled. `POST /onboarding/selfie` now returns `status=unavailable` with
`reasonCode=AI_SERVICE_UNAVAILABLE` via the existing fallback path. Source is kept under
`services/ai/`; restore the service definitions in `docker-compose.yml` to bring them back.

The first Docker milestone should be small and validated incrementally. Do not create all application containers in one giant change unless the current task explicitly asks for it.

## 6. Database Direction

Use PostgreSQL as the system of record and pgvector for embeddings.

Primary domains:

- users
- profiles
- preferences
- interests
- user_interests
- hobbies
- user_hobbies
- foods
- user_foods
- user_photos
- verification_records
- face_embeddings
- image_embeddings
- user_feature_vectors
- user_clusters
- likes
- matches
- blocks
- conversations
- conversation_members
- messages
- conversation_features
- recommendation_events
- notifications

Raw messages are immutable business records. AI-derived features are stored separately and versioned.

## 7. Recommendation Direction

Do not equate clustering with recommendation.

Recommended pipeline:

```text
Hard Filter
  -> Candidate Generation
  -> Vector / Feature Similarity
  -> Behavioral Signals
  -> Ranking
  -> Diversity / Safety rules
  -> Final candidates
```

Hard filters are deterministic and must be implemented in application/database logic, not delegated to an LLM.

## 8. AI User Feature Model

User features may include:

- basic profile
- dating preferences
- interests
- hobbies
- foods
- image features
- behavioral features
- conversation features
- geographic context

Persist model/feature versions so recommendation behavior remains auditable.

## 9. Face Verification Safety Boundary

Selfie verification must distinguish:

```text
Face Detection
Face Embedding
Liveness Detection
Identity Verification
Verification Decision
```

A successful face detection is not proof of identity.

Verification records should include status, score/confidence, model name/version, timestamp, and reason code where applicable.

Treat biometric data as sensitive. Do not invent retention, consent, export, or deletion policy; use the policy documents and explicit product decisions.

## 10. Conversation Intelligence

Store both:

1. Immutable/raw chat messages
2. AI-derived conversation features

Example derived features:

```json
{
  "topics": ["travel", "coffee", "photography"],
  "food_preferences": ["japanese", "italian"],
  "hobbies": ["photography"]
}
```

Every derived feature should track source, confidence, model version, feature version, and timestamp.

## 11. Harness Engineering Principles

The repository itself is the source of truth for agent behavior.

Codex must read:

- AGENTS.md
- relevant local AGENTS.md
- ARCHITECTURE.md
- relevant ADRs
- task document
- relevant tests/contracts

before making changes.

Agent changes must be small, scoped, testable, and reversible.

Do not:
- perform unrelated refactors
- upgrade dependencies without a task
- invent APIs
- invent product rules
- bypass tests
- expose production secrets
- add a second queue system without an ADR
- mix AI model implementation into NestJS

## 12. Task Execution Model

Each Codex task follows:

```text
Read -> Plan -> Change -> Validate -> Report
```

But the implementation unit should be deliberately small.

Preferred task size:
- one concern
- one or a few files
- one validation goal
- no unrelated cleanup

Example progression:

```text
Task 1.1: Create root package.json
Task 1.2: Create pnpm-workspace.yaml
Task 1.3: Create .gitignore and .env.example
Task 1.4: Create Docker Compose with PostgreSQL only
Task 1.5: Verify PostgreSQL + pgvector
Task 1.6: Add Redis
Task 1.7: Add MinIO
Task 1.8: Add Playwright scaffold
```

Codex must stop after the current task and report what changed.

## 13. Development Phases

### Phase 0 — Harness
Create and validate:
- AGENTS.md
- ARCHITECTURE.md
- SECURITY.md
- PRIVACY.md
- CONTRIBUTING.md
- ADR structure

### Phase 1 — Containerized Foundation
- Monorepo
- Docker Compose
- PostgreSQL + pgvector
- Redis + BullMQ foundation
- MinIO
- Playwright foundation
- CI skeleton

### Phase 2 — Backend Core
- NestJS
- Prisma
- health endpoints
- config
- logging
- error model

### Phase 3 — Frontend Core
- Next.js
- RWD shell
- design system
- API client
- route protection scaffold

### Phase 4 — Account / Profile / Preferences
- register/login
- profile
- preferences
- interest/hobby/food selection
- photo upload

### Phase 5 — Face Verification
- selfie upload
- liveness
- face embedding
- verification decision

### Phase 6 — Dating Interaction
- discovery
- like/pass
- match
- blocks

### Phase 7 — Chat
- conversations
- Socket.IO
- messages
- read receipts
- notifications

### Phase 8 — Recommendation AI
- embeddings
- candidate generation
- pgvector retrieval
- ranking
- clustering

### Phase 9 — Conversation AI
- conversation analysis
- feature extraction
- feature aggregation

### Phase 10 — Continuous Learning Loop
- behavior signals
- feature update
- re-clustering
- re-ranking
- offline evaluation
- model/version management

## 14. Initial Quality Gates

At minimum:

```text
Type Check
Lint
Unit Tests
Integration Tests
Build
API Contract Validation
Playwright E2E
Security Scan
AI Evaluation for AI changes
```

## 15. Working Agreement with Codex

The developer wants to inspect each small change manually. Therefore:

- Do not implement multiple future tasks in one turn.
- Do not create speculative files.
- Do not refactor unrelated code.
- At the end of every task, stop.
- Report exact files changed.
- Report commands run and their results.
- Ask for the next task only after the current task is accepted.

## 16. Current Starting Point

The previous manual Docker attempt established these intended root files:

```text
.env
.env.example
.gitignore
docker-compose.yml
infrastructure/postgres/init/
```

However, because development is now being moved fully to Codex, Codex should inspect the current repository state first rather than blindly recreating these files.

## 17. Current Task

**Task 1: Establish the controlled monorepo + containerized development foundation under Codex.**

The first Codex change must be tiny and should not build the whole system in one shot.

See `docs/tasks/00-codex-workflow.md` and `docs/tasks/01-phase-1-task-plan.md` for the exact workflow.
