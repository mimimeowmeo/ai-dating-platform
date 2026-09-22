# Architecture

## 1. Service Boundaries

### Web
Next.js application. Owns UI, browser state, client-side API access, and responsive design.

### API
NestJS application. Owns product business logic, authentication, **all database access (including pgvector reads/writes)**, realtime gateway, queue producers, and the consumer of AI results (see [ADR 0002](adr/0002-db-ownership.md)).

### AI
FastAPI application. Stateless AI compute: synchronous inference endpoints (reply suggestions, embeddings, chunking, topic spans, summaries, style profiles) and AI-specific schemas. **Holds no database credentials**; it receives data in requests and returns results.

### AI Worker
Python worker. Consumes asynchronous AI jobs from the `ai-jobs` queue and publishes results to the `ai-results` queue. It never writes to PostgreSQL directly.

### Ollama
Ollama Cloud (managed, `gemma4:31b`) is the primary model for reply suggestions and is used by the AI service/worker for background extraction (conversation summaries, style profiles); it needs `OLLAMA_API_KEY`. Gemini provides all embeddings (`gemini-embedding-2`) and is the reply fallback (`gemini-3.8-flash`). Every model in a fallback chain has its own timeout. A local Ollama (native macOS app; Docker on macOS has no GPU access) is optional for offline development. Decided after the 2026-09-23 benchmark in the reply suggestions spec (section 14).

### PostgreSQL
System of record for business data and vector data (pgvector).

### Redis
Caching, transient coordination, BullMQ queues, rate-limits, and realtime support where appropriate.

### MinIO / S3
Binary/media storage.

### Nginx
Reverse proxy / single local entry point.

## 2. Request Flows

### Standard API
```text
Browser -> Nginx -> Next.js or NestJS -> PostgreSQL / Redis
```

### Realtime Chat
```text
Browser -> Nginx -> Socket.IO / NestJS -> Redis / PostgreSQL
```

### Async AI
```text
NestJS -> BullMQ (ai-jobs) -> AI Worker -> BullMQ (ai-results) -> NestJS -> PostgreSQL
```

### Sync Verification
```text
Browser -> NestJS -> FastAPI AI -> NestJS -> Browser
```

### Sync Reply Suggestions
```text
Browser -> NestJS -> FastAPI AI /embed -> NestJS (pgvector search)
        -> FastAPI AI /reply-suggestions -> NestJS (persist) -> Browser
```

Details: [AI reply suggestions spec](../ai/REPLY-SUGGESTIONS-SPEC.md).

## 3. Dependency Direction

```text
Frontend
  -> API contract
  -> API

API
  -> application/domain services
  -> Prisma (incl. pgvector via raw SQL in one VectorStore service)
  -> infrastructure adapters
  -> AI service clients / queue producers / AI result consumers

AI
  -> model adapters (Gemini, Ollama)
  -> feature pipelines (pure functions)
```

Avoid circular dependencies.

## 4. Forbidden Patterns

- React component directly calling PostgreSQL.
- NestJS controller directly calling Prisma.
- AI model imported into NestJS process.
- AI service or AI worker connecting to PostgreSQL.
- Raw DB secrets in Git.
- Business logic buried inside generic utilities.
- LLM used for deterministic hard filters.
