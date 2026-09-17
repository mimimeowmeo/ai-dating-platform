# Architecture

## 1. Service Boundaries

### Web
Next.js application. Owns UI, browser state, client-side API access, and responsive design.

### API
NestJS application. Owns product business logic, authentication, data access orchestration, realtime gateway, and queue producers.

### AI
FastAPI application. Owns synchronous AI inference endpoints and AI-specific schemas.

### AI Worker
Python worker. Owns asynchronous AI jobs consumed from the queue.

### PostgreSQL
System of record for business data and vector data.

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
NestJS -> BullMQ -> Redis -> AI Worker -> PostgreSQL / MinIO
```

### Sync Verification
```text
Browser -> NestJS -> FastAPI AI -> NestJS -> Browser
```

## 3. Dependency Direction

```text
Frontend
  -> API contract
  -> API

API
  -> application/domain services
  -> Prisma
  -> infrastructure adapters
  -> AI service clients / queue producers

AI
  -> model adapters
  -> feature pipelines
  -> storage adapters
```

Avoid circular dependencies.

## 4. Forbidden Patterns

- React component directly calling PostgreSQL.
- NestJS controller directly calling Prisma.
- AI model imported into NestJS process.
- Raw DB secrets in Git.
- Business logic buried inside generic utilities.
- LLM used for deterministic hard filters.
