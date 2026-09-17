# AGENTS.md Seed Template

> This file is a seed for the repository root `AGENTS.md`. Codex should use it as the initial engineering contract.

## Project

AI Dating RWD Platform.

## Mandatory First Step

Before changing code:
1. Read root `AGENTS.md` and relevant local guidance.
2. Inspect git status.
3. Read the current task file.
4. Inspect existing patterns before creating new ones.

## Scope Rule

Implement only the current task.

Do not:
- add future features
- perform unrelated refactors
- change architecture without an ADR
- add infrastructure not required by the task
- add RabbitMQ
- put AI model code into NestJS

## Validation Rule

After changes:
- run task-specific validation
- run available static checks
- report exact results
- stop

## Architecture Rules

- Next.js owns browser UI.
- NestJS owns product API/business orchestration.
- Python owns AI model inference.
- PostgreSQL is the system of record.
- Redis/BullMQ is the initial queue system.
- MinIO/S3 stores binary media.
- API contracts must be explicit.

## Security

- Never commit secrets.
- Never expose biometric feature data to the browser unless explicitly required and approved.
- Do not invent retention or privacy policy.
