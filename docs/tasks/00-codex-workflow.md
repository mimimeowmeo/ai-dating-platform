# Codex Workflow — One Small Task at a Time

## Purpose

Codex is the implementation agent for this project. The human developer will inspect every increment.

## Rules

1. Read repository instructions before changing anything.
2. Inspect the current working tree before creating files.
3. Implement only the current numbered task.
4. Prefer one concern and a very small diff.
5. Do not pre-implement future tasks.
6. Do not perform unrelated refactors.
7. Do not add dependencies unless the current task needs them.
8. After the change, run only the validation relevant to that task, plus mandatory repository checks if available.
9. Report exact files changed.
10. Stop after the task. Do not continue to the next task.

## Required Task Output

```text
Task: <id>

Changed files:
- ...

Implementation summary:
- ...

Commands run:
- ...

Validation:
- PASS / FAIL

Notes / risks:
- ...
```

## Escalate to Human

Codex must stop and report when it encounters:
- architecture ambiguity
- security/privacy ambiguity
- destructive migration
- breaking API design choice
- production deployment or secret handling decision
- model threshold/policy decision
- requirement conflict

Normal code defects should be diagnosed and fixed by Codex within the current task when possible.
