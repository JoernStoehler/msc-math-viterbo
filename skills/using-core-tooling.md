---
name: using-core-tooling
description: This skill should be used every time for the minimal command set and day-to-day workflow glue.
last-updated: 2025-10-18
relevance: always
---

# Using Core Tooling (Always)

Minimal commands and glue you use on nearly every task.

## Quick Gate

- `just checks` — fast local gate (format → lint → type → smoke tests).
- `just ci` — CI parity (lint, type, smoke tests) before handoff when changes affect core modules or infra.

## Lint & Format

- `just lint` — Ruff lint and skills validation (non-mutating AGENTS.md check).
- `just fix` — apply Ruff format and auto-fixes, then rerun checks.

## Tests

- `just test` — incremental smoke-tier pytest. Use `INC_ARGS="--debug"` to inspect selection.
- `pytest -q -k "pattern"` — focus a subset when iterating.

## Environment

- `uv run ...` — run Python commands with dependencies from `uv.lock`.
- Use `rg` for fast code/search. Stream file reads ≤250 lines in shell.
- Provisioning auto-runs `just setup` and refreshes AGENTS.md; you rarely need to run it manually.

## Pointers

- Environment details and PDF ingestion: `skills/operating-environment.md`.
- Testing workflow details and troubleshooting: `skills/testing-and-troubleshooting.md`.
