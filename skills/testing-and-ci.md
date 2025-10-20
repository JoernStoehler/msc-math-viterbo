---
name: testing-and-ci
description: Use for detailed testing, static analysis, incremental selection, and CI parity/troubleshooting.
last-updated: 2025-10-17
---

# Testing & CI

## Instructions
- Use these mechanics when you need details beyond the Good Code Loop quickstart.
- Capture exact commands and outputs when failures occur; include them in task notes and PRs.

## Commands & Gates

- `just checks` — format (Ruff), lint (Ruff), type (Pyright), smoke tests (Pytest).
- `just fix` — apply Ruff auto-fixes; rerun `just checks` to confirm.
- `just test` — incremental smoke tests. Inspect selection with `INC_ARGS="--debug" just test`.
- `just ci` — CI parity run before handoff or PR when changes are substantial or cross-cutting.

## Incremental Selector

- Selection is computed by `scripts/inc_select.py` and invoked by `just test`.
- Debug decisions with `INC_ARGS="--debug"` to view impacted graph and chosen tests.
- Do not manually reset `.cache/inc_graph.json` mid-task; regressions are harder to diagnose.

## Static Analysis

- Pyright (basic) surfaces type issues; address quickly or document rationale for suppressions.
- Ruff enforces import ordering and selected lint rules; prefer `just fix` where auto-fixes exist.
- `just lint` runs metadata validation: `scripts/load_skills_metadata.py --quiet`. Fix warnings before handoff.

## Troubleshooting Patterns

- Prefer `uv run` for Python invocations to match the locked environment (e.g., `uv run python -m pytest ...`).
- For intermittent test failures, run a subset: `pytest -k "<pattern>" --maxfail=1 --disable-warnings`.
- If tests touch temp paths, ensure they stay within the workspace.
- Escalate when a regression persists after two targeted attempts or touches shared infrastructure (CI, devcontainer, base deps).

## CI Parity

- `just ci` mirrors the GitHub Actions workflow locally.
- Record runtime and notable failures; attach log excerpts if parity reveals flakiness.

## Related Skills

- `good-code-loop` — quickstart and review checklist.
- `environment-tooling` — environment troubleshooting when commands fail.
- `performance-discipline` — benchmarking after correctness is established.

