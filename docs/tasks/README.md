# Tasks Workflow (Parallel Development)

This folder holds small, self‑contained task briefs for parallel implementation on short‑lived branches. Each task should result in a focused PR (green CI) against `main`.

- Naming: `YYYY-MM-DD-task-<slug>.md`
- Branch: `task/<slug>` (or `feat/<scope>` when appropriate)
- Acceptance:
  - CI green (lint, type, smoke; deep/bench if perf‑sensitive)
  - Scope respected; docs/tests updated as needed
  - Minimal, coherent diff; clear PR description
- Handoff: include assumptions, edge cases, and any deviations from conventions in the PR body

Author briefs manually. For style and scope, learn from examples under `docs/tasks/archived/`.
