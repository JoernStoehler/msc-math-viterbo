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

## 2025-10-14 Triage

- Archived historical scaffolding briefs whose implementations now live in `src/viterbo/math/` (see `docs/tasks/archived/`).
- Confirmed the active backlog focuses on completing symplectic capacity solvers, dataset tooling, CI hardening, and smoke coverage.
- Relocated superseded workflow briefs under `docs/briefs/archive/` to avoid conflicting instructions.

### Active Backlog Snapshot

- `2025-10-13-task-math-ehz-capacity-and-min-action.md`
- `2025-10-13-task-datasets-atlas-tiny.md`
- `2025-10-13-task-datasets-collate-edge-cases.md`
- `2025-10-13-task-cpp-harness-and-docs.md`
- `2025-10-13-task-ci-cpu-torch-index-config.md`
- `2025-10-13-task-smoke-tests-definitions.md`

### Follow-ups

- [ ] Land the EHZ capacity + minimal action algorithms and unskip the associated tests.
- [ ] Finish the AtlasTiny dataset builder (generation + completion) and document the schema.
- [ ] Harden collate helpers for edge cases and extend smoke coverage.
- [ ] Enforce CPU-only Torch wheels in CI while preserving local GPU installs.
- [ ] Expand smoke definitions so new math modules gain fast regression tests.
