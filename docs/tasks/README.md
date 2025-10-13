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

## Recently archived

- 2025-10-13 Math/Geometry: vertex transforms, volume, regular n-gon
- 2025-10-13 Math/H-Rep: conversions and transforms
- 2025-10-13 Math/Random: polytope generators (H and V)
- 2025-10-13 Math/Symplectic: J, random symplectic matrices, Lagrangian product
- 2025-10-13 Define smoke tests for core algorithms
- 2025-10-13 Math/Symplectic: EHZ capacity (H/V) and minimal action cycle
