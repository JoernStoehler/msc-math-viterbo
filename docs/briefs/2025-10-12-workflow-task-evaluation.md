---
status: adopted
created: 2025-10-12
workflow: policy
summary: Updated task/experiment evaluation methodology for the modern program.
---

# Task & Experiment Evaluation (Modern)

## Workflow

1. Collect context from code, briefs, and recent PRs.
2. Clarify the task nucleus (core question, artefacts produced).
3. Map outcomes → information value; assign rough odds where helpful.
4. Score utilities by downstream leverage; estimate costs (agent time, compute, Owner time).
5. Decide and communicate; track dependencies inline; keep a short Status Log.

## Cadence

- Inner loop (<2 min): format, lint, type, smoke tests for touched modules.
- CI loop (<5 min): full smoke-tier via `just ci`.
- Deep loop (5–20+ min): broader unit/benchmarks; run locally before performance claims.

## Testing policy (recap)

- Every test declares exactly one goal marker and one suite marker; docstrings state the invariant.
- Deterministic seeds, x64; use clear assertions with context-appropriate tolerances.
