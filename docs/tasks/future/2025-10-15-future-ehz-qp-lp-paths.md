---
title: "Future: EHZ via convex QP and LP/SOCP relaxations"
created: 2025-10-15
status: idea
owner: TBD
priority: medium
labels: [future, math, capacity, optimisation]
deps:
  - src/viterbo/math/capacity_ehz/
  - docs/math/capacity_ehz.md
---

## Summary

Implement convex optimisation paths for the facet‑multiplier formulation of c_EHZ(P): a numerically robust QP for the weights β with affine constraints (β^T B = 0, β^T c = 1, β ≥ 0), and LP/SOCP relaxations that produce bounds and warm‑starts for the exact 4D method.

## Acceptance Criteria

- QP solver: stable preprocessing (QR on B, constraint elimination), active‑set or projected Newton method, strict feasibility handling, returns capacity and multipliers.
- LP/SOCP relaxations: documented bounds; unit tests verify monotonicity vs exact solver on small instances.
- API aligns with `capacity_ehz_via_qp` / `capacity_ehz_via_lp` stubs.
- Benchmarks show practical runtimes on F ≤ 50 with toleranced accuracy.

## Notes

- Evaluate off‑the‑shelf optimisers only if allowed; otherwise implement minimal custom routines.
- Warm‑start the exact 4D enumeration using large β entries to seed supports.

