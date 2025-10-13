---
title: "Math/Symplectic: J, random symplectic matrices, Lagrangian product"
created: 2025-10-13
status: ready
owner: TBD
branch: task/math-symplectic-foundations
priority: medium
labels: [task]
deps:
  - src/viterbo/math/symplectic.py
---

## Summary

Provide foundations for symplectic computations: standard symplectic form `J`, random symplectic matrix generator, and Lagrangian product of polytopes given vertices.

## Deliverables

- Implement `symplectic_form(d)` and validate block structure.
- Implement `random_symplectic_matrix(d, seed)` with checks `M.T @ J @ M == J` (within tolerance).
- Implement `lagrangian_product(vertices_P, vertices_Q)` returning `(vertices, normals, offsets)`.
- Add smoke tests (even `d`, shape checks, simple 2D/4D cases).

## Acceptance Criteria

- CI green (lint/type/smoke).
- Numerical checks pass within reasonable tolerances.
- Docstrings document shapes/dtypes and invariants.

