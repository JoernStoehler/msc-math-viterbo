---
title: minimal_action — EHZ capacities and cycles
---

# `viterbo.math.minimal_action`

Capacity solvers and minimal-action diagnostics with a 4D focus. Current
implementations cover planar (2D) smoke tests; higher-dimensional backends are
stubbed with references to planned algorithms.

Functions (implemented placeholders)

- `capacity_ehz_algorithm1(normals: (F, d), offsets: (F,)) -> ()` *(2D)*
- `capacity_ehz_algorithm2(vertices: (M, d)) -> ()` *(2D)*
- `capacity_ehz_primal_dual(vertices, normals, offsets) -> ()` *(2D consistency check)*
- `minimal_action_cycle(vertices, normals, offsets) -> (capacity, cycle)` *(2D)*
- `systolic_ratio(volume: (), capacity_ehz: (), symplectic_dimension: int) -> ()`

Stubs / planned work

- `capacity_ehz_haim_kislev(normals, offsets)` — general Haim–Kislev formula in 4D+
- `oriented_edge_spectrum_4d(vertices, normals, offsets, *, k_max)` — Hutchings-style spectrum
- `capacity_ehz_via_qp(normals, offsets)` — facet-multiplier convex QP
- `capacity_ehz_via_lp(normals, offsets)` — LP/SOCP relaxations for bounds

Notes

- Inputs must satisfy the even-dimension policy (`d = 2n`).
- Implementations preserve dtype/device and rely on helpers from
  `viterbo.math.polytope` and `viterbo.math.constructions`.
