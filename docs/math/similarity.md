---
title: similarity — comparing polytopes (stubs)
---

# `viterbo.math.similarity`

Distance measures between convex polytopes. The API is stubbed while 4D
experiments are scoped.

Stubs

- `hausdorff_distance(vertices_a, vertices_b)` — symmetric Hausdorff distance.
- `hausdorff_distance_under_symplectic_group(vertices_a, vertices_b)` — Hausdorff distance up to symplectomorphisms.
- `support_l2_distance(vertices_a, vertices_b, samples)` — Monte Carlo L2 distance between support functions.

Notes

- Implementations will reuse helpers from `viterbo.math.polytope` (support
  queries) and deterministic sampling strategies.
