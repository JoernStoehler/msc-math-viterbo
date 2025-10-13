---
title: "Math/Random: polytope generators (H and V)"
created: 2025-10-13
status: ready
owner: TBD
branch: task/math-random-polytopes
priority: medium
labels: [task]
deps:
  - src/viterbo/math/random_polytopes.py
  - src/viterbo/math/halfspaces.py
  - src/viterbo/math/convex_hull.py
---

## Summary

Implement two random polytope generators: (1) via random halfspaces and (2) via random vertices. Focus on reproducibility (seedable), removal of redundancies, and returning both V and H reps.

## Deliverables

- Implement `random_polytope_algorithm1(seed, num_facets, d)`.
- Implement `random_polytope_algorithm2(seed, num_vertices, d)`.
- Ensure both return `(vertices, normals, offsets)` and are deterministic for a fixed seed.
- Add smoke tests with small `d` (2, 3) verifying closed/valid polytopes.

## Acceptance Criteria

- CI green (lint/type/smoke).
- No implicit device moves; fixed seeds produce identical results.
- Document limits and potential degeneracies.

