---
title: "Algorithms: create stubs + signatures"
created: 2025-10-13
status: ready
owner: TBD
branch: task/algorithms-stub-sweep
priority: high
labels: [task]
deps:
  - src/viterbo/math/geometry.py
  - src/viterbo/math/halfspaces.py
  - src/viterbo/math/convex_hull.py
  - src/viterbo/math/incidence.py
---

## Summary

Add commented stubs (no implementations) and minimal docstrings across `viterbo.math` to cover core geometry/polytope algorithms, enabling parallel follow-up tasks per algorithm.

## Deliverables

- Expand `halfspaces.py` with vertex→H-rep and H-rep→vertex function signatures.
- Expand `convex_hull.py` with a hull API returning vertices/facets/incidence.
- Expand `incidence.py` with incidence matrices/graphs builders.
- Add TODOs and docstrings clarifying expected dtypes/shapes and invariants.
- Add a smoke test asserting importability and presence of functions.

## Acceptance Criteria

- CI green (lint/type/smoke).
- No heavy logic; strictly signatures and TODOs.

