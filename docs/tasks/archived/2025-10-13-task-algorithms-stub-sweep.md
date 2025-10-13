---
title: "Algorithms: create stubs + signatures"
created: 2025-10-13
status: draft
owner: TBD
branch: task/algorithms-stub-sweep
priority: high
labels: [task]
---

## Summary

Create placeholder modules and function signatures across `viterbo.math` to cover core geometry/polytope algorithms, enabling parallel follow-up tasks per algorithm.

## Deliverables

- Add commented stubs (no implementations) and minimal docstrings for:
  - `math/halfspaces.py`: vertex→H-rep; H-rep → vertex (signatures only)
  - `math/convex_hull.py`: convex hull API returning vertices/facets/incidence
  - `math/incidence.py`: builder utilities for incidence matrices/graphs
  - `math/geometry.py`: keep current helpers; add TODO hints for additional routines
- Add a single smoke test asserting imports and presence of functions
- Update MIGRATION.md “Task Matrix” with follow-up items per algorithm

## Acceptance Criteria

- CI green (lint/type/smoke)
- No heavy logic; strictly signatures and TODOs

