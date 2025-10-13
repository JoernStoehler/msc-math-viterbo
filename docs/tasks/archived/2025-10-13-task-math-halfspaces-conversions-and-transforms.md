---
title: "Math/H-Rep: conversions and transforms"
created: 2025-10-13
status: archived
owner: TBD
branch: task/math-halfspaces-conversions-transforms
priority: high
labels: [task]
deps:
  - src/viterbo/math/halfspaces.py
  - src/viterbo/math/convex_hull.py
---

## Summary

Implement H↔V conversions and simple H-rep transforms (linear map and translation) for convex polytopes. Keep implementations pure Torch (CPU baseline), with clear dtype/device semantics and docstrings.

## Deliverables

- Implement `vertices_to_halfspaces(vertices) -> (normals, offsets)`.
- Implement `halfspaces_to_vertices(normals, offsets) -> vertices`.
- Implement `matmul_halfspace(A, B, c)` and `translate_halfspace(t, B, c)`.
- Handle degenerate inputs gracefully with documented behaviour.
- Add smoke tests for shapes/dtypes and a couple of 2D/3D sanity cases.

## Acceptance Criteria

- CI green (lint/type/smoke).
- Deterministic on fixed inputs; no implicit device moves.
- Docstrings document shapes, dtype, and invariants.

## Status Log

- 2025-10-14 — Archived after verifying bidirectional conversions and transforms are implemented in `src/viterbo/math/halfspaces.py` with smoke coverage.

