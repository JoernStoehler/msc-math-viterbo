---
title: "Math/Geometry: vertex transforms, volume, regular n-gon"
created: 2025-10-13
status: archived
owner: TBD
branch: task/math-geometry-transforms-volume-ngon
priority: high
labels: [task]
deps:
  - src/viterbo/math/geometry.py
  - src/viterbo/math/convex_hull.py
---

## Summary

Add basic vertex-space transforms (linear map, translation), a reference `volume(vertices)` for convex polytopes, and a helper to create rotated regular n-gons in 2D.

## Deliverables

- Implement `matmul_vertices(A, vertices)` and `translate_vertices(t, vertices)`.
- Implement `volume(vertices)` using a robust hull-based approach.
- Implement `rotated_regular_ngon2d(k, angle)` returning `(vertices, normals, offsets)`.
- Smoke tests for 2D polygons and simple 3D polytopes.

## Acceptance Criteria

- CI green; functions accept tensors on caller's device and return tensors.
- `volume` consistent with known shapes (unit square/cube, regular polygons).
- Docstrings include shapes/dtypes and units where relevant.

## Status Log

- 2025-10-14 — Archived after verifying geometry transforms, `volume` (≤3D), and `rotated_regular_ngon2d` ship with smoke coverage; future high-dimensional work will spin out into new briefs.

