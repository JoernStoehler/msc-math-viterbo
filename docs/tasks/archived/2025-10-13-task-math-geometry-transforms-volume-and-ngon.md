---
title: "Math/Geometry: vertex transforms, volume, regular n-gon"
created: 2025-10-13
completed: 2025-10-20
status: done
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

## Delivered

- `matmul_vertices`, `translate_vertices`, and `volume` ship in `src/viterbo/math/geometry.py` with smoke coverage under `tests/test_math_geometry.py`.
- `rotated_regular_ngon2d` is available alongside normals/offset helpers, exercised in random polytope and geometry tests.

## Deliverables

- Implement `matmul_vertices(A, vertices)` and `translate_vertices(t, vertices)`.
- Implement `volume(vertices)` using a robust hull-based approach.
- Implement `rotated_regular_ngon2d(k, angle)` returning `(vertices, normals, offsets)`.
- Smoke tests for 2D polygons and simple 3D polytopes.

## Acceptance Criteria

- CI green; functions accept tensors on caller's device and return tensors.
- `volume` consistent with known shapes (unit square/cube, regular polygons).
- Docstrings include shapes/dtypes and units where relevant.

