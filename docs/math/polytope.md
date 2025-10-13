---
title: polytope — per-body geometry
---

# `viterbo.math.polytope`

Per-body geometry for fixed convex polytopes. Functions operate on vertex and
half-space representations, expose basic queries (support, distances, bounding
boxes), and convert between representations. All functions are Torch-first and
preserve dtype/device.

Functions (queries)

- `support(points: (N, D), direction: (D,)) -> ()`
- `support_argmax(points: (N, D), direction: (D,)) -> ((), int)`
- `pairwise_squared_distances(points: (N, D)) -> (N, N)`
- `halfspace_violations(points: (N, D), normals: (F, D), offsets: (F,)) -> (N, F)`
- `bounding_box(points: (N, D)) -> ((D,), (D,))`

Functions (representations)

- `vertices_to_halfspaces(vertices: (M, D)) -> (normals: (F, D), offsets: (F,))`
- `halfspaces_to_vertices(normals: (F, D), offsets: (F,)) -> (M, D)`
- `facet_vertex_incidence(vertices, normals, offsets) -> (F, M)` *(stub)*

Notes

- Deterministic ordering of vertices relies on lexicographic sorting on CPU.
- Tolerances scale with `sqrt(eps)` in the working dtype (never downcast).

