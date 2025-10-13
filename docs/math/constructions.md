---
title: constructions — polygon/polytope builders
---

# `viterbo.math.constructions`

Constructs new polytopes from scratch or by transforming existing ones.

Functions

- `rotated_regular_ngon2d(k: int, angle: float) -> (vertices: (k, 2), normals: (k, 2), offsets: (k,))`
- `matmul_vertices(A: (D, D), vertices: (M, D)) -> (M, D)`
- `translate_vertices(t: (D,), vertices: (M, D)) -> (M, D)`
- `matmul_halfspace(A: (D, D), normals: (F, D), offsets: (F,)) -> (normals, offsets)`
- `translate_halfspace(t: (D,), normals: (F, D), offsets: (F,)) -> (normals, offsets)`
- `lagrangian_product(vertices_P: (M1, n), vertices_Q: (M2, n)) -> (vertices: (M1·M2, 2n), normals: (F, 2n), offsets: (F,))`
- `random_polytope_algorithm1(seed, num_facets: int, dimension: int)`
- `random_polytope_algorithm2(seed, num_vertices: int, dimension: int)`

Notes

- Random generators operate in the unit ball, enforce reproducibility via an explicit generator, and return both V and H representations.
- Transform helpers (matmul/translate) exist for both vertex and half-space representations to keep pipelines symmetric.
