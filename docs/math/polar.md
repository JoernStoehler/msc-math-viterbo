---
title: polar — polarity and Mahler links (stubs)
---

# `viterbo.math.polar`

Helpers for polar bodies and Mahler-type quantities. These are stubs that align
with planned bridges between convex geometry and symplectic inequalities.

Functions

- `polar_from_halfspaces(normals: (F, d), offsets: (F,)) -> (normals_dual, offsets_dual)` (stub)
  - Computes an H-representation of the polar body `K^\circ` given `K = {x | Bx ≤ c}` once implemented.

- `mahler_product_approx(vertices: (M, d)) -> ()` (stub)
  - Returns an approximation to `vol(K) · vol(K^\circ)` using volume backends and a polar routine.

Notes

- Implementations are pending volume backends and robust polarity; the API surfaces shapes and invariants for early consumers.

