---
title: volume — polytope volumes and backends
---

# `viterbo.math.volume`

Torch-first utilities to evaluate volumes of convex polytopes in arbitrary
dimension, plus stubs for specialised backends. All functions preserve the dtype
and device of their inputs and remain side-effect free.

Functions

- `volume(vertices: (M, D)) -> ()`
  - General-purpose Gauss-divergence accumulator that works for any `D ≥ 1`. Uses
    polar/spherical decomposition for one dimension, shoelace for two, then
    recursively integrates facet measures (via `vertices_to_halfspaces`) for higher
    dimensions, enabling exact volumes for 4D polytopes such as hypercubes or
    Lagrangian products.

Stubs (planned)

- `volume_via_triangulation(vertices: (M, D)) -> ()`
  - Oriented-hull triangulation and signed simplex accumulation.

- `volume_via_lawrence(normals: (F, D), offsets: (F,), *, basis: (D, D)|None) -> ()`
  - Lawrence sign decomposition over facet bases; room for rational certification.

- `volume_via_monte_carlo(vertices: (M, D), normals: (F, D), offsets: (F,), *, samples: int, generator: Generator|int) -> ()`
  - Quasi–Monte Carlo rejection sampling with variance reduction.
