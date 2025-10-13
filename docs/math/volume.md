---
title: volume — polytope volumes and backends
---

# `viterbo.math.volume`

Volume for convex polytopes with planned higher-dimensional backends. Functions
are Torch-first, pure, and preserve input dtype/device.

Functions

- `volume(vertices: (M, D)) -> ()`
  - Convex hull volume for `D ∈ {1, 2, 3}` via interval length, polygon shoelace,
    and facet pyramids. Raises `NotImplementedError` for `D ≥ 4`.

Stubs (planned)

- `volume_via_triangulation(vertices: (M, D)) -> ()`
  - Oriented-hull triangulation and signed simplex accumulation.

- `volume_via_lawrence(normals: (F, D), offsets: (F,), *, basis: (D, D)|None) -> ()`
  - Lawrence sign decomposition over facet bases; room for rational certification.

- `volume_via_monte_carlo(vertices: (M, D), normals: (F, D), offsets: (F,), *, samples: int, generator: Generator|int) -> ()`
  - Quasi–Monte Carlo rejection sampling with variance reduction.

