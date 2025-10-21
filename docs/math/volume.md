---
title: volume — polytope volumes and backends
---

Status: Implemented (scope: volume(vertices) baseline helper; caveats: triangulation/Lawrence/Monte Carlo entry points remain stubbed)

# `viterbo.math.volume`

Torch-first utilities to evaluate volumes of convex polytopes. The primary
`volume(vertices)` helper currently covers low dimensions with specialised
routines and falls back to a facet-height accumulation for `D ≥ 3`.

## Implemented helper

- `volume(vertices: (M, D)) -> ()`
  - `D = 1`: returns the length of the interval spanned by the vertices.
  - `D = 2`: orders the polygon via a convex-hull sweep and applies the shoelace
    formula.
  - `D ≥ 3`: converts vertices to halfspaces, gathers facet vertices, measures
    each facet recursively, and accumulates `facet_measure * height / D` from the
    centroid. This assumes well-behaved vertex orderings and convex inputs; degenerate
    facets drop out via tolerances.

## Stubs (planned)

- `volume_via_triangulation(vertices)` — oriented-hull triangulation and signed
  simplex accumulation.
- `volume_via_lawrence(normals, offsets, *, basis=None)` — Lawrence sign
  decomposition over facet bases with potential rational certification.
- `volume_via_monte_carlo(vertices, normals, offsets, *, samples, generator)` —
  quasi–Monte Carlo rejection sampling with variance reduction.

## Notes

- The recursive facet accumulation inherits numerical stability limits from
  `vertices_to_halfspaces`; supply well-scaled vertices for high dimensions.
- All helpers preserve dtype/device and perform no implicit device transfers.
