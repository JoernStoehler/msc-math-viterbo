---
title: "future: implement facet-interior Minkowski billiard solver"
created: 2025-10-14
status: proposed
priority: medium
labels: [future, math, minkowski-billiard]
deps:
  - src/viterbo/math/capacity_ehz/
  - tests/math/test_minkowski_billiard.py
  - docs/tasks/2025-10-13-task-math-minkowski-billiard.md
---

## Summary

- Extend the planar Minkowski billiard search to allow bounce points in facet interiors so the solver matches Rudolf’s characterization for generic convex Lagrangian products.
- Parameterise bounce points via facet indices and barycentric coordinates, solving the stationarity conditions from Artstein-Avidan–Ostrover/Rudolf (≤3 bounces).
- Provide a fallback to vertex contacts (for degeneracies) and expose a clean API that mirrors the existing vertex-only solver.

## Acceptance Criteria

- New `minimal_action_cycle_lagrangian_product_generic` returns the minimal EHZ action for generic polygons and reduces to the vertex solver on symmetric cases.
- Tests cover synthetic instances where the minimiser uses interior facet points and confirm continuity with the vertex special case.
- Documentation updated to explain the new behaviour and remaining limitations in higher dimensions.

## Notes

- Likely approach: enumerate facet triples, solve the corresponding linear complementarity / convex program for barycentric parameters, and reject configurations violating the reflection rule.
- Consider reusing or extending existing optimisation utilities; performance target is to stay within milliseconds for small polygons.
