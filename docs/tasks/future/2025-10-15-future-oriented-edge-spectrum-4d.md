---
title: "Future: Oriented‑edge action spectrum in ℝ⁴ (Hutchings‑style)"
created: 2025-10-15
status: idea
owner: TBD
priority: low
labels: [future, math, spectrum]
deps:
  - src/viterbo/math/capacity_ehz/
  - docs/math/capacity_ehz.md
---

## Summary

Expose a Hutchings‑style oriented‑edge action spectrum for 4D polytopes. Given (V, B, c), compute the set of action values associated with oriented edges/faces under the symplectic structure, suitable for comparisons and regression tests.

## Acceptance Criteria

- API `oriented_edge_spectrum_4d(vertices, normals, offsets, *, k_max=None)` returns a sorted tensor of actions.
- Validates on symmetric bodies (e.g., cubes, products) against known patterns.
- Document relation to c_EHZ and when spectra capture capacity.

## Notes

- This is primarily a diagnostic/analysis tool; prioritise correctness over speed.

