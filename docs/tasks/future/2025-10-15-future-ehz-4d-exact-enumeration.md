---
title: "Future: Exact 4D EHZ via facet enumeration (Haim–Kislev)"
created: 2025-10-15
status: idea
owner: TBD
priority: high
labels: [future, math, capacity]
deps:
  - src/viterbo/math/capacity_ehz/
  - docs/math/capacity_ehz.md
---

## Summary

Implement the exact 4D EHZ capacity solver based on the Haim–Kislev facet‑multiplier formulation. Enumerate facet supports of size ≤5 (extreme rays of {β ≥ 0 : β^T B = 0}), normalise β^T c = 1, and evaluate the triangular quadratic form over permutations. Return the global maximiser and capacity c_EHZ(P) = 1 / (2 Q_max).

## Acceptance Criteria

- Deterministic solver for `B ∈ ℝ^{F×4}`, `c ∈ ℝ^F`, returning `()` capacity.
- Enumerates supports with rank(B_S) = 4 and |S| ≤ 5; handles degenerate nullspaces by enumerating extreme rays.
- Numerically stable (QR/orthonormal bases), toleranced comparisons, preserves dtype/device.
- Tests cover random small polytopes and known examples; cross‑check with vertex billiards on product cases (where applicable).
- Documentation updated with complexity/limitations; benchmarks added for small F.

## Notes

- Objective uses `W = B J B^T` with permutation sweep on lower‑triangular pairs.
- Consider pruning permutations using sign patterns or ordering heuristics.
- This task is the “exact in 4D” backbone; QP/LP variants tracked separately.

