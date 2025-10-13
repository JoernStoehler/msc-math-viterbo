---
title: "math: implement Minkowski two-bounce billiard solver"
created: 2025-10-13
status: proposed
owner: TBD
branch: task/math-minkowski-billiard
priority: high
labels: [task]
deps:
  - src/viterbo/math/minimal_action.py
  - src/viterbo/math/polytope.py
  - tests/math/test_minimal_action_invariants.py
---

## Summary

Lift the notebook’s two-bounce Minkowski billiard routine into `viterbo.math.minimal_action`. The goal is a reusable API that, given the Lagrangian product of planar polytopes, recovers the minimal-action Reeb orbit and EHZ capacity by inspecting the dual billiard in the polar.

## Deliverables

- New torch-first function (e.g., `minimal_action_cycle_lagrangian_product`) accepting `q`-vertices and `p`-halfspaces, returning the cycle and scalar capacity.
- Robust handling of diagonal selection (choose the minimiser, not maximiser) with clear error messages when no valid orbit exists.
- Unit tests covering the regular pentagon example and at least one synthetic polygon pair, confirming invariance under permutation and translation of vertices.
- Optional refactor of existing stubs in `minimal_action.py` to reuse the new routine for 4D Lagrangian products.

## Acceptance Criteria

- Tests pass and pin the known constant `2 cos(pi/10) (1 + cos(pi/5))`.
- API surfaces torch tensors on caller device/dtype, no implicit moves.
- Notebook `viterbo_counterexample_proposal.py` calls into the library function instead of shipping its own Minkowski logic.
