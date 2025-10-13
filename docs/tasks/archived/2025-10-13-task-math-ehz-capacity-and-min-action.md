---
title: "Math/Symplectic: EHZ capacity (H/V) and minimal action cycle"
created: 2025-10-13
completed: 2025-10-20
status: done
owner: TBD
branch: task/math-ehz-capacity-min-action
priority: medium
labels: [task]
deps:
  - src/viterbo/math/symplectic.py
  - src/viterbo/math/geometry.py
  - src/viterbo/math/halfspaces.py
---

## Summary

Implement placeholders into working algorithms for EHZ capacity both from H-rep and V-rep, and compute a minimal action cycle along with a `systolic_ratio` helper.

## Delivered

- Implemented planar EHZ capacity solvers bridging H/V representations with deterministic polygon area evaluation.
- Minimal action cycles now return counter-clockwise boundary traversals with capacity certificates.
- Added a `systolic_ratio` helper that respects the dimension parameter and guards invalid inputs.
- Expanded smoke tests in `tests/test_math_symplectic.py` and `tests/test_math_symplectic_invariants.py` to cover invariances, scaling, and ratio normalisations.

## Deliverables

- Implement `capacity_ehz_algorithm1(B, c)` and `capacity_ehz_algorithm2(vertices)`.
- Implement `minimal_action_cycle(vertices, B, c) -> (capacity, cycle)`.
- Implement `systolic_ratio(volume, capacity)` consistent with docstring definition.
- Add smoke tests on low-dimensional shapes with known/approximate values.

## Acceptance Criteria

- CI green (lint/type/smoke).
- Document assumptions, approximations, and numerical stability notes.
- Deterministic for fixed inputs/seeds; no implicit device moves.

