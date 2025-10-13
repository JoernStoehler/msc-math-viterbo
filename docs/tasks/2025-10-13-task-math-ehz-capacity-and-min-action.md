---
title: "Math/Symplectic: EHZ capacity (H/V) and minimal action cycle"
created: 2025-10-13
status: ready
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

## Deliverables

- Implement `capacity_ehz_algorithm1(B, c)` and `capacity_ehz_algorithm2(vertices)`.
- Implement `minimal_action_cycle(vertices, B, c) -> (capacity, cycle)`.
- Implement `systolic_ratio(volume, capacity)` consistent with docstring definition.
- Add smoke tests on low-dimensional shapes with known/approximate values.

## Acceptance Criteria

- CI green (lint/type/smoke).
- Document assumptions, approximations, and numerical stability notes.
- Deterministic for fixed inputs/seeds; no implicit device moves.

