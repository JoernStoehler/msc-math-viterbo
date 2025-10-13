---
title: "Define smoke tests for core algorithms"
created: 2025-10-13
status: completed
owner: TBD
branch: main
priority: high
labels: [task]
deps:
  - tests/test_smoke.py
  - tests/math/test_polytope_smoke.py
  - tests/math/test_volume_smoke.py
---

## Summary

Design minimal smoke tests (function presence, basic invariants, shapes/dtypes) for the implemented math modules so future enhancements land with immediate validation. Added dedicated smoke suites for `viterbo.math.polytope` primitives and the general `volume` helper.

## Deliverables

- Add `tests/math/test_polytope_smoke.py` to cover support queries, pairwise distances, halfspace violations, and bounding boxes.
- Add `tests/math/test_volume_smoke.py` to exercise 1D/2D/3D volumes and dtype preservation on simple shapes.

## Acceptance Criteria

- CI green (lint/type/smoke).
- Tests are light (<1s locally), readable, and stable across devices.
- New smoke modules pass standalone (`uv run pytest tests/math/test_polytope_smoke.py tests/math/test_volume_smoke.py`).
