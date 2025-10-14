---
title: "Define smoke tests for core algorithms"
created: 2025-10-13
completed: 2025-10-15
status: completed
owner: Codex agent
branch: task/smoke-tests-definitions
priority: high
labels: [task]
deps:
  - docs/tasks/2025-10-13-task-algorithms-stub-sweep.md
  - tests/test_smoke.py
  - tests/math/test_polytope_smoke.py
  - tests/math/test_volume_smoke.py
---

## Summary

Design minimal smoke tests (function presence, basic invariants, shapes/dtypes) for the implemented math modules so future enhancements land with immediate validation.

## Delivered

- `tests/test_smoke.py` exercises core geometry helpers, dataset/collate flows, and the demo probe to guarantee importability and fast end-to-end checks.
- Added pytest markers (`pytestmark = pytest.mark.smoke`) for easy incremental selection and CI gating.
- Added dedicated smoke suites `tests/math/test_polytope_smoke.py` and `tests/math/test_volume_smoke.py` covering support queries, pairwise distances, halfspace violations, and volume routines.

## Acceptance Criteria

- CI green (lint/type/smoke).
- Tests remain light (<1s locally), readable, and stable across devices.
- New smoke modules pass standalone (`uv run pytest tests/math/test_polytope_smoke.py tests/math/test_volume_smoke.py`).

## Status Log

- 2025-10-15 — Added dedicated smoke suites for polytope primitives and volume helper plus expanded integration coverage in `tests/test_smoke.py`; merged to `main`.

## Follow-on Checklist

- [ ] Extend `tests/test_smoke.py` (or sibling modules) whenever new math/dataset/model surfaces land, keeping runtime under ~1s.
- [ ] Document runtime expectations and deterministic seeds alongside new smoke coverage.
