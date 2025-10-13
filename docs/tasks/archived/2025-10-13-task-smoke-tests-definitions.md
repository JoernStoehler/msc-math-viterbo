---
title: "Define smoke tests for core algorithms"
created: 2025-10-13
completed: 2025-10-20
status: done
owner: TBD
branch: task/smoke-tests-definitions
priority: high
labels: [task]
deps:
  - docs/tasks/2025-10-13-task-algorithms-stub-sweep.md
  - tests/test_smoke.py
---

## Summary

Design minimal smoke tests (function presence, basic invariants, shapes/dtypes) for each stubbed algorithm so implementations can land with immediate validation.

## Delivered

- `tests/test_smoke.py` exercises core geometry helpers, dataset/collate flows, and the demo probe to guarantee importability and fast end-to-end sanity checks.
- Added pytest markers (`pytestmark = pytest.mark.smoke`) for easy incremental selection and CI gating.
- Documented invariants covered (support/violations, pairwise distances, dataset collates) to guide future smoke additions.

## Checklist for new coverage

- [ ] Extend `tests/test_smoke.py` (or a sibling `tests/test_smoke_<area>.py`) whenever a new math/dataset/model surface lands.
- [ ] Keep runtime under ~1s locally; prefer deterministic seeds and minimal tensor shapes.
- [ ] Cover import paths, shape/dtype assertions, and a thin vertical slice through the new functionality.

## Deliverables

- Establish a consolidated smoke module (`tests/test_smoke.py`) covering imports, geometry invariants, and dataset/model hooks.
- Ensure every addition documents runtime expectations and deterministic seeds.
- Capture the scope and checklist in this brief so future contributors know how to extend the suite.

## Acceptance Criteria

- CI green (lint/type/smoke).
- Tests are light (<1s locally), readable, and stable across devices.

