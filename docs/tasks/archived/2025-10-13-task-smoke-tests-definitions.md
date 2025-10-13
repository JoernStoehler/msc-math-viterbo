---
title: "Define smoke tests for core algorithms"
created: 2025-10-13
status: draft
owner: TBD
branch: task/smoke-tests-definitions
priority: high
labels: [task]
---

## Summary

Design minimal smoke tests (function presence, basic invariants, shapes/dtypes) for each stubbed algorithm so implementations can land with immediate validation.

## Deliverables

- Add/extend `tests/test_smoke_algorithms.py` to cover:
  - presence of functions and importability
  - deterministic behavior on fixed small inputs (where applicable)
- Document a short testing checklist in `docs/tasks/README.md`

## Acceptance Criteria

- CI green (lint/type/smoke)
- Tests are light (<1s locally), readable, and stable across devices

