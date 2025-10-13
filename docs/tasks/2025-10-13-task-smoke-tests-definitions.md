---
title: "Define smoke tests for core algorithms"
created: 2025-10-13
status: ready
owner: TBD
branch: task/smoke-tests-definitions
priority: high
labels: [task]
deps:
  - tests/test_smoke.py
---

## Summary

Design minimal smoke tests (function presence, basic invariants, shapes/dtypes) for the implemented math modules so future enhancements land with immediate validation.

## Deliverables

- Add/extend `tests/test_smoke.py` (or a dedicated `tests/test_smoke_algorithms.py`) to cover:
  - presence of functions and import paths
  - deterministic behaviour on small fixed inputs (where applicable)
  - dtype expectations in docstrings

## Acceptance Criteria

- CI green (lint/type/smoke).
- Tests are light (<1s locally), readable, and stable across devices.

