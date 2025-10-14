---
title: "Datasets: collate edge cases and masks"
created: 2025-10-13
status: completed
owner: Codex agent
branch: main
priority: medium
labels: [task]
deps:
  - src/viterbo/datasets/core.py
  - tests/test_smoke.py
---

## Summary

Harden `collate_list` and `collate_pad` for edge cases (empty batches, 1-sample batches, heterogeneous dtypes/devices) and extend smoke tests accordingly.

## Deliverables

- Add device/dtype checks and clear error messages.
- Ensure masks are correct for all batch shapes; document behaviour.
- Extend `tests/test_smoke.py` (or a new `tests/test_collate.py`).

## Acceptance Criteria

- CI green with additional tests.
- Collators robustly handle documented edge cases.

## Status Log

- 2025-10-15 — Hardened `collate_list`/`collate_pad` validation, added mask handling, and landed regression coverage in `tests/test_collate.py`; merged to `main`.
