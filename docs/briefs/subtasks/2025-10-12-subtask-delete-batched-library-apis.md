---
status: proposed
created: 2025-10-12
workflow: task
summary: Delete batched capacity/spectrum library APIs; keep per-instance calls only.
---

# Subtask: Delete batched library APIs

## Context

- Library-level batching/padding is no longer desired; batching belongs in ML pipelines.
- Existing batched functions (e.g., `ehz_capacity_batched`, `ehz_spectrum_batched`) and associated tests exist.

## Objectives

- Remove batched entry points from the `viterbo` library.
- Update tests and examples to use per-instance APIs only.

## Plan

1. Inventory batched APIs and their call sites in `tests/`.
2. Replace call sites with per-instance loops (vectorize at the caller if needed).
3. Remove batched implementations and exports from modules.
4. Update docs to reflect per-instance-only policy.

## Acceptance

- No `*_batched` exports in `viterbo`.
- Tests green using per-instance calls.
- README and ADRs match the no-batching policy.

