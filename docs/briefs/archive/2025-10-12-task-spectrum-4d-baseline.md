---
status: in-progress
created: 2025-10-12
workflow: task
summary: Maintain 4D spectrum baseline; defer ≥6D to next milestone.
---

# Task: EHZ Spectrum (4D Baseline)

## Context

- 4D oriented-edge spectrum is implemented and tested in the modern stack.
- Deterministic enumeration and ordering are required; ≥6D is deferred.

## Objectives

- Keep 4D spectrum stable and deterministic (fixtures, tolerances, graph metadata).
- Document the public API semantics and expected outputs (no padding at the library layer).
- Record ≥6D backlog items and prerequisites (combinatorics, CI/runtime limits, fixtures).

## Execution

1. Confirm graph builder and cycle enumeration determinism on reference polytopes.
2. Keep tests focused on invariants and action sequences (no xfail-by-design).
3. Track ≥6D planning in a separate backlog brief or milestone doc when scoped.

## Acceptance

- Smoke tests pass under CI; fixtures remain stable.
- API docstring clarity and brief alignment with library behaviour.

