---
status: completed
created: 2025-10-12
workflow: task
summary: Restore green CI by resolving lint and strict Pyright failures in the modern capacity stack.
---

# Subtask: Restore green CI for the modern capacity stack

## Context

- `just ci` was failing on the lint step due to an unused `viterbo.polytopes` import in `tests/viterbo/test_capacity.py`.
- After fixing lint, Pyright strict surfaced ~80 errors across the capacity and polytope modules. Root causes included
  - missing type coverage for JAX APIs in the local stubs,
  - untyped lists and unused variables inside capacity helpers, and
  - inconsistent exports (e.g. `viterbo.__init__`) that Pyright treated as unused imports.

## Objectives

1. Eliminate the lint violation in `tests/viterbo/test_capacity.py`.
2. Provide Pyright with the type information it needs (via stubs and annotations) so that `pyright -p pyrightconfig.strict.json` reports zero errors.
3. Keep runtime behaviour unchanged while tightening annotations (e.g. avoid mutating algorithms to appease the type checker).
4. Verify the full `just ci` pipeline succeeds locally.

## Plan

1. Remove the unused import from the capacity smoke test.
2. Expand the minimal JAX stubs with dynamic attribute fallbacks and missing signatures used in the library.
3. Tidy the capacity modules by removing redundant imports, annotating lists, fixing unused temporaries, and making re-exports explicit via `__all__` entries.
4. Address Pyright complaints in wrappers and helpers (e.g. casting SciPy results, converting JAX scalars to `float`).
5. Re-run `just ci` to confirm lint, Pyright, and smoke tests are green.

## Acceptance

- `just ci` completes successfully (lint, Pyright strict, pytest smoke tier).
- No new regressions in the capacity tests; runtime semantics unchanged.
- Type checker reports zero errors without disabling strict rules.
