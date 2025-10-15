---
title: "math: implement Minkowski two-bounce billiard solver"
created: 2025-10-13
status: completed
owner: Codex agent
branch: main
priority: high
labels: [task]
deps:
  - src/viterbo/math/capacity_ehz/
  - src/viterbo/math/polytope.py
  - tests/math/test_minimal_action_invariants.py
---

## Summary

 - Implemented `minimal_action_cycle_lagrangian_product`, following Rudolf’s ≤3 bounce guarantee for planar Minkowski billiards.
- Enumerates two- and three-bounce candidates, enforces strong billiard reflection conditions, and returns closed Reeb cycles with consistent device/dtype semantics.
- Counterexample notebook now calls the library API; the bespoke dual-polar helper has been removed.

## Validation

- `uv run pytest tests/math/test_minkowski_billiard.py`
- `just test-full`

## Notes

- Tests cover the regular pentagon constant, a synthetic instance where the optimal orbit needs three bounces, and permutation/translation invariance.
- The solver currently assumes bounce points coincide with supplied polygon vertices; extending to facet-interior contacts would require a continuous optimisation pass. A stub (`minimal_action_cycle_lagrangian_product_generic`) documents the intended follow-up.
