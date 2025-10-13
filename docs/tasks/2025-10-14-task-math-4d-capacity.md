---
title: "Math: 4D EHZ capacity and minimal action solvers"
created: 2025-10-14
status: draft
owner: TBD
priority: high
labels: [task, math, symplectic]
deps:
  - src/viterbo/math/minimal_action.py
  - src/viterbo/math/constructions.py
  - src/viterbo/math/volume.py
---

## Summary

Restore 4D coverage for symplectic invariants so datasets and notebooks can operate on the intended domain. Implement deterministic EHZ capacity and minimal-action cycle solvers that accept vertex and half-space representations in `R^4`.

## Background

- `volume` now supports arbitrary dimension (Gauss-divergence fallback), so missing pieces are the symplectic solvers.
- Current `minimal_action` implementations short-circuit at 2D, preventing 4D dataset rows from being completed.
- Atlas notebooks and downstream experiments depend on these invariants to study systolic ratios in the 4D setting.

## Deliverables

1. Implement a 4D EHZ capacity routine (e.g., Chaidez–Hutchings combinatorial billiards or a deterministic LP/QP formulation) exposed via `capacity_ehz_*` helpers.
2. Provide a minimal-action cycle extractor in 4D that returns an ordered orbit compatible with the dataset schema.
3. Update docstrings to describe supported dimensions, references, and numerical tolerances.
4. Add smoke tests under `tests/math/` covering canonical 4D bodies (hypercube, Lagrangian product, known counterexample) and verifying scaling/invariance.
5. Document limitations or approximations (e.g., combinatorial explosion, degeneracy handling) in `docs/math/minimal_action.md`.

## Acceptance Criteria

- `uv run python -m pytest tests/math -k minimal_action` passes with the new 4D coverage.
- Capacity/minimal-action helpers accept tensors on arbitrary devices/dtypes (within float64 baseline).
- Deterministic results for seeded generators and symmetric bodies (tests enforce specific values or tolerances).
- Docs updated to point to algorithm references and usage constraints.

## Progress Log

- 2025-10-14: Added regression tests covering 4D volume computations, H/V round-trips, and generator outputs. Capacity/minimal-action solvers remain stubbed; follow-up work will enable the remaining acceptance criteria.
