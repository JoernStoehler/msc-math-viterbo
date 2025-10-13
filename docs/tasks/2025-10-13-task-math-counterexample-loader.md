---
title: "math: expose 2024 counterexample geometry helper"
created: 2025-10-13
status: proposed
owner: TBD
branch: task/math-counterexample-loader
priority: medium
labels: [task]
deps:
  - src/viterbo/math/constructions.py
  - src/viterbo/math/minimal_action.py
  - notebooks/viterbo_counterexample_proposal.py
---

## Summary

Graduate the ad-hoc geometry builders in the counterexample notebook into an importable helper inside `viterbo.math`. The API should return the vertices, half-space data, capacity, systolic ratio, and minimal Reeb cycle for the Haim–Kislev–Ostrover pentagon product.

## Deliverables

- Public `counterexample_pentagon_product()` (or similar) exposing the 4D polytope, its `q/p` factors, and precomputed invariants as torch tensors.
- Documentation of shapes/dtypes and normalization conventions (include the volume and capacity formulas used).
- Lightweight regression test ensuring the helper reproduces the known capacity and systolic ratio constants.
- Update the notebook to consume the new helper instead of reconstructing the geometry manually.

## Acceptance Criteria

- `import viterbo.math.constructions as constructions` (or dedicated module) provides a single call to retrieve the counterexample data with no side effects.
- Tests verify the returned tensors live on caller device/dtype (CPU float64 default) and match the analytic values from the literature to 1e-9 relative error.
- Notebook diff shows only the import/API usage switch (no duplicated geometry logic remains).
