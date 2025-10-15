---
title: "Datasets: AtlasTiny 4D upgrade"
created: 2025-10-14
status: draft
owner: TBD
priority: medium
labels: [task, datasets]
deps:
  - src/viterbo/datasets/atlas_tiny.py
  - notebooks/atlas_tiny_profile.py
  - docs/math/volume.md
  - docs/math/capacity_ehz.md
---

## Summary

Refocus AtlasTiny on 4D symplectic polytopes now that core volume support is in place. Replace the current planar rows with deterministic 4D examples and refresh profiling guidance.

## Background

- AtlasTiny presently emits 2D polygons because the 4D symplectic stack was incomplete.
- Volume computations are now dimension-agnostic, but EHZ/minimal-action solvers still block 4D completion (see `2025-10-14-task-math-4d-capacity`).
- Profiling notebook and docs should highlight the 4D targets once the math layer is upgraded.

## Deliverables

1. Swap `atlas_tiny_generate()` to produce a curated set of 4D polytopes (e.g., hypercube, scaled Lagrangian product, counterexample prototype) with exact V/H data.
2. Update `atlas_tiny_complete_row()` gating so 4D invariants are computed via the new math helpers; drop 2D-specific conditionals.
3. Refresh `notebooks/atlas_tiny_profile.py` text and defaults to reflect the 4D dataset and expected hotspots.
4. Add smoke tests asserting dataset length, dimensionality, determinism, and presence of derived tensors (volume, capacity, minimal-action cycle, systolic ratio).
5. Update relevant docs (`docs/math/volume.md`, dataset briefs) to cross-reference 4D usage.

## Acceptance Criteria

- `uv run python -m pytest tests -k atlas_tiny` passes with new assertions.
- Profiling notebook runs without modifications and reports 4D math hotspots.
- Documentation outlines the 4D focus and links to the math task for deeper algorithms.
