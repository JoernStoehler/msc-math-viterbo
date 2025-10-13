---
title: "Datasets: AtlasTiny scaffolding and torch Dataset build"
created: 2025-10-13
status: ready
owner: TBD
branch: task/datasets-atlas-tiny
priority: medium
labels: [task]
deps:
  - src/viterbo/datasets/atlas_tiny.py
  - src/viterbo/math/geometry.py
  - src/viterbo/math/symplectic.py
---

## Summary

Implement a small synthetic dataset of polytopes (AtlasTiny) with geometric and symplectic attributes, using math utilities for completion. Build as a `torch.utils.data.Dataset`; avoid external dataset frameworks.

## Deliverables

- Implement `atlas_tiny_generate()` to yield rows with geometry fields populated (vertices/normals/offsets) via existing generators.
- Implement `atlas_tiny_complete_row()` to fill `volume`, `capacity_ehz`, `systolic_ratio`, and `minimal_action_cycle` using `viterbo.math` APIs.
- Implement `atlas_tiny_build()` to return a `torch.utils.data.Dataset` (`AtlasTinyDataset`) of completed rows.
- Add smoke tests that build a tiny dataset and check length/types.

## Acceptance Criteria

- CI green (lint/type/smoke) with no external dataset dependency.
- Deterministic generation given fixed seeds.
- Clear docstrings for shapes/dtypes and invariants.
