---
title: "Datasets: AtlasTiny ragged rows and collate helper"
created: 2025-10-13
status: completed
owner: Codex agent
branch: main
priority: medium
labels: [task]
deps:
  - src/viterbo/datasets/atlas_tiny.py
  - src/viterbo/math/geometry.py
  - src/viterbo/math/symplectic.py
---

## Summary

Implement a small synthetic dataset of polytopes (AtlasTiny) with geometric and symplectic attributes, using math utilities for completion. Return completed rows as typed dictionaries and provide padding/collate utilities instead of introducing a custom Dataset subclass.

## Deliverables

- Implement `atlas_tiny_generate()` to yield rows with geometry fields populated (vertices/normals/offsets) via existing generators.
- Implement `atlas_tiny_complete_row()` to fill `volume`, `capacity_ehz`, `systolic_ratio`, and `minimal_action_cycle` using `viterbo.math` APIs.
- Implement `atlas_tiny_build()` to return a list of completed row dictionaries and add a `atlas_tiny_collate_pad()` helper for default-batching scenarios.
- Add smoke tests that build the tiny dataset and exercise the collate helper.

## Acceptance Criteria

- CI green (lint/type/smoke) with no external dataset dependency.
- Deterministic generation for repeated builds.
- Clear docstrings for shapes/dtypes, derived invariants, and collate semantics.

## Status Log

- 2025-10-15 — Implemented deterministic generators/completion/collate helpers in `atlas_tiny.py` with smoke coverage under `tests/datasets/test_atlas_tiny.py`; merged to `main`.
