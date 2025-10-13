---
status: backlog
created: 2025-10-12
workflow: task
summary: Wire HF Datasets builders directly now that Polars wrappers are gone.
---

# Subtask: Remove Polars, migrate atlas to HF Datasets

## Context

- The atlas is standardised on HF Datasets (row-first, Parquet-backed, cloud-ready).
- Legacy Polars wrappers (`viterbo/_wrapped/polars_io.py`) and the `viterbo.atlas` module referenced in older docs have already
  been deleted during the namespace cleanup.
- Builders currently live under `src/viterbo/datasets2/` with an `atlas_tiny` preset that only materialises geometry and volume
  columns.

## Objectives

- Ensure builders return the documented quantities (capacity, spectrum, tags, provenance) per sample.
- Keep local Parquet storage under `artefacts/datasets/` (single file or sharded directory) with clear versioning guidance.
- Update CLI scripts and notebooks to import the existing builders from `viterbo.datasets2` and rely on HF Datasets’ save/load helpers.

## Plan

1. Confirm `_row_from_sample` (or equivalent helper) populates the full schema using `viterbo.math` (implemented for `atlas_tiny`).
2. Update placeholder notebooks and `scripts/build_atlas_small.py` to use the builders and HF Datasets save/load helpers directly.
3. Document dataset presets (`atlas_tiny`, future `atlas_small`) and retention policy in the project guide.

## Acceptance

- CI green; tests cover building and reloading datasets with HF Datasets only.
- Example notebooks run end-to-end with HF-backed atlas builders from `viterbo.datasets2`.
- CLI scripts and docs reference the correct import paths; dataset presets are documented with storage locations.

