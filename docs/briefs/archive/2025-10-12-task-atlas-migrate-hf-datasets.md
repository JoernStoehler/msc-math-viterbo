---
status: proposed
created: 2025-10-12
workflow: task
summary: Implement atlas on HF Datasets; deprecate Polars-specific helpers.
---

# Task: Atlas migration to HF Datasets

## Goals

- Provide a row-first HF Datasets atlas with Parquet backing built directly from `viterbo.datasets2`.
- Support build → compute quantities per row → save/load → consume flows using native HF Datasets
  primitives (no additional adapter layer).

## Deliverables

1. Row builders assembling quantities (`capacity_ehz`, `spectrum_topk`, `volume`, Reeb cycles,
   systolic ratios, tags, provenance) across every algorithm implemented in `viterbo.math`.
2. Smoke tests and placeholder notebooks updated to exercise the HF-backed atlas via native HF
   Datasets methods.
3. Storage path: `artefacts/datasets/atlas.parquet` (single file or sharded directory); versioned
   snapshots optional.

## Execution Plan

1. Update notebooks (`modern_atlas_builder.py`, `modern_atlas_consumer.py`) to call the existing
   builders directly and rely on HF Datasets’ save/load helpers.
2. Document that Polars wrappers are already gone; focus remaining cleanup on wiring the builders
   through the CLI and notebooks.

## Open Questions

- Decide on snapshot cadence (per-commit vs milestone) to manage repository size.
- Optional cloud storage integration once local scale exceeds Git LFS comfort.
- Clarify whether additional presets beyond `atlas_tiny` (e.g., `atlas_small`) live alongside the
  tiny builder or under separate entry points.

