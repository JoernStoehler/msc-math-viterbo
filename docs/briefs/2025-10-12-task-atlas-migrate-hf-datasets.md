---
status: proposed
created: 2025-10-12
workflow: task
summary: Implement atlas on HF Datasets; deprecate Polars-specific helpers.
---

# Task: Atlas migration to HF Datasets

## Goals

- Provide a row-first HF Datasets atlas with Parquet backing and thin adapters in `viterbo.atlas`.
- Support build → compute quantities per row → save/load → consume flows without library-level batching.

## Deliverables

1. `viterbo.atlas` adapters: `build_dataset`, `append_rows`, `save_dataset`, `load_dataset`, `map_quantities`.
2. Row builder assembling quantities (`capacity_ehz`, `spectrum_topk`, `volume`) from per-instance APIs.
3. Smoke tests and placeholder notebooks updated to exercise the HF-backed atlas.
4. Storage path: `artefacts/datasets/atlas.parquet` (single file or sharded directory); versioned snapshots optional.

## Execution Plan

1. Implement adapters; write small tests covering append/load and simple projections.
2. Update notebooks (`modern_atlas_builder.py`, `modern_atlas_consumer.py`) to call the HF adapters.
3. Mark Polars-specific helpers as deprecated in docs; remove after consumers switch.

## Open Questions

- Decide on snapshot cadence (per-commit vs milestone) to manage repository size.
- Optional cloud storage integration once local scale exceeds Git LFS comfort.

