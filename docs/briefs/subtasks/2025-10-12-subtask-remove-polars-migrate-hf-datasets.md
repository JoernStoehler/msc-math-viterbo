---
status: proposed
created: 2025-10-12
workflow: task
summary: Remove Polars adapters and migrate atlas to Hugging Face Datasets.
---

# Subtask: Remove Polars, migrate atlas to HF Datasets

## Context

- We decided to standardize the atlas on HF Datasets (row-first, Parquet-backed, cloud-ready).
- Polars-specific helpers live under `viterbo/_wrapped/polars_io.py` and `viterbo.atlas` consumes Polars types.

## Objectives

- Remove Polars-only adapters from the public path.
- Introduce HF Datasets adapters in `viterbo.atlas` for build/load/save/map.
- Keep local Parquet storage under `artefacts/datasets/` (single file or sharded directory).

## Plan

1. Add HF adapters in `viterbo.atlas`:
   - `build_dataset(rows_iterable) -> Dataset`
   - `append_rows(dataset, rows_iterable) -> Dataset`
   - `save_dataset(dataset, path) -> None`
   - `load_dataset(path) -> Dataset`
   - `map_quantities(dataset, fn) -> Dataset`
2. Update placeholder notebooks to exercise HF-backed atlas.
3. Deprecate and then remove Polars wrappers (`viterbo/_wrapped/polars_io.py`).
4. Purge Polars imports from `viterbo.atlas` and tests; adjust tests to HF Datasets.

## Acceptance

- CI green; tests use HF Datasets only.
- No imports from Polars remain under `src/viterbo/`.
- Example notebooks run end-to-end with HF-backed atlas.

