---
status: adopted
created: 2025-10-12
workflow: adr
summary: Adopt Hugging Face Datasets as the atlas backbone (row-first, Parquet-backed, cloud-ready).
---

# ADR: Atlas Data Spine on HF Datasets

## Decision

- Use Hugging Face Datasets (HF Datasets) for the atlas store.
- Represent rows as plain dicts; dataset size target is O(1e5) rows.
- Persist via Parquet under the hood; support local artifacts and future cloud storage.

## Why HF Datasets (vs Polars-only)

- Row-first ergonomics (dicts) for per-polytope computations; integrates well with ML.
- Built-in Parquet I/O; efficient columnar projections; memory mapping; simple sharding.
- Cloud-friendly storage without bespoke infra; compatible with on-disk artifacts.

## Row Model (v0)

Required keys (illustrative; evolve as needed):

- `polytope_id: str`
- `dimension: int`
- `normals: list[list[float]]` (nullable)
- `offsets: list[float]` (nullable)
- `vertices: list[list[float]]` (nullable)
- `capacity_ehz: float | None`
- `spectrum_topk: list[float] | None`
- `volume: float | None`
- `tags: dict[str, object]` (e.g., `csym`, `family`, `scale`, `normalized`)
- `provenance: dict[str, object]` (git SHA, seeds, tool versions)

## Interfaces (library side)

Thin adapters under `viterbo.atlas`:

- `build_dataset(rows_iterable) -> Dataset`
- `append_rows(dataset, rows_iterable) -> Dataset`
- `save_dataset(dataset, path) -> None`
- `load_dataset(path) -> Dataset`
- `map_quantities(dataset, fn) -> Dataset` (per-row compute; no library batching)

Keep the interface explicit; no global singletons. Conversions to JAX arrays originate in
consumers (ML pipelines or task scripts), not within library kernels.

## Storage Locations

- Default: `artefacts/datasets/atlas.parquet` (single-file or sharded directory).
- Optionally maintain snapshot versions `artefacts/datasets/atlas-YYYYMMDD.parquet` or directories.

## Migration Plan

1. Implement HF adapters (`build/load/save/map`) and smoke tests.
2. Provide a row builder that consumes `viterbo.types.Polytope` and current quantities.
3. Update placeholder notebooks to exercise the HF-backed atlas.
4. Mark Polars-specific helpers as deprecated in docs; remove once consumers switch.

