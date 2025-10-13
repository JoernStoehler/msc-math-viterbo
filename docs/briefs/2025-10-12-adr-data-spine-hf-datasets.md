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
- `volume: dict[str, dict[str, float]]` (e.g., `{"halfspaces": {"reference": 1.0, "fast": 1.0}, "vertices": {"reference": 1.0}}`)
- `capacity_ehz: dict[str, dict[str, float | dict[str, float | int | str]]]` (facet-normal, Reeb, symmetry-reduced, support-relaxation, and MILP variants)
- `spectrum_topk: dict[str, list[float]]`
- `reeb_cycles: dict[str, object]` (simple-cycle representatives and oriented-edge diagnostics)
- `systolic_ratio: dict[str, float]`
- `tags: dict[str, object]` (e.g., `csym`, `family`, `scale`, `normalized`)
- `provenance: dict[str, object]` (git SHA, seeds, tool versions)

## Interfaces (library side)

Dataset builders in `src/viterbo/datasets2/` return HF `Dataset` instances directly. Callers use
the standard HF Datasets API (`Dataset.from_list`, `.map`, `.save_to_disk`, `.load_from_disk`, …)
without a project-specific adapter layer. Keep imports explicit and avoid helper singletons.

## Current implementation snapshot (2025-10)

- Only the `atlas_tiny` builder is present today (`src/viterbo/datasets2/atlas_tiny.py`). It now
  materialises every quantity family (volume, EHZ capacity, spectrum, Reeb cycles, systolic ratios)
  across all implemented algorithms, returning `NaN` when a solver is not available for a
  particular dimension.
- There is no `viterbo.atlas` package. CLI entry points still import the legacy name and fail until
  they call the live builders under `viterbo.datasets2`.
- Polars wrappers referenced in older docs have already been removed.

## Storage Locations

- Default: `artefacts/datasets/atlas.parquet` (single-file or sharded directory).
- Optionally maintain snapshot versions `artefacts/datasets/atlas-YYYYMMDD.parquet` or directories.

## Migration Plan

1. Update placeholder notebooks and CLI scripts to call the `viterbo.datasets2` builders directly
   instead of the removed `viterbo.atlas` module.
2. Expand coverage beyond `atlas_tiny` once additional presets (`atlas_small`, …) have an agreed
   specification and implementation plan.

