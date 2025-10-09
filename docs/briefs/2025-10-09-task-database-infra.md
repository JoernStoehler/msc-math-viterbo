---
status: proposed
created: 2025-10-09
workflow: task
summary: Database infrastructure for polytopes, quantities, and consumable datasets (natural scale, Parquet+Polars, typed interfaces).
---

# Database Infra — Polytopes, Quantities, Datasets (v0)

## Context

- Target domain: convex polytopes in R^{2n} with 4D focus, some 2D/6D. Generators will evolve (structured, random, products, tangential), and quantities will expand (EHZ, spectra, distances, tags).
- Scale policy: Preserve natural scale by default; `vol=1` normalization is optional and recorded (`normalized`, `scale`).
- Representations: We support H-rep, V-rep, and LagrangianProductPolytopes; algorithms accept any via lightweight multi-dispatch with on-demand conversion.
- Goal: A simple, typesafe, extendable data spine where generators → quantities → storage → consumers interoperate without heavy abstractions.

## Objectives

- Storage format and stack (MVP)
  - Use Parquet on disk with Polars (PyArrow backend) for typed IO and lazy scans.
  - Single-table design: one Parquet file per dataset (no early partitioning). Refactors are cheap since datasets are reproducible.
  - Arrays live in Parquet list columns; prefer Arrow FixedSizeList for inner dimension when available; revisit offloading only if rows grow large.
- MVP row model (schema v0, rows = polytopes)
  - `polytope_id: str`
  - `generator: str` (family name or generator id)
  - `dimension: int`
  - `hrep_normals: list[vec_f64_dim]` (nullable)
  - `hrep_offsets: list[f64]` (nullable)
  - `vrep_vertices: list[vec_f64_dim]` (nullable)
  - `is_lagrangian_product: bool`
  - `volume: f64`
  - `capacity_ehz: f64`
  - `systolic_ratio: f64`
  - `min_action_orbit: list[i32]` (facet indices; raw word, no canonicalization)
- Interfaces (minimal, typed)
  - Imperative functions: `ensure_dataset(path)`, `append_rows(path, rows)`, `scan_lazy(path)`, `load_rows(path, columns=None)`.
  - `log_row(poly, quantities) -> dict` small builder; no ORMs or extra layers.
  - Polars interop lives under `viterbo/_wrapped/polars_io.py` (e.g., `scan_parquet`, `rows_to_polars`, `materialize_to_jnp`).
  - Converters: `to_hrep(poly)`, `to_vrep(poly)`, `to_product(poly)` are explicit and side-effect free.
- Applicability
  - Not all fields apply to all polytopes; keep nullable columns and fill only when available.
- Validation & CI
  - Unit tests: schema roundtrip, append/scan correctness, simple query (e.g., `systolic_ratio > 1`).
  - Markers per AGENTS.md; Pyright strict; Ruff lint.

## Quantitative Size & Throughput (guesstimates)

- Geometry scale
  - 4D product K_m × T_k: facets ≈ m + k (typical m,k in 4..32), vertices ≈ m·k (e.g., 64–1024).
  - Random H-reps: typical facets 16–128; vertices vary widely (often O(m^2) worst-case but much smaller in practice).
- Row footprint (Parquet, list columns)
  - Small (products, short certificates): ~10–50 KB/row.
  - Medium (random, longer spectra/certs): ~50–150 KB/row.
  - Working sets: 1e2 rows ~ 5–15 MB; 1e3 rows ~ 50–150 MB; 1e5 rows ~ 5–15 GB.
- IO & bandwidth
  - Polars lazy scans on predicates (e.g., `dimension==4`, `generator in {...}`) read only needed columns.
  - Feature-only reads remain light (<1–5 KB/row) and interactive at 1e5 rows on SSDs for projected columns.
- ML/training
  - Baseline: 1e3–1e4 rows, feature vectors 64–1k dims → tens of MB; fits in memory on dev hardware.
  - Scale-up: 1e5 rows likely requires feature-only materialization and sharded batches from Parquet.

## Execution

1. Implement `src/viterbo/exp1/store.py`
   - Imperative functions: `ensure_dataset(path)`, `append_rows(path, rows)`, `scan_lazy(path)`, `load_rows(path, columns=None)`, and `select_halfspaces_volume_capacity(path, polytope_ids=None)`.
   - Single Parquet file per dataset; optional metadata sidecar `dataset.json` with `dataset_id`, `schema_version`, `created`, `tool_versions`.
2. Implement `src/viterbo/exp1/store.py`
   - Add `log_row(poly, quantities) -> dict` assembling the MVP row; stable key ordering.
3. Add `src/viterbo/_wrapped/polars_io.py`
   - Converters `rows_to_polars(rows)`, `read_parquet`, `write_parquet`, `scan_parquet`, and a helper `materialize_to_jnp(lf, columns)` to materialize selected columns to JAX arrays.
4. Minimal E2E harness (smoke)
   - Generate a tiny K_m×T_k grid (≤ 20 rows), compute current quantities (volume, capacity_ehz, systolic_ratio, min_action_orbit), append to dataset, run a Polars query (`systolic_ratio > 1` count), and materialize a small feature subset to JAX arrays.
5. Tests (smoke tier)
   - Roundtrip append→scan→materialize; schema field presence; null handling for inapplicable quantities.
   - Deterministic ordering under a fixed sort; strict type hints; Ruff/Pyright clean.

## Interfaces (concrete v0 signatures)

```python
from typing import Iterable, Mapping, Sequence
import polars as pl

# Storage (imperative)
def ensure_dataset(path: str) -> None: ...
def append_rows(path: str, rows: Iterable[Mapping[str, object]]) -> None: ...
def scan_lazy(path: str) -> pl.LazyFrame: ...
def load_rows(path: str, columns: Sequence[str] | None = None) -> pl.DataFrame: ...

# Logging
def log_row(poly, quantities: dict) -> dict: ...

# Converters
def to_hrep(poly) -> tuple[list[list[float]], list[float]]: ...
def to_vrep(poly) -> list[list[float]]: ...

# Interop wrapper
from viterbo._wrapped import polars_io
```

Notes:
- Keep APIs explicit and small; no global singletons. Prefer passing paths explicitly.
- Avoid ORMs. Use Polars expressions for queries through `_wrapped/polars_io`; expose only thin adapters for JAX array materialization.

## Dependencies / Unlocks

- Upstream
  - exp1 core polytope types and converters (`HalfspacePolytope`, `VertexPolytope`, `LagrangianProductPolytope`).
  - Summaries/tags stubs (vol, counts, csym, anisotropy placeholder).
- Downstream
  - Playbook P.1 micro-grid (reads/writes rows; generates heatmaps/phase diagrams).
  - ML baseline (feature scans; JAX arrays); certificates checker (stores cert blobs).

## Acceptance Criteria

- A dataset can be created, appended to, and scanned via Polars; simple filters and projections work lazily.
- MVP columns exist with expected dtypes; nullable fields are handled gracefully.
- E2E smoke: tiny generation → compute current quantities → append → query → feature materialization into JAX arrays.
- Tests: smoke tier, markers per AGENTS.md; Pyright strict and Ruff clean; defer DuckDB and partitioning.

## Open Questions (escalate if blocking)

- When/if row sizes exceed ~200 KB, do we offload long arrays (e.g., spectra/certs) to sidecar blobs referenced by `blob_id`? (Deferred.)
- Any consumer pressure to add convenience columns like `n_facets`/`n_vertices` for faster filters, or keep computing via `arr.lengths()`? (Deferred.)

## Best Practice: Array Shapes in Parquet/Polars

- Store `dimension: int` explicitly.
- Use nested lists with a fixed-size inner array for coordinates (Arrow FixedSizeList; Polars `Array` dtype if available). This encodes point/normal dimensionality without extra columns.
- Rely on list lengths for counts (`len(hrep_normals)`, `len(vrep_vertices)`) rather than dedicated columns; Polars provides `arr.lengths()` for efficient queries. Add count columns later only if profiling shows value.

## Status Log

- 2025-10-09 — initial draft (v0) with Parquet+Polars proposal and v0 schema; pending discussion/approval.
- 2025-10-09 — updated to imperative storage functions and `_wrapped/polars_io` location; removed DatasetHandle.
