# AtlasTiny v1 — Schema, Backends, Timings, Reproducibility

AtlasTiny is a minimal but diverse roster of low‑dimensional polytopes for demos, tests, and benchmarks. It focuses on deterministic generators, float64 CPU numerics, and explicit provenance. The roster is intentionally small; extend by adding generators/configs that preserve reproducibility and schema guarantees.

- Purpose: compact dataset to exercise geometry, volume, and symplectic invariants without heavy compute.
- Diversity: 2D polygons (regular, random) and 4D examples (simplex, products, mixed).
- Extension: add rows via new generator labels/configs; add new quantities/backends as new nullable columns.

## Schema Reference (v1)

Columns are stable for v1; all numeric scalars are float64 and geometry uses ragged nested lists. Nullability rules are noted.

- Identity/meta
  - `polytope_id` (string) — human‑readable stable identifier.
  - `generator` (string) — canonical generator label (see below).
  - `generator_config` (string) — JSON with params and seeds (sorted keys).
  - `dimension` (int64) — ambient dimension D.
  - `num_vertices` (int64) — vertex count M.
  - `num_facets` (int64) — facet count F.
- Geometry (ragged; nested lists)
  - `vertices` (list[list[float64]]) — shape (M, D).
  - `normals` (list[list[float64]]) — shape (F, D) outward facet normals.
  - `offsets` (list[float64]) — shape (F,) facet offsets with `normals`.
  - `minimal_action_cycle` (list[list[float64]]) — shape (C, D); empty list when missing.
- Quantities (nullable float64 scalars)
  - `volume` — always present; 2D “area”.
  - `capacity_ehz` — present in 2D and select 4D products; else null.
  - `systolic_ratio` — present iff `capacity_ehz` present; else null.
- Backend labels (nullable strings)
  - `volume_backend` — `area2d` for D=2, else `facets` (≥3D).
  - `capacity_ehz_backend` — `area2d` in 2D; `minkowski_lp3` in 4D product cases; null otherwise.
  - `systolic_ratio_backend` — `formula` when computed; null otherwise.
- Timings (nullable float64, seconds)
  - `time_generator` — always present.
  - `time_volume_area2d` — present for D=2; null otherwise.
  - `time_volume_facets` — present for D≥3; null for D=2.
  - `time_capacity_area2d` — present when 2D capacity computed; else null.
  - `time_capacity_minkowski_lp3` — present for 4D `minkowski_lp3`; else null.
  - `time_systolic_ratio` — present when ratio computed; else null.

## Generator Labels and Configs

Canonical labels (grammar) with examples; configs are JSON strings with sorted keys and numeric types coercible to Python primitives. All generators output float64 CPU tensors.

- `unit_square` — config `{}`.
- `triangle_area_one` — config `{}`.
- `regular_ngon` — config `{"k": int, "angle": float}`; e.g. `{"k":5,"angle":0.0}`.
- `random_polygon` — config `{"seed": int, "k": int}`; deterministic by seed.
- `regular_simplex` — config `{"d": int}`; e.g. `{"d":4}`.
- `lagrangian_product` — config variants:
  - `{"variant":"pentagon_product_counterexample", "factors":[...]}` — fixed.
  - `{"variant":"noisy_pentagon_product","seed_q":int,"seed_p":int,"amp":float}`.
- `mixed_nonproduct_from_product` — config `{"base": str, "seed_q": int, "seed_p": int, "amp": float, "mix": str}`.
- `random_polytope_algorithm1` — H‑rep: `{"seed": int, "num_facets": int, "dimension": int}`.
- `random_polytope_algorithm2` — V‑rep: `{"seed": int, "num_vertices": int, "dimension": int}`.

Reproducibility
- Numerics: all tensors are `float64` on CPU for dataset construction and I/O.
- Seeds: stored in `generator_config`; reconstruction uses only this JSON.
- Provenance: Parquet exports include `metadata.json` with Git commit hash and UTC build time.

## Backends and Mapping Rules

- Volume
  - D=2 → `area2d` backend (polygon area from vertices).
  - D≥3 → `facets` backend (general volume routine from vertices/facets).
- Capacity (EHZ)
  - D=2 → `area2d` backend via minimal action cycle on polygon.
  - D=4 product cases → `minkowski_lp3` (specialized product routine when available); may be null for non‑product or unsupported cases.
- Systolic ratio
  - `formula` when `capacity_ehz` is present; null otherwise.

Extending backends
- Add new backends by: computing the quantity, storing the value, setting a new backend label, and adding a `time_*` column. New nullable columns are non‑breaking for v1.

Links: math details live under Math API pages (`math/volume.md`, `math/symplectic.md`, `math/capacity_ehz.md`). This page focuses on dataset plumbing.

## I/O and Artefacts

- Artefact layout: `artefacts/datasets/atlas-tiny/v1/`
  - `data.parquet` — dataset rows and columns.
  - `dataset_info.json` — minimal HF‑style info (features summary, row count).
  - `metadata.json` — commit hash, build timestamp, schema version, summary.
  - `README.md` — brief schema summary and reproducibility notes.
- Module: `viterbo.datasets.atlas_tiny_io`
  - `atlas_tiny_to_hf(rows)` — list of completed rows → HF `Dataset`.
  - `atlas_tiny_save_parquet(ds, out_dir)` — write Parquet + companion files.
  - `atlas_tiny_load_parquet(path)` — Parquet (file or dir) → HF `Dataset`.
  - `atlas_tiny_rows_from_hf(ds)` — HF rows → in‑memory rows (float64 tensors).

Build notebook
- Use `notebooks/atlas_tiny_build.py` to build rows and save Parquet in one step:
  - `uv run python notebooks/atlas_tiny_build.py`
  - Outputs to `artefacts/datasets/atlas-tiny/v1/` by default.

## Code Snippets

Build rows in memory

```python
from viterbo.datasets.atlas_tiny import atlas_tiny_build, atlas_tiny_collate_pad

rows = atlas_tiny_build()  # list of completed rows (float64 CPU tensors)

# Optional: pad to batch tensors for a model
batch = atlas_tiny_collate_pad(rows)
V = batch["vertices"]   # (B, V_max, D)
N = batch["normals"]    # (B, F_max, D)
vol = batch["volume"]   # (B,)
```

Convert to HF and save Parquet

```python
import os
from viterbo.datasets.atlas_tiny import atlas_tiny_build
from viterbo.datasets.atlas_tiny_io import atlas_tiny_to_hf, atlas_tiny_save_parquet

rows = atlas_tiny_build()
ds = atlas_tiny_to_hf(rows)

out_dir = os.path.join("artefacts", "datasets", "atlas-tiny", "v1")
atlas_tiny_save_parquet(ds, out_dir)
```

Load Parquet and batch for use

```python
from viterbo.datasets.atlas_tiny import atlas_tiny_collate_pad
from viterbo.datasets.atlas_tiny_io import atlas_tiny_load_parquet, atlas_tiny_rows_from_hf

loaded = atlas_tiny_load_parquet("artefacts/datasets/atlas-tiny/v1/")
rows = atlas_tiny_rows_from_hf(loaded)
batch = atlas_tiny_collate_pad(rows)
```

## Versioning Policy

- Schema version: `v1` for the columns listed here.
- Non‑breaking updates: add new quantities/backends as new nullable columns and `time_*` fields; existing columns remain unchanged.
- Breaking changes: rename/remove columns, change dtypes/nullability, or alter semantics → bump to `v2` and write to `artefacts/datasets/atlas-tiny/v2/`.
- Reproducibility guarantees: float64 CPU numerics, deterministic generators via stored JSON configs, and commit hash captured during Parquet export.

