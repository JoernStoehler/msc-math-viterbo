Status: Implemented (scope: datasets & collation deep-dive; caveats: reflects repository state on 2025-10-20)

# Review — Datasets & Collation Deep-Dive

Provenance
- Topic: Datasets & Collation Deep-Dive
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 87fdebc
- Timebox: 60 minutes
- Scope: dataset schema/invariants, ragged batching & collate helpers, deterministic seeds & reproducibility, Atlas Tiny performance, roadmap for main dataset
- Version: v1.0

Context
- Focused pass over dataset helpers and collation to ensure schema clarity, ragged batching invariants, determinism, and a quick performance probe on Atlas Tiny. Roadmap notes added for the planned “main”/HF-backed dataset based on briefs.

Inputs & Method
- Loaded skills per AGENTS.md boot sequence (read-only): `uv run python scripts/load_skills_metadata.py` — printed skill summaries.
- Searched code: `rg -n "collate|dataset|batch|seed|reproduc|atlas|tiny" -S src`
- Read key modules: `src/viterbo/datasets/atlas_tiny.py`, `src/viterbo/datasets/core.py`, `src/viterbo/math/{polytope.py,volume.py,constructions.py,symplectic.py}`, tests under `tests/`.
- Probed Atlas Tiny runtime: `uv run python - << 'PY' ... atlas_tiny_build(); atlas_tiny_collate_pad(...) ... PY` (numbers below).
- Verified docs build: `uv run mkdocs build --strict` — OK.

Findings (unsorted)
- Atlas Tiny schema is explicit via TypedDicts: `AtlasTinyRaggedRow` (polytope_id, generator, vertices, normals, offsets) and `AtlasTinyRow` adds derived scalars/tensors (volume, capacity_ehz?, systolic_ratio?, minimal_action_cycle?, sizes). Source: `src/viterbo/datasets/atlas_tiny.py:20` and `src/viterbo/datasets/atlas_tiny.py:30`.
- Invariants in `atlas_tiny_complete_row`: enforces shapes, shared device/dtype across vertices/normals/offsets; computes `volume` for all dims; for 2D also computes `(capacity, cycle)` and `systolic_ratio`. Good guardrails, clear error messages. Source: `src/viterbo/datasets/atlas_tiny.py:74`.
- Dtype choice: Atlas Tiny forces `dtype=torch.float64` for the generated rows (volume/capacity/cycle). Accurate and stable, but potentially slower than float32 for larger batches. Consider parameterizing dtype/device in `atlas_tiny_generate/build` to allow float32 fast paths while keeping tests on float64.
- Devices: Atlas Tiny keeps everything on the current CPU device (math stack often CPU due to SciPy use). Collation respects the row device/dtype. No hidden device moves in dataset helpers; math functions also preserve dtype/device except where SciPy forces CPU (see vertices→halfspaces).
- Ragged batching helpers in `viterbo.datasets.core` are well-factored: `collate_list` returns python lists; `collate_pad` returns padded `(B, K_max, D)` with boolean mask `(B, K_max)` and stacked `direction (B, D)`. They validate consistent `dim`, device, dtype per batch and allow zero-length K. Sources: `src/viterbo/datasets/core.py:157` and `src/viterbo/datasets/core.py:172`.
- `_validate_ragged_batch` centralizes checks; error messages are specific (dtype/device/dim mismatches, empty batch). This is aligned with `skills/data-collation.md` guidance.
- `RaggedPointsDataset` uses an internal `torch.Generator(cpu)` seeded in `__init__`. Deterministic per-instance sampling is confirmed (same seed ⇒ same sequence). Probed: equal points/directions for `idx=0` and `idx=1` across two datasets seeded identically. Source: `src/viterbo/datasets/core.py:22` and probe below.
- Multi-worker note: with `num_workers>0`, each worker pickles the Dataset with the same generator state; without a `worker_init_fn` that perturbs `dataset._rng`, workers may replay identical sequences for the same indices. Recommend: in DataLoader, set `worker_init_fn` to call `dataset._rng.manual_seed(base_seed + worker_id)` or derive from `torch.initial_seed()` to ensure determinism without duplication.
- Atlas Tiny generation is deterministic (no RNG): uses `rotated_regular_ngon2d` followed by conversion to H-rep. Tests confirm determinism: `tests/datasets/test_atlas_tiny.py:test_atlas_tiny_generate_deterministic` and `test_atlas_tiny_build_returns_completed_rows`.
- Atlas Tiny collator `atlas_tiny_collate_pad` pads variable-length vertices/facets/cycles to per-batch maxima and returns masks. Scalars are `(B,)` tensors; missing capacities/systolic use `nan` (tensors), while cycle absent is an empty padded row with `cycle_mask=False`. This mix (None for cycle inside rows, NaN for scalars in batch) is pragmatic for PyTorch but will need normalization when storing in HF Datasets (e.g., sentinel empty arrays or Arrow nulls).
- Capacity/volume computations: 2D capacity via polygon area; `minimal_action_cycle` returns ordered vertices for 2D; 4D supports certain Lagrangian products. Volume uses a facet-sum formula leveraging `vertices_to_halfspaces` (SciPy ConvexHull). Deterministic, Torch-first APIs with explicit dtype/device conversions where needed.
- Potential perf hotspot: `vertices_to_halfspaces` calls SciPy’s `ConvexHull` on CPU float64, then converts back to Torch tensors. Fine for Atlas Tiny, but the main dataset (HF-backed) should avoid recomputing conversions redundantly (e.g., don’t compute H-rep from V-rep if already available; cache per row).
- `viterbo.models.demo.run_probe` honors both ragged list and padded collates; masked path iterates only valid rows per sample, matching collate contract. This is a good exemplar for downstream consumers.
- Tests cover key invariants: collate shape/masks, dtype/device mismatch errors, deterministic Atlas Tiny build, smoke path through DataLoader + model probe. Seed determinism is thoroughly covered for math generators (`random_polytope_*`, `random_symplectic_matrix`) via typed `seed: int | torch.Generator` patterns.
- Reproducibility policy matches skills: datasets provide collate functions; math is batching-agnostic and pure; conversions (e.g., `.numpy()`) are localized in a few places where external libs require it.
- Naming and layout align with `skills/repo-layout.md`: datasets thin adapters over math; math remains pure; models consume collated batches.
- Atlas Tiny baseline perf (CPU, float64): build≈0.226s for 3 rows; collate≈1.7ms; shapes `(3, 6, 2)` for vertices. This is already “instant” for local iteration; suggests plenty of headroom for adding a few dozen rows for richer benchmarks.
- Dataloader determinism knobs (not yet codified in code): advise setting `generator` on `DataLoader` for shuffling determinism and using `worker_init_fn` to seed dataset RNGs per worker predictably.
- For the main dataset (HF Datasets/Parquet), schema should avoid Python `None` in row fields for interoperability; prefer empty arrays for cycles and NaN for scalars, or use Arrow nullable fields explicitly via Feature types.
- Atlas Tiny currently fixes `dtype=torch.float64`. For the main dataset, recommend storing float32 for bulk numeric columns unless high-precision is required; run tiny benchmarks to validate error/precision tradeoffs.
- Roadmap alignment: briefs propose `viterbo.datasets2` with `schema.py`, `converters.py`, `quantities.py`, and builders `atlas_tiny`/`atlas_small` with Parquet storage under `artefacts/datasets/atlas.parquet`. Current code is consistent in spirit; migrating Tiny to produce an HF Dataset with the same per-row quantities is straightforward once converters are in place.
- HF Datasets integration consideration: variable-length tensors map to `Sequence(feature=Value(dtype))` with fixed inner dimension for vertices/cycles; ensure consistent dimension fields (`dimension`, `num_vertices`, `num_facets`) to validate shape on load.
- Batch invariants in `core.collate_pad`: single device/dtype per batch enforced; zero-length samples supported; well-suited for `DataLoader` with `drop_last=False`. These are robust defaults.
- Potential extension: expose a unified `CollatePolicy` enum or small helpers that return both `collate_fn` and a descriptor of batch shapes for downstream modules to assert against.
- Performance hygiene: when scaling beyond Tiny, measure vectorization gains of padding vs list-collate; for large raggedness, list-collate may avoid padding overhead. Consider micro-benchmarks in `tests/performance/`.
- Torch determinism: if end-to-end determinism is critical (esp. with CUDA), document `torch.use_deterministic_algorithms(True)` knobs in user-facing docs; current math ops are CPU-friendly and mostly deterministic.

Actions (first pass)
- Parameterize `atlas_tiny_generate/build` with `dtype` and `device` (defaults keep float64 CPU); reflect in tests.
- Add `worker_init_fn` example in docs/snippets to seed `RaggedPointsDataset._rng` per worker deterministically.
- Draft `viterbo.datasets2/schema.py` Feature spec for vertices/normals/offsets/cycle and scalars, including handling of nulls/empties.
- Implement `converters.py` sketch: row⇄Torch helpers (keeping math pure), with tests for round-trip fidelity.
- Add a tiny benchmark that compares list-collate vs pad-collate for representative raggedness; document guidance in `skills/data-collation.md`.
- Normalize Atlas Tiny row “missing” values toward HF-friendly representation (empty arrays/NaN or explicit nullable features) and document.

Cross-References
- Skills: `skills/data-collation.md`, `skills/repo-layout.md`, `skills/performance-discipline.md`
- Code: `src/viterbo/datasets/atlas_tiny.py:20`, `src/viterbo/datasets/core.py:22`, `src/viterbo/math/polytope.py:1`, `src/viterbo/math/volume.py:1`
- Briefs: `docs/briefs/archive/2025-10-12-task-atlas-migrate-hf-datasets.md:1`, `docs/briefs/archive/2025-10-13-task-datasets-slimming.md:1`

Validation
- `uv run python scripts/load_skills_metadata.py` — OK (printed skill summaries).
- `uv run mkdocs build --strict` — OK (site built; several pages intentionally not in nav).
- Probe: `uv run python - << 'PY'` (Atlas Tiny perf)
  - Result: `{'rows': 3, 'vertices_max': 6, 'facets_max': 6, 'dim': 2, 'vertices_batch_shape': (3, 6, 2), 'volume_dtype': 'torch.float64', 'build_s': 0.225998, 'collate_s': 0.001679}`
- Probe: `uv run python - << 'PY'` (RaggedPointsDataset determinism)
  - Result: equal points/directions for identical seeds at idx 0 and idx 1.

Limitations
- Did not run full test suite or GPU-specific paths; focused on datasets/collation surfaces and 2D Tiny flow.
- Did not modify code to parameterize dtype/device (action proposed); current review is docs-only as requested.
- Did not validate HF Datasets converters (not present yet); roadmap items inferred from briefs.

Status
- Draft

Pre-Submit Checklist
- Do not add nav/index links (per guardrails).
- `uv run mkdocs build --strict` green — yes.
- Actions captured for follow-up tasks.
- Provenance filled.
