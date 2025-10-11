status: in-progress
created: 2025-10-10
workflow: task
summary: Roadmap for implementing viterbo.modern fully and decommissioning legacy modules.
---

# Task Brief: Migrate to the viterbo.modern Architecture

## 1. Objective

Deliver a production-ready implementation of the `viterbo.modern` namespace, exercise it end to end, and retire the legacy code paths without breaking existing workflows.

## 2. Current Context

- `viterbo.modern` exposes small JAX-first modules (`polytopes`, `atlas`, `basic_generators`, `volume`, `capacity`, `spectrum`, `cycles`).
- The `atlas` module is the thin adapter for tabular IO (currently Polars), with row helpers (`atlas_pl_schema`, `as_polytope`, `as_cycle`).
- `polytopes.incidence_matrix` provides the first reference primitive with strict float64 tolerances (rtol=1e-12, atol=0.0). Geometry builders:
  - `build_from_vertices` uses SciPy Qhull equations to robustly produce outward normals/offsets and deduplicated hull vertices.
  - `build_from_halfspaces` returns a Polytope with given halfspaces and an empty vertex set (vertex enumeration TBD).
- Tests have been migrated to the modern surfaces and now assert intended behaviours instead of expecting `NotImplementedError`; unimplemented functions therefore cause failing tests by design.
- Legacy functionality still lives under `src/viterbo/` and its consumers/tests/notebooks.
- CI expects smoke-level pytest coverage, Ruff linting, and Pyright type checking.

## 3. Work Plan

1. **Map the surfaces**
   - Confirm module boundaries and public signatures for: `polytopes` (geometry ops), `atlas` (dataset adapter), `basic_generators` (sampling), and quantity modules (`volume`, `capacity`, `spectrum`, `cycles`).
   - Catalogue missing primitives for parity with legacy code; prefer readable reference first, then fast variants.
2. **Replace stubs with implementations**
   - Implement pure JAX-first math routines (polytopes, generators, capacities, spectra, cycles) with explicit padding strategies.
   - Build out `atlas` helpers for dataframe materialisation and row conversions; plan a follow-up to swap Polars for HF datasets once the schemas stabilise.
   - Keep notebooks executable end-to-end by updating them to call the modern surfaces.
3. **Design validation**
- Draft a test matrix covering unit correctness, geometric invariants, numerical stability, and batching behaviour.
- Tests should describe intended behaviour and therefore fail until implementations land (do not assert `NotImplementedError`).
- Add/expand pytest suites with proper goal/suite markers; include property-based tests where useful.
- Plan performance benchmarks if any new algorithms are asymptotically heavier than the legacy versions.
4. **Quality gates**
   - Ensure Ruff, Pyright, and `pytest` (smoke tier) pass locally and in CI; add deep/longhaul runs if risk dictates.
   - Execute the notebooks end to end (script mode) to confirm dataset build/consume flows succeed.
5. **Review cycle**
   - Conduct an internal review of API ergonomics, padding semantics, and documentation before requesting maintainer review.
   - Address feedback, update docstrings/tutorial notebooks, and confirm all automation stays green.
6. **Migration and cleanup**
   - Remove deprecated legacy modules, tests, and notebooks once the modern replacements are verified.
   - Update docs, briefs, and any orchestration scripts to point to `viterbo.modern` as the canonical entry point.
   - Record migration notes or ADRs if major architectural decisions change.

## 4. Deliverables

- Fully implemented `viterbo.modern` package with comprehensive tests and benchmarks.
- Updated notebooks illustrating atlas generation and consumption using the new APIs (`atlas` instead of `datasets`/`converters`).
- CI evidence (lint, typecheck, pytest) demonstrating readiness.
- Migration commits removing obsolete code and updating documentation/briefs.

## 5. Dependencies & Open Questions

- Data layer direction: plan to replace Polars with Hugging Face Datasets. Define the mapping from `atlas` schema to HF features and batching semantics.
- Confirm intermediate storage conventions (Arrow/Parquet) until HF migration completes.
- Coordinate with maintainers on timing for removing legacy entry points.
- Determine whether additional tooling (e.g., benchmarking harnesses) is required before the cleanup phase.

## 7. Status Update (2025-10-11)

- Implemented
  - `polytopes.build_from_vertices`: Qhull-backed normals/offsets + hull vertex selection; incidence computed.
  - `polytopes.build_from_halfspaces`: now enumerates vertices via SciPy HalfspaceIntersection using a Chebyshev-center interior point; preserves provided halfspaces; computes incidence.
  - `basic_generators`: sphere/ball (d+1 points), halfspace and halfspace-tangent; return `Polytope` objects with float64 data.
  - `volume.volume_reference`: exact in 2D via shoelace over hull-ordered vertices; Qhull volume for d>2.
  - `capacity.ehz_capacity_reference`: 2D uses area; higher-even dims use facet-normal Haim–Kislev reference (ported into modern; no legacy deps).
  - `cycles.minimum_cycle_reference`: 4D oriented-edge Reeb cycle extraction implemented directly over modern incidence/vertices (no legacy deps); other dims TBD.
  - `spectrum.ehz_spectrum_reference`: 4D oriented-edge baseline implemented (Euclidean PL length); DFS bounded by `head`.
  - `symplectic.random_symplectic_matrix`: random Sp(2n) sampler via expm of Hamiltonian; added `standard_symplectic_matrix`.

- Deferred
  - Batching across quantities (capacity/spectrum/cycles/volume) — shape-only placeholders remain where needed; not critical now.
  - Spectrum general algorithms (2n) — 4D baseline is in; generalization pending (see notes below).

- Tests
  - Modern tests expanded:
    - Builders (H/V), 2D properties (capacity=area, rotation), 4D invariants (scaling, inclusion, symplectic invariance), 4D spectrum smoke, batched spectrum NaN padding.
    - Property-based tests (2D) and 2D baselines as JSON under `tests/_baselines/`.

## 8. Algorithm Notes (for follow-up tasks)

- Capacity (chosen reference now)
  - Facet-normal (Haim–Kislev) reference: enumerate subsets of size 2n+1, solve Reeb measures, maximize quadratic form over permutations/dynamic programming, take min over subsets. Implementation lives under `viterbo.symplectic.capacity.facet_normals.reference`.
  - 2D equivalence: c_EHZ equals area for planar domains (used in `modern.capacity`).
  - Candidates for later: symmetry-reduced variants; MILP formulation; Reeb-cycle constrained variants; support-function relaxations.

- Cycles (initial 4D)
  - Oriented-edge graph (Chaidez–Hutchings): build graph from 4D facet triples; extract a simple cycle and validate. Reference lives under `exp1.reeb_cycles` and `symplectic.capacity.reeb_cycles`.
  - General-dimension strategy TBD; options include dual graphs, normal fan walks, or MILP-based cycle search.

- Spectrum (blocked)
  - 4D baseline implemented: oriented-edge cycles + Euclidean PL action; DFS bounded by `head`.
  - Candidates for general `2n`: (i) admissible Reeb orbit enumeration; (ii) billiard-based discrete actions on PL boundaries; (iii) facet-subset induced stationary measures. Define ordering, multiplicities, and regularization.

- Volume (batched)
  - Prefer SciPy Qhull per sample when vertices are available; consider HalfspaceIntersection in 2D/3D when only halfspaces are given. Avoid Monte Carlo.

### Padding Semantics (proposed)

- General approach: use in-band masks with NaN padding where applicable so we can apply `jnp.nan*` reductions (`nanmin/mean/sum`) and avoid dynamic shapes.
- Neutral values: for some algorithms, `0.0` or `+/-inf` can be neutral with respect to `min`/`max` aggregations — document per-function when this is stable.
- Control flow: prefer masked `lax.scan`/`vmap` over early returns; JIT/XLA requires static shapes. Keep masks explicit and consistent across batched functions.
- Incidence/indices: use `-1` for padded indices and boolean masks for validity; pair with NaN-padded float arrays when mixing continuous and discrete states.

## 9. Next Steps

1. Generalize spectrum beyond 4D; remove the legacy 2D spectrum xfail.
2. Implement true batched paths (capacity, cycles, spectrum, volume) using in-band NaN/(-1) semantics.
3. Add curated 4D baselines under `tests/_baselines/` (with tolerances) once values are agreed.
4. Evaluate performance; add modern benchmarks mirroring existing symplectic ones.

## 10. Handoff Summary (2025-10-11)

- modern is decoupled from legacy (`exp1`/`symplectic`); algorithms ported where needed.
- Geometry and quantities implemented per above; spectrum has a 4D baseline.
- Padding policy: in-band invalidation only (NaN floats; -1 indices). No separate masks.
- Tests: smoke + deep tiers cover builders, capacity invariants (2D/4D), spectrum smoke/batched, and baselines.
- Open items: general spectrum (2n), batched implementations, 4D numeric baselines.

## 6. Success Criteria

- All legacy consumers can switch to `viterbo.modern` without regressions in coverage, performance, or maintainability.
- CI remains green post-migration, and notebooks execute without manual intervention.
- Documentation reflects the new architecture, and no deprecated references remain in the repo.
- Geometry predicates use float64 with strict tolerances; behaviour is documented where relevant.
