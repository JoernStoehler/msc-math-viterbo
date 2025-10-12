status: in-progress
created: 2025-10-10
workflow: task
summary: Roadmap for implementing flat viterbo namespace and decommissioning legacy modules.
---

# Task Brief: Migrate to the flat `viterbo` Architecture

## 1. Objective

Deliver a production-ready implementation of the flat `viterbo` namespace, exercise it end to end, and retire the legacy code paths without breaking existing workflows.

## 2. Current Context

- `viterbo` exposes small JAX-first modules (`polytopes`, `atlas`, `basic_generators`, `volume`, `capacity`, `spectrum`, `cycles`).
- The `atlas` module still targets Polars, but we intend to replace Polars with Hugging Face Datasets so that ~1 GB tables stay memory-resident while remaining easy to share and reuse via on-disk artefacts and integrated ML dataloaders.
- `polytopes.incidence_matrix` provides the first reference primitive with strict float64 tolerances (rtol=1e-12, atol=0.0). Geometry builders:
  - `build_from_vertices` uses SciPy Qhull equations to robustly produce outward normals/offsets and deduplicated hull vertices.
  - `build_from_halfspaces` returns a Polytope with given halfspaces and an empty vertex set (vertex enumeration TBD).
- Tests have been migrated to the modern surfaces and now assert intended behaviours instead of expecting `NotImplementedError`; unimplemented functions therefore cause failing tests by design.
- Capacity, spectrum, and cycle modules still contain heuristic or placeholder implementations; the promised Haim–Kislev subset solver and related utilities never landed despite earlier status notes. There is no separate `viterbo.modern` artefact to recover.
- 2D Reeb cycles are trivial; the modern roadmap focuses on 4D first and ≥6D second. We no longer plan to implement 2D spectrum, capacity, or cycle machinery.
- Batched helpers (e.g., `*_batched`) are not needed in the near term and should be removed until the batching roadmap restarts.
- Legacy functionality still lives under `src/viterbo/` and its consumers/tests/notebooks.
- CI expects smoke-level pytest coverage, Ruff linting, and Pyright type checking. Per repository policy we avoid defining `__all__` re-export lists.

## 3. Work Plan

1. **Map the surfaces**
   - Confirm module boundaries and public signatures for: `polytopes` (geometry ops), `atlas` (dataset adapter), `basic_generators` (sampling), and quantity modules (`volume`, `capacity`, `spectrum`, `cycles`).
   - Catalogue missing primitives for parity with legacy code; prefer readable reference first, then fast variants.
2. **Replace stubs with implementations**
   - Implement pure JAX-first math routines (polytopes, generators, capacities, spectra, cycles) with explicit padding strategies where they remain relevant. Defer batched APIs until the roadmap revives that need.
   - Build out `atlas` helpers for dataframe materialisation and row conversions; execute the migration from Polars to Hugging Face Datasets once schemas stabilise.
   - Keep notebooks executable end-to-end by updating them to call the modern surfaces.
3. **Design validation**
   - Draft a test matrix covering unit correctness, geometric invariants, numerical stability, and non-batched control flows.
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
- Update docs, briefs, and any orchestration scripts to point to the flat `viterbo` namespace as the canonical entry point.
   - Record migration notes or ADRs if major architectural decisions change.

## 4. Deliverables

- Fully implemented flat `viterbo` package with comprehensive tests and benchmarks.
- Updated notebooks illustrating atlas generation and consumption using the new APIs (`atlas` instead of `datasets`/`converters`).
- CI evidence (lint, typecheck, pytest) demonstrating readiness.
- Migration commits removing obsolete code and updating documentation/briefs.

## 5. Dependencies & Open Questions

- Data layer direction: replace Polars with Hugging Face Datasets. Define the mapping from `atlas` schema to HF features and confirm how we persist and reload intermediate computations.
- Confirm intermediate storage conventions (Arrow/Parquet) until HF migration completes.
- Coordinate with maintainers on timing for removing legacy entry points.
- Determine whether additional tooling (e.g., benchmarking harnesses) is required before the cleanup phase.

## 7. Status Update (2025-10-11)

- Implemented
  - `polytopes.build_from_vertices`: Qhull-backed normals/offsets + hull vertex selection; incidence computed.
  - `polytopes.build_from_halfspaces`: enumerates vertices via SciPy HalfspaceIntersection using a Chebyshev-center interior point; preserves provided halfspaces; computes incidence.
  - `basic_generators`: sphere/ball (d+1 points), halfspace and halfspace-tangent; return `Polytope` objects with float64 data.
  - `volume.volume_reference`: exact in 2D via shoelace over hull-ordered vertices; Qhull volume for d>2.
  - `symplectic.random_symplectic_matrix`: random Sp(2n) sampler via expm of Hamiltonian; added `standard_symplectic_matrix`.

- Missing / inaccurate
  - `capacity.ehz_capacity_reference` currently falls back to a support-radius heuristic instead of the Haim–Kislev facet-subset solver. The modern module must be rewritten to match the desired algorithm.
  - `cycles.minimum_cycle_reference` still mirrors the legacy placeholder behaviour; the robust 4D oriented-edge extractor should be ported and validated. Higher dimensions remain unexplored.
  - `spectrum.ehz_spectrum_reference` is limited to a partial 4D baseline and retains unimplemented batching helpers. 2D variants should be dropped, and higher dimensions should follow the facet-subset plan once capacity parity is achieved.

- Deferred / removed
  - Batched helpers (e.g., `*_batched`) will be deleted until the roadmap requires them again.
  - Spectrum general algorithms (2n) remain future work pending the capacity rewrite and validated cycle extraction.

- Tests
  - Modern tests cover builders (H/V) and basic capacity invariants, but they still assume the placeholder heuristics. Once the reference algorithms land we must update or replace these expectations. Batched tests can be dropped alongside the removed APIs.

## 8. Algorithm Notes (for follow-up tasks)

- Capacity (required direction)
- Facet-normal (Haim–Kislev) reference: enumerate subsets of size 2n+1, solve Reeb measures, maximize quadratic form over permutations/dynamic programming, take min over subsets. Port the authoritative implementation into `viterbo.capacity`.
  - 2D equivalence: c_EHZ equals area for planar domains, but the roadmap no longer prioritises specialised 2D capacity code.
  - Candidates for later: symmetry-reduced variants; MILP formulation; Reeb-cycle constrained variants; support-function relaxations.

- Cycles (initial 4D focus)
  - Oriented-edge graph (Chaidez–Hutchings): build graph from 4D facet triples; extract a simple cycle and validate. Reference lives under `exp1.reeb_cycles` and `symplectic.capacity.reeb_cycles`.
  - 2D Reeb cycles are trivial and out of scope.
  - General-dimension strategy TBD; options include dual graphs, normal fan walks, or MILP-based cycle search.

- Spectrum (blocked)
  - 4D baseline remains incomplete until the cycle extractor is ported faithfully; revisit once capacity parity is achieved.
  - 2D spectrum work is dropped.
  - Candidates for general `2n`: (i) admissible Reeb orbit enumeration; (ii) billiard-based discrete actions on PL boundaries; (iii) facet-subset induced stationary measures. Define ordering, multiplicities, and regularization.

- Volume
  - Prefer SciPy Qhull per sample when vertices are available; consider HalfspaceIntersection in 2D/3D when only halfspaces are given. Avoid Monte Carlo.

### Padding Semantics (reference)

- General approach: use in-band masks with NaN padding where applicable so we can apply `jnp.nan*` reductions (`nanmin/mean/sum`) and avoid dynamic shapes.
- Neutral values: for some algorithms, `0.0` or `+/-inf` can be neutral with respect to `min`/`max` aggregations — document per-function when this is stable.
- Control flow: prefer masked `lax.scan`/`vmap` over early returns; JIT/XLA requires static shapes. Keep masks explicit and consistent across functions once batching resurfaces.
- Incidence/indices: use `-1` for padded indices and boolean masks for validity; pair with NaN-padded float arrays when mixing continuous and discrete states.

## 9. Next Steps

- Rebuild `viterbo.capacity` around the Haim–Kislev facet-subset solver, including any combinatorial helpers it requires.
- Port the robust 4D Reeb cycle extractor and align the 4D spectrum baseline with it; document ≥6D expectations as future work.
- Remove batched helpers from the public API and tests until demand resurfaces; ensure padding policy notes remain for future reference.
- Plan and execute the migration from Polars to Hugging Face Datasets in `atlas` once the modern algorithms are stable.
- Evaluate performance; add modern benchmarks mirroring existing symplectic ones.

## 10. Handoff Summary (2025-10-11)

- modern is decoupled from legacy (`exp1`/`symplectic`) but several promised algorithms remain missing in `capacity`, `cycles`, and `spectrum`.
- Geometry builders are in place; quantity modules require substantial follow-up (capacity rewrite, cycle port, spectrum alignment) before we can decommission legacy code.
- Padding policy: in-band invalidation only (NaN floats; -1 indices). No separate masks.
- Tests: smoke + deep tiers cover builders and heuristic capacities; they need revision once the reference algorithms land.
- Open items: capacity rewrite, cycle extractor port, 4D spectrum alignment, ≥6D roadmap definition, Polars → Hugging Face migration.

## 6. Success Criteria

- All legacy consumers can switch to `viterbo` without regressions in coverage, performance, or maintainability.
- CI remains green post-migration, and notebooks execute without manual intervention.
- Documentation reflects the new architecture, and no deprecated references remain in the repo.
- Geometry predicates use float64 with strict tolerances; behaviour is documented where relevant.
