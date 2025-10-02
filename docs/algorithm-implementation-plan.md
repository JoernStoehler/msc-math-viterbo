# Implementation Plan for Priority Algorithms

This document scopes the implementation work for the algorithms called out as high-value in the
capacity and volume notes. The plan emphasises separation between mathematically faithful
reference implementations and performance-oriented fast variants, with shared infrastructure,
validation, and profiling strategies that keep both aligned.

Status tags: unmarked items remain in progress; entries labelled ``[done]`` are completed and
``[blocked]`` items require follow-up decisions or dependencies.

## 1. Scope and guiding conventions

- Algorithms covered: the six EHZ-capacity approaches and six polytope-volume methods marked as
  promising in `docs/convex-polytope-cehz-capacities.md` and `docs/convex-polytope-volumes.md`.
- Code layout: colocate the reference and optimised variants under domain-specific packages
  (for example `src/viterbo/symplectic/capacity_algorithms/`). Provide thin wrappers in
  `src/viterbo/symplectic/` or `src/viterbo/geometry/` when a stable public API is required.
- Reference vs fast:
  - Reference modules optimise for clarity and correctness, relying on straightforward numerical
    methods (dense linear algebra, exact arithmetic libraries, generic solvers).
  - Fast modules focus on asymptotic improvements (sparser data structures, caching, incremental
    updates) only after profiling demonstrates a bottleneck. They may require additional optional
    dependencies guarded by feature flags.
- Validation philosophy:
  - Cross-check against analytical values for cubes, simplices, cross-polytopes, and Lagrangian
    products with known capacities/volumes.
  - Compare algorithmic outputs after symplectic/affine normalisations to ensure invariance.
  - Use reference implementations as oracles for the fast counterparts on random polytopes in low
    dimension, and compare different algorithms where theoretical equality/ordering holds.
- Tooling:
  - Integrate open-source solvers instead of hand-rolling LP/MILP/SDP routines (`cvxpy` with
    default ECOS/SCS backends, `pulp` or `mip` for MILP, `networkx` for graph searches, `pygurobi`
    behind an optional extra if licensed).
  - Reuse existing half-space and vertex utilities in `viterbo.geometry.polytopes` for conversions and
    canonicalisation.

## 2. Shared infrastructure tasks

| Task | Description | Dependencies | Validation |
| --- | --- | --- | --- |
| [done] Geometry backends | Extend `viterbo.geometry.polytopes` with converters between H/V-representations, facet adjacency, and normal fan traversal utilities. Cache combinatorial data keyed by hashable polytope fingerprints. | `scipy`, `pycddlib` (optional) | Unit tests covering cubes, simplices, products; deterministic hashing |
| [done] Symplectic form helpers | Centralise symplectic matrix builders, support-function evaluations, and Minkowski sums in `viterbo.symplectic.core`. | `numpy` | Compare with closed-form values on axis-aligned boxes |
| [done] Solver abstraction | Wrap LP/MILP/SDP calls behind thin adapters that accept problem objects and choose a backend (`cvxpy`, `pulp`, `sdpa-python`). Allow dependency injection for tests. | `cvxpy`, `pulp`, `sdpa-python` (optional) | Mock adapters in tests; smoke-test simple problems with known optima |
| [done] Test fixtures | Expand `tests/geometry/_polytope_samples.py` with representative polytopes (toric, centrally symmetric, product). Provide expected capacities/volumes when available. | none | Verified values from literature |
| Benchmark harness | Add `tests/performance/` cases that exercise each fast implementation with pytest-benchmark markers and profile hooks. | `pytest-benchmark`, `pytest-line-profiler` | Baseline run records stored under `.benchmarks/` |

## 3. EHZ capacity algorithms

### 3.1 Facet-normal optimisation (Leipold–Vallentin)
- **Reference (`facet_normals_reference.py`)**
  - [done] Extract the reference enumeration algorithm into a dedicated module with typed facet-subset containers.
  - [blocked] Replace the permutation search by the `cvxpy` quadratic-program formulation and investigate rational arithmetic for degeneracy debugging.
- **Fast (`facet_normals_fast.py`)**
  - [done] Implement dynamic-programming order search and reuse cached subset data provided by the reference helper.
  - [blocked] Integrate rank-one nullspace updates and QP backends (`quadprog`/`osqp`).
  - [blocked] Add cone prefilters and symmetry heuristics to prune infeasible subsets.
- **Validation**
  - [done] Cross-check reference and fast algorithms on cubes/cross-polytopes and assert symplectic invariance in regression tests.
  - Compare against the current `viterbo.symplectic.capacity.compute_ehz_capacity` implementation on 4D samples.
- **Profiling triggers**
  - Benchmark facet counts 8–14 in 4D, inspect time per subset; optimise only if fast version yields >2× improvement.

### 3.2 MILP relaxation (Krupp)
- **Reference (`ehz_milp_reference.py`)**
  - Formulate the MILP with `cvxpy` + `cvxpy.MIP` or `pulp`, exploring facet subsets up to size `2n`. Return upper/lower bounds and certificates.
  - Allow solver selection via configuration; default to CBC through `pulp` for availability.
- **Fast (`ehz_milp_fast.py`)**
  - Implement cutting-plane refinement: start from relaxation without integrality, then branch only on promising facets.
  - Exploit symmetry reduction for centrally symmetric polytopes by grouping variables.
- **Validation**
  - Verify MILP bounds sandwich the facet-normal optimum on benchmark polytopes.
  - For cases where MILP returns exact solution, ensure equality with reference EHZ values.
- **Dependencies**
  - Introduce optional extras `milp` that pull `pulp`, `mip`, and allow plugging commercial solvers.

### 3.3 Combinatorial Reeb orbit enumeration (Chaidez–Hutchings, 4D)
- **Reference (`reeb_cycles_reference.py`)**
  - Implement orientation-augmented edge graph builder using `networkx`. Enumerate admissible cycles via depth-first search with pruning.
  - Compute combinatorial actions using rational arithmetic from facet data.
- **Fast (`reeb_cycles_fast.py`)**
  - Encode admissibility transitions as bitsets and perform cycle enumeration via Johnson’s algorithm with pruning on action lower bounds.
  - Cache edge-to-facet incidence to reuse across transformations.
- **Validation**
  - Cross-check against published Chaidez–Hutchings examples (24-cell, perturbed cubes).
  - Confirm minimal action matches facet-normal optimisation outputs in 4D.
- **Profiling**
  - Run pytest benchmark on families of random 4-polytopes with 16–24 facets; inspect cycle counts.

### 3.4 Minkowski billiard shortest path (Rudolf)
- **Reference (`minkowski_billiards_reference.py`)**
  - Build normal fan as a directed graph and run exhaustive search over paths up to length `n+1`, computing lengths via support functions from `viterbo.symplectic.core`.
  - Use `itertools.product` for low-dimensional enumeration.
- **Fast (`minkowski_billiards_fast.py`)**
  - Apply dynamic programming on the normal fan with memoised path prefixes and prune using triangle inequality bounds.
  - For Lagrangian products, separate computations on `K` and `T` fans; reuse convolution-style updates.
- **Validation**
  - Compare outputs with MILP bounds for product polytopes.
  - Validate against known billiard trajectories in cubes × cross-polytopes.

### 3.5 Support-function convex relaxations (Haim-Kislev)
- **Reference (`support_relaxation_reference.py`)**
  - Implement smoothing via discrete convolution on the support function sampled on a dense grid. Solve the resulting convex optimisation with `cvxpy`.
- **Fast (`support_relaxation_fast.py`)**
  - Replace grid sampling with adaptive refinement based on gradient norms; use GPU-friendly array ops if `cupy` is available.
  - Integrate line-search continuation on the smoothing parameter ε to accelerate convergence.
- **Validation**
  - Show monotone decrease of the upper bound sequence toward the exact capacity (as verified by reference algorithms) on sample polytopes.
  - Stress-test invariance under translations/scalings.

### 3.6 Symmetry-reduced search (Artstein-Avidan–Ostrover)
- **Reference (`symmetry_reduced_reference.py`)**
  - Take facet pairing information and solve the reduced optimisation directly with `cvxpy`, verifying β-pair constraints explicitly.
- **Fast (`symmetry_reduced_fast.py`)**
  - Integrate with the fast facet-normal solver but restrict enumeration to representative orbits under the symmetry group.
  - Use integer relation detection to identify paired facets automatically.
- **Validation**
  - Cross-compare capacities from the reduced solver with unreduced facet-normal optimisation on centrally symmetric benchmarks.

## 4. Volume algorithms

### 4.1 Lawrence sign-decomposition
- **Reference (`lawrence_reference.py`)**
  - Implement vertex enumeration using `pycddlib`, compute signed cone integrals with exact rationals (`fractions.Fraction`).
- **Fast (`lawrence_fast.py`)**
  - Cache LU decompositions of vertex tangent matrices; vectorise cone volume calculations using NumPy.
- **Validation**
  - Test on cubes, simplices, permutohedra; compare with triangulation output.

### 4.2 Beneath–Beyond triangulation
- **Reference (`beneath_beyond_reference.py`)**
  - Wrap existing `lrs` or `pycddlib` incremental hull enumeration to obtain triangulations; compute simplex volumes via determinants.
- **Fast (`beneath_beyond_fast.py`)**
  - Maintain incremental determinant updates and reuse adjacency caches between insertions.
  - Provide streaming interface for high-vertex polytopes in moderate dimensions (≤8).
- **Validation**
  - Confirm triangulation volume equals Lawrence method within tolerance; ensure triangulation produces oriented simplices covering the polytope.

### 4.3 Barvinok generating functions
- **Reference (`barvinok_reference.py`)**
  - Use `pybarvinok` bindings (optional dependency) to obtain short rational generating functions and evaluate derivatives symbolically.
- **Fast (`barvinok_fast.py`)**
  - Precompute unimodular cone decompositions and parallelise evaluation using multiprocessing; allow caching of intermediate series expansions.
- **Validation**
  - Compare with Lawrence volumes for fixed dimensions (d ≤ 6).
  - Validate coefficient extraction by checking lattice-point counts on dilates.

### 4.4 Randomised hit-and-run FPRAS
- **Reference (`hitandrun_reference.py`)**
  - Implement vanilla hit-and-run walk with isotropic rounding using `scipy.linalg.svd`. Provide reproducible RNG seeding.
- **Fast (`hitandrun_fast.py`)**
  - Integrate Lovász–Vempala accelerated cooling schedule, vectorised sampling, and optional `numba`-accelerated kernels.
  - Support parallel chains to reduce variance.
- **Validation**
  - Statistical tests comparing estimates against exact volumes for low-dimensional polytopes; ensure concentration bounds hold.
  - Cross-validate randomised estimates with deterministic methods on the same inputs.

### 4.5 Moment-based SDP relaxations
- **Reference (`moment_sdp_reference.py`)**
  - Encode Lasserre hierarchy using `cvxpy` with the `SCS` solver; generate moment matrices and extract bounds.
- **Fast (`moment_sdp_fast.py`)**
  - Switch to `mosek`/`sdpa-python` when available; exploit sparsity and chordal decomposition for larger orders.
  - Implement warm-start continuation in the relaxation order.
- **Validation**
  - Verify monotone bound convergence on benchmark polytopes; compare mid-range bounds with hit-and-run estimates.

### 4.6 Four-dimensional specialised exact method
- **Reference (`four_d_reference.py`)**
  - Implement facet-ridge graph traversal using `networkx`; compute shellings and accumulate simplex volumes with exact rationals.
- **Fast (`four_d_fast.py`)**
  - Precompute ridge adjacency using NumPy arrays; apply caching for repeated volume queries on related polytopes (e.g., during optimisation loops).
- **Validation**
  - Ensure the 4D specialised result matches general-purpose exact methods; add regression tests for known 4-polytopes (24-cell, 4D cube, simplex).

## 5. Testing, benchmarking, and documentation roadmap

1. **Testing layers**
   - Unit tests per module validating mathematical identities and edge cases.
   - Property-based tests using `hypothesis` (optional dependency) for random polytopes subject to invariance checks.
   - Integration tests that compare outputs across algorithms (e.g., Viterbo inequality `c_EHZ^n ≤ n!·vol`).
2. **Benchmark suites**
   - For each fast implementation, add `@pytest.mark.benchmark` tests and profile runs via `make profile` before deciding on micro-optimisations.
   - Record baseline metrics in the repository wiki or docs for trend tracking.
3. **Documentation**
   - Update `docs/convex-polytope-cehz-capacities.md` and `docs/convex-polytope-volumes.md` with implementation status tables.
   - Provide API reference snippets in `docs/README.md` once modules are stable.
4. **Dependency management**
   - Introduce optional extras in `pyproject.toml` (`milp`, `sdp`, `barvinok`, `sampling`) so users can install only the stacks they need.
   - Document solver installation pitfalls (e.g., `pybarvinok` prerequisites) in `docs/README.md`.
5. **Release criteria**
   - Reference implementations must match literature benchmarks within numerical tolerance.
   - Fast implementations should demonstrate measurable speed-ups (>2×) on profiling benchmarks before being advertised as default.
   - All new modules require docstrings, jaxtyping annotations, and coverage in tests/benchmarks.

## 6. Open questions and sequencing

- Prioritise EHZ facet-normal and Reeb enumeration first to unlock validation pathways for other capacity algorithms.
- Evaluate the practicality of Barvinok and SDP dependencies early; if bindings prove unstable, consider delegating these paths to CLI wrappers instead of Python APIs.
- Decide whether to expose fast implementations behind feature flags (e.g., environment variable `VITERBO_USE_FAST=1`) or via explicit API choice.
- Defer deep optimisation passes until profiling reveals hotspots; focus on correctness and cross-validation harnesses in the initial milestone.

This plan should feed into task breakdowns for upcoming sprints and guide dependency onboarding discussions before implementation begins.
