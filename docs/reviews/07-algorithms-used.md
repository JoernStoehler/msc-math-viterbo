# Review 07 — Algorithms Used

Provenance
- Topic: 07 — Algorithms used
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 96e7f5a
- Timebox: ~90–120 minutes
- Scope: Survey of algorithm choices, specs, correctness notes, complexity/performance, and fallback/approximation policies across implemented math/geometry, EHZ capacity modules, constructions, runtime and C++ backends. Excludes exhaustive proof checking and bench profiling beyond repo docs; stubs noted but not expanded beyond their contracts.
- Version: v0.1

Context
- Goal: capture an unsorted, exhaustive pass over algorithms present in the repository, focusing on what is implemented, how it works, expected complexity, invariants/correctness guards, and any fallbacks or approximation modes.
- Timing: performed alongside Topic 6 (Math used) and Topic 8 (Review process) to consolidate a baseline for next implementation steps (volume backends, 4D capacities).

Inputs & Method
- Ran: `uv run python scripts/load_skills_metadata.py --quiet` (sync skills index per AGENTS.md boot sequence)
- Searched source: `rg --files src | sed -n '1,200p'`, `sed -n` reads of math and capacity modules
- Read docs: `docs/math/*.md` (polytope, volume, capacity_ehz, oriented_edge_spectrum_4d, cstar_constant_spec, constructions, symplectic, polar, similarity)
- Built docs: `uv run mkdocs build --strict` (green prior to edits; verified again after adding this file)

Findings (unsorted)
- Torch-first geometry conventions are consistent: every function preserves input `dtype`/`device`, uses no implicit device transfers, and picks tolerances as `tol = max(√eps(dtype), 1e-9)` throughout. Examples: `src/viterbo/math/volume.py:40`, `src/viterbo/math/polytope.py:71`, `src/viterbo/math/capacity_ehz/algorithms.py:37`.
- Support function helpers:
  - `support(points, direction)` computes `max_i ⟨p_i, v⟩` in O(N·D) via a single matmul and reduce-max; stable and shape-checked. `src/viterbo/math/polytope.py:23`.
  - `support_argmax(points, direction)` yields both value and index in O(N·D); uses `torch.max` to return an integer index. `src/viterbo/math/polytope.py:34`.
  - Correctness relies on convexity only insofar as a maximiser is among given vertices; no convex hull computed here (caller’s responsibility).
- Pairwise distances: `pairwise_squared_distances(points)` uses the standard x^2+y^2−2xy trick; clamps negatives to zero to counter FP error. Complexity O(N^2·D) time and O(N^2) memory. `src/viterbo/math/polytope.py:48`.
- Halfspace violations: `halfspace_violations(points, B, c)` returns ReLU(Bx−c). Vectorised O(N·F·D). `src/viterbo/math/polytope.py:63`.
- Axis-aligned bounding box: `bounding_box(points)` returns per-dim min/max; O(N·D). `src/viterbo/math/polytope.py:70`.
- H/V conversions:
  - `vertices_to_halfspaces(vertices)` requires full-dimensional input (rank D, ≥D+1 vertices). In 1D returns trivial two facets; for D≥2 calls SciPy’s `ConvexHull` (Qhull) on CPU float64 and then renormalises equations so each normal has unit length; deduplication via pairwise `allclose` with tol; returns back on original device/dtype. Complexity dominated by Qhull; worst case O(M^{⌈D/2⌉}) but very fast in low D. `src/viterbo/math/polytope.py:95`, `docs/math/polytope.md:22`.
  - `halfspaces_to_vertices(B, c)` enumerates all D-sized facet subsets, solves `B_I x = c_I`, filters infeasible candidates (`max(Bx−c) ≤ tol`), deduplicates, then lexicographically orders on CPU. Complexity combinatorial in F choose D; intended for small D/F (planar and 4D scale with modest F). `src/viterbo/math/polytope.py:139`.
  - Correctness notes: requires positive offsets (0 ∈ int), numerical guarding via `torch.linalg.solve` try/except and tolerance filters; raises when no feasible vertices found.
- Volume algorithms: `viterbo.math.volume.volume(vertices)` implements a tiered approach. `docs/math/volume.md:9`.
  - D=1: interval length from min/max. `src/viterbo/math/volume.py:28`.
  - D=2: convex-hull ordering via centroid-angles, then shoelace formula. Handles duplicate vertices via `torch.unique`. `src/viterbo/math/volume.py:8`, `:31`.
  - D≥3: converts to H-rep, for each facet: (i) collects facet vertices via mask `⟨v, n⟩ ≈ λ`; (ii) computes (d−1)-measure by projecting facet vertices into their affine span using SVD and recursing into `volume` on embedded coordinates; and (iii) accumulates `area_facet * |height| / d` from the centroid (pyramid decomposition / divergence theorem flavour). `src/viterbo/math/volume.py:46`.
  - Correctness: assumes convex input and well-separated facets; degenerate faces drop out by tol checks; heights signed but magnitude taken; overall volume positive. Numerical stability bounded by hull conversion and SVD rank decisions.
  - Not implemented stubs: `volume_via_triangulation`, `volume_via_lawrence`, `volume_via_monte_carlo` with docs describing target specs and roles (exact triangulations, sign-decomposition, MC estimator). `src/viterbo/math/volume.py:79`.
- Symplectic linear algebra: `symplectic_form(d)` builds the canonical J; `random_symplectic_matrix(d, seed)` constructs a matrix in Sp(2n) by QR for an orthogonal block A and symmetric shears U/L, returning `upper @ block_a @ lower` which is provably symplectic. Deterministic under given seed/generator. `src/viterbo/math/symplectic.py:12`, `:24`. Documented in `docs/math/symplectic.md:9`.
- Constructions and transforms:
  - `rotated_regular_ngon2d(k, angle)`: places k points on the unit circle at `angle + 2π i/k`, computes outward edge normals, flips sign to ensure positive offsets; returns both V and H reps. `src/viterbo/math/constructions.py:23`.
  - `matmul_vertices(A, V)` and `translate_vertices(t, V)` apply linear maps and translations to vertices; simple batched ops with dimension checks. `src/viterbo/math/constructions.py:57`, `:69`.
  - `matmul_halfspace(A, B, c)` transforms H-rep under x↦Ax via `A^{−T} B`, then re-normalises each row to unit length and scales offsets accordingly; rejects zero-norm rows. Ensures invariance for invertible A. `src/viterbo/math/constructions.py:79`.
  - `translate_halfspace(t, B, c)` updates offsets as `c' = c + B t`. `src/viterbo/math/constructions.py:100`.
  - `lagrangian_product(V_P, V_Q)`: builds `Q×T ⊂ ℝ^{2n}` when `dim(P)=dim(Q)`: H-rep becomes block-diagonal concat of their facets; vertices are the cartesian product. `src/viterbo/math/constructions.py:108`.
  - Random generators: `random_polytope_algorithm1/2` sample directions/points uniformly in the unit ball (radius via `U^{1/d}`), guarantee containment of the origin (axis-aligned box added in alg1), and round-trip through H/V to clean duplicates; reproducible via an explicit torch `Generator`. `src/viterbo/math/constructions.py:130`.
- Planar polygon helpers (capacity context):
  - Vertex ordering CCW by centroid-angle sort, then area via shoelace; used as EHZ placeholder for 2D. `src/viterbo/math/capacity_ehz/common.py:39`, `:55`.
  - Simple geometric predicates: `satisfies_reflection_at_vertex` checks whether a given direction exposes a polygon vertex using `support`. `src/viterbo/math/capacity_ehz/common.py:68`.
- Lagrangian-product Minkowski billiards (≤3 bounces):
  - `minimal_action_cycle_lagrangian_product(Q_vertices, T_normals, T_offsets, max_bounces∈{2,3})` enumerates all 2-bounce and 3-bounce candidates over ordered polygon `Q` and polygon `T` (recovered from H-rep), tests discrete reflection constraints, computes action via support values, and returns the minimal-action cycle with deterministic tie-breaking by lexicographic comparison of flattened cycle coordinates. `src/viterbo/math/capacity_ehz/lagrangian_product.py:1`.
  - Complexity: O(|Q|^2 + |Q|^3) support queries; each query O(|T|·2) via `support_argmax`. Works well for small polygons; generalisation to facet-interior contacts is planned.
  - Correctness: specialisation of Artstein–Avidan–Ostrover discrete billiards for `Q×T⊂ℝ^4`; uses strong reflection (vertex-exposes) checks per bounce. Tolerances scale with dtype.
- EHZ front-ends and invariants:
  - `capacity_ehz_algorithm1(B,c)`: 2D placeholder returns polygon area; in 4D forwards to `capacity_ehz_algorithm2(V)` after recovering vertices. `src/viterbo/math/capacity_ehz/algorithms.py:12`.
  - `capacity_ehz_algorithm2(V)`: 2D returns polygon area; 4D attempts a cartesian split (`split_lagrangian_product_vertices`) and, if successful, runs the product billiard solver; otherwise falls back to Chaidez–Hutchings oriented-edge spectrum (`oriented_edge_spectrum_4d`) to approximate a minimal cycle. `src/viterbo/math/capacity_ehz/algorithms.py:33`.
  - `capacity_ehz_primal_dual(V,B,c)`: 2D/4D cross-check between primal (vertex billiards) and dual (halfspace placeholder) backends; raises on mismatch; returns the agreed scalar. Acts as a consistency guard. `src/viterbo/math/capacity_ehz/algorithms.py:53`.
  - `minimal_action_cycle(V,B,c)`: front-end returning both capacity and a representative minimal cycle; in 4D only supports exact Lagrangian products (cartesian vertex structure). `src/viterbo/math/capacity_ehz/cycle.py:1`.
- Chaidez–Hutchings oriented-edge algorithm (R^4):
  - Implemented as `oriented_edge_spectrum_4d(V,B,c, *, k_max=None, rotation_cap≈4π, use_cF_budgets=False, cF_constant=None, use_memo=None, memo_grid≈1e−6, memo_buckets=32)` and wrapped with a per-call wall-clock guard `@enforce_time_budget()` defaulting to `VITERBO_SOLVER_TIMEOUT=30s`. `src/viterbo/math/capacity_ehz/stubs.py:232` and `src/viterbo/runtime.py:1`.
  - Pipeline: (1) vertex–facet incidence; (2) enumerate 2-faces with local 2D QR bases; (3) facet Reeb vectors `R_i=(2/λ_i)J n_i`; (4) oriented edges between adjacent faces with affine transfers and linear action forms; (5) depth-first search over cycles with (a) polar-rotation pruning and (b) optional per-face budget pruning `Σ_F c_F ≤ R` using certified `C*(X)`; (6) cycle evaluation by solving a 2D fixed-point for the return map and summing edge actions; (7) return minimal admissible action. Detailed spec in `docs/math/oriented_edge_spectrum_4d.md:1`.
  - Invariants: determinism (CPU float64), translation/rotation/scale invariance by unit-normal normalisation and coordinate-free constructions (see `docs/math/cstar_constant_spec.md:1`).
  - Performance: worst-case exponential in cycle length; rotation caps and `c_F` budgets are key. Optional heuristics (`use_memo=True`) quantise the 2×2 transfer matrix and track coarse budget “dominance” to prune DFS; these are explicitly flagged as heuristic and may lose minimality (non-certified). `src/viterbo/math/capacity_ehz/stubs.py:298`.
  - Fallback/approx policy: when a 4D input is not a product, `capacity_ehz_algorithm2` defers to this CH-based solver for an approximate minimal cycle and returns its action as capacity (documented caveat). `docs/math/capacity_ehz.md:116`.
  - Certified per-face budgets: `compute_cF_constant_certified(B,c,faces)` constructs `C*(X)` deterministically using (i) closed-form bounds for pairs and safe relaxations for triples/quadruples and (ii) an adaptive 1D search with Lipschitz certificates to lower-bound `N_ann(X)`. Returns a small positive constant; never over-prunes by construction. `src/viterbo/math/capacity_ehz/stubs.py:848`, `docs/math/cstar_constant_spec.md:1`.
  - Numerical guards: rank checks, SVD/QR fallbacks, tolerance-aware comparisons; informative `ValueError`s if the face graph is under-populated or transfers degenerate. `src/viterbo/math/capacity_ehz/stubs.py:580`.
- Haim–Kislev facet-multiplier program (variational EHZ):
  - Implemented as `capacity_ehz_haim_kislev(B,c)` and exact in d=2 (area). For d≥4, enumerates facet supports with size between `d+1` and `min(F, d+2)` (extreme-ray bound, allowing one extra for degeneracy), solves nullspaces of `B_S^⊤`, normalises feasible non-negative `β` with `β^⊤ c = 1`, and evaluates the triangular sum `Σ_{i>j} β_i β_j ω(n_i,n_j)` to keep antisymmetry from cancelling. `src/viterbo/math/capacity_ehz/stubs.py:130`.
  - For `dim(nullspace)=1`: candidate rays are ± the unit vector; clamp negatives to 0, reject all-zeros; residual feasibility check in ∞-norm. `src/viterbo/math/capacity_ehz/stubs.py:36`.
  - For `dim(nullspace)=2`: parameterises `β = rows·γ(θ)` with `γ=(cosθ, sinθ)`; builds angular boundaries from row directions; samples each interval on a 32-point grid with feasibility checks; de-duplicates by normalising and allclose. `src/viterbo/math/capacity_ehz/stubs.py:72`.
  - Triangular sum maximisation across permutations is done by DP over subsets in O(2^k k^2), with k≤d+2 (≤6 in 4D); tractable. `src/viterbo/math/capacity_ehz/stubs.py:114`.
  - Returns `c_EHZ = 0.5 / best_value`. Raises on infeasibility or degenerate inputs (odd d, too few facets, non-positive offsets). Tolerances mirror the rest of the code.
  - Complexity drivers: combinations of supports `C(F, k)` with k≤5/6 (4D), each with a small DP and constant-time skew form lookup; practical for small F.
- Product detection in 4D: `split_lagrangian_product_vertices(V, tol)` validates that the 4D vertex set equals the cartesian product of two planar polygons by uniqueness and membership checks; otherwise raises `NotImplementedError` to force CH fallback. `src/viterbo/math/capacity_ehz/common.py:14`.
- Dataset and batching algorithms (for completeness):
  - `RaggedPointsDataset` generates ragged point clouds with random directions; simple Gaussian sampling with reproducible generator. `src/viterbo/datasets/core.py:1`.
  - Collators: `collate_list` keeps ragged lists; `collate_pad` builds zero-padded `(B,K_max,D)` plus masks. Both perform rigorous batch validation (device/dtype/shape) via `_validate_ragged_batch`. `src/viterbo/datasets/core.py:48`.
- C++ extension backends and fallbacks:
  - `_cpp.add_one` and `_cpp.affine_scale_shift` provide trivial elementwise ops via Torch extensions; loaded lazily with `torch.utils.cpp_extension.load`, `USE_NINJA=0` fallback. On any safe import/build error (OSError/RuntimeError/ImportError), return `None` and use pure-Torch fallbacks. `src/viterbo/_cpp/__init__.py:1`.
  - Fallback policy is explicit and deterministic; functional parity maintained (`x+1`, `x*scale+shift`). `src/viterbo/_cpp/__init__.py:33`, `:54`.
- Runtime/time-budget policy: `@enforce_time_budget(seconds=None)` enforces a per-call wall-clock limit via POSIX `setitimer` and `SIGALRM` where available; on platforms without signals, wraps function and throws ex post if elapsed time exceeds the budget. Default budget reads env `VITERBO_SOLVER_TIMEOUT` (30s). Used to guard long-running solvers (e.g., CH DFS). `src/viterbo/runtime.py:1`, `src/viterbo/math/capacity_ehz/stubs.py:232`.
- Stubs present (future work targets): `convex_hull.py` (hull + triangulations), `incidence.py` (incidence/adjacency graphs), `similarity.py` (Hausdorff/support L2), `polar.py` (polarity, Mahler product), and additional volume and capacity backends. These have shapes and invariants documented for early consumers. `src/viterbo/math/*.py` and `docs/math/*.md`.
- Tolerance and determinism discipline:
  - Systematic `tol = max(√eps, 1e-9)` pattern; `torch.unique` used to remove duplicate vertices/coords before ordering; lexicographic ordering on CPU to stabilise V-reps; QR/SVD for bases and projections; clamping of negatives to zero in squared distances and non-negativity enforcement in multipliers. Multiple modules follow this rubric.
- Complexity hotspots and guardrails:
  - `halfspaces_to_vertices` combinatorics; acceptable for D≤4 and modest F.
  - `capacity_ehz_haim_kislev` subset enumeration with DP; controlled by k≤d+2 (≤6 in 4D) and practical F.
  - CH oriented-edge DFS is the main exponential; rotation caps and budgets (optionally memo) are the primary performance controls; certified budgets guarantee completeness when memo is off and time budget permits.
- Approximation/fallback modes summarised:
  - 2D: placeholders return polygon area for capacity and minimal cycle (acts as a smoke test, not a general EHZ formula).
  - 4D product: exact minimal cycle via ≤3-bounce vertex billiards (finite enumeration) — returns both cycle and action.
  - 4D general: fallback to CH oriented-edge spectrum; certified pruning ensures no over-pruning when memo is disabled; otherwise, turning on memo is a heuristic speed-up that may lose optimality.
  - Missing backends (triangulation/Lawrence/MC volumes; QP/LP capacity relaxations) are stubs with clear intended roles.
- Invariants and scaling:
  - Many algorithms are invariant under translation/rotation/scale once inputs are normalised (unit normals, positive offsets, 0∈int). This is explicitly stated for the CH budgets and implicitly relied upon in H-rep transforms. `docs/math/cstar_constant_spec.md:153`.
- Error handling philosophy: raise `ValueError` with precise messages on shape/device/dtype mismatches, degeneracies (rank, norm=0), or infeasibility; prefer early validation (e.g., dataset collators). This makes failure modes explicit for callers.
- Testing hooks and determinism: extension loaders expose `clear_extension_caches()`, random generators accept seeds/generators, and CPU float64 is used for deterministic geometry where needed (e.g., CH and `vertices_to_halfspaces`). `src/viterbo/_cpp/__init__.py:62`.

Actions (first pass)
- Implement `volume_via_triangulation(vertices)` using convex hull + signed simplex accumulation; cross-check with current facet-height method in D=3 and D=4 on small cases.
- Wire a deterministic 4D exact `capacity_ehz_4d_exact` wrapper around `capacity_ehz_haim_kislev` and add a product-vs-variational cross-check similar to `capacity_ehz_primal_dual`.
- Expose a safe, optional `use_memo=True` preset in `capacity_ehz_algorithm2` behind a flag for large 4D inputs; document the heuristic status prominently.
- Add `facet_vertex_incidence(V,B,c)` implementation to replace ad hoc masks in multiple places and to centralise tolerances and ordering.
- Implement `polar_from_halfspaces` for well-conditioned polytopes and a basic `mahler_product_approx` using existing volume backends; restrict to D≤4 initially.
- Consider a compact CPU-only QP solver path for `capacity_ehz_via_qp` in 4D using `torch.linalg` primitives for small F, gated behind a feature flag.

Cross-References
- Topics: T06 (Math used), T08 (Review process)
- Files: `src/viterbo/math/polytope.py:1`, `src/viterbo/math/volume.py:1`, `src/viterbo/math/constructions.py:1`, `src/viterbo/math/symplectic.py:1`, `src/viterbo/math/capacity_ehz/algorithms.py:1`, `src/viterbo/math/capacity_ehz/common.py:1`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:1`, `src/viterbo/math/capacity_ehz/stubs.py:1`, `docs/math/capacity_ehz.md:1`, `docs/math/oriented_edge_spectrum_4d.md:1`, `docs/math/cstar_constant_spec.md:1`

Validation
- `uv run mkdocs build --strict` — OK (locally after adding this review)
- `just checks` — N/A (docs-only change)

Limitations
- Did not run numerical benchmarks; performance/complexity notes are based on code structure and documented algorithms.
- Did not verify proofs; correctness comments are engineering-level (invariants, guards, shape/tol) and references to cited results.
- Focused on implemented modules; stubs listed only by their intended roles.

Status
- Draft

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` under Published reviews — done
- `uv run mkdocs build --strict` green — verified
- `just checks` green (or N/A for docs-only) — N/A
- Actions extracted — yes
- Cross-refs noted — yes
- Provenance filled — yes
- Title matches pattern — yes
