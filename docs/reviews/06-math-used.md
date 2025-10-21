Status: Implemented (scope: math usage review snapshot; caveats: reflects repository state on 2025-10-20)

# Review 06 — Math Used (Foundations, Invariants, Cutoffs, Warnings)

Provenance
- Topic: 06 — Math used
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 4dab5e0
- Timebox: ~90–120 minutes
- Scope: Mathematical foundations referenced/implemented; invariants; algorithmic cutoffs and their citations; correctness warnings; tests that lock behavior; only documentation changes
- Version: v0.1

Context
- Thoroughly review the mathematical layer (`src/viterbo/math/*`) and associated docs/tests to surface conventions, invariants, proofs/citations for algorithmic cutoffs, and any correctness warnings. Summarize as an unsorted, long list for later consolidation.

Inputs & Method
- Repo survey with `rg` across `docs/math/`, `src/viterbo/math/`, and `tests/math/`.
- Read shaping docs: `docs/math/index.md`, `docs/math/capacity_ehz.md`, `docs/math/oriented_edge_spectrum_4d.md`, `docs/math/cstar_constant_spec.md`.
- Spot‑read implementations: polytope geometry, constructions, volume, symplectic helpers, EHZ algorithms/solvers.
- Confirm build and doc integrity: `uv run mkdocs build --strict`.

Findings (unsorted)
- Torch‑first contract everywhere in math: functions return tensors, preserve caller `dtype`/`device`, and avoid implicit device transfers. Documented in docs and consistently implemented in code and tests (docs/math/index.md:1, src/viterbo/math/*, tests set default float64).
- Float64 default in tests and docs to improve numerical stability; tolerances scale like `tol = max(√ε, 1e-9)` across modules. Codified many times, e.g., `src/viterbo/math/polytope.py:93`, `src/viterbo/math/volume.py:51`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:31`, `src/viterbo/math/capacity_ehz/stubs.py:183`.
- Even‑dimension policy for symplectic operations: ambient `d = 2n` enforced. Errors raised otherwise. See `docs/math/index.md:1`, `docs/math/symplectic.md:1`, `src/viterbo/math/symplectic.py:15`, `src/viterbo/math/capacity_ehz/algorithms.py:33`.
- Polytope support function semantics: `h_T(v) = max⟨p, v⟩` over vertices; maximizer attained at a vertex for polytopes. Implemented by `support`/`support_argmax` and used by EHZ solvers (src/viterbo/math/polytope.py:18, src/viterbo/math/capacity_ehz/lagrangian_product.py:99).
- Halfspace representation and conversions adhere to outward normals with positive offsets for bounded polytopes; origin in interior conventions are assumed by capacity solvers. See `src/viterbo/math/polytope.py:81` (flip orientation using centroid), `src/viterbo/math/capacity_ehz/stubs.py:210` (offset positivity checks), docs `docs/math/capacity_ehz.md:1`.
- Vertices→halfspaces uses SciPy ConvexHull under the hood, moved to CPU float64, and normalizes equations to unit normals; it deduplicates near‑duplicate facets via `torch.allclose` under tolerance to maintain deterministic counts (src/viterbo/math/polytope.py:104–139).
- Halfspaces→vertices enumerates all `D`‑tuples of facets, solves linear systems for intersections, filters infeasible points with `Bx ≤ c`, deduplicates, and returns vertices in lexicographic order for determinism (src/viterbo/math/polytope.py:141–187).
- Deterministic ordering and lexicographic semantics for vertices to stabilize downstream algorithms and comparisons. Used broadly (src/viterbo/math/polytope.py:62, src/viterbo/math/volume.py:12).
- Pairwise distance helper defines squared Euclidean distances, clamping to zero from below to mitigate small negative round‑off (src/viterbo/math/polytope.py:40–59).
- Halfspace violation helper returns `relu(Bx − c)`; used to assess constraint slack; sign conventions clear in docstring (src/viterbo/math/polytope.py:61–77).
- Basic constructions cover deterministic rotated regular `n`‑gons in 2D with consistent outward normals and offsets; outwardness enforced by flipping if offset < 0. Verified by tests that offsets equal `cos(π/k)` for regular polygons (src/viterbo/math/constructions.py:34–78, tests/math/test_polytope.py:31–49).
- Linear transforms on vertices vs H‑reps are kept symmetric: `matmul_vertices` vs `matmul_halfspace` (which solves for transformed normals and renormalizes) and `translate_vertices` vs `translate_halfspace` (offsets become `c' = c + B t`). Tests validate equivalence by round‑trips (src/viterbo/math/constructions.py:80–145, tests/math/test_polytope_representation.py:1).
- Lagrangian product builder returns `Q × T` vertices and block‑diagonal H‑rep, preserving dtype/device; used to build 4D products for EHZ solvers and tests (src/viterbo/math/constructions.py:147–203).
- Random polytope generators sample directions in the unit ball with reproducible `torch.Generator` inputs; they return both V and cleaned H representations (src/viterbo/math/constructions.py:206–259).
- Volume in 1D, 2D, ≥3D follows classical formulas: segment length; shoelace/convex‑hull sweep in 2D; for `D≥3` a facet‑height accumulation from centroid with recursive facet measure calculation, invoking `vertices_to_halfspaces` and projecting facets to affine spans. Stability guards for degeneracy (src/viterbo/math/volume.py:1–90).
- Volume invariants tested: translation invariance, scaling homogeneity `vol(sK)=s^d vol(K)`, and known shape values for cubes/hypercubes (tests/math/test_minimal_action_invariants.py:18–64, tests/math/test_polytope.py:17–48).
- Symplectic form `J = [[0, I], [−I, 0]]` returns exact structure; validated (tests/math/test_symplectic.py:6–14, src/viterbo/math/symplectic.py:13–27).
- Random symplectic matrix generator uses QR for `A`, symmetric shear blocks, and block factors; property `M^T J M = J` asserted with tolerances (src/viterbo/math/symplectic.py:29–55, tests/math/test_symplectic.py:16–24).
- EHZ capacity solvers expose three front ends: `capacity_ehz_algorithm1(B,c)`, `capacity_ehz_algorithm2(V)`, and `capacity_ehz_primal_dual(V,B,c)` (src/viterbo/math/capacity_ehz/algorithms.py:1–74).
- 2D EHZ capacity equals polygon area (placeholder standing in for the exact formula in 2D). Both H and V pipelines reduce to area via `polygon_area` (src/viterbo/math/capacity_ehz/algorithms.py:22–35, 58–72; docs/math/capacity_ehz.md:1).
- 4D handling in `algorithm2`: if the vertices form a Lagrangian product (cartesian vertex grid), split factors `(Q,P)` and run the Minkowski billiard search specialized to products (≤3 bounces). Otherwise, fall back to Chaidez–Hutchings oriented‑edge spectrum (src/viterbo/math/capacity_ehz/algorithms.py:36–57).
- Primal–dual consistency checks: in 2D and 4D (where both paths apply), `capacity_ehz_primal_dual` raises if `algorithm1` and `algorithm2` disagree beyond tight tolerances (src/viterbo/math/capacity_ehz/algorithms.py:74–105). This is a guardrail for correctness.
- Minimal action cycle front‑end returns the boundary polygon in 2D (ordered CCW) with area as capacity; in 4D product case defers to the specialized billiard solver and returns the actual cycle in R^4 (src/viterbo/math/capacity_ehz/cycle.py:1–41).
- Lagrangian product Minkowski billiards solver implements ≤3 bounce enumeration following Rudolf’s finite‑bounce results for planar polygons: enumerate 2‑ and 3‑bounce candidates on `q` with support points on `p`, enforce vertex reflection rules, and take minimal action with deterministic tie‑breakers (src/viterbo/math/capacity_ehz/lagrangian_product.py:1–163; docs/math/capacity_ehz.md:63–110).
- Vertex reflection rule (strong form) is directly encoded: at bounce `q_i`, `q_i` maximizes `⟨x, p_next − p_prev⟩`. Implemented checks in two‑bounce and three‑bounce routines mirror the discrete billiard condition (src/viterbo/math/capacity_ehz/lagrangian_product.py:117–156; docs/math/capacity_ehz.md:37–48).
- Determinism: in case of equal actions, tie‑break via lexicographic comparison of the flattened cycle coordinates to pick a canonical representative (src/viterbo/math/capacity_ehz/lagrangian_product.py:44–62, 89–115, 134–163).
- Product split detection in 4D is rigorous: exact cartesian vertex structure with membership checks and tolerance; raises NotImplementedError if the 4D polytope is not a Lagrangian product, triggering the oriented‑edge fallback (src/viterbo/math/capacity_ehz/common.py:13–53; src/viterbo/math/capacity_ehz/algorithms.py:45–57).
- Chaidez–Hutchings oriented‑edge algorithm (R^4) is implemented with documented differences and warnings: the solver enumerates 2‑faces, builds oriented edges with affine transfers and action coefficients, then DFS searches cycles with rotation‑number pruning and optional per‑face budgets (src/viterbo/math/capacity_ehz/stubs.py:248–859; docs/math/oriented_edge_spectrum_4d.md:1–210).
- Face enumeration: 2‑faces are recognized via vertex–facet incidence and rank‑2 tests in a local basis; failures raise descriptive errors (src/viterbo/math/capacity_ehz/stubs.py:566–612).
- Reeb vectors: for facet `i`, `R_i = (2/λ_i) J n_i` with `λ_i>0` and standard `J`; used to form edge transfers and hitting‑time formulas (src/viterbo/math/capacity_ehz/stubs.py:320–329; docs/math/oriented_edge_spectrum_4d.md:12–20).
- First‑hit domain admissibility is enforced per edge by sample checks inside the 2‑face against competitor facets; edges without a valid open domain are discarded (src/viterbo/math/capacity_ehz/stubs.py:736–762).
- Per‑edge time‑of‑flight is affine in local coordinates: `t(u) = a0 + a1·u` with coefficients derived from target facet data; action accumulation uses linear forms with domain checks (src/viterbo/math/capacity_ehz/stubs.py:786–824; docs/math/oriented_edge_spectrum_4d.md:22–38).
- Rotation‑number pruning: computes the polar rotation angle of accumulated 2×2 transfers and prunes paths exceeding a cap (default `4π`), per CH. This is an algorithmic cutoff with explicit parameterization (src/viterbo/math/capacity_ehz/stubs.py:329–355, 394–414; docs/math/oriented_edge_spectrum_4d.md:38–48).
- Per‑face budgets (CH Theorem 1.12(v)) implemented via a certified constant `C*(X)` times the face angle `θ(F) = arccos⟨ν_i, ν_j⟩`. Budgets prune by ensuring the sum used ≤ rotation‑number cap; correctness is certified by construction (src/viterbo/math/capacity_ehz/stubs.py:355–373; docs/math/cstar_constant_spec.md:1–225; docs/math/oriented_edge_spectrum_4d.md:38–64).
- `C*(X)` builder `compute_cF_constant_certified` follows the spec: computes conservative lower bounds for `N_ann(X)` via adaptive Lipschitz‑certified search; bounds `D_min(X)` and `U_max(X)` via closed‑form/relaxations on active sets; returns `(N_ann * D_min) / (2π * U_max)`; deterministic CPU float64 (src/viterbo/math/capacity_ehz/stubs.py:881–1136; docs/math/cstar_constant_spec.md:1–225).
- Invariance of `C*(X)`: translation/rotation/scale invariance due to unit normals, offset normalization, and S^3 distances; spec documents proofs and implementation underscores the invariance (docs/math/cstar_constant_spec.md:148–195; docs/math/oriented_edge_spectrum_4d.md:66–88).
- Heuristic memoization is explicitly flagged as non‑certified and requires budgets; enabling it may accelerate but can be non‑minimal. Guardrails enforce `use_cF_budgets=True` when memo is on (src/viterbo/math/capacity_ehz/stubs.py:371–393).
- Time budgets: oriented‑edge solver is wrapped with `@enforce_time_budget()` (default 30s wall clock, platform‑aware); tests and docs specify usage; additional per‑process CPU cap via `sitecustomize.py` (~180s) (src/viterbo/math/capacity_ehz/stubs.py:239, src/viterbo/runtime.py:1–74, sitecustomize.py:14–60).
- Haim–Kislev facet‑multiplier program implemented for general polytopes (≥4D cases supported by enumeration bounds). Enumerates supports up to size `d+2` (exact bound `≤ d+1` for extreme rays; code permits `d+2` defensively), solves nullspaces, normalizes multipliers `β` with `β^T c = 1`, and maximizes triangular sums over permutations of `W = B J B^T` to get `c_EHZ = 1/(2·max)`. Tolerances and feasibility checks are explicit (src/viterbo/math/capacity_ehz/stubs.py:160–239; docs/math/capacity_ehz.md:111–176).
- 4D exactness of HK approach is justified by extreme‑support size ≤5 after equality constraints; docs provide the proof sketch and algorithm outline (docs/math/capacity_ehz.md:138–176).
- Oriented‑edge and Haim–Kislev are kept independent: tests ensure oriented‑edge does not call HK; algorithm2 is allowed to fall back to oriented‑edge for non‑product 4D polytopes (tests/math/test_oriented_edge_guard_no_hk.py:1–26; tests/math/test_capacity_oriented_edge.py:24–59).
- Equivalence in product cases: oriented‑edge matches Minkowski product solver on square×square; tested and asserted (tests/math/test_capacity_oriented_edge.py:7–23).
- Global invariants validated by tests: capacity is invariant under symplectic maps (via `random_symplectic_matrix`), translation invariance for both H and V forms, and quadratic scaling under uniform scaling in 2D (tests/math/test_minimal_action_invariants.py:66–123).
- Minimal action cycle front‑end returns cycles that translate covariantly and transform symplectically covariantly (cycle maps by `M`); tested (tests/math/test_minimal_action_invariants.py:125–160).
- Minkowski billiards three‑bounce stricter candidate sometimes beats two‑bounce; both diagnosed and deterministic; a real‑world non‑regular test case exhibits this (tests/math/test_minkowski_billiard.py:1–57).
- Pentagon×rotated‑pentagon product constant reproduces known capacity formula `2 cos(π/10) (1+cos(π/5))`; both 2‑ and 3‑bounce routines agree and return 4D cycles of expected structure (tests/math/test_minkowski_billiard.py:9–30; tests/polytopes.py:260–301).
- 2D/4D smoke invariants: EHZ capacity equals polygon area in 2D; product cases route consistently across algorithms; primal–dual validates (tests/math/test_minimal_action.py:1–57; tests/math/test_capacity_ehz_haim_kislev.py:1–53).
- `systolic_ratio(vol, c_ehz, 2n)` implements Viterbo’s systolic ratio `vol(K) / c^n` with explicit positivity and dimension checks; reference constants tested (tests/math/test_minimal_action.py:59–80; src/viterbo/math/capacity_ehz/ratios.py:1–24).
- Geometry helpers and volume backends document planned algorithms: triangulation, Lawrence’s sign decomposition, Monte‑Carlo via quasi‑MC — all stubbed and referenced for future correctness checks (docs/math/volume.md:1–44; src/viterbo/math/volume.py:92–122).
- Polar/Mahler links stubbed: polarity from H‑rep and Mahler product approximation are placeholders for future bridges; shapes and invariants predeclared (docs/math/polar.md:1–26).
- Similarity metrics stubs (Hausdorff, symplectic‑Hausdorff, support L2) planned to reuse support queries and deterministic sampling; shape contracts documented (docs/math/similarity.md:1–23).
- Determinism posture: many operations explicitly use CPU float64 internally (e.g., ConvexHull conversions, C* builder, edge enumeration) and lexicographic/quantized memoization to keep results reproducible across runs and hardware (src/viterbo/math/polytope.py:118–139; src/viterbo/math/capacity_ehz/stubs.py:902–913, 403–414).
- Numerical guards: epsilons/hooks — `sqrt(eps)` tolerances, feasibility tolerances for nullspaces `feas_tol = 10·tol`, regularization in fixed‑point solves for cycles, domain checks with multi‑point sampling per 2‑face; ValueErrors raised with clear messages when preconditions fail (src/viterbo/math/capacity_ehz/stubs.py:562–613, 814–859, 862–880, 1007–1015).
- Safety warnings documented clearly in oriented‑edge page: differences to CH code (no undocumented optimizations), budgets are certified lower bounds (can under‑prune but never over‑prune), memoization is optional and non‑certified, and timeouts bound runtime (docs/math/oriented_edge_spectrum_4d.md:66–108).
- Scaling/translation invariance of oriented‑edge and HK paths is tested explicitly; translation is handled by recomputing H‑rep post‑translation to keep offsets consistent (tests/math/test_capacity_oriented_edge.py:61–87; tests/math/test_capacity_ehz_haim_kislev.py:1–53).
- C++ extensions exist but are not part of math‑layer logic reviewed here; Python fallbacks used; performance tests wrap C++ stubs but do not affect correctness of math invariants (src/viterbo/_cpp/*, tests/performance/*).
- Cross‑module architecture boundary respected: math layer is “pure” (no I/O, no global state), datasets/models build on it; tests enforce shape/dtype/device behavior and numerics (docs/math/index.md:1; skills/good-code-loop.md; tests across math/*).
- Repository explicitly ties 4D oriented‑edge pruning to CH Theorem 1.12(v), Lemma 5.13, and §6.2; c* spec reproduces the denominator/annest bound derivation and quaternionic invariance conventions for R^4 (docs/math/cstar_constant_spec.md:1–225).
- 4D HK program cites extreme‑support size arguments and equivalence to variational formulation, with references to Haim–Kislev; oriented‑edge product equivalence justified via Artstein‑Avidan–Ostrover and Rudolf’s finite‑bounce bound (docs/math/capacity_ehz.md:1–210; refs inline).
- Tests include seeded random and non‑regular polygons to ensure algorithms are robust away from symmetric cases; these probe degeneracy and ensure tolerance usage is effective (tests/polytopes.py:172–201, 302–389).
- Error contracts: informative messages for odd ambient dimension, insufficient facets (`d+1` minimal), non‑positive offsets (unbounded), degenerate faces, or failure to locate feasible multipliers. These surface correctness problems early and explicitly (src/viterbo/math/capacity_ehz/stubs.py:186–239, 566–612, 1016–1040; tests assert some of these).
- Planar area routines order vertices CCW to guarantee positive area; combined with `abs()` ensures sign stability (src/viterbo/math/capacity_ehz/common.py:40–53, 55–68; src/viterbo/math/volume.py:21–37).
- Cycle reconstruction for product billiards stitches R^4 polylines that are closed and verified numerically at endpoints; smoke tests confirm closure and dimensions (tests/math/test_minkowski_billiard.py:31–57).
- Lexicographic normal forms help produce deterministic equality in tests that compare sets (vertices/facets) after transforms; e.g., sorting rows by dimensions to check equality up to ordering (tests/math/test_polytope_representation.py:1–35).
- Systolic references and counterexample constants appear in tests/papers: e.g., Ostrover, Haim–Kislev & Ostrover counterexample (docs/papers/2024-ostrover-counterexample-viterbo/main.tex; tests use the derived numeric ratio formula in 4D).
- Docs clearly separate implemented vs planned parts, calling out placeholders and intended citations for future backends (e.g., Lawrence 1996/2000 style sign decomposition; quasi‑MC convergence expectations) (docs/math/volume.md:1–44; docs/math/similarity.md; docs/math/polar.md).

Actions (first pass)
- Add a short “Assumptions and invariants” block at the top of each math doc summarizing tolerance policy, device/dtype, and even‑dimension rule to reduce repetition and improve discoverability.
- Cross‑link `docs/math/capacity_ehz.md` directly to `docs/math/oriented_edge_spectrum_4d.md` and `docs/math/cstar_constant_spec.md` sections that define rotation pruning and budgets; add anchors for quick navigation.
- Expand `docs/math/volume.md` with a brief error/degeneracy subsection (facet rank checks, duplicate vertices) aligning with code guards in `volume.py`.
- Add a “References” footer to `docs/math/polytope.md` with citations for support functions and V/H conversions (e.g., Ziegler; Preparata–Shamos for 2D hull ordering) to back the documented semantics.
- Minor doc tweak: note that `capacity_ehz_haim_kislev` currently enumerates up to `d+2` facets for safety but proofs bound extreme support by `d+1`; add a TODO to tighten once degenerate‐case guard is proven unnecessary.
- Add a quick start example in `docs/math/symplectic.md` that composes `random_symplectic_matrix` with a small 2D polygon to demonstrate symplectic invariance of 2D capacity numerically (bridges theory with practice).

Cross-References
- docs/math/index.md:1
- docs/math/polytope.md:1
- docs/math/volume.md:1
- docs/math/symplectic.md:1
- docs/math/capacity_ehz.md:1
- docs/math/oriented_edge_spectrum_4d.md:1
- docs/math/cstar_constant_spec.md:1
- src/viterbo/math/polytope.py:1
- src/viterbo/math/constructions.py:1
- src/viterbo/math/volume.py:1
- src/viterbo/math/symplectic.py:1
- src/viterbo/math/capacity_ehz/algorithms.py:1
- src/viterbo/math/capacity_ehz/cycle.py:1
- src/viterbo/math/capacity_ehz/common.py:1
- src/viterbo/math/capacity_ehz/lagrangian_product.py:1
- src/viterbo/math/capacity_ehz/stubs.py:160
- src/viterbo/runtime.py:1
- tests/math/test_minimal_action_invariants.py:1
- tests/math/test_minkowski_billiard.py:1
- tests/math/test_capacity_oriented_edge.py:1
- tests/math/test_oriented_edge_guard_no_hk.py:1
- tests/math/test_capacity_ehz_haim_kislev.py:1
- tests/math/test_polytope.py:1
- tests/math/test_polytope_representation.py:1
- tests/math/test_symplectic.py:1

Validation
- Loaded skills metadata: `uv run python scripts/load_skills_metadata.py --quiet` — OK
- Built docs strictly: `uv run mkdocs build --strict` — OK (pre‑change baseline); will rerun after publish.

Limitations
- Did not attempt to prove new theorems; relied on internal docs and literature citations included in repository.
- Performance characteristics and bench data not analyzed; focus was on mathematical correctness/invariants and algorithmic cutoffs.
- Did not run the full test suite here; task scope is documentation. The tests referenced give strong coverage signals for invariants.

Status
- Finalized
