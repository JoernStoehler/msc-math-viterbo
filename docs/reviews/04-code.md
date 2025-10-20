---
title: Review 04 — Code
---

# Review 04 — Code

Provenance
- Topic: 04 — Code
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 96e7f5a
- Timebox: ~120 minutes
- Scope: src/viterbo, tests/, scripts/, runtime, C++ extension shim; skimmed docs and Justfile for conventions
- Version: v0.1

Context
- Goal: Thorough code review focusing on correctness, readability, typing, error handling, test posture, device/dtype discipline, and adherence to repo conventions. Unfiltered, unsorted list of observations. Redundancy OK.
- Why now: Baseline code quality and risks before deep math/algorithm work; ensure guardrails (typing, tests, docs build) catch regressions.

Inputs & Method
- Commands run (representative):
  - `uv run python scripts/load_skills_metadata.py --quiet`
  - `rg --files -n`
  - `rg -n "def |class |@" src/viterbo`
  - `sed -n '1,200p' <file>` for targeted reads (≤250 lines)
  - `git rev-parse --short HEAD`
  - `uv run mkdocs build --strict` (post‑write)
- Reviewed modules: `src/viterbo/**`, tests under `tests/**`, build/config files (`pyproject.toml`, `pyrightconfig*.json`, `Justfile`, `mkdocs.yml`), skills metadata tooling.

Findings (unsorted)
- `src/viterbo/runtime.py:19` defines `TimeBudgetExceededError` deriving from `RuntimeError`; appropriate exception class for time budget violations.
- `src/viterbo/runtime.py:23` `_default_timeout` respects explicit argument, falls back to env `VITERBO_SOLVER_TIMEOUT`; invalid env values safely default to 30.0 (good resilience, clear env override).
- `src/viterbo/runtime.py:33` `enforce_time_budget` cleanly disables when `timeout <= 0` (returns identity decorator) — nice ergonomics.
- `src/viterbo/runtime.py:51` signal support detection is robust (`setitimer` and `SIGALRM` presence check) to gate the signal path.
- `src/viterbo/runtime.py:55` slow path (no signals) times after execution and raises if exceeded; ensures some guard on platforms lacking itimers; message notes limitation (post‑execution detect) — good clarity.
- `src/viterbo/runtime.py:67` signal path installs a handler, arms `ITIMER_REAL`, restores previous handler and prior timer values; correctly handles nested timers by re‑arming if previous delay > 0.0 — careful state restoration.
- `src/viterbo/runtime.py:69` `_handle_timeout` raises the custom error; handler properly typed with `FrameType | None` and excluded from coverage (signal path) — sensible.
- `src/viterbo/_cpp/__init__.py:13` imports `torch` and `load` from `torch.utils.cpp_extension`; shim is isolated and opt‑in.
- `src/viterbo/_cpp/__init__.py:24` `_load_extension` sets `USE_NINJA=0` to prefer legacy build if Ninja isn’t installed — pragmatic for broader environments.
- `src/viterbo/_cpp/__init__.py:33` wraps extension build in `_SAFE_EXCEPTIONS` (OSError, RuntimeError, ImportError) and returns `None` on failure — allows Python fallback without crashing.
- `src/viterbo/_cpp/__init__.py:41` and `:57` `@lru_cache(maxsize=1)` memoize loaders to avoid rebuild/import overhead; `clear_extension_caches` provided for tests — solid testability hook.
- `src/viterbo/_cpp/__init__.py:45` `add_one` falls back to `x + 1` when ext unavailable; maintains dtype/device semantics since Torch broadcasting preserves both — correct baseline behavior.
- `src/viterbo/_cpp/__init__.py:64` `affine_scale_shift` returns `x * scale + shift` as fallback; float/ints cast to float for ext call — consistent with Torch semantics.
- `tests/test_cpp_extension.py` covers ext presence, correctness vs baseline, and forced fallback with monkeypatch; good smoke posture; no explicit test for non‑contiguous tensors — optional enhancement.
- `src/viterbo/math/polytope.py:20` `support` uses `points @ direction` and `max()` — assumes both are float and shaped `(N,D)` and `(D,)`; tests exercise basic behavior; device/dtype preservation guaranteed by Torch; good.
- `src/viterbo/math/polytope.py:33` `support_argmax` returns `(val, idx:int)` — clean API; uses `torch.max(..., dim=0)` which on 1D returns correct shape.
- `src/viterbo/math/polytope.py:48` `pairwise_squared_distances` implements the standard x^2 + y^2 − 2xy formula; clamps min to 0 in‑place to avoid tiny negative numerical noise — robust.
- `src/viterbo/math/polytope.py:58` `halfspace_violations` computes positive parts `relu(Bx - c)`; shape math is clear; returns `(N,F)`; naming consistent with math.
- `src/viterbo/math/polytope.py:66` `bounding_box` returns per‑dim min/max; works for any `N>=1`; no explicit empty handling — relying on callers/tests to ensure non‑empty input (OK given domain).
- `src/viterbo/math/polytope.py:77` `_ensure_full_dimension` checks 2D tensor, `M > D`, and affine rank equals `D` via `matrix_rank(vertices - mean)`; explicit errors — helpful for early failures.
- `src/viterbo/math/polytope.py:87` `_lexicographic_order` constructs a stable lexicographic order by repeated argsort from last dim to first; complexity O(D N log N) and avoids Python sorting — acceptable for small M.
- `src/viterbo/math/polytope.py:96` `_pairwise_unique` dedups facets via `torch.allclose` on normals and offsets with `atol=tol, rtol=0.0`; linear scan O(F^2) — fine at present scales; consider hashing later.
- `src/viterbo/math/polytope.py:106` `_facet_from_indices` computes candidate facet normal via SVD nullspace of edge diffs, orients outwards using centroid, checks support tightness and cardinality; returns `None` on degeneracy; dead code currently (not called) — planned future use.
- `src/viterbo/math/polytope.py:134` `vertices_to_halfspaces`:
  - Validates full dimension; tolerances as `max(sqrt(eps), 1e-9)` — reasonable numeric guard.
  - 1D special case returns `[+1,-1]` normals with offsets `[max, -min]` — correct.
  - Otherwise uses `scipy.spatial.ConvexHull` on CPU float64; normalises equations so normals are unit length; then dedups; returns tensors cast back to original `device` and `dtype` — correct device/dtype discipline.
  - Potential issue: SciPy dependency ties to CPU and NumPy; acceptable per `pyproject.toml` (`scipy>=1.15.2`) but precludes use‑only‑Torch environments; okay for now given test posture.
- `src/viterbo/math/polytope.py:171` `halfspaces_to_vertices`:
  - Validates shapes, `dim>0`.
  - Iterates all `itertools.combinations(F, D)` to solve intersection points; robust but combinatorial; OK for small F.
  - Uses `torch.linalg.solve` and feasibility check `max(Bv - c) <= tol` and dedups with `allclose`.
  - Errors with “no feasible vertices” — clear.
  - Orders vertices lexicographically (CPU helper) to provide deterministic output — good for tests.
- `src/viterbo/math/polytope.py:218` `facet_vertex_incidence` stub; noted in docs `docs/math/polytope.md` — consistent with roadmap.
- `src/viterbo/math/constructions.py:15` `_make_generator` returns CPU generator seeded from int; if a `torch.Generator` passed, returns as‑is — clear API.
- `src/viterbo/math/constructions.py:23` `_sample_in_unit_ball` draws Gaussian, normalises to directions, and radius via `U^(1/d)` — correct method to sample uniformly in unit ball.
- `src/viterbo/math/constructions.py:33` `rotated_regular_ngon2d` builds unit circle vertices and per‑edge outward normals by normal to edge; orientation correction ensures positive offsets; returns vertices/normals/offsets — consistent and deterministic.
- `src/viterbo/math/constructions.py:71` `matmul_vertices` validates shapes and square matrix; uses `vertices @ A^T` — correct linear map for row‑major vertices.
- `src/viterbo/math/constructions.py:82` `translate_vertices` checks 1D translation, dims match; returns broadcast addition — fine.
- `src/viterbo/math/constructions.py:93` `matmul_halfspace` computes transformed normals using `solve(A^T, B^T)^T` (i.e., B A^{-1}); re‑normalises row normals and scales offsets consistently — correct; raises if zero norm rows occur (singular transform) — good error.
- `src/viterbo/math/constructions.py:114` `translate_halfspace` uses `c' = c + B t`; returns cloned normals to avoid aliasing; consistent with other APIs.
- `src/viterbo/math/constructions.py:128` `lagrangian_product` builds H‑rep block diagonal for P×Q and cartesian vertex set; validates dims/dtype/device match — robust; calls V→H for each factor — OK.
- `src/viterbo/math/constructions.py:160` `random_polytope_algorithm1` and `:187` `algorithm2` provide simple sampling schemes; both canonicalise via V→H and back to remove redundancies — good hygiene.
- `src/viterbo/math/similarity.py` all functions are stubs raising `NotImplementedError` — clearly marked; doc page exists but notes “stubs”; safe.
- `src/viterbo/math/polar.py` stubs with early input validation; comments align to intended math; exceptions are precise with shapes; good placeholder discipline.
- `src/viterbo/math/volume.py:8` imports only `vertices_to_halfspaces`; keeps module narrow.
- `src/viterbo/math/volume.py:12` `_convex_hull_order_2d` unique + centroid angle sort — returns polygon order; assumes convex input points; OK for vertices of convex polygons.
- `src/viterbo/math/volume.py:20` `_project_onto_affine_span` SVD chooses rank by `s>tol` and returns coordinates in basis — reuses later to compute facet measures; consistent numeric tolerance.
- `src/viterbo/math/volume.py:38` `_facet_measure` dedups coords, handles low cardinality and low dimension early (returns 0); calls `volume` recursively — elegant and dimension‑agnostic.
- `src/viterbo/math/volume.py:54` `volume`:
  - Validates shapes/dim>0, picks `tol = max(sqrt(eps), 1e-9)` like elsewhere — consistent.
  - 1D returns length; 2D uses shoelace formula on ordered hull — standard and fast.
  - d≥3: V→H, chooses `centre = mean(vertices)` which lies in convex hull; decomposes into pyramids per facet: `vol += area(facet)*height/d`; absolute at end — standard polytope volume formula; correct if centroid is inside body (true for mean of vertices).
  - Potential performance cost: two conversions (V→H, then mask per facet and recursion) — acceptable for modest sizes.
  - Numerical robustness: tolerances applied to mask facet membership via `isclose`; consistent with rest.
- `src/viterbo/math/volume.py:90` `volume_via_*` stubs left with `NotImplementedError` — clear roadmap.
- `src/viterbo/math/symplectic.py:14` `symplectic_form` constructs standard J = [[0,I],[-I,0]]; validates even dimension; uses default dtype; correct.
- `src/viterbo/math/symplectic.py:24` `random_symplectic_matrix` generates via QR(A) and symmetric shears; composes `upper @ block_a @ lower`; tests use it for invariance; no direct test that `M.T @ J @ M == J` — add a dedicated property test.
- `src/viterbo/math/capacity_ehz/common.py` provides robust helpers:
  - `point_exists` toleranced equality; `split_lagrangian_product_vertices` validates cartesian product structure (4D only) and returns planar factors; clear error messages when not product — great UX for fallback path.
  - `validate_planar_vertices`, `validate_halfspaces_planar` ensure planar constraints and positivity of offsets — consistent with domain assumptions.
  - `order_vertices_ccw` and `polygon_area` are straightforward and reused in tests — DRY.
  - `satisfies_reflection_at_vertex` calls polytope.support to enforce reflection condition; tolerances consistent — good separation of concerns.
- `src/viterbo/math/capacity_ehz/lagrangian_product.py` implements ≤3‑bounce discrete search cleanly:
  - Validates inputs (planar, positive offsets, bounce cap 2 or 3) with clear errors.
  - Consistently uses `tol = max(sqrt(eps), 1e-9)` — numeric policy is uniform across modules.
  - Two‑bounce and three‑bounce helpers isolate reflection checks and lexicographic tie‑break — deterministic minimal action selection; good for tests/reproducibility.
  - Uses `support_argmax` for P‑factor supports and strict checks for Q reflections; linear scans over vertices — OK at small scales.
  - Returns capacity and explicit cycle as `(action, (C×4) points)`; dtype/device preserved back to caller’s tensors.
- `src/viterbo/math/capacity_ehz/cycle.py` top‑level `minimal_action_cycle`:
  - Validates shapes and even dimension; supports 2D (returns area and ordered boundary) and 4D Lagrangian products (auto‑detects product factors with `split_lagrangian_product_vertices`).
  - Converts 4D P‑factor vertices to H‑rep; defers to product search; converts back to original dtype/device — consistent API.
- `src/viterbo/math/capacity_ehz/algorithms.py` EHZ algorithms align with placeholders:
  - `capacity_ehz_algorithm1` uses H‑rep input; in 2D returns polygon area via V‑rep; in 4D delegates to `capacity_ehz_algorithm2` — consistent front‑end shape.
  - `capacity_ehz_algorithm2` uses V‑rep; for 4D tries product detection; on failure, falls back to `stubs.oriented_edge_spectrum_4d` (exp search) — graceful degrade path.
  - `capacity_ehz_primal_dual` cross‑checks V vs H implementations for 2D and selected 4D; raises on mismatch — good invariant enforcement.
- `src/viterbo/math/capacity_ehz/ratios.py:10` `systolic_ratio` validates scalars, positivity, even symplectic dimension; returns `vol / c^n` — correct definition; tests cover 2D/4D normalisations.
- `src/viterbo/math/capacity_ehz/stubs.py` is large and partially implemented, not just stubs:
  - Provides Haim–Kislev facet‑multiplier programme `capacity_ehz_haim_kislev` with careful numeric conditioning, nullspace detection via SVD, candidate support multipliers enumeration; returns `0.5 / best_value` consistent with literature; good numeric hygiene and informative errors on infeasibility.
  - `oriented_edge_spectrum_4d` is implemented with numerous guardrails: time budget via `@enforce_time_budget()`, rotation‑angle pruning, optional Chaidez–Hutchings per‑face budgets, heuristic memoisation (quantised transfer and dominance), and deterministic DFS traversal with canonical cycle dedup; docstring warns about experimental nature — excellent.
  - Backwards‑compat parameter shimming (`verified` → `use_memo`) includes deprecation warning with `stacklevel=2` — thoughtful API stability.
  - Uses typing for local helpers and narrow keys in memo — maintainable.
  - Places CPU fallbacks and numpy conversions only where needed; otherwise Torch‑first.
  - Potential missing coverage: deep tests for `oriented_edge_spectrum_4d` are absent (only smoke falls through via algorithms path) — consider adding targeted unit tests with tiny polytopes to exercise pruning/memoisation toggles deterministically.
- `src/viterbo/datasets/core.py`:
  - `Sample` dataclass typed with `torch.Tensor` fields; frozen for hashability — good.
  - `RaggedPointsDataset` uses CPU `torch.Generator` for reproducibility; devices/dtypes are parameters; returns per‑sample random `points` and `direction` — suitable for smoke tests.
  - `_validate_ragged_batch` centralises checks for collators; returns `(dim, device, dtype, lengths)`; error messages include indices and expected shapes — great DX.
  - `collate_list` returns Python lists preserving ragged shape — keeps semantics simple.
  - `collate_pad` creates zero‑padded batch and boolean mask; supports zero‑length samples (`k==0`) and returns consistent shapes; masks are `torch.bool`; passes tests — solid.
- `src/viterbo/datasets/atlas_tiny.py`:
  - Deterministic tiny atlas generator using `rotated_regular_ngon2d` and V→H; rows carry both V and H reps — ideal for test fixtures and demos.
  - `atlas_tiny_complete_row` attaches `volume`, optional `capacity`/`cycle` (2D only currently), and `systolic_ratio`; validates ndims, device, dtype consistency thoroughly.
  - `atlas_tiny_collate_pad` pads batch across vertices/facets/cycle lengths; returns masks and per‑row scalar tensors; uses `nan` for missing capacities/ratios — clear sentinel values for “not available”.
- `src/viterbo/models/demo.py` `run_probe` handles both ragged list batches and padded batches with or without masks; moves tensors to `device` parameter if given, else infers from batch entries; accumulates scalar `avg_support`; good example and used by smoke test.
- `src/viterbo/__init__.py` intentionally minimal, avoids re‑exports; consistent with ruff config ignoring F401 for init files.
- Typing posture:
  - Project‑wide `pyrightconfig.json` uses basic mode with `reportMissingImports: error`; `pyrightconfig.strict.json` available for strict sweeps — healthy two‑tier setup.
  - Code uses explicit typing for public APIs and internal helpers; keeps `torch.Tensor` annotations and explicit returns; utilities like `ParamSpec`/`TypeVar` in runtime decorator for typing of wrapped callables — thoughtful use.
  - Some modules are stubs and raise literals; acceptable while under construction; consider `typing.NoReturn` returns for pure stubs if desired (optional).
- Linting and style:
  - `pyproject.toml` sets Ruff with focused rule set, ignores for tests and package `__init__` files, Google docstring convention; line length 100; repo largely adheres.
  - Docstrings present for most public functions; stubs include comments marking intent and invariants.
  - Imports are absolute (ban relative configured) — consistent across codebase.
- Device/dtype discipline:
  - Most math functions preserve dtype/device and avoid implicit host/device moves; where CPU dependencies exist (ConvexHull), tensors are cast to CPU float64 and results cast back — explicit and documented.
  - Constructions default to CPU/default dtype for generation; callers often recast — consistent with tests.
  - Dataset collators maintain dtype/device per batch and enforce consistency — excellent.
- Error handling:
  - Functions validate shapes early with precise error messages (naming offending arg and expected shape) — improves diagnosis.
  - Tolerances consistently computed as `max(sqrt(eps), 1e-9)` — coherent numeric policy; appears in polytope, constructions, volume, capacity modules.
  - ValueError used for user/input misuse; NotImplementedError for stubs — correct exception taxonomy.
- Tests overview:
  - Smoke tests cover imports, basic geometry, dataset usage, model probe; good high‑level sanity.
  - Collate tests cover empty batch, dtype mismatch, device mismatch (CUDA‑gated), zero‑length samples — well‑designed inputs.
  - Minimal action/capacity tests validate invariance (translation, symplectic maps), scaling laws, and equality between V and H solver variants; also test cycle covariances (translation, symplectic) — strong mathematical behavior coverage in 2D.
  - Oriented‑edge 4D spectrum is only reached indirectly via algorithms; no direct tests for pruning/memoisation/time budget — area for future tests with small fixtures.
  - C++ extension smoke/benchmark tests gated on availability; benchmark tests skipped if `pytest-benchmark` missing — graceful.
  - `tests/polytopes.py` provides well‑curated standard polytopes and 4D product factories — useful fixtures; used in multiple tests.
- Documentation and MkDocs:
  - `mkdocs.yml` has `strict: true` and includes “Reviews” index; math API pages document stubs appropriately.
  - `docs/reviews/README.md` prescribes adding a bullet per review; cross‑linking policy clear; this review follows it.
- Build/tooling:
  - `Justfile` provides comprehensive workflows (lint, type, tests, docs); `ci` mirrors GH Actions; `ci-cpu` allows Torch CPU wheel install explicitly — well‑thought CI parity.
  - `scripts/load_skills_metadata.py` maintains AGENTS.md sections, validates frontmatter; robust CLI with `--quiet`, `--fix`, `--check`; good repository hygiene.
  - `pyproject` pins a modern stack: Python 3.12, Torch 2.5.1, SciPy/Numpy modern; optional dev deps include pyright, ruff, mkdocs, pytest‑benchmark; aligns with usage.
- Potential improvements (non‑blocking):
  - Add property test ensuring `random_symplectic_matrix(d,seed)` satisfies `M.T @ J @ M == J` within numeric tolerance for multiple seeds/dims.
  - Add tiny deterministic fixtures and direct unit tests for `oriented_edge_spectrum_4d` to lock rotation pruning and memoisation logic across toggles (`use_memo`, budgets, caps).
  - Consider CPU‑fallback path for V→H when SciPy missing (e.g., qhull via `torch` isn’t available; maybe optional pure‑Torch hull for 2D only to reduce dependency surface in lightweight deployments).
  - `halfspaces_to_vertices`: warn or early‑exit for obviously infeasible systems (e.g., check whether convex hull of intersections exists) — may be expensive; optional.
  - `volume`: surface an optional argument to choose backend (`method="pyramids|triangulation|lawrence|monte_carlo"`) and dispatch; current stub functions could be routed via the main function when implemented.
  - `datasets.core.RaggedPointsDataset`: add bounds on `dim>=1` and `num_samples>=0` with clear errors; currently asserted only for min/max points.
  - `models.demo.run_probe`: if `device` is set and batch tensors live on another device, the code moves inputs per sample; consider moving once per batch for efficiency; also support non‑blocking transfer optional param.
  - Add end‑to‑end docstring examples using `doctest`‑style minimal snippets for key APIs (support, V↔H, volume) under docs; helpful for new readers.
  - Typing: consider exporting `typing.Protocol` for “Polygon2D” or “Polytope” if common shapes of tensors are passed around frequently to aid discoverability; optional.
  - Expand error messages with actual shapes when mismatched (some already do; keep it consistent across all modules).
  - Tolerance policy centralisation: a small helper like `eps_tol(tensor) -> float` to unify `sqrt(eps)` policy and make it easier to modify globally.
  - Consider `torch.no_grad()` wrappers for pure geometry routines that never need autograd, to avoid inadvertent graph creation in user code.
  - For algorithms exploring combinatorial states (lagrangian_product search), optional `progress`/`callback` hook could help debugging in long runs.
  - Add `pytest.mark.slow` or time‑budgeted variants for any future heavy 4D algorithms to keep smoke tier fast.
  - Add test for `datasets.core._validate_ragged_batch` error messages to remain stable (golden messages) given they are user‑facing.
  - In `atlas_tiny_collate_pad`, consider returning integer `counts` per row for vertices/facets/cycle to avoid recomputing masks downstream; optional convenience.
  - In `_cpp` shim: expose a function to report build errors or logs when extension fails for debugging; currently silent fallback — good for UX, but debugging sometimes needs more detail (make it opt‑in or env‑gated).
  - In `capacity_ehz.algorithms.capacity_ehz_primal_dual`, tolerances are fixed at `1e-8`; consider tying to dtype (`sqrt(eps)`) like elsewhere for consistency.
  - `symplectic.random_symplectic_matrix`: add docstring note on distribution (not Haar on Sp(2n)); label as “random construction for tests”, not general purpose — reduces misuse risk.
  - `volume._convex_hull_order_2d`: if given non‑vertex interior points, ordering still returns a simple polygon possibly self‑intersecting; consider first eliminating non‑extreme points (convex hull) for correctness in worst cases (tests use true vertices so fine today).
  - `polytope.halfspaces_to_vertices`: when `F` is large vs `D`, combinations may be expensive; consider early facet grouping or linear programming feasibility check to prune candidate combinations.
  - Add `dtype`/`device` explicitness to docstrings consistently (“Torch‑first; outputs preserve dtype/device”); most already have it — maintain consistency.
  - Add a simple benchmark in `tests/performance` for V↔H conversions and volume in 3D/4D to catch egregious slowdowns.
  - Review `pyproject` `numpy>=2.3.3` compatibility with SciPy pin for the CI matrix; currently aligned, but document minimum versions in README for contributors.
  - Consider exposing a public “tolerances” module to avoid duplication of `max(sqrt(eps), 1e-9)` across files.

Actions (first pass)
- Add property test that `random_symplectic_matrix` preserves `J` for several seeds/dims.
- Add a tiny deterministic 4D product case and one non‑product case to directly test `oriented_edge_spectrum_4d` with `use_memo` True/False and finite `rotation_cap`.
- Factor a `numeric_tol(t: torch.Tensor) -> float` helper and use in polytope/constructions/volume/capacity for uniformity.
- Consider a 2D pure‑Torch convex hull fallback to reduce SciPy dependency for simple cases.
- Add device move optimisation in `models.demo.run_probe` to move entire batch once when padded.

Cross-References
- Topics: T03 (Repo and code architecture), T06 (Math used), T07 (Algorithms used)
- Files: `src/viterbo/math/polytope.py:134`, `src/viterbo/math/volume.py:54`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:32`, `src/viterbo/math/capacity_ehz/stubs.py:241`, `src/viterbo/runtime.py:33`, `pyproject.toml:1`, `mkdocs.yml:18`, `docs/math/polytope.md:1`

Validation
- `uv run mkdocs build --strict` — OK (post‑write)
- `uv run python scripts/load_skills_metadata.py --quiet` — OK
- Tests not executed in this pass to keep scope docs‑only (smoke tier previously green in CI flow); recommend `just precommit` before merging code changes.

Limitations
- Did not run full `just ci` or full pytest tiers; review emphasised static inspection and targeted reads; performance characteristics inferred from patterns, not benchmarked.
- Did not deep‑review numerical stability for extreme degeneracies; current tolerance policy likely sufficient at present scales.
- 4D oriented‑edge algorithm correctness not proven here; treated as experimental with strong guardrails; needs dedicated tests.

Status
- Draft

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` under Published reviews — Yes
- `uv run mkdocs build --strict` green — Yes
- `just checks` green — N/A (docs‑only changes in this review)
- Actions extracted — Yes
- Cross-refs noted — Yes
- Provenance filled — Yes
- Title matches — Yes

