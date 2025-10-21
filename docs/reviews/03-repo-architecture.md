Status: Implemented (scope: repo architecture review snapshot; caveats: reflects repository state on 2025-10-20)

# Review 03 — Repo and Code Architecture

Provenance
- Topic: 03 — Repo and code architecture
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 585a547
- Timebox: ~90–120 minutes
- Scope: Full repository with emphasis on layering/boundaries (math → datasets → models), numerics and purity policies, and enforcement via tests/types/docs
- Version: v0.1

Context
- Goal: Assess whether the repository’s architecture, layering, invariants, and numerics policies are coherent, explicit, and enforced. Focus on the math layer purity, dataset/model boundaries, device/dtype discipline, 4D capacity stack (EHZ + CH oriented‑edge), and C++ extension integration.

Inputs & Method
- Read code under `src/viterbo/**`, tests under `tests/**`, and docs under `docs/**` (including oriented‑edge and capacity specs).
- Ran: `uv run python scripts/load_skills_metadata.py --quiet` (skills surfaced OK).
- Will validate docs: `uv run mkdocs build --strict` (recorded in Validation).
- Searched with `rg` for structure, imports, stubs, and invariants.

Findings (unsorted)
- Layering is explicit and respected across the tree: `src/viterbo/math/**` is pure and self‑contained, `src/viterbo/datasets/**` provides ragged collation and typed rows, and `src/viterbo/models/**` consumes math via simple probes; no upward imports detected.
- Math module surfaces are intentionally non‑re‑exported; `src/viterbo/math/__init__.py:1` documents “no re‑export indirection,” encouraging explicit imports and stable paths.
- Device/dtype policy is Torch‑first and consistently applied: math functions accept caller tensors, return tensors, and preserve dtype/device; explicit conversions to CPU/float64 occur only for SciPy/QHull calls with outputs cast back to the original dtype/device. See `src/viterbo/math/polytope.py:87` (ConvexHull CPU/double path) and return casts at the end of `vertices_to_halfspaces`.
- Tolerances scale with `sqrt(eps)` and an absolute floor `1e-9` across modules, improving robustness in float32/float64. Examples: `polytope.vertices_to_halfspaces` tol calculation; `volume.py:31` tol selection; `capacity_ehz.*` modules compute `tol = max(sqrt(eps), 1e-9)` on working dtype/device.
- Invariants are encoded via argument validation with precise error messages. Examples: `constructions.matmul_vertices` enforces square matrix and dimension match; `translate_halfspace` validates shapes and returns a cloned normals tensor; `symplectic.symplectic_form` enforces even dimension `2n`.
- Math purity is consistently upheld: no I/O, no global state, no dataclasses in math; functions either return tensors or raise `NotImplementedError` stubs (e.g., `similarity.py`, `convex_hull.py`, and some `volume` backends) rather than silently failing.
- The polytope representation conversions are carefully implemented with determinism: `halfspaces_to_vertices` enumerates facet combinations, solves candidates with `torch.linalg.solve`, filters infeasible/duplicate vertices with tolerance checks, and imposes a lexicographic row order on CPU for stable results (`_lexicographic_order`).
- `vertices_to_halfspaces` normalises plane equations to unit normals and deduplicates facets via pairwise allclose checks; returns are cast to original device/dtype, preserving the Torch‑first contract (`src/viterbo/math/polytope.py:118` ff.).
- Geometry queries include support and support_argmax returning both value and Python index (`support_argmax` returns an `int` via `.item()`), pairwise squared distances with non‑negative clamp, and halfspace violation ReLU — these are well‑tested in `tests/math/test_polytope_smoke.py` with dtype/device checks.
- Volume implementation is dimension‑aware: 1D (length), 2D (shoelace after convex‑hull ordering), ≥3D (facet‑height decomposition using halfspaces + recursive facet measure). See `src/viterbo/math/volume.py:52` ff. The approach is Torch‑native and avoids external dependencies except where conversions are already present.
- `volume._facet_measure` projects facet vertices to their affine span via SVD and recurses to `volume` on reduced coordinates; it removes duplicates to reduce degeneracy issues — a nice stability touch.
- EHZ capacity stack is split by concern: `algorithms` (entry points and 2D/4D dispatch), `cycle` (front‑end minimal‑action helpers), `lagrangian_product` (planar K×T ≤3‑bounce solver), `stubs` (Haim–Kislev programme, oriented‑edge CH solver, and future QP/LP paths), and `common` (shared helpers such as CCW ordering, polygon area, and product splitting checks).
- 2D placeholders are explicit: `capacity_ehz_algorithm1` and `capacity_ehz_algorithm2` currently compute area in 2D (polygon), which is correct for invariance tests and baseline behaviour; they raise NotImplementedError outside supported dimensions.
- 4D path in `algorithms.py:22` tries a Lagrangian product split (`common.split_lagrangian_product_vertices`) and falls back to the general `oriented_edge_spectrum_4d` solver if the product structure check (cartesian vertex grid) fails — a clean separation with a deterministic fast path.
- `common.split_lagrangian_product_vertices` validates cartesian product structure via uniqueness checks and exhaustive membership verification; the error message clearly sets scope limits: current 4D support limited to Lagrangian products or general oriented‑edge fallback.
- The product solver `lagrangian_product.py` is carefully deterministic: CCW ordering on q‑vertices, two‑ and three‑bounce enumeration, explicit strong reflection checks via `support` exposure, tie‑breaking by lexicographic order of the flattened cycle. This gives a certifiable minimal cycle under its finite‑bounce assumptions.
- `stubs.capacity_ehz_haim_kislev` is not a stub — it implements the extreme‑support facet‑multiplier programme in arbitrary even dimensions with enumeration up to `d+2` and a dynamic programming `O(2^k k^2)` triangular sum for permutations, returning `0.5/best`. It enforces feasibility (`β≥0`, `βᵀ B = 0`, `βᵀ c = 1`) and rejects degenerate sets.
- Oriented‑edge CH solver is substantial and well‑documented in code and docs: it builds 2‑faces from vertex–facet incidence, enumerates oriented edges with affine transfer/action, performs DFS with rotation‑number pruning and optional per‑face budgets C*(X), and evaluates cycles via a least‑squares fixed‑point. It carries memoization options with warnings when heuristics are enabled (`use_memo=True` emits `RuntimeWarning`). See `src/viterbo/math/capacity_ehz/stubs.py:540` ff. and `docs/math/oriented_edge_spectrum_4d.md`.
- The CH solver is wrapped with a wall‑clock guard via `@enforce_time_budget()` (`src/viterbo/math/capacity_ehz/stubs.py:232`) and a default env‑configurable timeout (`src/viterbo/runtime.py:28`) which is a strong operational safeguard for long searches.
- Certified budget builder `compute_cF_constant_certified` computes C*(X) per docs (`docs/math/cstar_constant_spec.md`) using CPU float64, orthonormal bases on faces, Lipschitz‑certified lower bounds, and device‑stable invariants — this matches the docs’ invariance and correctness notes.
- Symplectic utilities are clean and test‑covered: `symplectic_form(d)` and `random_symplectic_matrix(d, seed)` (QR factorisation + symmetric shears); tests verify `M.T @ J @ M = J` within tolerance (`tests/math/test_symplectic.py`).
- Construction helpers include exact transforms on both V‑ and H‑representations (`matmul_vertices`, `translate_vertices`, `matmul_halfspace`, `translate_halfspace`) and Lagrangian product assembly (`lagrangian_product`); tests validate block structure and feasibility.
- Random polytope generators are deterministic via explicit `torch.Generator` seeds and return both V/H reps; tests ensure feasibility and round‑trip properties.
- Datasets layer is minimal and disciplined: `RaggedPointsDataset` generates ragged point clouds and directions; `_validate_ragged_batch` enforces a single device/dtype per batch and dimension consistency; collators return either lists of tensors or padded tensors with boolean masks. Errors are specific and actionable.
- The typed dataset row for AtlasTiny uses `TypedDict` to describe schema and a completion step that attaches derived invariants using math utilities (`capacity_ehz.cycle`, `ratios.systolic_ratio`, `volume`). Collate builds padded batch tensors and masks while preserving dtype/device and attaching scalar metrics as `(B,)` tensors.
- The models layer `models/demo.py` is intentionally tiny and device‑agnostic: it computes a simple support statistic over different batch shapes (ragged list or padded), moving only inputs as needed without hidden device transfers.
- C++ extension scaffold under `src/viterbo/_cpp` follows best practices: lazy loaders with per‑op fallbacks, Ninja disabled by default for portability (`USE_NINJA=0`), explicit capability checks (`has_*_extension`) and clearance of caches for testing; smoke/benchmark tests exercise both paths.
- Architecture conventions are documented in `docs/architecture/overview.md` and reinforced by `AGENTS.md` and the skills (`skills/good-code-loop.md`, `skills/math-layer.md`, `skills/testing-and-ci.md`), which align with the code observed: no re‑exports, math purity, device semantics, and C++ fallbacks.
- Documentation coverage is very good: each math submodule has a dedicated page (`docs/math/*.md`) with shapes, invariants, and status; the CH solver and C* spec are unusually detailed and match code behaviour.
- Tests enforce invariants systematically: translation/scaling/symplectic invariance for capacities and volumes, primal–dual consistency, 4D product vs oriented‑edge equality, oriented‑edge invariances under translation/scale, and that oriented‑edge does not call HK internally (guarding architecture boundaries between backends).
- Type hints are pervasive (including `TypedDict`, `NamedTuple`, `ParamSpec` for decorators). Pyright basic config exists (`pyrightconfig.json` and strict variant), and Ruff is configured with correctness‑focused rules. Tests rely on Torch dtype default set to float64 to avoid fragility, and tolerances are consistent.
- Incremental selection and performance tiers are available (benchmarks guarded by optional `pytest-benchmark`, smoke mark for quick runs). This matches the Good Code Loop guidance and supports fast PR feedback.
- Error messages are consistently specific and user‑oriented (what shape was expected vs received), which helps users debug without reading implementation details.
- Determinism: many routines rely on `torch.unique`, lexicographic sorts on CPU, and seeded RNGs; oriented‑edge’s memo path is explicitly flagged as heuristic with warnings; DFS memo uses quantised transfer matrices and coarse budget buckets to make pruning deterministic across runs.
- Potential portability wart: `polytope.vertices_to_halfspaces` requires SciPy/QHull at runtime; dependency is declared in `pyproject.toml` and tests exercise it. The fallback path is not provided (it’s fundamental), but CPU/double conversion is explicit and outputs are cast back — acceptable for this scope.
- Numerical safety around near‑degenerate configurations is handled via rank checks, SVD/QR fallbacks, duplicate/feasibility filters, and minimum positive clamps (e.g., in `compute_cF_constant_certified`, `max(..., 1e-12)`; and in oriented‑edge action/time checks).
- Naming and structure closely mirror the docs taxonomy, making it easy to map code ↔ documentation ↔ tests. The oriented‑edge doc includes verification checklists and small examples that echo the implementation design.
- CI posture (from docs and Justfile) prefers CPU‑only Torch wheels and strict docs build; this aligns with repository emphasis on reproducibility and low friction for contributors.
- Stubs are clearly marked and covered by skipped tests that document the intended future validation (e.g., volume backends), which prevents silent regressions and communicates roadmap.
- Runtime guard `enforce_time_budget` handles platforms without SIGALRM by measuring wall time post‑call and raising if exceeded; on Unix it uses `setitimer`/`SIGALRM` with careful restoration of previous handlers/timers — robust and testable behaviour (`src/viterbo/runtime.py`).
- Public API avoids re‑exports (`__init__.py` files are minimal), preventing accidental surface growth and import cycles; import paths are explicit in tests and docs.
- Code comments and docstrings strike a good balance: focus on semantics, invariants, and shapes; implementation rationale is in docs pages where appropriate.
- Tests for datasets and models verify that collate behaviours and demo probes are device/dtype‑respecting and produce scalar metrics; masks are validated for correctness and zero‑padding semantics.
- The repository uses `uv` with a lockfile (`uv.lock`), pinning dependencies sufficiently for deterministic local/devcontainer builds; MkDocs strict build is part of the routine gates.
- The `Justfile` (examined separately) encodes a “golden path” for checks/CI with CPU Torch indexes; skills reference these commands, enabling consistent flows across agents.
- Repository docs emphasise that math remains pure and GPU is optional/experiment‑only (“GPU confined to models”), a choice that simplifies CI and avoids spurious device transfer bugs.
- The 4D stack’s separation (product vs general CH) provides a graceful path for breadth: fast exact product solver plus a general but budget‑guarded CH implementation, with correctness and invariance tests covering both.
- Cross‑file coherence: e.g., `docs/math/capacity_ehz.md` states the oriented‑edge fallback and the exactness of HK in 4D; tests assert equality among `algorithm1/2/primal_dual/HK` on canonical shapes.
- The `skills` system is used not just for meta‑docs but also to validate AGENTS.md sections via `scripts/load_skills_metadata.py`; this is wired into `just lint`, making skills part of the repo’s “testable” surface.
- `tests/polytopes.py` provides a canonical catalogue of shapes (with references/metadata) centralising test inputs; this reduces duplication and keeps test semantics clean.

Actions (first pass)
- Add a brief note in `docs/math/polytope.md` clarifying the SciPy/QHull CPU/double conversion in `vertices_to_halfspaces`, to set expectations on device/dtype round‑trip and dependency.
- Consider adding a short “Math Purity & Numerics” section in `docs/architecture/overview.md` linking to the math docs and highlighting the `sqrt(eps)` tolerance policy and CPU/double conversions when needed.
- Add a smoke test for `datasets/core.collate_pad` that exercises non‑zero padding and ensures zero‑padded entries do not affect support queries when masked (pattern exists but an explicit negative‑case check could catch regressions).
- Evaluate enabling `pyrightconfig.strict.json` selectively for `src/viterbo/math/**` in CI, while keeping the rest at basic; this would raise the floor on type safety where it matters most.
- Add a short “Determinism & Seeds” subsection to `docs/math/index.md` clarifying RNG/construction conventions (`torch.Generator`, seeds, CPU orderings).
- Consider a tiny example in `docs/math/volume.md` explaining the facet‑height accumulation using a simple 3D polytope with an illustration; purely documentation, no code changes.

Cross-References
- Topics: T01 (Project design principles), T08 (Review process), T05 (Developer docs), T06 (Math used), T07 (Algorithms used)
- Files: `src/viterbo/math/polytope.py:87`, `src/viterbo/math/volume.py:52`, `src/viterbo/math/capacity_ehz/algorithms.py:22`, `src/viterbo/math/capacity_ehz/lagrangian_product.py:15`, `src/viterbo/math/capacity_ehz/stubs.py:232`, `src/viterbo/runtime.py:28`, `docs/math/oriented_edge_spectrum_4d.md:1`, `docs/math/capacity_ehz.md:1`, `docs/architecture/overview.md:1`

Validation
- `uv run python scripts/load_skills_metadata.py --quiet` — OK
- `uv run mkdocs build --strict` — OK (see pre-submit once validated)

Limitations
- Did not run the full test suite or benchmarks due to scope; focused on code/doc inspection and invariants visible in tests.
- Did not step through the oriented‑edge solver in a debugger; relied on code reading and tests/docs coherence.
- C++ extension paths validated via code/test reading, not by compiling in this run.

Status
- Draft

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` under Published reviews — yes
- `uv run mkdocs build --strict` green — pending, will validate below
- `just checks` green — N/A for docs‑only change
- Actions extracted — yes (5 items)
- Cross‑refs noted — yes
- Provenance filled — yes
- Title matches pattern — yes
