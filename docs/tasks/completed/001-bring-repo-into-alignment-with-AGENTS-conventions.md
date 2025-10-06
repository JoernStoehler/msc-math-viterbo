# Task 001 — Bring repo into alignment with AGENTS conventions

## Task idea (one‑liner)

Align the codebase and configs with AGENTS.md: absolute imports only, Google docstrings, strict
typing, no `__all__`, bounded caches, and CI green under the golden path.

## Acceptance criteria

- CI is green locally and on GitHub (`just ci`): Ruff format/lint (Google docstrings, no relative
  imports), Pyright strict, tests, benchmarks job unaffected.
- No relative imports anywhere in `src/` (aggregators included). Verification:
  `rg -n "^from \\." src` returns no results.
- No `__all__` anywhere in the repo. Verification: `rg -n "^__all__\s*="` returns no results.
- Public functions/classes have Google‑style docstrings (concise, with shape tokens for arrays).
  Internal helpers are encouraged to follow; tiny locals can omit.
- Remove custom array typedefs (e.g., `Vector`) and annotate directly with `jaxtyping` + shape
  tokens from AGENTS vocabulary.
- Caches are explicit and bounded with a hard limit, have a `clear()` API (or context manager), and
  a way to disable in tests; invalidation rules documented.
- Numerical assertions specify tolerances; use the most readable assertion for the case
  (`math.isclose`, `numpy.isclose`, `pytest.approx`, or `numpy.testing.assert_allclose` for arrays).
- Waivers (if any) live in `waivers.toml`, are time‑bounded, and pass the CI checker
  (`scripts/check_waivers.py`).

## Starter notes (longlist, actionable)

Use small PRs. Keep changes scoped; prefer one theme per PR (imports vs. docstrings vs. cache).
Always run `just ci`.

### 1) Absolute imports only (aggregators included)

Refactor relative imports to absolute across the project. Start with these files:

- `src/viterbo/__init__.py` (uses relative imports)
- `src/viterbo/geometry/__init__.py` (relative imports)
- `src/viterbo/symplectic/__init__.py` (relative imports)
- `src/viterbo/examples/__init__.py` (relative imports; removed in 2025-10-05 cleanup)
- (updated) `src/viterbo/symplectic/capacity/__init__.py`
- (updated) `src/viterbo/symplectic/capacity/facet_normals/fast.py`
- (updated) `src/viterbo/symplectic/capacity/facet_normals/reference.py`
- (updated) `src/viterbo/symplectic/capacity/facet_normals/subset_utils.py`
- `src/viterbo/geometry/polytopes/reference.py` (currently uses `.halfspaces`)
- `src/viterbo/geometry/volume/reference.py` (currently uses `.halfspaces`)

Verification:

- `rg -n "^from \\." src` → no matches.
- `rg -n "\bfrom \\..+ import|\bimport \\..+" src` → no matches.

### 2) Remove `__all__` everywhere

Delete all `__all__` declarations (leaf modules and aggregators). Curate the public surface via
explicit imports in `viterbo/__init__.py` only.

- Likely locations:
  - `src/viterbo/geometry/polytopes/reference.py`
  - `src/viterbo/geometry/halfspaces.py`
  - `src/viterbo/symplectic/capacity/facet_normals/fast.py`
  - `src/viterbo/symplectic/capacity/facet_normals/reference.py`
  - `src/viterbo/symplectic/__init__.py`
  - `src/viterbo/__init__.py`

Verification: `rg -n "^__all__\s*="` → no matches.

### 3) Google docstrings for public APIs

Convert remaining modules to Google style (public functions/classes; concise; shape tokens for
arrays). We already aligned some; prioritize the rest:

- `src/viterbo/optimization/solvers.py` (ensure Google style, concise sections)
- `src/viterbo/examples/hello.py` (short Google docstring; removed in 2025-10-05 cleanup)
- `src/viterbo/symplectic/systolic.py` (ensure Google style)
- `src/viterbo/geometry/__init__.py` and `src/viterbo/symplectic/__init__.py` (module docstrings
  optional; keep short)

Ruff enforces Google docstrings; fix D2xx/D4xx quickly.

### 4) Replace custom array typedefs

AGENTS forbids typedefs like `Vector`. Remove and adjust signatures to use `jaxtyping.Float[...]`
directly.

- Find: `rg -n "\bVector\b" src`
- Update `src/viterbo/symplectic/core.py` and any exports in `src/viterbo/__init__.py`.
- Ensure call sites reflect the inline `jaxtyping` types.

### 5) Bounded caches with clear() and test toggles

`src/viterbo/geometry/polytopes/reference.py` defines `_POLYTOPE_CACHE`.

- Add a hard size limit (e.g., LRU; simple deque or OrderedDict is fine).
- Provide `clear_polytope_cache()` and an opt‑out (e.g., env var `VITERBO_DISABLE_CACHE` or function
  arg) for tests.
- Document invalidation rules in the module docstring.

### 6) Shape token normalization

- Use AGENTS vocabulary: `"dimension"`, `"num_facets"`, `"num_vertices"`, etc.
- Fix stray spaces and single‑letter tokens where semantics are clear (e.g., `" d "` →
  `" dimension "`).
- Quick scan: `rg -n "\" d\"|\" m\"|\" n\"| num_facets\"| num_vertices\"| num_polytopes\"" src`

### 7) Numerical assertion hygiene in tests

- Ensure explicit tolerances (default `rtol=1e-9`, `atol=0.0`) and readable assertions.
- Arrays → `numpy.testing.assert_allclose` or `pytest.approx`; scalars →
  `math.isclose`/`numpy.isclose`/`pytest.approx`.
- Keep RNG seeds fixed in tests and benchmarks.

### 8) Aggregator curation (public API)

- In `viterbo/__init__.py`, switch to absolute imports and remove `__all__`.
- Export the curated public surface via explicit imports only.
- Avoid re‑exporting private helpers.

### 9) Docs cleanup (golden path)

- Remove/adjust references to optional dependencies (e.g., cvxpy) if they imply optional backends
  are supported.
- Keep pointers in AGENTS.md to `docs/02-project-roadmap.md`, `docs/13-symplectic-quantities.md`,
  and `docs/convex-polytope-cehz-capacities.md`.

### 10) Waivers (only if needed)

- Add time‑boxed entries to `waivers.toml` for temporary policy exceptions (docstring rollout,
  import refactor staging, etc.).
- CI will fail on expired waivers via `scripts/check_waivers.py`.

## Commands (quick reference)

- Run all checks: `just ci`
- Grep relative imports: `rg -n "^from \\." src`
- Grep **all**: `rg -n "^__all__\s*="`
- Benchmarks (fast paths): `just bench`

---

Owner: maintainer/PI • Priority: High • Milestone: Policy alignment (Week 1–2)
