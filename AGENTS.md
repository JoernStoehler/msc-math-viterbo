# AGENTS.md

Purpose (fact): This repository uses a **single AGENTS.md** for **all tasks**.  
Authority (fact): **This file is the source of truth** for conventions and workflows. If any other doc contradicts this file, follow **AGENTS.md**.

## 0) Roles & scope (facts)

- Maintainer (PI):
  - Spawns/manages Codex agents; writes task briefs; merges PRs.
  - Owns environment/DevOps (devcontainer, CI, perf infra) and makes research/architecture decisions tied to the thesis.
  - Approves policy waivers and larger directional changes.
- Codex agents (ephemeral, per-task):
  - Implement focused changes (feature/fix/refactor/docs/tests/benchmarks) within the golden path.
  - Open PRs, respond to review, and iterate until CI is green. Agents do not merge PRs.
  - May be invoked for reviews via `@codex review` and provide inline suggestions.
- Scope & decision policy:
  - Agents avoid reconfiguring the environment or making architectural/research decisions without an explicit brief.
  - When in doubt, escalate instead of guessing.
- Escalation triggers (choose one channel: PR description, `Needs-Unblock: <topic>`, or issue):
  - Ambiguous or missing acceptance criteria; unclear invariants.
  - Environment/DevOps changes; policy conflicts; need for a waiver.
  - Research or architecture choices that affect more than the current task.
  - Performance regressions beyond accepted thresholds; inability to reproduce CI locally.
- Lifecycle & context:
  - Agents run in fresh, ephemeral containers (Codex Cloud) with this AGENTS.md and the task brief. The maintainer merges or closes PRs.
  - The project targets a 6‑month thesis submission; prioritize reproducibility, small PRs, and deterministic results. See the roadmap docs for details.

### Task briefs (one-liner checklist)

Every task brief should include: scope, acceptance criteria, links to context (files/docs), constraints (perf/interfaces), expected tests/benchmarks, and escalation triggers.

## 1) Facts: Conventions the repo follows

- **Language & runtime**: Python **3.12+**.
- **Package layout**: `src/viterbo/` (library code), `tests/` (unit & perf), `docs/` (overview + references), `.devcontainer/` (container & lifecycle), `.github/` (CI), `tmp/` (ignored scratch).
- **Dependency manager**: **uv** with `uv sync` (lockfile‑driven). Commit `uv.lock`. Use `uv run`/`uv sync` instead of raw `pip`.
- **Formatting & lint**: **Ruff** (format + lint). Target line length **100**. No unused imports; no wildcard imports; no reformatting suppression except where strictly necessary.
- **Type checking**: **Pyright** in **strict** mode (treat warnings as errors). Keep both `src/` and `tests/` type-clean.
- **Type checking policy**: Strict with zero silent waivers. Inline suppressions require a one-line justification and a TODO to remove.
- **Docs**: Google docstring style (fact). All public functions/classes carry Google-style docstrings. Include shape tokens from the vocabulary for all array args/returns. Prefer Google docstrings for internal helpers as well; tiny local helpers or throwaway closures can omit. Examples only when they add clarity.
- **Arrays & shapes**: **jaxtyping** for explicit shapes/dtypes. **No custom array typedefs** (no `Vector`, `FloatMatrix`, etc.). Prefer semantic shape names (`"num_facets"`, `"dimension"`, `"num_polytopes"`).
- **Dtypes**: Default to **float64** for numeric stability unless a function clearly documents another dtype.
- **Functional core**: Math code is **pure** (no I/O, no hidden state). Side-effects live in thin adapters (imperative shell).
- **Errors**: Fail fast with precise exceptions (`ValueError`, `TypeError`). Do not silently coerce incompatible shapes/dtypes.
- **Logging**: Use `logging` (module loggers). No `print` in library code. No secrets in logs.
- **Determinism**: Tests are deterministic. If randomness is unavoidable: seed explicitly and assert invariants, not exact bit-patterns.
- **Numerical testing**: Use explicit tolerances (default `rtol=1e-9`, `atol=0.0` unless a function states otherwise). Choose the most readable assertion for the case: `math.isclose`, `numpy.isclose`, `pytest.approx`, or `numpy.testing.assert_allclose` for arrays.
- **Environments**: Single golden‑path environment (plus Codex Cloud devcontainer). Required deps include NumPy and SciPy; avoid optional dependency branches.
- **Imports**: Absolute imports everywhere (no relative imports), including within package submodules and aggregators. No circular imports; refactor to break cycles.
- **Performance policy**: Micro-optimizations only after correctness. Bench only for code paths tagged performance-critical.
- **Security**: No secrets in code or logs; config via env vars; avoid echoing env or using `set -x` where secrets may appear.
- **Branching**: `feat/<scope>`, `fix/<scope>`, `refactor/<scope>`. Small, scoped changes.
- **Commits**: Conventional Commits style (e.g., `feat: add EHZ estimator for polytopes`).
- **Releases**: None planned (MSc thesis). Tag milestones only.

**Follow these conventions throughout all tasks.**

## 2) Shape vocabulary (facts)

Use the following **shape symbols** consistently in type annotations and docstrings:

- `"dimension"` — ambient Euclidean dimension (often `2n` for `R^{2n}`).
- `"num_facets"` — number of facets of a polytope.
- `"num_vertices"` — number of vertices.
- `"num_polytopes"` — batch count across multiple polytopes.
- `"num_samples"` — sample count (generic data).
- `"k"` / `"m"` / `"n"` — generic axes where semantics are not domain-critical.

If two parameters must share a dimension, **reuse the same symbol** in annotations.

## 3) Code style & typing (facts + one concise example)

- Prefer small, composable, **pure** functions with explicit types.
- Arrays: use `jaxtyping.Float[np.ndarray, "<shape>"]` (or `Int[...]` etc.).
- Return scalars as Python `float`/`int` only when the meaning is unambiguous and documented.
- Document units and coordinate frames when relevant.

#### Minimal example (Google docstring + jaxtyping)

```python
import numpy as np
from jaxtyping import Float

def ehz_capacity(
    facets: Float[np.ndarray, "num_facets dimension"],
    normals: Float[np.ndarray, "num_facets dimension"],
) -> float:
    """Estimate EHZ capacity for a convex polytope.

    Args:
      facets: Facet vertex data, shape (num_facets, dimension). Units: coordinates.
      normals: Outward facet normals, shape (num_facets, dimension). Must align with `facets`.

    Returns:
      Scalar capacity estimate.

    Raises:
      ValueError: If shapes are inconsistent or dimension < 2.
      TypeError: If arrays are not floating point.
    """
    if facets.ndim != 2 or normals.ndim != 2:
        raise ValueError("facets/normals must be 2D arrays: (num_facets, dimension)")
    if facets.shape != normals.shape:
        raise ValueError("facets and normals must have identical shapes")
    if facets.shape[1] < 2:
        raise ValueError("dimension must be >= 2")

    facets = facets.astype(np.float64, copy=False)
    normals = normals.astype(np.float64, copy=False)

    # placeholder structure for demonstration:
    # ... compute support numbers, actions, and minimal closed characteristic ...
    capacity = float(np.maximum(0.0, np.mean(np.einsum("fd,fd->f", facets, normals))))
    return capacity
```

## 4) Workflows (imperative, concise)

### 4.1 Setup (once per environment)

1. Use the devcontainer.
2. Run:

   * `bash .devcontainer/post-create.sh` (one-time)
   * `bash .devcontainer/post-start.sh` (each boot)
3. Install deps: `make setup` (uses `uv sync` with a lockfile)

### 4.2 Daily development

1. Read the task and scan relevant modules and tests.
2. Plan the **minimal** change (one feature OR one fix OR one refactor).
3. Implement small, pure functions in `src/viterbo/`. Keep I/O at the edges.
4. Add or adjust tests next to the code (deterministic, minimal fixtures).
5. Run locally: use the commands in Quick reference. `make ci` mirrors CI.

### 4.3 Performance-sensitive changes

1. Only if a change touches a marked fast path.
2. Run:

   * `pytest tests/performance -q --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks`
3. Compare autosaved vs. current and record the delta.
4. If regression > 10%, iterate or document a waiver and open a follow-up issue.

### 4.4 Pre-PR checks

* Keep diffs focused (≈ ≤300 LOC when practical).
* Ensure types, tests, and docs are updated.
* Ensure `make ci` is **green locally**.

### 4.5 Pull request (concise content)

* State **scope**, **files touched**, **what you read**, **what you changed**, **how you tested** (paste summaries of Ruff/Pyright/pytest), and **perf delta** if applicable.
* Brief **limitations** and **follow-ups**.
* Keep the PR small; split if needed.

### 4.6 When blocked

* If progress stalls after a focused attempt (≈ 60–90 minutes) due to missing invariants, unclear specs, or environment issues, open a **draft PR** titled `Needs-Unblock: <topic>` listing blockers and a proposed fallback.

## 5) Testing (facts + short rules)

* Organize by feature/module; prefer small, explicit fixtures.
* No hidden I/O in tests; temporary files clean up via fixtures.
* Numerical: use explicit tolerances (default `rtol=1e-9`, `atol=0.0`). Choose the most readable assertion: `math.isclose`, `numpy.isclose`, `pytest.approx`, or `numpy.testing.assert_allclose` for arrays.
* Property-based tests are welcome where invariants are cleanly expressible (e.g., monotonicity, symmetry).
* Avoid brittle tests tied to incidental internal representations.

## 6) Performance (facts)

* Benchmarks live in `tests/performance/` and **reuse the same fixtures** as correctness tests.
* Autosave results under `.benchmarks/` for comparisons in PRs.
* Perf hygiene for fast paths:
  - Run `make bench` (autosave enabled) and include a short delta summary in the PR (compare against latest artifact or baseline branch).
  - Keep RNG seeds fixed and note any environment constraints that influence results.
  - If regression > 10% and not justified, add a time‑boxed waiver entry in `waivers.toml` while investigating.

## 7) Numeric stability (facts)

* Prefer operations with predictable conditioning; avoid subtractive cancellation when a stable algebraic form exists.
* Prefer `@` and `einsum` with explicit indices over ambiguous `dot`.
* Normalize or rescale inputs when it improves stability; document such preconditions.
* For tolerance negotiation, bias toward **slightly stricter** thresholds first; relax with justification if necessary.

## 8) Error handling & validation (facts)

* Validate **shape**, **dtype**, and **domain** constraints at function boundaries; fail early with clear messages.
* Do not catch and silence exceptions in library code; allow callers to observe errors.
* Use `NotImplementedError` only for intentionally incomplete optional paths; avoid placeholders elsewhere.

## 9) I/O boundaries & state (facts)

* Library functions are pure; any filesystem, network, or device interaction occurs in thin wrapper modules.
* No global mutable state. If caching is necessary: keep it explicit and bounded (hard size limit), key by explicit invariants, provide a `clear()` API (or context manager) and a way to disable in tests. Document invalidation rules.

## 10) Imports & structure (facts)

* Absolute imports only across `src/viterbo/` (no relative imports), including aggregators.
* No `__all__` anywhere. Curate the public API by explicit imports in `viterbo/__init__.py`; avoid wildcard re‑exports within this project.
* Wildcard imports within this project are disallowed. Wildcard imports from well‑known third‑party packages are discouraged but permitted with an inline justification comment when they materially improve ergonomics.
* Internal helpers live in private modules (leading underscore) and are not imported into public namespaces.
* Keep module size modest; split by cohesive concerns.

## 11) Security & privacy (facts)

* Credentials/config via environment variables only.
* Never print or log secrets.
* Do not upload private data to third-party services in CI or benchmarks.

## 12) CI & Branch protection (facts)

* `make ci` mirrors GitHub Actions (format check → lint → strict typecheck → tests).
* Branch protection requires all checks to pass before merge.
* Concurrency cancels in-progress runs per ref to save CI time.
* Perf-critical changes: include a benchmark delta summary or a documented waiver.
* CI fails on expired policy waivers using `scripts/check_waivers.py` and `waivers.toml`.

## 13) Quick reference: common commands (exact text)

```bash
# install (dev)
make setup

# format + lint + typecheck + unit tests (fast loop)
make format && make lint && make typecheck && make test

# full local mirror of CI
make ci

# performance (when touching fast paths)
pytest tests/performance -q --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks
```

## 14) What NOT to do (hard nos)

* Do **not** introduce custom array aliases (`Vector`, `FloatMatrix`, …). Use jaxtyping with explicit shapes.
* Do **not** merge PRs without local `make ci` passing.
* Do **not** add dependencies without necessity and a clearly documented rationale. Prefer a single golden path over optional backends.
* Do **not** hide I/O or mutation inside math helpers.
* Do **not** weaken types/tests to “make CI green”; fix the root cause or raise blockers.
* Do **not** use relative imports or `__all__` anywhere.

## 15) Scope of this file (fact)

* This AGENTS.md applies to **all tasks** and **all agents**. It is intentionally concise, declarative for conventions, and imperative only for workflows. Maintain consistency with these facts unless this file is updated.

## 16) Environment assurance (facts)

* Maintainers ensure tool configs are correct and pre-baked into the devcontainer, `pyproject.toml`, CI, and `Makefile`.
* New contributors should rely on the provided commands (`make setup`, `make format`, `make lint`, `make typecheck`, `make test`, `make ci`) without re‑validating tool configuration.
* If you notice a mismatch (tools disagree or the golden path breaks), do not hand‑tune your local setup. Open an issue or a draft PR (`Needs-Unblock: <topic>`) describing the mismatch; maintainers will fix the environment.
* Avoid bespoke local tweaks. The project values a single **golden path** that keeps everyone fast and aligned.

## 17) Policy waivers (facts)

* Deviations from this policy are tracked centrally in `waivers.toml` (repo root).
* Each waiver must include: `id`, `summary`, `owner`, `scope`, `created`, `expires` (YYYY‑MM‑DD), `justification`, and `removal_plan`.
* Waivers are time‑bounded and should be minimized; prefer fixing root causes quickly.

## 18) Policy enforcement (maintainers)

* Enforcement lives in repo configs; contributors need not re‑validate.
* Mapping (non-exhaustive):
  * Google docstrings → Ruff pydocstyle (convention=google).
  * Absolute imports / no relatives → Ruff tidy-imports (ban-relative-imports=all).
  * Strict typing → Pyright strict; CI treats diagnostics as errors.
  * Format/lint → Ruff format + lint gates in CI.
  * Waiver expiry → scripts/check_waivers.py validates `waivers.toml`.

### Pointers to context (optional reading)

* Roadmap: `docs/02-project-roadmap.md`
* Symplectic quantities overview: `docs/13-symplectic-quantities.md`
* Capacity algorithms: `docs/convex-polytope-cehz-capacities.md`
