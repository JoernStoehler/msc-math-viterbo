# AGENTS.md

Single authoritative policy. This file distils the important takeaways from configs and reference 
documentation, so agents can start their work quickly. If another doc conflicts, follow AGENTS.md.

## 0) Roles & Scope (facts)

- Maintainer (PI)
  - Approves task briefs (agents draft and iterate with the PI).
  - Merges PRs; owns environment/DevOps and research/architecture decisions.
  - Approves policy waivers and larger directional changes.
- Codex agents (ephemeral)
  - Implement focused changes (feature/fix/refactor/docs/tests/benchmarks) on the golden path.
  - Draft task briefs when useful and escalate uncertainties early.
  - Open PRs and iterate until CI is green. The PI merges PRs.
- Escalation triggers (choose one channel: PR description, `Needs-Unblock: <topic>`, or issue)
  - Ambiguous acceptance criteria, unclear invariants, or competing interpretations.
  - Environment/DevOps changes; policy conflicts; need for a waiver.
  - Cross‑task research/architecture choices.
  - Performance regressions beyond thresholds; inability to reproduce CI locally.

## 1) Sources of Truth & Reference Materials (facts)

- AGENTS.md: rules, workflows, basic context.
- Configuration files:
  - Dependencies: `pyproject.toml`, `uv.lock` (pinned).
  - Formatting, Linting: `pyproject.toml` (Ruff format + lint). Lint focuses on correctness and
    policy, not cosmetic style. Auto‑fixable styling (e.g., import sorting) is allowed; non‑fixable
    whitespace/empty‑line nags are disabled.
  - Typing: `pyrightconfig.json` (strict; diagnostics are errors).
  - Pytest defaults: `pytest.ini` (markers, session timeout, smoke filter).
  - CI: `.github/workflows/ci.yml`.
  - Environment: `.devcontainer/` (`devcontainer.json`, `post-create.sh`, `post-start.sh`).
- Golden-path task runner: `Justfile` recipes.
- Policy Waivers: `waivers.toml`.
- Reference docs: `docs/` (math, architecture, decisions).
- Task briefs: `docs/tasks/` (drafts, scheduled, completed, dependencies and priorities).
- Source code: `src/viterbo/` (library), `tests/viterbo/` (unit/integration tests),
  `tests/performance/viterbo/` (benchmarks).
- Thesis: `thesis/` (LaTeX source).
- Weekly progress reports: `progress-reports/` (drafts, sent mails).

## 2) Environment & Tooling

- Provisioned environment: Agents start in a ready‑to‑use devcontainer with deps installed and
  x64 JAX enabled (`JAX_ENABLE_X64=1`). No manual setup required.
- Dependency manager: uv. Use `uv run`, `uv sync`, and `uv add`. Commit `uv.lock`.
- Verifying locally (only if replicating outside the provisioned env):
  - One‑time: `bash .devcontainer/post-create.sh`
  - Each boot: `bash .devcontainer/post-start.sh`
  - Install: `just setup`

## 3) Coding Conventions (facts)

- Docstrings: Google style for public APIs; prefer Google for non‑trivial internals.
- JAX‑first: library code uses `jax.numpy`; return JAX arrays from public APIs.
- Arrays & shapes: jaxtyping with explicit shapes/dtypes.
  - Import: `from jaxtyping import Array, Float`; annotate as `Float[Array, " <shape>"]`.
  - Use a leading space in the shape string to avoid erroneous Ruff F821 warnings/errors.
  - Use shape literals, not constants. No custom typedefs (`Vector`, `FloatMatrix`, …).
  - Examples of shape tokens: semantic `" num_facets dimension"`, algebraic `" B k n dimension"`.
- Dtypes: default to float64; document and justify deviations; never downcast silently.
- Purity: math code is pure (no I/O, no hidden state). Keep side‑effects in thin adapters.
- Validation philosophy: prefer type/shape clarity and tests over redundant runtime assertions.
  - Use `jaxtyping` + `beartype`/`jaxtyped` in tests for shape/name validation.
  - Add explicit checks only for domain constraints that tests can’t reasonably cover.
- Logging: use `logging` (module loggers). Logs may be elided under JIT; avoid logging patterns
  that cause tracing errors; prefer logging in non‑jitted paths.
- Imports & structure: absolute imports; no wildcard imports; no `__all__`. Keep modules modest;
  split by cohesive concerns. Curate public API in `viterbo/__init__.py` via explicit imports.
- Security: credentials/config via env vars; never print or log secrets.
- Branching & commits: `feat/<scope>`, `fix/<scope>`, `refactor/<scope>`; Conventional Commits.
- API policy (v0.x): breaking changes allowed; update tests/docs in the same PR.

## 4) JAX‑first specifics (facts)

- Two variants when performance matters:
  - Reference: readable, trusted, JAX‑first; Python control flow allowed; no JIT requirement.
  - Fast: performance-optimized, jit‑able; prefer `jax.jit`, `vmap`, `lax` instead of Python loops on hot paths.
- Precision: x64 is mandatory; do not downcast silently.
- RNG: prefer JAX PRNG keys (`jax.random.PRNGKey`) or integer seeds; split keys locally; avoid
  hidden global state.
- Plotting/IO: convert to NumPy (`np.asarray`) at call sites (examples/tests), not within library
  code.
- Interop boundary: SciPy/NumPy calls live only under `viterbo/_wrapped/` (e.g., spatial Qhull,
  scipy.optimize.linprog, byte hashing). Library code stays JAX‑first and should not import SciPy.

## 5) Minimal example (Google docstring + jaxtyping)

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def ehz_capacity(
    facets: Float[Array, " num_facets dimension"],
    normals: Float[Array, " num_facets dimension"],
) -> float:
    """Estimate EHZ capacity for a convex polytope.

    Args:
      facets: Facet data, shape (num_facets, dimension). Units: coordinates.
      normals: Outward facet normals, shape (num_facets, dimension). Must align with `facets`.

    Returns:
      Scalar capacity estimate.
    """
    facets = jnp.asarray(facets, dtype=jnp.float64)
    normals = jnp.asarray(normals, dtype=jnp.float64)
    capacity = jnp.maximum(0.0, jnp.mean(jnp.einsum("fd,fd->f", facets, normals)))
    return float(capacity)
```

## 6) Testing (facts)

- Structure: organize by feature/module; explicit fixtures; no hidden I/O; clean up temp files.
- Tolerances: choose context‑appropriate tolerances; document rationale. Typical float64 ranges are
  `rtol` ~ 1e‑9–1e‑12 and `atol` near 0.0 for well‑conditioned problems, but adjust as needed.
- Assertions: use clear options such as `pytest.approx`, `numpy.testing.assert_allclose`,
  `numpy.isclose`, or `math.isclose` depending on the case.
- Justfile toggles for pytest-driven targets:
  - Testmon caching is enabled by default. Set `USE_TESTMON=0` to disable for a run.
  - `PYTEST_ARGS="..."` forwards additional selectors/markers (e.g., `PYTEST_ARGS="-k smoke"`).
- Property‑based tests: welcome when invariants are cleanly expressible (e.g., monotonicity,
  symmetry). Prefer Hypothesis.
- Shape/name validation in tests: enable `jaxtyping` + `beartype`/`jaxtyped` during tests when
  valuable; avoid cluttering library code with repetitive checks.
- Pytest tiering: layer `smoke`, `deep`, `longhaul` markers on top of `benchmark`,
  `line_profile`, `slow`. `just test` runs smoke with a 10 s per-test timeout, a hard 60 s session
  cap, `--maxfail=1`, and a slow-test summary (`--durations=15`). `just test-deep` runs the deep
  tier; `just test-longhaul` is manual/scheduled. See
  `docs/testing-decision-criteria.md` for decision guidance and re-tiering criteria.
- Invariant baselines live under `tests/_baselines/` as JSON; update values only with
  maintainer sign-off and record the rationale in the PR/task brief.

## 7) Performance (facts)

- Benchmarks live in `tests/performance/` and reuse correctness fixtures. Keep RNG seeds fixed.
- Run `just bench`; include a brief delta in PRs vs baseline/artifact.
- Bench tiers: `just bench` (smoke/CI), `just bench-deep` (pre-merge),
  `just bench-longhaul` (scheduled); archive longhaul runs in reports or
  task briefs.
- Scheduled CI runs `just test-longhaul` and `just bench-longhaul` weekly; longhaul failures block
  merges until resolved or waived.
- Profiling: `just profile` / `just profile-line` wrap `uv run` and write to
  `.profiles/`; notebooks are out of scope.
- If regression > 10% without justification, add a time‑boxed waiver to `waivers.toml` and open a
  follow‑up issue.

## 8) Workflows (imperative)

Daily development (in provisioned env)

1. Read the task and scan relevant modules/tests.
2. Plan the minimal change (one feature OR one fix OR one refactor). Write a short plan (≈4–7 steps).
3. Implement pure functions in `src/viterbo/`; keep I/O at the edges. Add/adjust tests.
4. Run: `just precommit-fast` during quick loops (wraps `lint-fast` + `test-incremental`); finish with
   `just precommit` (alias for `just precommit-slow`) before handing off or requesting review. Use
   `just help` for per-target tips, toggles, and related workflows (testmon caching is on by default;
   set `USE_TESTMON=0` to disable, `PYTEST_ARGS="..."` forwards selectors/markers).
   - Lint tiers: `just lint-fast` runs Ruff E/F/B essentials (ignores jaxtyping F722); `just lint`
     mirrors CI (Ruff policy set + Prettier).
   - Typechecking tiers: `just typecheck-fast` targets `src/viterbo`; `just typecheck` covers the
     entire repo.

Short loops may use direct commands (`uv run pytest path/to/test.py`, custom markers, etc.). Close
each handoff by re-running the golden-path targets above.

Pre‑PR checks

- Keep diffs focused and coherent. Ensure types, tests, and docs are updated.
- Ensure `just ci` is green locally (includes smoke tests with coverage). If the golden path breaks,
  do not hand‑tune — escalate.
- Run `just precommit` (slow tier) before requesting review; `just precommit-fast` is for local
  iteration only.
- Run `just coverage` when you need local HTML reports before requesting review.
- Before landing performance-sensitive changes, run `just test-deep` and
  `just bench-deep`; only run the longhaul tiers when maintainer asks.

PR content

- Scope, files touched, what you read, what you changed, how you tested (Ruff/Pyright/pytest
  summaries), perf delta (if applicable), limitations, clarifications (with assumptions), follow‑ups
  (H/M/L). Include file path references when helpful (e.g., `path/to/file.py:42`).

Blocked?

- After 60–90 minutes of focused effort, open a draft PR `Needs-Unblock: <topic>` with blockers and
  a proposed fallback.

## 9) CI & Branch Protection (facts)

- `just ci` mirrors GitHub Actions: dedicated jobs run format/lint, typecheck, and smoke+coverage
  tests while the weekly schedule executes the longhaul tiers.
- Branch protection: all checks must pass before merge; concurrency cancels in‑progress runs per ref.
- Enforcement lives in repo configs; contributors need not re‑validate tool configuration.

## 10) Policy Waivers (facts)

- Deviations live in `waivers.toml` with: `id`, `summary`, `owner`, `scope`, `created`, `expires`
  (YYYY‑MM‑DD), `justification`, `removal_plan`. CI fails on expiry via `scripts/check_waivers.py`.

## 11) Process: Docs, Thesis, Weekly Mail (facts)

- Docs authoring (`docs/`): concise Markdown documenting decisions, math references, and overviews;
  include relative links to modules/tests; avoid committing rendered artefacts unless requested.
- Thesis authoring (`thesis/`): single entry `thesis/main.tex`; include chapters via
  `\include{chapters/<name>}`; figures under `thesis/figures/`; prefer vector formats; use
  `thesis/macros.tex` for recurring notation; commit sources only (no build artefacts).
- Weekly progress mail (`progress-reports/`): use the scaffold and prompt in that folder; drafts
  named `YYYY-MM-DD-weekly-mail.md`; British English; short, outcome‑led paragraphs and bullets.

## 12) Scope & Enforcement (facts)

- This file applies to all tasks and agents. Maintain a single golden path.
- There must be exactly one `AGENTS.md` at the repository root; do not add per‑folder variants.
- If tools disagree or the golden path breaks, open `Needs-Unblock` instead of hand‑tuning.
