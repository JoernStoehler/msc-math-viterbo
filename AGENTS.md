# AGENTS.md — Onboarding & Conventions (Source of Truth)

This file is the single source of truth for onboarding, workflows, and conventions. If anything elsewhere contradicts this file,
update those docs to match this.

**Quick Commands**
- `make setup` — install the package with development dependencies (pyproject extras) via uv
- `make format` — format code with Ruff
- `make lint` — lint code with Ruff
- `make typecheck` — static analysis via Pyright
- `make test` — run the pytest suite
- `make ci` — run formatting checks, lint, typecheck, and tests

**Workflows**
- Setup
  - Use Python 3.12+.
  - Use `uv` for dependency management (preinstalled by the devcontainer scripts).
  - Run `make setup` after cloning or when dependencies change.
- Daily Dev
  - Implement small, composable, mostly pure functions in `src/viterbo/`.
  - Keep side effects explicit (functional core, imperative shell).
  - Add or adjust pytest coverage for every behavior change.
  - Provide docstrings for public APIs, include shape information inline where helpful.
  - Annotate NumPy arrays with jaxtyping types and inline shape comments (`# shape: (n,)`).
  - Run `make format`, `make lint`, and `make test` locally before committing.
  - Use the performance workflows below whenever you touch optimized kernels.
- Local CI Check
  - Run `make ci` to mirror the GitHub Actions workflow (ruff check/format, pyright, pytest).
  - Resolve all lint and typecheck warnings; treat warnings as errors.
- Devcontainer (Codex Cloud)
  - One-time: `bash .devcontainer/post-create.sh`
  - Every boot: `bash .devcontainer/post-start.sh`
  - The devcontainer installs `uv` and keeps a warm cache for fast installs.
- Before PR
  - Ensure tests cover changes and public APIs have docstrings.
  - Keep diffs minimal and scoped.
  - No secrets in logs; env-only auth.

**Conventions**
- Tech Stack
  - Python 3.12+, NumPy, jaxtyping, pytest, Ruff, Pyright.
  - Project packaging via `pyproject.toml` with setuptools and editable installs.
  - Dependency manager: `uv`.
- Coding
  - Prefer pure functions for math core; isolate I/O in thin shells.
  - Use type hints everywhere practical; leverage jaxtyping for array shapes/dtypes.
  - Do **not** introduce custom NumPy array aliases (e.g., ``FloatMatrix``); annotate arrays directly with jaxtyping types.
  - Include docstrings for public symbols with examples when practical.
  - Maintain 4-space indentation across Python files.
  - Keep tests deterministic; prefer small, targeted assertions.
- Project Layout
  - `src/viterbo/` — Python package with functional core helpers.
  - `tests/` — pytest test suite.
    - `tests/performance/` centralizes benchmarks that reuse the same fixtures as regression tests.
  - `docs/` — overview + numbered references (project, math, tooling). Onboarding lives in this file.
  - `tmp/` — ignored scratch area for copied papers/notes you intend to incorporate later; do not commit raw notes.
  - `.devcontainer/` — container config and lifecycle scripts.
  - `.github/` — workflows and templates.
- CI
  - GitHub Actions on push/PR (Linux only): checkout, setup Python 3.12, cache pip, install deps, run Ruff format check, Ruff lint, Pyright, pytest.
  - Concurrency cancels in-progress runs per ref.
  - Coverage policy: upload-only TBD; currently no coverage upload or threshold.

**Performance Workflows**
- Install tooling via `make setup` (includes pytest-benchmark, pytest-profiling, pytest-line-profiler, scalene, and py-spy).
- Benchmarks live in `tests/performance/` and double as correctness checks.
  - `make bench` → run only benchmarked tests with autosaved stats under `.benchmarks/`.
  - `make profile` → wrap the same suite in `cProfile` for call graph exploration (`pytest --profile`).
  - `make profile-line` → activate `line_profiler` for the fast kernel (`pytest --line-profile viterbo.ehz_fast.compute_ehz_capacity_fast`).
- Mark new optimized workloads with `@pytest.mark.benchmark` and (optionally) `@pytest.mark.line_profile("path.to.function")`.
- Prefer extending `tests/_polytope_samples.py` when adding datasets so benchmarks and regression tests stay aligned.
- For deep dives outside pytest, use the installed standalone tools:
  - `scalene path/to/script.py` for CPU/GPU/memory sampling.
  - `py-spy top --pid <pid>` or `py-spy record -o profile.svg -- python script.py` for low-overhead profiling.
- Security
  - Never log secrets; environment variables only; avoid `set -x` and echoing env.

**Release**
- No releases/semver process planned; this is a MSc thesis repository.

**References**
- Project goal: `docs/01-project-goal.md`
- Roadmap: `docs/02-project-roadmap.md`
- Docs overview: `docs/README.md`
- Reading list: `docs/12-math-reading-list.md`
- Thesis topics: `docs/11-math-thesis-topics.md`
- Working with Codex: `docs/21-tool-working-with-codex-agents.md`

**Continuous Improvement**
- If you encounter ambiguity or friction, file an issue with a concrete change proposal. When you fix a discrepancy between this
file and another doc, prefer aligning the other doc to this file.
