Status: Implemented (scope: testing & CI deep-dive; caveats: reflects repository state on 2025-10-20)

# Review — Testing & CI Deep-Dive

Provenance
- Topic: Testing & CI Deep-Dive
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 87fdebc
- Timebox: ~60 minutes
- Scope: test tiers (smoke/deep/longhaul), selection/incremental (scripts/inc_select.py), coverage targets/enforcement, Pyright basic/strict posture, Ruff rules, CI parity and timing SLOs
- Version: v0.1

Context
- Deep pass over testing and CI posture to capture what exists, what’s missing, and where enforcement gaps are. Focus per ticket brief; unfiltered notes below.

Inputs & Method
- Read configs and scripts:
  - `pytest.ini:1`
  - `pyproject.toml:1`
  - `pyrightconfig.json:1`, `pyrightconfig.strict.json:1`
  - `scripts/inc_select.py:1`
  - `Justfile:1`, `.github/workflows/ci.yml:1`, `.github/workflows/docs.yml:1`
  - Skills: `skills/testing-and-ci.md:1`, `skills/good-code-loop.md:1`
  - SLO reference: `docs/charter.md:1`
- Ran commands (results captured under Validation):
  - `uv run python scripts/load_skills_metadata.py --quiet`
  - `uv run ruff check --no-cache .`
  - `uv run pyright -p pyrightconfig.json`
  - `mkdir -p .cache && uv run --script scripts/inc_select.py --debug > .cache/impacted_nodeids.txt 2> .cache/selector.log || true`

Findings (unsorted)
- Test tiers exist and are wired via markers:
  - `pytest.ini:1` defines `markers` for `smoke`, `deep`, `longhaul` and sets `timeout = 120` seconds globally.
  - `Justfile:1` provides `test` (smoke, incremental), `test-full` (smoke), `test-deep` (smoke or deep), `test-longhaul`.
  - CI runs smoke by default, then deep, and longhaul on schedule: `.github/workflows/ci.yml:1`.
- Incremental selection is implemented and documented:
  - `scripts/inc_select.py:1` builds a module-level import graph and content hashes; selects dirty test files + previous failures from `.cache/last-junit.xml`.
  - Exit codes: 0 emit selection; 2 skip (no changes, no prior failures); 3 advise full run (large impact). Selection threshold `SELECTION_THRESHOLD = 0.4` of total test files.
  - Conservative “plumbing” invalidation forces full run when `conftest.py`, `pytest.ini`, `pyproject.toml`, `uv.lock`, or the selector itself change; accepts extra `--plumbing` patterns.
  - Persists graph to `.cache/inc_graph.json`; normalizes relative paths; union of previous/current edges to avoid under-selection after import churn.
  - Just recipe `_pytest-incremental` mirrors intended contract including creating `.cache` and passing `--junitxml` so prior failures are tracked: `Justfile:240`.
- Default local test flow prefers incremental selection:
  - `just test` uses incremental selector; when nothing changed and no failures cached, it skips invoking pytest (fast inner loop). Observed skip behavior matches docs (see Validation).
- Coverage configuration present but enforcement missing:
  - `pyproject.toml:100` sets `[tool.coverage.run]` with `source = ["src/viterbo"]` and `branch = true`.
  - `Justfile:198` `coverage` target runs smoke tier with coverage and reports `term-missing`, HTML, XML; no `--cov-fail-under` and no `[tool.coverage.report] fail_under` found.
  - CI (`just ci`, `.github/workflows/ci.yml:1`) does not run coverage or enforce coverage floors.
  - The Charter specifies coverage floors (Math ≥95%, Datasets ≥60%): `docs/charter.md:1`; not currently enforced in CI or `just ci`.
- Pyright posture is explicit and split basic/strict:
  - Base `pyrightconfig.json:1` includes `src`, sets `typeCheckingMode: "basic"`, and treats missing imports as errors; `extraPaths` include `tests` for resolution.
  - `pyrightconfig.strict.json:1` extends base and sets strict mode, elevating several unknown-type reports; provided as an optional deep sweep (`just type-strict`).
  - `Justfile:1` `type` runs Pyright (basic) against `src/viterbo` only; `type-strict` runs repo-wide.
- Ruff rules are curated for correctness and policy over style:
  - `pyproject.toml:50` selects `E7,E9,F,D,I,UP,ANN401,PGH003,TID252,RUF100,B006,B008,BLE001,T20`; ignores several docstring formatters (`D100,D104,D203,D212,D401,D417,D202,D205,D301`).
  - Google docstring convention configured; per-file ignores disable docstring checks in tests; `__init__.py` ignores include `F401,F841,ANN401` as needed for package exports: `pyproject.toml:79`.
  - Format settings standardized (double quotes, LF, width 100); `just fix` and `just format` wire Ruff format and autofix.
- CI parity commands are available and mostly mirrored:
  - `just ci` runs: `uv sync --extra dev`, `ruff check .`, `pyright -p pyrightconfig.json`, `pytest -q -m smoke`, and `mkdocs build --strict` — mirrors `.github/workflows/ci.yml` ‘ci-cpu’ job except CI installs CPU-only torch via pip to system first.
  - CI leverages `ci-cpu` to install `torch==2.5.1` CPU wheels against the CPU index for reliability; locally the pre-push hook forces the same CPU-only indexes only when no CUDA runtime is present, otherwise it preserves developer GPU wheels.
  - UV cache (`~/.cache/uv`) is cached in CI; no explicit pip/pytest caches beyond that.
- Timing and SLOs:
  - Charter sets CI SLO p95 ≤ 7 minutes: `docs/charter.md:1`.
  - No runtime capture/reporting in CI beyond raw step durations; tests do not pass `--durations` to summarize slowest nodes; no explicit per-tier wall-clock budgets codified in recipes.
  - `pytest-timeout` plugin is present and `timeout=120` is configured globally in `pytest.ini`; `SMOKE_TEST_TIMEOUT` env var exists in `Justfile` but is not wired into pytest flags (unused).
- Test layout and tiers appear coherent:
  - Smoke tests exercise core math, datasets, and C++ fallback plumbing: `tests/test_smoke.py:1`, `tests/test_cpp_extension.py:1`, `tests/test_collate.py:1`.
  - A `deep` placeholder exists to verify CI marker wiring: `tests/test_deep_placeholder.py:1`.
  - Benchmarks live under `tests/performance`; smoke/deep/longhaul bench recipes exist and CI runs smoke benches.
- Docs build parity is explicit and strict:
  - `mkdocs.yml:1` sets `strict: true` and CI/docs workflow builds with `uv run mkdocs build --strict`.
  - Nav deliberately excludes per-review pages; new files should build without nav additions. Strict mode will fail on broken links/references only.
- Skills and Good Code Loop align with current wiring:
  - `skills/testing-and-ci.md:1` and `skills/good-code-loop.md:1` describe incremental selection, `just checks`, and CI parity expectations; metadata loader `scripts/load_skills_metadata.py` is integrated into `just lint`.
- Minor polish opportunities noticed:
  - `_pytest-incremental` already writes JUnit XML; consider adding `-q -ra` flags parity to match `pytest.ini addopts` for consistency (currently `-q` only is passed in recipes; `-ra` appears only in `pytest.ini`).
  - Consider adding a `just ci-with-coverage` that enforces coverage floors and uploads XML/HTML as artifacts for PRs.
  - Consider exposing `--durations=10` for smoke in CI to track regressions.

Actions (first pass)
- Add coverage enforcement aligned with Charter:
  - Option A: add `[tool.coverage.report] fail_under = 95` scoped to `src/viterbo/math/**` via `--cov-config` per-path is tricky; simpler: add `--cov-fail-under` to `Justfile:198` and a second target for datasets with 60. In CI, run both selectors or a combined floor at a conservative global value plus per-path checks in a follow-up.
  - Wire `just ci` to generate coverage (XML/HTML) and fail under agreed floors; upload as artifacts in CI.
- Wire `SMOKE_TEST_TIMEOUT` into pytest invocation so CI/local can override global timeout without editing `pytest.ini` (e.g., `pytest --timeout=${SMOKE_TEST_TIMEOUT}`).
- Add `--durations=10` to smoke tier in CI to surface slowest tests; consider storing a brief durations log as an artifact.
- Consider promoting `pyright strict` for specific math modules by adding a `pyrightconfig.math-strict.json` and a `just type-math-strict` target; keep basic as repo gate.
- Add a small `docs` note in `skills/testing-and-ci.md` about interpreting incremental selector exit codes (0/2/3) and the plumbing-change behavior observed.

Cross-References
- Topics: Good Code Loop; Testing & CI (skills)
- Files: `pytest.ini:1`, `pyproject.toml:1`, `scripts/inc_select.py:1`, `Justfile:1`, `.github/workflows/ci.yml:1`, `docs/charter.md:1`, `skills/testing-and-ci.md:1`

Validation
- `uv run python scripts/load_skills_metadata.py --quiet` — OK (no output).
- `uv run ruff check --no-cache .` — OK (All checks passed!).
- `uv run pyright -p pyrightconfig.json` — OK (0 errors).
- Incremental selector dry-run: `mkdir -p .cache && uv run --script scripts/inc_select.py --debug > .cache/impacted_nodeids.txt 2> .cache/selector.log || true` — OK; observed `[inc] plumbing: scripts/inc_select.py, tests/conftest.py` then `plumbing changed; fall back to full run`; snapshot persisted at `.cache/inc_graph.json`; selection file empty as expected.
- `uv run mkdocs build --strict` — OK (site built; orphan page notice expected for unlinked reviews).

Limitations
- Did not run full pytest tiers to avoid installing torch/wheels in this pass; relied on selector behavior and config reads. Coverage floors not measured empirically in this pass.
- CI timings taken from config intent; no fresh run timings captured here.

Status
- Draft

Pre-Submit Checklist
- mkdocs strict build green (post‑add) — pending
- No changes to README/index/nav (guardrail) — respected
- Actions extracted — yes
- Cross‑refs noted — yes
- Provenance filled — yes
- Title pattern — ok
