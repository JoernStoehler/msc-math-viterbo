set shell := ["bash", "-euo", "pipefail", "-c"]
set export := true

default: checks

UV := env_var_or_default("UV", "uv")
PRETTIER := 'npx --yes prettier@3.3.3'

SMOKE_TEST_TIMEOUT := env_var_or_default("SMOKE_TEST_TIMEOUT", "10")
TESTMON_CACHE := env_var_or_default("TESTMON_CACHE", ".testmondata")
USE_TESTMON := env_var_or_default("USE_TESTMON", "1")
PYTEST_ARGS := env_var_or_default("PYTEST_ARGS", "")
ARGS := env_var_or_default("ARGS", "")
RUN_DIR := env_var_or_default("RUN_DIR", "")
FAST_ENV := "FAST=1 JAX_DISABLE_JIT=true JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false"

PYTEST_SMOKE_FLAGS := "-m \"not deep and not longhaul\" --timeout=$SMOKE_TEST_TIMEOUT --timeout-method=thread"
PYTEST_DEEP_FLAGS := "-m \"not longhaul\""
PYTEST_LONGHAUL_FLAGS := "-m \"longhaul\""

PRETTIER_PATTERNS := '"README.md" "docs/**/*.{md,mdx}" "progress-reports/**/*.md" "**/*.{yml,yaml,json}"'

BENCHMARK_STORAGE := ".benchmarks"
BENCH_FLAGS := "--benchmark-only --benchmark-autosave --benchmark-storage={{BENCHMARK_STORAGE}}"

PROFILES_DIR := ".profiles"

# Show the frequently used recipes.
help:
    @echo "Common development commands (tips follow their primary description):"
    @just --list

# Run an arbitrary command inside the project environment (uv run).
run *ARGS:
    $UV run {{ARGS}}

# Open an interactive shell with the project environment on PATH.
# Note: exits when the shell session ends.
shell:
    bash -l

# Sync project dependencies (dev extras included).
# Tip: Run after pulling dependency changes; idempotent under uv.
sync:
    @echo "Syncing project dependencies via uv (dev extras)."
    $UV sync --extra dev

# Install the package with development dependencies.
# Tip: Backwards-compatible alias for `just sync`.
setup: sync

# Run quiet Ruff formatter and Prettier writes.
# Tip: Safe to run before commits; pairs with `just lint` or `just precommit`.
fix:
    @echo "Applying Ruff format and autofix to src/, tests/, and scripts/."
    $UV run ruff format src tests scripts
    $UV run ruff check src tests scripts --fix

# Run quiet Ruff formatter and Prettier writes.
# Tip: Safe to run before commits; pairs with `just lint` or `just precommit`.
format:
    @echo "Formatting source and docs with Ruff + Prettier."
    $UV run ruff format --quiet .
    {{PRETTIER}} --log-level warn --write {{PRETTIER_PATTERNS}}

# Full Ruff lint and Prettier validation.
# Tip: Mirrors CI linting; use when editing Markdown alongside Python.
lint:
    @echo "Running Ruff lint and Prettier check (CI parity)."
    $UV run ruff check .
    $UV run python scripts/check_test_metadata.py
    {{PRETTIER}} --log-level warn --check {{PRETTIER_PATTERNS}}

# Summarise pytest test metadata (markers + docstrings).
test-metadata:
    @echo "Usage: just test-metadata [ARGS='--marker goal_math tests/path']; forwards ARGS to report_test_metadata."
    $UV run python scripts/report_test_metadata.py {{ARGS}}

# Quick, sensible defaults: full lint → fast type → parallel tests.
checks:
    @echo "Running checks: lint → type → test (parallel)."
    just lint
    just type
    USE_TESTMON=0 just test

# Quick test run using FAST settings (no JIT/GPU, single-process when testmon is enabled).
test-fast:
    @echo "Running quick pytest (FAST mode)."
    @testmon_flags=(); parallel_flags=(); \
    if [[ "${USE_TESTMON,,}" != "0" && "${USE_TESTMON,,}" != "false" && "${USE_TESTMON,,}" != "no" ]]; then \
        testmon_flags=(--testmon); parallel_flags=(-p no:xdist); \
    else \
        parallel_flags=(-n auto); \
    fi; \
    {{FAST_ENV}} TESTMONDATA="{{TESTMON_CACHE}}" $UV run pytest "${testmon_flags[@]}" {{PYTEST_SMOKE_FLAGS}} "${parallel_flags[@]}" {{PYTEST_ARGS}}

# Full repository loop: strict lint/type and full pytest tier (no FAST env overrides).
# Tip: Run before reviews or when validating significant refactors.
# (legacy) full-loop use case is covered by `ci` below.

# Run strict Pyright analysis.
# Tip: Full repository sweep; matches CI `just ci`.
type:
    @echo "Running Pyright (basic) against src/viterbo."
    $UV run pyright -p pyrightconfig.json src/viterbo

type-strict:
    @echo "Running Pyright across the entire repository (strict)."
    $UV run pyright -p pyrightconfig.strict.json

# Smoke-tier pytest with enforced timeouts.
# Tip: Testmon cache is on by default; set `USE_TESTMON=0` to disable, add selectors via `PYTEST_ARGS`.
test:
    @echo "Running smoke-tier pytest (testmon cache: {{USE_TESTMON}}; set USE_TESTMON=0 to disable)."
    @testmon_flags=(); parallel_flags=(); \
    if [[ "${USE_TESTMON,,}" != "0" && "${USE_TESTMON,,}" != "false" && "${USE_TESTMON,,}" != "no" ]]; then \
        testmon_flags=(--testmon); parallel_flags=(-p no:xdist); \
    else \
        parallel_flags=(-n auto); \
    fi; \
    TESTMONDATA="{{TESTMON_CACHE}}" $UV run pytest "${testmon_flags[@]}" {{PYTEST_SMOKE_FLAGS}} "${parallel_flags[@]}" {{PYTEST_ARGS}}

# Smoke + deep tiers.
# Tip: Ideal before review; combine with `just bench-deep` for performance-sensitive work.
test-deep:
    @echo "Running smoke + deep pytest tiers."
    @testmon_flags=(); parallel_flags=(); \
    if [[ "${USE_TESTMON,,}" != "0" && "${USE_TESTMON,,}" != "false" && "${USE_TESTMON,,}" != "no" ]]; then \
        testmon_flags=(--testmon); parallel_flags=(-p no:xdist); \
    else \
        parallel_flags=(-n auto); \
    fi; \
    TESTMONDATA="{{TESTMON_CACHE}}" $UV run pytest "${testmon_flags[@]}" {{PYTEST_DEEP_FLAGS}} "${parallel_flags[@]}" {{PYTEST_ARGS}}

# Longhaul pytest tier (manual).
# Tip: Scheduled weekly; coordinate with maintainer before running locally.
test-longhaul:
    @echo "Running longhaul pytest tier (expect multi-hour runtime)."
    @testmon_flags=(); parallel_flags=(); \
    if [[ "${USE_TESTMON,,}" != "0" && "${USE_TESTMON,,}" != "false" && "${USE_TESTMON,,}" != "no" ]]; then \
        testmon_flags=(--testmon); parallel_flags=(-p no:xdist); \
    else \
        parallel_flags=(-n auto); \
    fi; \
    TESTMONDATA="{{TESTMON_CACHE}}" $UV run pytest "${testmon_flags[@]}" {{PYTEST_LONGHAUL_FLAGS}} "${parallel_flags[@]}" {{PYTEST_ARGS}}

# Run smoke, deep, and longhaul sequentially.
# Removed `test-all` in favor of explicit invocations.

## (Removed) test-incremental: prefer `just test` or `just test-fast`.

# Unit vs integration convenience selectors.
test-unit:
    $UV run pytest -m "not integration and not deep and not longhaul" {{PYTEST_ARGS}}

test-integration:
    $UV run pytest -m "integration and not deep and not longhaul" {{PYTEST_ARGS}}

# Smoke-tier benchmarks.
# Tip: Use `PYTEST_ARGS="-k case"` to focus on a specific benchmark.
bench:
    @echo "Running smoke-tier benchmarks; results saved under {{BENCHMARK_STORAGE}}."
    $UV run pytest tests/performance -m "smoke" {{BENCH_FLAGS}} {{PYTEST_ARGS}}

# Deep-tier benchmarks.
# Tip: Pair with `just test-deep` during pre-merge performance validation.
bench-deep:
    @echo "Running deep-tier benchmarks (longer runtime)."
    $UV run pytest tests/performance -m "deep" {{BENCH_FLAGS}} {{PYTEST_ARGS}}

# Longhaul benchmarks (manual).
# Tip: Schedule with maintainer; archives feed performance baselines.
bench-longhaul:
    @echo "Running longhaul benchmarks (expect extended runtime)."
    $UV run pytest tests/performance -m "longhaul" {{BENCH_FLAGS}} {{PYTEST_ARGS}}

# Profile deep-tier benchmarks (callgrind + svg).
# Tip: After running, inspect `.profiles/` artifacts locally; keep runs out of Git.
profile:
    @mkdir -p "{{PROFILES_DIR}}"
    @echo "Running profile tier (callgrind + SVG) into {{PROFILES_DIR}}."
    $UV run pytest tests/performance -m "deep" --profile --profile-svg --pstats-dir="{{PROFILES_DIR}}" {{PYTEST_ARGS}}

# Line profile the fast EHZ kernel.
profile-line:
    @mkdir -p "{{PROFILES_DIR}}"
    @echo "Running line profiler for compute_ehz_capacity_fast into {{PROFILES_DIR}}."
    $UV run pytest tests/performance -m "deep" --line-profile viterbo.symplectic.capacity.facet_normals.fast.compute_ehz_capacity_fast --pstats-dir="{{PROFILES_DIR}}" {{PYTEST_ARGS}}

# Package build & publish
build:
    @echo "[skip] Build disabled: research project is not published as a package."
    @echo "      If you need a wheel/sdist for local experiments, run: uv build"
    @true

publish:
    @echo "[skip] Publish disabled: research project is not published to PyPI/TestPyPI."
    @echo "      For internal sharing, use artefact tarballs or Git tags."
    @true

# Bump semantic version and tag a release. Usage: just release patch|minor|major
release LEVEL:
    @echo "Bumping version: {{LEVEL}}"
    $UV version --bump {{LEVEL}}
    @new_ver=$($UV version --short); \
    echo "New version: $$new_ver"; \
    git add pyproject.toml uv.lock || true; \
    git commit -m "chore(release): v$$new_ver" || true; \
    git tag -a "v$$new_ver" -m "Release v$$new_ver" || true; \
    echo "Created tag v$$new_ver";

# Dependency helpers
dep-add *PKGS:
    @if [ -z "{{PKGS}}" ]; then echo "Usage: just dep-add PKG[ PKG2 ...]" >&2; exit 2; fi
    $UV add {{PKGS}}

dep-rm *PKGS:
    @if [ -z "{{PKGS}}" ]; then echo "Usage: just dep-rm PKG[ PKG2 ...]" >&2; exit 2; fi
    $UV remove {{PKGS}}

dep-upgrade:
    @echo "Upgrading lockfile to latest compatible versions and syncing environment."
    $UV lock --upgrade
    $UV sync --extra dev

lock:
    $UV lock
# Smoke-tier tests with coverage reports.
# Tip: Generates HTML at `htmlcov/index.html`; testmon cache is on by default.
coverage:
    @echo "Running smoke-tier tests with coverage (HTML + XML reports)."
    @testmon_flags=(); parallel_flags=(); \
    if [[ "${USE_TESTMON,,}" != "0" && "${USE_TESTMON,,}" != "false" && "${USE_TESTMON,,}" != "no" ]]; then \
        testmon_flags=(--testmon); parallel_flags=(-p no:xdist); \
    else \
        parallel_flags=(-n auto); \
    fi; \
    TESTMONDATA="{{TESTMON_CACHE}}" $UV run pytest "${testmon_flags[@]}" {{PYTEST_SMOKE_FLAGS}} "${parallel_flags[@]}" --cov=src/viterbo --cov-report=term-missing --cov-report=html --cov-report=xml {{PYTEST_ARGS}}

precommit-fast: checks

# Full formatter, lint, typecheck, and smoke tests.
# Tip: Run before review or handoff; matches the golden-path expectation.
precommit-slow: format lint type-strict test

# Alias for precommit-slow.
# Tip: Convenience alias for the slow gate.
precommit: precommit-slow

# Run the CI command set locally.
# Tip: Mirrors GitHub Actions; expect coverage artefacts and longer runtime.
ci:
    @echo "Running CI parity: sync deps, lint, type-strict, pytest (durations summary)."
    $UV sync --extra dev
    $UV run python scripts/check_waivers.py
    $UV run ruff check .
    {{PRETTIER}} --log-level warn --check {{PRETTIER_PATTERNS}}
    $UV run pyright -p pyrightconfig.strict.json
    $UV run pytest {{PYTEST_SMOKE_FLAGS}} -n auto --durations=20 {{PYTEST_ARGS}}

# CI plus longhaul tiers and benchmarks.
# Tip: Reserved for scheduled runs; coordinate with the maintainer before executing.
ci-weekly: ci test-longhaul bench bench-longhaul

# Build MkDocs site with strict checks.
docs-build:
    $UV run mkdocs build --strict

# Serve MkDocs locally on :8000.
docs-serve:
    $UV run mkdocs serve -a 0.0.0.0:8000

# Run MkDocs build in strict mode (fails on warnings).
docs-check-links: docs-build

# Remove cached tooling outputs.
clean:
    rm -rf .pytest_cache .ruff_cache .pyright .pyright_cache build dist *.egg-info

# Remove benchmarking, profiling, and coverage artefacts.
clean-artefacts:
    rm -rf "{{BENCHMARK_STORAGE}}" "{{PROFILES_DIR}}" htmlcov .coverage* .artefacts

# Clean virtual environment and caches (CAUTION: removes .venv).
distclean:
    rm -rf .venv .pytest_cache .ruff_cache .pyright .pyright_cache "{{BENCHMARK_STORAGE}}" "{{PROFILES_DIR}}" htmlcov .coverage* .artefacts

# Train the toy logistic regression experiment.
train-logreg:
    @set -a; \
    if [ -f .env ]; then . ./.env; fi; \
    set +a; \
    if [ -z "${WANDB_API_KEY:-}" ]; then \
        echo "WANDB_API_KEY is not set. Add it to .env before training." >&2; \
        exit 1; \
    fi; \
    WANDB_API_KEY="${WANDB_API_KEY}" $UV run python scripts/train_logreg_toy.py {{ARGS}}

# Evaluate a toy logistic regression run.
evaluate-logreg:
    @if [ -z "${RUN_DIR:-}" ]; then \
        echo "RUN_DIR must reference the run directory to evaluate." >&2; \
        exit 1; \
    fi
    $UV run python scripts/evaluate_logreg_toy.py --run-dir "${RUN_DIR}" {{ARGS}}

# Package a toy logistic regression run for sharing.
publish-logreg:
    @if [ -z "${RUN_DIR:-}" ]; then \
        echo "RUN_DIR must reference the run directory to publish." >&2; \
        exit 1; \
    fi
    @mkdir -p artefacts/published/logreg-toy
    @tar czf artefacts/published/logreg-toy/$(basename "${RUN_DIR}").tar.gz -C "$(dirname "${RUN_DIR}")" "$(basename "${RUN_DIR}")"
