set shell := ["bash", "-euo", "pipefail", "-c"]
set export := true

default: checks

UV := env_var_or_default("UV", "uv")
HOME_DIR := env_var_or_default("HOME", "")
UV_PROJECT_ENVIRONMENT_DEFAULT := if HOME_DIR == "" {
    ".cache/uv/project-envs/msc-math-viterbo"
} else {
    HOME_DIR + "/.cache/uv/project-envs/msc-math-viterbo"
}
UV_PROJECT_ENVIRONMENT := env_var_or_default(
    "UV_PROJECT_ENVIRONMENT",
    UV_PROJECT_ENVIRONMENT_DEFAULT,
)
UV_LINK_MODE := env_var_or_default("UV_LINK_MODE", "hardlink")

SMOKE_TEST_TIMEOUT := env_var_or_default("SMOKE_TEST_TIMEOUT", "10")
PYTEST_ARGS := env_var_or_default("PYTEST_ARGS", "")
ARGS := env_var_or_default("ARGS", "")
RUN_DIR := env_var_or_default("RUN_DIR", "")
INC_ARGS := env_var_or_default("INC_ARGS", "")
JUST_USE_SELECTOR := env_var_or_default("JUST_USE_SELECTOR", "")

PYTEST_SMOKE_FLAGS := "-m smoke --durations=10"
PYTEST_DEEP_FLAGS := "-m \"smoke or deep\""
PYTEST_LONGHAUL_FLAGS := "-m longhaul"


PYTEST_JUNIT := ".cache/last-junit.xml"



BENCHMARK_STORAGE := ".benchmarks"
BENCH_FLAGS := "--benchmark-only --benchmark-autosave --benchmark-storage={{BENCHMARK_STORAGE}}"

PROFILES_DIR := ".profiles"

# Show the frequently used recipes.
help:
    @echo "Common development commands (tips follow their primary description):"
    @just --list

# Verify VK worktree hygiene before running heavier flows.
# Fails fast on:
# - detached HEAD
# - wrong branch for this worktree (expects vk/<worktree-name>)
# - rebase state present in this worktree's .git dir
preflight:
    @git_dir=`git rev-parse --git-dir 2>/dev/null || echo ""`; \
    if [ -z "$$git_dir" ]; then echo "[vk-preflight] Not a git repo; skipping."; exit 0; fi; \
    branch=`git rev-parse --abbrev-ref HEAD`; \
    expected=""; \
    if echo "$git_dir" | grep -q "/worktrees/"; then name=`echo "$git_dir" | awk -F'/worktrees/' '{print $2}'`; expected="vk/$name"; fi; \
    rebase_active=0; \
    if [ -d "$$git_dir/rebase-merge" ] || [ -d "$$git_dir/rebase-apply" ]; then rebase_active=1; fi; \
    echo "[vk-preflight] branch=$branch"; \
    if [ -n "$expected" ]; then echo "[vk-preflight] expected=$expected"; fi; \
    if [ "$branch" = "HEAD" ]; then echo "[vk-preflight] detached=1"; else echo "[vk-preflight] detached=0"; fi; \
    echo "[vk-preflight] rebase_active=$rebase_active"; \
    if git show-ref --quiet refs/heads/main; then \
      ab=`git rev-list --left-right --count HEAD...main 2>/dev/null || echo ""`; \
      if [ -n "$ab" ]; then echo "[vk-preflight] ahead_behind_vs_main=$ab"; fi; \
    fi; \
    if [ "$branch" = "HEAD" ]; then echo "[vk-preflight] ERROR: detached HEAD; attach to assigned VK branch."; exit 2; fi; \
    if [ -n "$expected" ] && [ "$branch" != "$expected" ]; then echo "[vk-preflight] ERROR: wrong branch for this worktree; expected $expected"; exit 2; fi; \
    if [ "$rebase_active" = "1" ]; then echo "[vk-preflight] ERROR: rebase state detected; abort with: git rebase --abort"; exit 2; fi

# Run an arbitrary command inside the project environment (uv run).
run *ARGS:
    $UV run {{ARGS}}

# Open an interactive shell with the project environment on PATH.
# Note: exits when the shell session ends.
shell:
    bash -l

# Fetch a paper by arXiv ID and store text under docs/papers.
# Usage: just fetch-arxiv ID=1712.03494 STATUS=useful
fetch-arxiv ID STATUS="useful":
    @echo "Fetching arXiv:{{ID}} (status={{STATUS}})"
    $UV run python scripts/fetch_paper.py --arxiv {{ID}} --status {{STATUS}}

# Fetch a paper by DOI and store text if an OA PDF is found.
# Usage: just fetch-doi DOI=10.1007/s00039-019-00486-4 STATUS=background
fetch-doi DOI STATUS="background":
    @echo "Fetching DOI:{{DOI}} (status={{STATUS}})"
    $UV run python scripts/fetch_paper.py --doi {{DOI}} --status {{STATUS}}

# Sync project dependencies (dev extras included).
# Tip: Run after pulling dependency changes; idempotent under uv.
sync:
    @echo "Syncing project dependencies via uv (dev + data extras)."
    $UV sync --extra dev --extra data

# Execute Jupytext notebooks and render to docs/notebooks as Markdown.
# Tip: View directly on GitHub or in the MkDocs site.
notebooks-md PATTERN="*.py":
    @echo "Rendering notebooks to Markdown under docs/notebooks (pattern={{PATTERN}})."
    $UV run --extra dev python scripts/render_notebooks.py --to md --out docs/notebooks --pattern {{PATTERN}} --index docs/notebooks/index.md

# Execute Jupytext notebooks and render single-file HTML under docs/notebooks/html.
notebooks-html PATTERN="*.py":
    @echo "Rendering notebooks to single-file HTML under docs/notebooks/html (pattern={{PATTERN}})."
    $UV run --extra dev python scripts/render_notebooks.py --to html --out docs/notebooks/html --pattern {{PATTERN}}

# Install the package with development dependencies, then refresh AGENTS.md sections.
setup:
    @echo "Syncing project dependencies via uv (dev + data extras)."
    $UV sync --extra dev --extra data
    @echo "Refreshing AGENTS.md skills sections."
    $UV run python scripts/load_skills_metadata.py --fix --quiet

# Run quiet Ruff formatter and autofixes.
# Tip: Safe to run before commits; pairs with `just lint` or `just precommit`.
fix:
    @echo "Applying Ruff format and autofix to src/, tests/, and scripts/."
    $UV run ruff format src tests scripts
    $UV run ruff check src tests scripts --fix

# Run quiet Ruff formatter across the repo.
# Tip: Safe to run before commits; pairs with `just lint` or `just precommit`.
format:
    @echo "Formatting source with Ruff."
    $UV run ruff format --quiet .

# Full Ruff lint and metadata validation.
# Tip: Mirrors CI linting.
lint:
    @echo "Validating skill metadata and AGENTS.md sections."
    $UV run --extra dev python scripts/load_skills_metadata.py --quiet --check
    @echo "Running Ruff lint."
    $UV run --extra dev ruff check .

# Summarise pytest test metadata (markers + docstrings).
# (Removed) test-metadata: legacy JAX-era test annotations utility.

# Quick, sensible defaults: lint → type → incremental tests (serial).
# (Removed) checks: prefer `just precommit`.

# Full repository loop: strict lint/type and full pytest tier handled by `ci`.

# Run strict Pyright analysis.
# Tip: Full repository sweep; matches CI `just ci`.
type:
    @echo "Running Pyright (basic) against src/viterbo."
    $UV run pyright -p pyrightconfig.json src/viterbo

# (Optional) type-strict: available for deep sweeps.
type-strict:
    @echo "Running Pyright strict across the repository."
    $UV run pyright -p pyrightconfig.strict.json

# Smoke-tier pytest with enforced timeouts.
# Default: full run; set JUST_USE_SELECTOR=1 to opt into the incremental selector.
test:
    just preflight
    @if [ "{{JUST_USE_SELECTOR}}" = "1" ]; then \
        echo "Selector opt-in detected (JUST_USE_SELECTOR=1); running incremental smoke-tier pytest."; \
        just _pytest-incremental "{{PYTEST_SMOKE_FLAGS}}" "smoke-tier pytest"; \
    else \
        echo "Selector disabled; running full smoke-tier pytest."; \
        echo "Tip: run 'just test-fast' or set JUST_USE_SELECTOR=1 to enable the selector."; \
        $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} {{PYTEST_ARGS}}; \
    fi

# Smoke-tier pytest without C++-marked tests.
test-no-cpp:
    just preflight
    @echo "Running smoke-tier pytest without cpp-marked tests."
    $UV run pytest -q -m 'smoke and not cpp' --durations=10 {{PYTEST_ARGS}}

# Incremental smoke-tier run (selector opt-in). Alias: `JUST_USE_SELECTOR=1 just test`.
test-fast:
    JUST_USE_SELECTOR=1 just test

# Full smoke-tier run (serial, no impacted selection).
test-full:
    @echo "Running full smoke-tier pytest (serial)."
    $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} {{PYTEST_ARGS}}

# Smoke + deep tiers (full serial).
# Tip: Ideal before review; combine with `just bench-deep` for performance-sensitive work.
test-deep:
    @echo "Running smoke + deep pytest tiers (serial)."
    $UV run pytest -q {{PYTEST_DEEP_FLAGS}} --durations=10 {{PYTEST_ARGS}}

# Longhaul pytest tier (manual, full serial).
# Tip: Scheduled weekly; coordinate with maintainer before running locally.
test-longhaul:
    @echo "Running longhaul pytest tier (serial; expect multi-hour runtime)."
    $UV run pytest -q {{PYTEST_LONGHAUL_FLAGS}} --durations=10 {{PYTEST_ARGS}}

# Run smoke, deep, and longhaul sequentially.
# Removed `test-all` in favor of explicit invocations.

## Legacy alias: `test-incremental` delegates to `just test-fast`.

# Unit vs integration convenience selectors.
test-unit:
    $UV run pytest -m "not integration and not deep and not longhaul" {{PYTEST_ARGS}}

test-integration:
    $UV run pytest -m "integration and not deep and not longhaul" {{PYTEST_ARGS}}

# Smoke-tier benchmarks.
# Tip: Use `PYTEST_ARGS="-k case"` to focus on a specific benchmark.
bench:
    just preflight
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
    @echo "[placeholder] Add project-specific profiling when needed."
    @true

# Line profile the fast EHZ kernel.
profile-line:
    @echo "[placeholder] Add line profiling targets when needed."
    @true

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
    @echo "Running smoke-tier tests with coverage."
    $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} --cov=src/viterbo --cov-report=term-missing --cov-report=html --cov-report=xml {{PYTEST_ARGS}}

precommit-fast: fix lint type test

# Full formatter, lint, typecheck, and smoke tests.
# Tip: Run before review or handoff; matches the golden-path expectation.
precommit-slow: format lint type test

# Alias for precommit-slow.
# Tip: Convenience alias for the slow gate.
precommit: precommit-slow

# Run the CI command set locally.
# Tip: Mirrors GitHub Actions; expect coverage artefacts and longer runtime.
ci:
    just preflight
    @echo "Running CI: sync deps (dev + data), lint, type (basic), smoke-tier tests, docs build."
    $UV sync --extra dev --extra data
    $UV run ruff check .
    $UV run pyright -p pyrightconfig.json
    $UV run pytest {{PYTEST_SMOKE_FLAGS}} -q {{PYTEST_ARGS}}
    $UV run mkdocs build --strict

# Local gate: format → lint → type → smoke tests (full by default; set JUST_USE_SELECTOR=1 for incremental)
checks:
    just preflight
    @echo "Running checks: format → lint → type → test (full smoke; set JUST_USE_SELECTOR=1 for incremental)."
    just format
    just lint
    just type
    just test

# Legacy alias for incremental smoke tests (opt-in).
test-incremental:
    just test-fast

_pytest-incremental FLAGS DESCRIPTION:
    @mkdir -p .cache
    @echo "Running {{DESCRIPTION}} (incremental selector)."
    @sel_status=0; $UV run --script scripts/inc_select.py {{INC_ARGS}} > .cache/impacted_nodeids.txt || sel_status=$?; \
    if [ -s .cache/impacted_nodeids.txt ] && [ "$sel_status" = "0" ]; then \
        $UV run pytest -q {{FLAGS}} --durations=10 --junitxml {{PYTEST_JUNIT}} @.cache/impacted_nodeids.txt {{PYTEST_ARGS}}; \
    elif [ "$sel_status" = "2" ]; then \
        echo "Selector: no changes and no prior failures — skipping pytest run."; \
    else \
        $UV run pytest -q {{FLAGS}} --durations=10 --junitxml {{PYTEST_JUNIT}} {{PYTEST_ARGS}}; \
    fi

# CI flow with CPU-only torch wheel using uv pip (bypasses lock for torch).
ci-cpu:
    @echo "Installing CPU-only torch (2.5.1) and project dev deps into system site-packages..."
    PIP_INDEX_URL="${PIP_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}" \
    UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://download.pytorch.org/whl/cpu}" \
    UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.org/simple}" \
    UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}" \
        $UV pip install --system "torch==2.5.1"
    PIP_INDEX_URL="${PIP_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}" \
    UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://download.pytorch.org/whl/cpu}" \
    UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.org/simple}" \
    UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}" \
        $UV pip install --system -e ".[dev,data]"
    @echo "Validating torch build is CPU-only."
    python scripts/verify_cpu_torch.py
    @echo "Running lint/type/smoke tests with coverage and docs build (system Python)."
    ruff check .
    pyright -p pyrightconfig.json
    python -m pytest -q {{PYTEST_SMOKE_FLAGS}} --cov=src/viterbo --cov-report=term-missing --cov-report=xml --cov-fail-under=85 {{PYTEST_ARGS}}
    mkdocs build --strict

# System-Python variants for CI (avoid uv-run creating new envs)
test-sys:
    @echo "Running smoke-tier pytest (system Python)."
    python -m pytest -q {{PYTEST_SMOKE_FLAGS}} {{PYTEST_ARGS}}

test-deep-sys:
    @echo "Running smoke+deep pytest tiers (system Python)."
    python -m pytest -q {{PYTEST_DEEP_FLAGS}} --durations=10 {{PYTEST_ARGS}}

bench-sys:
    @echo "Running smoke-tier benchmarks (system Python)."
    python -m pytest tests/performance -m "smoke" {{BENCH_FLAGS}} {{PYTEST_ARGS}}


# CI plus longhaul tiers and benchmarks.
# Tip: Reserved for scheduled runs; coordinate with the maintainer before executing.
ci-weekly: ci test-longhaul bench bench-longhaul

# Build MkDocs site with strict checks.
docs-build:
    just preflight
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

# Clean Torch extension build cache (user-level)
ext-clean:
    rm -rf ~/.cache/torch_extensions || true

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

# Cloudflare Workers — VK helpers
CF_DIR := ".devcontainer/cloudflare"

cf-deploy-sanitizer:
    @echo "Deploying VK API sanitizer worker (routes: /api/*)."
    cd {{CF_DIR}} && wrangler -c wrangler-sanitizer.toml deploy

cf-tail-sanitizer:
    @echo "Tailing VK API sanitizer worker logs."
    cd {{CF_DIR}} && wrangler -c wrangler-sanitizer.toml tail

cf-deploy-font:
    @echo "Deploying VK font/CSS injector worker (routes: /*)."
    cd {{CF_DIR}} && wrangler -c wrangler.toml deploy

cf-tail-font:
    @echo "Tailing VK font/CSS worker logs."
    cd {{CF_DIR}} && wrangler -c wrangler.toml tail

cf: cf-deploy-sanitizer cf-deploy-font

cf-tail:
    @echo "Tailing both Workers (sanitizer + font)."
    cd {{CF_DIR}} && (wrangler -c wrangler-sanitizer.toml tail & wrangler -c wrangler.toml tail & wait)
