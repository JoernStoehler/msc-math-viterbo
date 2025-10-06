UV ?= uv
JAX_ENABLE_X64 ?= 1
export JAX_ENABLE_X64

# Tool invocation shortcuts (prefer uv to leverage the managed virtualenv).
PYTEST := $(UV) run pytest
PYRIGHT := $(UV) run pyright
RUFF := $(UV) run ruff
PRETTIER := npx --yes prettier@3.3.3
MKDOCS := $(UV) run mkdocs

# Override-friendly toggles and caches.
SMOKE_TEST_TIMEOUT ?= 10
TESTMON_CACHE ?= .testmondata
USE_TESTMON ?= 1
# Supply extra pytest args with e.g. `PYTEST_ARGS="-k pattern" make test`.
PYTEST_ARGS ?=

# Set USE_TESTMON=0 (or no/false) to disable pytest-testmon caching on standard test targets.
PYTEST_TESTMON_FLAGS = $(if $(filter-out 0 false no,$(USE_TESTMON)),--testmon --testmondata $(TESTMON_CACHE),)

# Pytest selector presets mirror pytest.ini markers; extend with PYTEST_ARGS when needed.
PYTEST_SMOKE_FLAGS := -m "not deep and not longhaul" --timeout=$(SMOKE_TEST_TIMEOUT) --timeout-method=thread
PYTEST_DEEP_FLAGS := -m "not longhaul"
PYTEST_LONGHAUL_FLAGS := -m "longhaul"

.PHONY: help \
        setup \
        format \
        lint \
        lint-fast \
        typecheck \
        typecheck-fast \
        test \
        test-deep \
        test-longhaul \
        test-all \
        test-incremental \
        bench \
        bench-deep \
        bench-longhaul \
        profile \
        profile-line \
        coverage \
        precommit-fast \
        precommit-slow \
        precommit \
        docs-build \
        docs-serve \
        docs-check-links \
        ci \
        ci-weekly \
        clean \
        clean-artefacts \
        train-logreg \
        evaluate-logreg \
        publish-logreg

#------------------------------------------------------------------------------
# Self-documenting help
#------------------------------------------------------------------------------
## help                 Show the frequently used targets.
help:
	@echo "Common development commands (tips follow their primary description):"
	@awk '/^## /{sub(/^## /, ""); print}' $(MAKEFILE_LIST)

#------------------------------------------------------------------------------
# Core QA workflow
#------------------------------------------------------------------------------
## setup                Install the package with development dependencies.
##   Tip: Run after pulling dependency changes; idempotent under uv.
setup:
	@echo "Syncing project dependencies via uv (dev extras)."
	$(UV) sync --extra dev

# Formatter configuration keeps Markdown/JSON/YAML consistent with docs guidance.
PRETTIER_PATTERNS := "README.md" "docs/**/*.{md,mdx}" "progress-reports/**/*.md" "**/*.{yml,yaml,json}"

## format               Run quiet Ruff formatter and Prettier writes.
##   Tip: Safe to run before commits; pairs with `make lint` or `make precommit`.
format:
	@echo "Formatting source and docs with Ruff + Prettier."
	$(RUFF) format --quiet .
	$(PRETTIER) --log-level warn --write $(PRETTIER_PATTERNS)

## lint                 Full Ruff lint and Prettier validation.
##   Tip: Mirrors CI linting; use when editing Markdown alongside Python.
lint:
	@echo "Running Ruff lint and Prettier check (CI parity)."
	$(RUFF) check .
	$(PRETTIER) --log-level warn --check $(PRETTIER_PATTERNS)

## lint-fast            Minimal Ruff diagnostics (E/F/B006/B008).
##   Tip: Catches runtime errors quickly; run `make lint` for policy/doc coverage.
lint-fast:
	@echo "Running Ruff fast lint (E/F/B006/B008, ignores jaxtyping F722)."
	$(RUFF) check src tests --select E9 --select F --select B006 --select B008 --ignore F722

## typecheck            Run strict Pyright analysis.
##   Tip: Full repository sweep; matches CI `make ci`.
typecheck:
	@echo "Running Pyright across the entire repository."
	$(PYRIGHT)

## typecheck-fast       Quick library-only Pyright pass.
##   Tip: Focuses on `src/viterbo`; run `make typecheck` before review.
typecheck-fast:
	@echo "Running Pyright against src/viterbo only."
	$(PYRIGHT) src/viterbo

#------------------------------------------------------------------------------
# Pytest tiers
#------------------------------------------------------------------------------
## test                 Smoke-tier pytest with enforced timeouts.
##   Tip: Testmon cache is on by default; set `USE_TESTMON=0` to disable, add selectors via `PYTEST_ARGS`.
test:
	@echo "Running smoke-tier pytest (testmon cache: $(USE_TESTMON); set USE_TESTMON=0 to disable)."
	$(PYTEST) $(PYTEST_TESTMON_FLAGS) $(PYTEST_SMOKE_FLAGS) $(PYTEST_ARGS)

## test-deep            Smoke + deep tiers.
##   Tip: Ideal before review; combine with `make bench-deep` for performance-sensitive work.
test-deep:
	@echo "Running smoke + deep pytest tiers."
	$(PYTEST) $(PYTEST_TESTMON_FLAGS) $(PYTEST_DEEP_FLAGS) $(PYTEST_ARGS)

## test-longhaul        Longhaul pytest tier (manual).
##   Tip: Scheduled weekly; coordinate with maintainer before running locally.
test-longhaul:
	@echo "Running longhaul pytest tier (expect multi-hour runtime)."
	$(PYTEST) $(PYTEST_TESTMON_FLAGS) $(PYTEST_LONGHAUL_FLAGS) $(PYTEST_ARGS)

## test-all             Run smoke, deep, and longhaul sequentially.
test-all: test test-deep test-longhaul

## test-incremental     Smoke tier with pytest-testmon cache.
##   Tip: Keeps testmon warm during tight loops; clear cache by deleting `$(TESTMON_CACHE)`.
test-incremental:
	@echo "Running smoke-tier pytest with testmon cache warmup."
	$(PYTEST) --testmon --testmondata $(TESTMON_CACHE) --maxfail=1 $(PYTEST_SMOKE_FLAGS) $(PYTEST_ARGS)

#------------------------------------------------------------------------------
# Benchmarks and profiling
#------------------------------------------------------------------------------
# Benchmark harness configuration (pytest-benchmark caches live under $(BENCHMARK_STORAGE)).
BENCHMARK_STORAGE := .benchmarks
BENCH_FLAGS := --benchmark-only --benchmark-autosave --benchmark-storage=$(BENCHMARK_STORAGE)

## bench                Smoke-tier benchmarks.
##   Tip: Use `PYTEST_ARGS="-k case"` to focus on a specific benchmark.
bench:
	@echo "Running smoke-tier benchmarks; results saved under $(BENCHMARK_STORAGE)."
	$(PYTEST) tests/performance -m "smoke" $(BENCH_FLAGS) $(PYTEST_ARGS)

## bench-deep           Deep-tier benchmarks.
##   Tip: Pair with `make test-deep` during pre-merge performance validation.
bench-deep:
	@echo "Running deep-tier benchmarks (longer runtime)."
	$(PYTEST) tests/performance -m "deep" $(BENCH_FLAGS) $(PYTEST_ARGS)

## bench-longhaul       Longhaul benchmarks (manual).
##   Tip: Schedule with maintainer; archives feed performance baselines.
bench-longhaul:
	@echo "Running longhaul benchmarks (expect extended runtime)."
	$(PYTEST) tests/performance -m "longhaul" $(BENCH_FLAGS) $(PYTEST_ARGS)

# Profiling outputs go to $(PROFILES_DIR); inspect with kcachegrind/qcachegrind.
PROFILES_DIR := .profiles

## profile              Profile deep-tier benchmarks (callgrind + svg).
##   Tip: After running, inspect `.profiles/` artifacts locally; keep runs out of Git.
profile:
	@mkdir -p $(PROFILES_DIR)
	@echo "Running profile tier (callgrind + SVG) into $(PROFILES_DIR)."
	$(PYTEST) tests/performance -m "deep" --profile --profile-svg --pstats-dir=$(PROFILES_DIR) $(PYTEST_ARGS)

## profile-line         Line profile the fast EHZ kernel.
profile-line:
	@mkdir -p $(PROFILES_DIR)
	@echo "Running line profiler for compute_ehz_capacity_fast into $(PROFILES_DIR)."
	$(PYTEST) tests/performance -m "deep" --line-profile viterbo.symplectic.capacity.facet_normals.fast.compute_ehz_capacity_fast --pstats-dir=$(PROFILES_DIR) $(PYTEST_ARGS)

#------------------------------------------------------------------------------
# Coverage and pre-commit gates
#------------------------------------------------------------------------------
## coverage             Smoke-tier tests with coverage reports.
##   Tip: Generates HTML at `htmlcov/index.html`; testmon cache is on by default.
coverage:
	@echo "Running smoke-tier tests with coverage (HTML + XML reports)."
	$(PYTEST) $(PYTEST_TESTMON_FLAGS) $(PYTEST_SMOKE_FLAGS) --cov=src/viterbo --cov-report=term-missing --cov-report=html --cov-report=xml $(PYTEST_ARGS)

## precommit-fast       Lint essentials and incremental smoke tests.
##   Tip: Use during tight dev loops; relies on the testmon cache.
precommit-fast: lint-fast test-incremental

## precommit-slow       Full formatter, lint, typecheck, and smoke tests.
##   Tip: Run before review or handoff; matches the golden-path expectation.
precommit-slow: format lint typecheck test

## precommit            Alias for precommit-slow.
##   Tip: Convenience alias for the slow gate.
precommit: precommit-slow

#------------------------------------------------------------------------------
# Continuous integration presets
#------------------------------------------------------------------------------
## ci                   Run the CI command set locally.
##   Tip: Mirrors GitHub Actions; expect coverage artefacts and longer runtime.
ci:
	@echo "Running full CI suite locally (check waivers, format, lint, typecheck, coverage)."
	$(UV) run python scripts/check_waivers.py
	$(RUFF) format --check .
	$(RUFF) check .
	$(PRETTIER) --log-level warn --check $(PRETTIER_PATTERNS)
	$(PYRIGHT)
	$(PYTEST) $(PYTEST_SMOKE_FLAGS) --cov=src/viterbo --cov-report=term-missing --cov-report=xml --cov-report=html

## ci-weekly            CI plus longhaul tiers and benchmarks.
##   Tip: Reserved for scheduled runs; coordinate with the maintainer before executing.
ci-weekly: ci test-longhaul bench bench-longhaul

#------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------
## docs-build           Build MkDocs site with strict checks.
docs-build:
	$(MKDOCS) build --strict

## docs-serve           Serve MkDocs locally on :8000.
docs-serve:
	$(MKDOCS) serve -a 0.0.0.0:8000

## docs-check-links     Run MkDocs build in strict mode (fails on warnings).
docs-check-links:
	$(MKDOCS) build --strict

#------------------------------------------------------------------------------
# Hygiene
#------------------------------------------------------------------------------
## clean                Remove cached tooling outputs.
clean:
	rm -rf .pytest_cache .ruff_cache .pyright .pyright_cache build dist *.egg-info

## clean-artefacts      Remove benchmarking, profiling, and coverage artefacts.
clean-artefacts:
	rm -rf $(BENCHMARK_STORAGE) $(PROFILES_DIR) htmlcov .coverage* .artefacts

#------------------------------------------------------------------------------
# Toy logistic regression helpers
#------------------------------------------------------------------------------
## train-logreg         Train the toy logistic regression experiment.
train-logreg:
	@set -a; \
	  if [ -f .env ]; then . ./.env; fi; \
	  set +a; \
	  if [ -z "$$WANDB_API_KEY" ]; then \
	    echo "WANDB_API_KEY is not set. Add it to .env before training." >&2; \
	    exit 1; \
	  fi; \
	  WANDB_API_KEY="$$WANDB_API_KEY" $(UV) run python scripts/train_logreg_toy.py $(ARGS)

## evaluate-logreg      Evaluate a toy logistic regression run.
evaluate-logreg:
	@if [ -z "$$RUN_DIR" ]; then \
	  echo "RUN_DIR must reference the run directory to evaluate." >&2; \
	  exit 1; \
	fi
	$(UV) run python scripts/evaluate_logreg_toy.py --run-dir "$$RUN_DIR" $(ARGS)

## publish-logreg       Package a toy logistic regression run for sharing.
publish-logreg:
	@if [ -z "$$RUN_DIR" ]; then \
	  echo "RUN_DIR must reference the run directory to publish." >&2; \
	  exit 1; \
	fi
	@mkdir -p artefacts/published/logreg-toy
	@tar czf artefacts/published/logreg-toy/$$(basename "$$RUN_DIR").tar.gz -C "$$(dirname "$$RUN_DIR")" "$$(basename "$$RUN_DIR")"
