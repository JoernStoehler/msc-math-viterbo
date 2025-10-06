UV ?= uv
JAX_ENABLE_X64 ?= 1
export JAX_ENABLE_X64

PYTEST := $(UV) run pytest
PYRIGHT := $(UV) run pyright
RUFF := $(UV) run ruff
PRETTIER := npx prettier
MKDOCS := $(UV) run mkdocs

SMOKE_TEST_TIMEOUT ?= 10
TESTMON_CACHE ?= .testmondata

PYTEST_SMOKE_FLAGS := -m "not deep and not longhaul" --timeout=$(SMOKE_TEST_TIMEOUT) --timeout-method=thread
PYTEST_DEEP_FLAGS := -m "not longhaul"
PYTEST_LONGHAUL_FLAGS := -m "longhaul"
BENCHMARK_STORAGE := .benchmarks
BENCH_FLAGS := --benchmark-only --benchmark-autosave --benchmark-storage=$(BENCHMARK_STORAGE)
PROFILES_DIR := .profiles
PRETTIER_PATTERNS := "README.md" "docs/**/*.{md,mdx}" "progress-reports/**/*.md" "**/*.{yml,yaml,json}"
PRETTIER_IGNORES := "!node_modules/**" "!.benchmarks/**" "!site/**"

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
	@echo "Common development commands:"
	@grep -E '^## [a-z]' $(MAKEFILE_LIST) | sed 's/^## //'

#------------------------------------------------------------------------------
# Core QA workflow
#------------------------------------------------------------------------------
## setup                Install the package with development dependencies.
setup:
	$(UV) sync --extra dev

## format               Run Ruff formatter and Prettier writes.
format:
	$(RUFF) format .
	$(PRETTIER) --write $(PRETTIER_PATTERNS) $(PRETTIER_IGNORES)

## lint                 Full Ruff lint and Prettier validation.
lint:
	$(RUFF) check .
	$(PRETTIER) --check $(PRETTIER_PATTERNS) $(PRETTIER_IGNORES)

## lint-fast            Minimal Ruff diagnostics (E/F/B006/B008).
lint-fast:
	$(RUFF) check src tests --select E9 --select F --select B006 --select B008

## typecheck            Run strict Pyright analysis.
typecheck:
	$(PYRIGHT)

## typecheck-fast       Quick verifytypes sweep for public API.
typecheck-fast:
	$(PYRIGHT) --verifytypes viterbo

#------------------------------------------------------------------------------
# Pytest tiers
#------------------------------------------------------------------------------
## test                 Smoke-tier pytest with enforced timeouts.
test:
	$(PYTEST) $(PYTEST_SMOKE_FLAGS)

## test-deep            Smoke + deep tiers.
test-deep:
	$(PYTEST) $(PYTEST_DEEP_FLAGS)

## test-longhaul        Longhaul pytest tier (manual).
test-longhaul:
	$(PYTEST) $(PYTEST_LONGHAUL_FLAGS)

## test-all             Run smoke, deep, and longhaul sequentially.
test-all: test test-deep test-longhaul

## test-incremental     Smoke tier with pytest-testmon cache.
test-incremental:
	$(PYTEST) --testmon --testmondata $(TESTMON_CACHE) --maxfail=1 $(PYTEST_SMOKE_FLAGS)

#------------------------------------------------------------------------------
# Benchmarks and profiling
#------------------------------------------------------------------------------
## bench                Smoke-tier benchmarks.
bench:
	$(PYTEST) tests/performance -m "smoke" $(BENCH_FLAGS)

## bench-deep           Deep-tier benchmarks.
bench-deep:
	$(PYTEST) tests/performance -m "deep" $(BENCH_FLAGS)

## bench-longhaul       Longhaul benchmarks (manual).
bench-longhaul:
	$(PYTEST) tests/performance -m "longhaul" $(BENCH_FLAGS)

## profile              Profile deep-tier benchmarks (callgrind + svg).
profile:
	@mkdir -p $(PROFILES_DIR)
	$(PYTEST) tests/performance -m "deep" --profile --profile-svg --pstats-dir=$(PROFILES_DIR)

## profile-line         Line profile the fast EHZ kernel.
profile-line:
	@mkdir -p $(PROFILES_DIR)
	$(PYTEST) tests/performance -m "deep" --line-profile viterbo.symplectic.capacity.facet_normals.fast.compute_ehz_capacity_fast --pstats-dir=$(PROFILES_DIR)

#------------------------------------------------------------------------------
# Coverage and pre-commit gates
#------------------------------------------------------------------------------
## coverage             Smoke-tier tests with coverage reports.
coverage:
	$(PYTEST) $(PYTEST_SMOKE_FLAGS) --cov=src/viterbo --cov-report=term-missing --cov-report=html --cov-report=xml

## precommit-fast       Lint essentials and incremental smoke tests.
precommit-fast: lint-fast test-incremental

## precommit-slow       Full formatter, lint, typecheck, and smoke tests.
precommit-slow: format lint typecheck test

## precommit            Alias for precommit-slow.
precommit: precommit-slow

#------------------------------------------------------------------------------
# Continuous integration presets
#------------------------------------------------------------------------------
## ci                   Run the CI command set locally.
ci:
	$(UV) run python scripts/check_waivers.py
	$(RUFF) format --check .
	$(RUFF) check .
	$(PRETTIER) --check $(PRETTIER_PATTERNS) $(PRETTIER_IGNORES)
	$(PYRIGHT)
	$(PYTEST) $(PYTEST_SMOKE_FLAGS) --cov=src/viterbo --cov-report=term-missing --cov-report=xml --cov-report=html

## ci-weekly            CI plus longhaul tiers and benchmarks.
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
