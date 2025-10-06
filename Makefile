UV ?= uv
JAX_ENABLE_X64 ?= 1
export JAX_ENABLE_X64

SMOKE_TEST_TIMEOUT ?= 10
SMOKE_SUITE_TIMEOUT ?= 60
TESTMON_CACHE ?= .testmondata

.PHONY: help setup format lint lint-fast typecheck typecheck-fast test test-deep test-longhaul test-all test-incremental bench bench-deep bench-longhaul profile profile-line coverage precommit-fast precommit-slow precommit docs-build docs-serve docs-check-links ci ci-weekly clean clean-artefacts train-logreg evaluate-logreg publish-logreg

help:
	@echo "Common development commands:"
	@echo "  make setup     # install the package with development dependencies"
	@echo "  make format    # format code with ruff"
	@echo "  make lint      # lint code with ruff"
	@echo "  make lint-fast # lint essentials (E/F/B006/B008 only)"
	@echo "  make typecheck # run pyright static analysis"
	@echo "  make typecheck-fast # quick pyright verifytypes sweep"
	@echo "  make test      # run smoke-tier pytest suite with enforced timeouts"
	@echo "  make test-deep # run smoke+deep pytest tiers"
	@echo "  make test-longhaul # run longhaul pytest tier (manual)"
	@echo "  make test-all  # run smoke, deep, and longhaul sequentially"
	@echo "  make test-incremental # run smoke tier using pytest-testmon cache"
	@echo "  make bench     # run smoke-tier benchmarks"
	@echo "  make bench-deep # run deep-tier benchmarks"
	@echo "  make bench-longhaul # run longhaul benchmarks (manual)"
	@echo "  make profile   # profile deep-tier benchmarks"
	@echo "  make profile-line # line-profile deep-tier kernel"
	@echo "  make coverage  # run smoke-tier tests with coverage reports"
	@echo "  make precommit-fast # lint essentials and incremental smoke tests"
	@echo "  make precommit-slow # full format/lint/typecheck/smoke"
	@echo "  make docs-build  # build MkDocs site with strict checks"
	@echo "  make docs-serve  # serve MkDocs locally on :8000"
	@echo "  make docs-check-links # build with htmlproofer link checks"
	@echo "  make ci        # run the CI command set"
	@echo "  make ci-weekly # run CI plus longhaul tiers"
	@echo "  make clean     # remove cached tooling outputs"
	@echo "  make clean-artefacts # remove benchmarking, profiling, and coverage artefacts"

setup:
	$(UV) sync --extra dev

format:
	$(UV) run ruff format .
	npx prettier --write "README.md" "docs/**/*.{md,mdx}" "progress-reports/**/*.md" "**/*.{yml,yaml,json}" "!node_modules/**" "!.benchmarks/**" "!site/**"

lint:
	$(UV) run ruff check .
	npx prettier --check "README.md" "docs/**/*.{md,mdx}" "progress-reports/**/*.md" "**/*.{yml,yaml,json}" "!node_modules/**" "!.benchmarks/**" "!site/**"

lint-fast:
	$(UV) run ruff check src tests --select E9 --select F --select B006 --select B008

typecheck:
	$(UV) run pyright

typecheck-fast:
	$(UV) run pyright --verifytypes viterbo

test:
	$(UV) run pytest -m "not deep and not longhaul" --timeout=$(SMOKE_TEST_TIMEOUT) --timeout-method=thread

test-deep:
	$(UV) run pytest -m "not longhaul"

test-longhaul:
	$(UV) run pytest -m "longhaul"

test-all: test test-deep test-longhaul

test-incremental:
	$(UV) run pytest --testmon --testmondata $(TESTMON_CACHE) --maxfail=1 -m "not deep and not longhaul" --timeout=$(SMOKE_TEST_TIMEOUT) --timeout-method=thread

bench:
	$(UV) run pytest tests/performance -m "smoke" --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks

bench-deep:
	$(UV) run pytest tests/performance -m "deep" --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks

bench-longhaul:
	$(UV) run pytest tests/performance -m "longhaul" --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks

profile:
	@mkdir -p .profiles
	$(UV) run pytest tests/performance -m "deep" --profile --profile-svg --pstats-dir=.profiles

profile-line:
	@mkdir -p .profiles
	$(UV) run pytest tests/performance -m "deep" --line-profile viterbo.symplectic.capacity.facet_normals.fast.compute_ehz_capacity_fast --pstats-dir=.profiles

coverage:
	$(UV) run pytest -m "not deep and not longhaul" --timeout=$(SMOKE_TEST_TIMEOUT) --timeout-method=thread --cov=src/viterbo --cov-report=term-missing --cov-report=html --cov-report=xml

precommit-fast: lint-fast test-incremental

precommit-slow: format lint typecheck test

precommit: precommit-slow

ci:
	$(UV) run python scripts/check_waivers.py
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	npx prettier --check "README.md" "docs/**/*.{md,mdx}" "progress-reports/**/*.md" "**/*.{yml,yaml,json}" "!node_modules/**" "!.benchmarks/**" "!site/**"
	$(UV) run pyright
	$(UV) run pytest -m "not deep and not longhaul" --timeout=$(SMOKE_TEST_TIMEOUT) --timeout-method=thread --cov=src/viterbo --cov-report=term-missing --cov-report=xml --cov-report=html

ci-weekly: ci test-longhaul bench bench-longhaul
docs-build:
	$(UV) run mkdocs build --strict

docs-serve:
	$(UV) run mkdocs serve -a 0.0.0.0:8000

docs-check-links:
	$(UV) run mkdocs build --strict

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .pyright_cache build dist *.egg-info

clean-artefacts:
	rm -rf .benchmarks .profiles htmlcov .coverage* .artefacts

train-logreg:
	@set -a; \
	  if [ -f .env ]; then . ./.env; fi; \
	  set +a; \
	  if [ -z "$$WANDB_API_KEY" ]; then \
	    echo "WANDB_API_KEY is not set. Add it to .env before training." >&2; \
	    exit 1; \
	  fi; \
	  WANDB_API_KEY="$$WANDB_API_KEY" $(UV) run python scripts/train_logreg_toy.py $(ARGS)

evaluate-logreg:
	@if [ -z "$$RUN_DIR" ]; then \
	  echo "RUN_DIR must reference the run directory to evaluate." >&2; \
	  exit 1; \
	fi
	$(UV) run python scripts/evaluate_logreg_toy.py --run-dir "$$RUN_DIR" $(ARGS)

publish-logreg:
	@if [ -z "$$RUN_DIR" ]; then \
	  echo "RUN_DIR must reference the run directory to publish." >&2; \
	  exit 1; \
	fi
	@mkdir -p artefacts/published/logreg-toy
	@tar czf artefacts/published/logreg-toy/$$(basename "$$RUN_DIR").tar.gz -C "$$(dirname "$$RUN_DIR")" "$$(basename "$$RUN_DIR")"
