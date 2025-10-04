UV ?= uv
JAX_ENABLE_X64 ?= 1
export JAX_ENABLE_X64

.PHONY: help setup format lint typecheck test bench profile profile-line ci clean

help:
	@echo "Common development commands:"
	@echo "  make setup     # install the package with development dependencies"
	@echo "  make format    # format code with ruff"
	@echo "  make lint      # lint code with ruff"
	@echo "  make typecheck # run pyright static analysis"
	@echo "  make test      # run pytest test suite"
	@echo "  make bench     # run pytest benchmarks in tests/performance"
	@echo "  make profile   # run pytest with cProfile on performance suite"
	@echo "  make profile-line # run pytest with line_profiler on the fast kernel"
	@echo "  make ci        # run the CI command set"

setup:
	$(UV) sync --extra dev

format:
	$(UV) run ruff format .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run pyright

test:
	$(UV) run pytest

bench:
	$(UV) run pytest tests/performance --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks

profile:
	$(UV) run pytest tests/performance --profile

profile-line:
	$(UV) run pytest tests/performance --line-profile viterbo.symplectic.capacity_fast.compute_ehz_capacity_fast

ci:
	$(UV) run python scripts/check_waivers.py
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run pyright
	$(UV) run pytest

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .pyright_cache build dist *.egg-info
