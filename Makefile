UV ?= uv

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
	$(UV) sync --extra dev --system

format:
	ruff format .

lint:
	ruff check .

typecheck:
	pyright

test:
	pytest

bench:
	pytest tests/performance --benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks

profile:
	pytest tests/performance --profile

profile-line:
	pytest tests/performance --line-profile viterbo.symplectic.capacity_fast.compute_ehz_capacity_fast

ci:
	ruff format --check .
	ruff check .
	pyright
	pytest

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .pyright_cache build dist *.egg-info
