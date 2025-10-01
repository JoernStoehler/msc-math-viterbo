UV ?= uv

.PHONY: help setup format lint typecheck test ci clean

help:
@echo "Common development commands:"
@echo "  make setup     # install the package with development dependencies"
@echo "  make format    # format code with ruff"
@echo "  make lint      # lint code with ruff"
@echo "  make typecheck # run pyright static analysis"
@echo "  make test      # run pytest test suite"
@echo "  make ci        # run the CI command set"

setup:
$(UV) pip install --system -e .[dev]

format:
ruff format .

lint:
ruff check .

typecheck:
pyright

test:
pytest

ci:
ruff format --check .
ruff check .
pyright
pytest

clean:
rm -rf .pytest_cache .ruff_cache .pyright .pyright_cache build dist *.egg-info
