"""Benchmark harness for the MILP reference solver."""

from __future__ import annotations

import math

import pytest

from viterbo.geometry.polytopes import simplex_with_uniform_weights
from viterbo.symplectic.capacity.milp.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference_milp,
)


@pytest.fixture(scope="module")
def benchmark_polytope():
    return simplex_with_uniform_weights(4, name="milp-benchmark-simplex")


@pytest.mark.goal_performance
def test_reference_solver_benchmark(benchmark, benchmark_polytope) -> None:
    """Benchmark MILP reference solver runtime on a small instance."""

    B, c = benchmark_polytope.halfspace_data()

    result = benchmark(lambda: compute_ehz_capacity_reference_milp(B, c))
    assert math.isfinite(result.upper_bound)
